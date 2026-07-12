(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.WinnerClient = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const REQUIRED_LANDMARK_INDICES = [
    1, 13, 14, 33, 61, 133, 144, 152, 153, 158,
    160, 263, 291, 362, 373, 380, 385, 387, 468, 473,
  ];
  const VALID_FPS = new Set([10, 15, 20, 30]);
  const TRANSIENT_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

  function numberOrNull(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function fixed(value, digits) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : '-';
  }

  function clamp01(value) {
    return Math.min(1, Math.max(0, value));
  }

  function extractRequiredLandmarks(faceLandmarks) {
    if (!faceLandmarks) return null;
    const out = {};
    for (const index of REQUIRED_LANDMARK_INDICES) {
      const point = faceLandmarks[index] || faceLandmarks[String(index)];
      if (!point) return null;
      const x = numberOrNull(point.x);
      const y = numberOrNull(point.y);
      if (x === null || y === null) return null;
      out[String(index)] = [clamp01(x), clamp01(y)];
    }
    return out;
  }

  function createFramePacket({ seq, timestampMs, width, height, faceLandmarks }) {
    const landmarks = extractRequiredLandmarks(faceLandmarks);
    return {
      seq,
      timestamp_ms: Number(timestampMs) || 0,
      width: Math.max(0, Math.round(Number(width) || 0)),
      height: Math.max(0, Math.round(Number(height) || 0)),
      face_detected: Boolean(landmarks),
      landmarks: landmarks || undefined,
    };
  }

  class LandmarkBatcher {
    constructor(options) {
      const opts = options || {};
      this.maxFrames = opts.maxFrames || 4;
      this.flushMs = opts.flushMs || 200;
      this.now = opts.now || (() => Date.now());
      this.schedule = opts.schedule || ((fn, delay) => setTimeout(fn, delay));
      this.clear = opts.clear || ((id) => clearTimeout(id));
      this.onFlush = opts.onFlush || (() => {});
      this.canFlush = opts.canFlush || (() => true);
      this.retryMs = opts.retryMs || 25;
      this.queue = [];
      this.firstQueuedAt = 0;
      this.timer = null;
    }

    canAccept() {
      return this.queue.length < this.maxFrames;
    }

    enqueue(frame) {
      if (!this.canAccept()) return false;
      if (!this.queue.length) {
        this.firstQueuedAt = this.now();
        this._scheduleFlush(this.flushMs);
      }
      this.queue.push(frame);
      if (this.queue.length >= this.maxFrames) this.flush();
      return true;
    }

    maybeFlush(force) {
      if (this.queue.length && (force || this.now() - this.firstQueuedAt >= this.flushMs)) {
        this.flush();
      }
    }

    flush() {
      if (!this.queue.length) return null;
      if (!this.canFlush()) {
        this._scheduleFlush(this.retryMs);
        return null;
      }
      if (this.timer !== null) {
        this.clear(this.timer);
        this.timer = null;
      }
      const frames = this.queue.splice(0, this.maxFrames);
      this.firstQueuedAt = this.queue.length ? this.now() : 0;
      if (this.queue.length) this._scheduleFlush(this.flushMs);
      this.onFlush(frames);
      return frames;
    }

    _scheduleFlush(delay) {
      if (this.timer !== null) return;
      this.timer = this.schedule(() => {
        this.timer = null;
        this.flush();
      }, delay);
    }

    reset() {
      if (this.timer !== null) this.clear(this.timer);
      this.queue = [];
      this.firstQueuedAt = 0;
      this.timer = null;
    }
  }

  class ActiveSessionClock {
    constructor(options) {
      const opts = options || {};
      this.now = opts.now || (() => (typeof performance !== 'undefined' ? performance.now() : Date.now()));
      this.reset();
    }

    reset() {
      this.elapsedMs = 0;
      this.startedAtMs = null;
      this.active = false;
    }

    start() {
      this.elapsedMs = 0;
      this.startedAtMs = this.now();
      this.active = true;
    }

    resume() {
      if (this.active) return;
      this.startedAtMs = this.now();
      this.active = true;
    }

    pause() {
      if (!this.active) return;
      this.elapsedMs += Math.max(0, this.now() - this.startedAtMs);
      this.startedAtMs = null;
      this.active = false;
    }

    activeMs() {
      if (!this.active) return this.elapsedMs;
      return this.elapsedMs + Math.max(0, this.now() - this.startedAtMs);
    }

    timestampMs(sourceMode, mediaElement) {
      if (sourceMode === 'file') {
        const seconds = Number(mediaElement && mediaElement.currentTime);
        return Number.isFinite(seconds) ? Math.max(0, seconds * 1000) : 0;
      }
      return this.activeMs();
    }
  }

  class RunGenerationGuard {
    constructor() {
      this.generation = 0;
    }

    capture() {
      return this.generation;
    }

    invalidate() {
      this.generation += 1;
      return this.generation;
    }

    isCurrent(generation) {
      return Number(generation) === this.generation;
    }
  }
  class WinnerApiError extends Error {
    constructor(message, status, body, options) {
      super(message);
      this.name = 'WinnerApiError';
      this.status = status || 0;
      this.body = body || null;
      this.cancelled = Boolean(options && options.cancelled);
    }
  }

  class WinnerApiClient {
    constructor(options) {
      const opts = options || {};
      this.baseUrl = opts.baseUrl || '';
      this.fetchImpl = opts.fetchImpl || (typeof fetch !== 'undefined' ? fetch.bind(globalThis) : null);
      this.sourceMode = opts.sourceMode || 'camera';
      this.targetFps = VALID_FPS.has(Number(opts.targetFps)) ? Number(opts.targetFps) : 20;
      this.maxRetries = opts.maxRetries === undefined ? 1 : opts.maxRetries;
      this.retryDelayMs = opts.retryDelayMs === undefined ? 180 : opts.retryDelayMs;
      this.requestTimeoutMs = opts.requestTimeoutMs === undefined ? 2500 : opts.requestTimeoutMs;
      this.maxFrameGapMs = opts.maxFrameGapMs === undefined ? 3000 : Number(opts.maxFrameGapMs);
      this.sleep = opts.sleep || ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
      this.onSessionReset = opts.onSessionReset || (() => {});
      this.sessionId = null;
      this.nextBatchSeq = 1;
      this.lastFrameTimestampMs = null;
      this.activeControllers = new Set();
      this.lifecycleAbortedControllers = new WeakSet();
      this.cancelGeneration = 0;
    }

    setMode(sourceMode, targetFps) {
      this.sourceMode = sourceMode === 'file' ? 'file' : 'camera';
      if (VALID_FPS.has(Number(targetFps))) this.targetFps = Number(targetFps);
    }

    async requestSession() {
      return this._request('/api/v1/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_mode: this.sourceMode, target_fps: this.targetFps }),
      });
    }

    activateSession(body, reason) {
      if (!body || !body.session_id) throw new Error('Session response is missing session_id');
      this.sessionId = body.session_id;
      this.nextBatchSeq = 1;
      this.lastFrameTimestampMs = null;
      this.onSessionReset({ reason: reason || 'created', response: body });
      return body;
    }

    async createSession(reason) {
      return this.activateSession(await this.requestSession(), reason);
    }

    async resetSession() {
      if (!this.sessionId) return null;
      try {
        return await this._request('/api/v1/sessions/' + encodeURIComponent(this.sessionId) + '/reset', {
          method: 'POST',
        });
      } catch (error) {
        if (error && error.status === 404) {
          return this.createSession('lost');
        }
        throw error;
      }
    }

    async deleteSessionId(id, options) {
      if (!id) return null;
      const requireConfirmation = Boolean(options && options.requireConfirmation);
      const clearLocal = () => {
        if (this.sessionId !== id) return;
        this.sessionId = null;
        this.nextBatchSeq = 1;
        this.lastFrameTimestampMs = null;
      };
      try {
        const body = await this._request('/api/v1/sessions/' + encodeURIComponent(id), { method: 'DELETE' });
        clearLocal();
        return body;
      } catch (error) {
        if (requireConfirmation && (!error || error.status !== 404)) throw error;
        clearLocal();
        return null;
      }
    }

    async deleteSession() {
      return this.deleteSessionId(this.sessionId);
    }

    async deleteSessionKeepalive() {
      if (!this.sessionId) return null;
      const id = this.sessionId;
      this.sessionId = null;
      this.nextBatchSeq = 1;
      this.lastFrameTimestampMs = null;
      this._requireFetch();
      try {
        return await this.fetchImpl(this.baseUrl + '/api/v1/sessions/' + encodeURIComponent(id), {
          method: 'DELETE',
          keepalive: true,
        });
      } catch (_) {
        return null;
      }
    }

    async sendBatch(frames) {
      if (!frames || !frames.length) return null;
      if (frames.length > 4) throw new Error('Frame batch exceeds max 4');
      if (!this.sessionId) await this.createSession('created');
      if (this._hasTimestampDiscontinuity(frames)) {
        await this.deleteSessionId(this.sessionId, { requireConfirmation: true });
        await this.createSession('gap');
        return null;
      }
      const batchSeq = this.nextBatchSeq;
      const payload = { batch_seq: batchSeq, frames };
      try {
        const decision = await this._requestWithRetry(
          '/api/v1/sessions/' + encodeURIComponent(this.sessionId) + '/frames',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          },
        );
        this.nextBatchSeq += 1;
        this.lastFrameTimestampMs = Number(frames[frames.length - 1].timestamp_ms);
        return decision;
      } catch (error) {
        if (error && error.status === 404) {
          await this.createSession('lost');
          return null;
        }
        throw error;
      }
    }

    async _requestWithRetry(path, options) {
      let lastError = null;
      const cancelGeneration = this.cancelGeneration;
      for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
        try {
          return await this._request(path, options);
        } catch (error) {
          if (this.cancelGeneration !== cancelGeneration) throw this._cancelledError();
          lastError = error;
          if (!this._isTransient(error) || attempt >= this.maxRetries) break;
          await this.sleep(this.retryDelayMs * (attempt + 1));
          if (this.cancelGeneration !== cancelGeneration) throw this._cancelledError();
        }
      }
      throw lastError;
    }

    async _request(path, options) {
      this._requireFetch();
      const requestOptions = { ...(options || {}) };
      let controller = null;
      let timeoutId = null;
      if (!requestOptions.signal && this.requestTimeoutMs > 0 && typeof AbortController !== 'undefined') {
        controller = new AbortController();
        requestOptions.signal = controller.signal;
        this.activeControllers.add(controller);
        timeoutId = setTimeout(() => controller.abort(), this.requestTimeoutMs);
      }
      let response;
      let body;
      try {
        response = await this.fetchImpl(this.baseUrl + path, requestOptions);
        body = await this._readJson(response);
        if (controller && controller.signal.aborted) throw new Error('aborted');
      } catch (error) {
        const cancelled = Boolean(controller && this.lifecycleAbortedControllers.has(controller));
        const message = cancelled
          ? 'SERVER REQUEST CANCELLED'
          : (controller && controller.signal.aborted ? 'SERVER REQUEST TIMEOUT' : (error.message || 'SERVER UNAVAILABLE'));
        throw new WinnerApiError(message, 0, null, { cancelled });
      } finally {
        if (timeoutId !== null) clearTimeout(timeoutId);
        if (controller) this.activeControllers.delete(controller);
      }
      if (!response.ok) {
        throw new WinnerApiError(body.error || 'SERVER UNAVAILABLE', response.status, body);
      }
      return body;
    }

    _hasTimestampDiscontinuity(frames) {
      let previous = this.lastFrameTimestampMs;
      for (const frame of frames) {
        const current = Number(frame && frame.timestamp_ms);
        if (Number.isFinite(previous) && Number.isFinite(current) && current - previous > this.maxFrameGapMs) {
          return true;
        }
        previous = current;
      }
      return false;
    }

    abortActiveRequests() {
      this.cancelGeneration += 1;
      for (const controller of this.activeControllers) {
        this.lifecycleAbortedControllers.add(controller);
        controller.abort();
      }
      this.activeControllers.clear();
    }

    _cancelledError() {
      return new WinnerApiError('SERVER REQUEST CANCELLED', 0, null, { cancelled: true });
    }

    async _readJson(response) {
      try {
        return await response.json();
      } catch (_) {
        return {};
      }
    }

    _isTransient(error) {
      if (error && error.cancelled) return false;
      return !error || error.status === 0 || TRANSIENT_STATUS.has(error.status);
    }

    _requireFetch() {
      if (!this.fetchImpl) throw new Error('Fetch API is unavailable');
    }
  }

  class AudioCommandState {
    constructor() {
      this.reset();
    }

    consume(commands) {
      let doubleCount = 0;
      for (const raw of commands || []) {
        const command = String(raw || 'none');
        if (command === 'double') doubleCount += 1;
        else if (command === 'continuous_start') this.continuousRequested = true;
        else if (command === 'continuous_stop') this.continuousRequested = false;
      }
      return { doubleCount, continuousRequested: this.continuousRequested };
    }

    reset() {
      this.continuousRequested = false;
    }
  }

  function canBeginLifecycle(starting, resetting) {
    return !Boolean(starting) && !Boolean(resetting);
  }

  function audioCommandsForRender(decision, freshServerResponse) {
    if (!freshServerResponse) return [];
    const d = decision || {};
    if (Array.isArray(d.audio_commands) && d.audio_commands.length) {
      return d.audio_commands.slice();
    }
    return [d.audio_command || 'none'];
  }

  function canCaptureFrame(running, resetting) {
    return Boolean(running) && !Boolean(resetting);
  }

  function pageLifecycleResetState() {
    return {
      running: false,
      paused: false,
      processing: false,
      resetting: false,
      starting: false,
      sendInFlight: false,
    };
  }
  function normalizeState(decision) {
    if (!decision) return 'STOPPED';
    return String(decision.state || decision.label || 'ALERT').toUpperCase();
  }

  function serializeHistoryEntry(decision, at) {
    const d = decision || {};
    const m = d.metrics || {};
    const time = at || new Date();
    return {
      time: formatDateTime(time),
      iso_time: time.toISOString(),
      state: normalizeState(d),
      label: d.label || normalizeState(d),
      probability: fixed(d.probability, 4),
      threshold: fixed(d.threshold, 4),
      guard: d.hybrid_guard || '-',
      profile: d.profile || '-',
      model_hash: d.model_hash || '-',
      visual_alert_mode: d.visual_alert_mode || '-',
      audio_command: d.audio_command || 'none',
      runtime_alert_semantic: d.runtime_alert_semantic || '-',
      ear: fixed(m.ear, 4),
      ear_threshold: fixed(m.ear_threshold, 4),
      mar: fixed(m.mar, 4),
      mar_threshold: fixed(m.mar_threshold, 4),
      p5: fixed(m.perclos_short, 4),
      p60: fixed(m.perclos, 4),
      blink: m.blink_frequency === undefined ? '-' : m.blink_frequency,
      yawn: m.yawn_frequency === undefined ? '-' : m.yawn_frequency,
      pitch: fixed(m.pitch, 2),
      head_nod: m.head_nod_detected ? 'NOD' : '-',
      reasons: Array.isArray(d.reasons) ? d.reasons.join(';') : '',
    };
  }

  function historyToCsv(history) {
    const cols = [
      'time', 'iso_time', 'state', 'label', 'probability', 'threshold', 'guard',
      'profile', 'model_hash', 'visual_alert_mode', 'audio_command',
      'runtime_alert_semantic', 'ear', 'ear_threshold', 'mar', 'mar_threshold',
      'p5', 'p60', 'blink', 'yawn', 'pitch', 'head_nod', 'reasons',
    ];
    const esc = (value) => '"' + String(value === undefined || value === null ? '' : value).replaceAll('"', '""') + '"';
    return [cols.join(',')].concat((history || []).map((row) => cols.map((col) => esc(row[col])).join(','))).join('\n');
  }

  function escapeHtml(value) {
    return String(value === undefined || value === null ? '' : value)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  function formatDateTime(date) {
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(date.getDate())}/${pad(date.getMonth() + 1)}/${date.getFullYear()} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  return {
    REQUIRED_LANDMARK_INDICES,
    extractRequiredLandmarks,
    createFramePacket,
    LandmarkBatcher,
    ActiveSessionClock,
    WinnerApiClient,
    WinnerApiError,
    RunGenerationGuard,
    AudioCommandState,
    canBeginLifecycle,
    audioCommandsForRender,
    canCaptureFrame,
    pageLifecycleResetState,
    serializeHistoryEntry,
    historyToCsv,
    escapeHtml,
    normalizeState,
  };
});
