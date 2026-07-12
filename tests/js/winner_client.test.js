'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  REQUIRED_LANDMARK_INDICES,
  extractRequiredLandmarks,
  LandmarkBatcher,
  ActiveSessionClock,
  WinnerApiClient,
  RunGenerationGuard,
  AudioCommandState,
  canBeginLifecycle,
  audioCommandsForRender,
  canCaptureFrame,
  pageLifecycleResetState,
  serializeHistoryEntry,
  historyToCsv,
  escapeHtml,
} = require('../../static/winner_client.js');

function makeLandmarks() {
  const lms = Array.from({ length: 500 }, (_, index) => ({
    x: index / 1000,
    y: index / 2000,
    z: -index / 3000,
    extra: 'ignored',
  }));
  return lms;
}

test('extractRequiredLandmarks sends only the agreed normalized indices', () => {
  const landmarks = extractRequiredLandmarks(makeLandmarks());
  assert.deepEqual(Object.keys(landmarks).map(Number), REQUIRED_LANDMARK_INDICES);
  assert.equal(Object.keys(landmarks).length, 20);
  assert.deepEqual(landmarks['1'], [0.001, 0.0005]);
  assert.equal(landmarks['2'], undefined);
  assert.equal(landmarks['33'].extra, undefined);
});

test('extractRequiredLandmarks returns null when a required point is missing', () => {
  const lms = makeLandmarks();
  delete lms[473];
  assert.equal(extractRequiredLandmarks(lms), null);
});

test('extractRequiredLandmarks clamps x and y and never transmits z', () => {
  const lms = makeLandmarks();
  lms[1] = { x: -0.25, y: 1.25, z: 2.5 };

  const landmarks = extractRequiredLandmarks(lms);
  assert.deepEqual(landmarks['1'], [0, 1]);
});

test('LandmarkBatcher flushes at max four frames and after 200ms', () => {
  const flushed = [];
  let now = 0;
  const batcher = new LandmarkBatcher({
    maxFrames: 4,
    flushMs: 200,
    now: () => now,
    schedule: (fn, delay) => ({ fn, delay }),
    clear: () => {},
    onFlush: (frames) => flushed.push(frames),
  });

  batcher.enqueue({ seq: 1 });
  batcher.enqueue({ seq: 2 });
  batcher.enqueue({ seq: 3 });
  assert.equal(flushed.length, 0);
  batcher.enqueue({ seq: 4 });
  assert.deepEqual(flushed.map((batch) => batch.map((frame) => frame.seq)), [[1, 2, 3, 4]]);

  batcher.enqueue({ seq: 5 });
  now = 199;
  batcher.maybeFlush();
  assert.equal(flushed.length, 1);
  now = 200;
  batcher.maybeFlush();
  assert.deepEqual(flushed.map((batch) => batch.map((frame) => frame.seq)), [[1, 2, 3, 4], [5]]);
});

test('LandmarkBatcher keeps at most one batch while the sender is busy', () => {
  const flushed = [];
  let senderReady = false;
  const batcher = new LandmarkBatcher({
    maxFrames: 4,
    flushMs: 200,
    canFlush: () => senderReady,
    schedule: () => null,
    clear: () => {},
    onFlush: (frames) => flushed.push(frames),
  });

  assert.equal(batcher.enqueue({ seq: 1 }), true);
  assert.equal(batcher.enqueue({ seq: 2 }), true);
  assert.equal(batcher.enqueue({ seq: 3 }), true);
  assert.equal(batcher.enqueue({ seq: 4 }), true);
  assert.equal(batcher.enqueue({ seq: 5 }), false);
  assert.equal(batcher.queue.length, 4);
  assert.equal(flushed.length, 0);

  senderReady = true;
  batcher.maybeFlush(true);
  assert.deepEqual(flushed.map((batch) => batch.map((frame) => frame.seq)), [[1, 2, 3, 4]]);
  assert.equal(batcher.queue.length, 0);
});

test('WinnerApiClient can request and discard a detached session without clobbering active state', async () => {
  const deleted = [];
  const client = new WinnerApiClient({
    fetchImpl: async (url, options) => {
      if (options.method === 'DELETE') {
        deleted.push(url);
        return { ok: true, status: 200, json: async () => ({ ok: true }) };
      }
      return { ok: true, status: 201, json: async () => ({ session_id: 'candidate-session' }) };
    },
  });

  const candidate = await client.requestSession();
  assert.equal(client.sessionId, null);
  await client.deleteSessionId(candidate.session_id);
  assert.equal(client.sessionId, null);
  assert.equal(deleted.length, 1);

  client.activateSession(candidate, 'created');
  assert.equal(client.sessionId, 'candidate-session');
});

test('AudioCommandState preserves continuous intent across mute and clears it on stop', () => {
  const state = new AudioCommandState();
  assert.deepEqual(state.consume(['continuous_start']), { doubleCount: 0, continuousRequested: true });
  assert.equal(state.continuousRequested, true);
  assert.deepEqual(state.consume(['double']), { doubleCount: 1, continuousRequested: true });
  assert.deepEqual(state.consume(['continuous_stop']), { doubleCount: 0, continuousRequested: false });
  state.consume(['continuous_start']);
  state.reset();
  assert.equal(state.continuousRequested, false);
});

test('lifecycle start is blocked while another start or reset is active', () => {
  assert.equal(canBeginLifecycle(false, false), true);
  assert.equal(canBeginLifecycle(true, false), false);
  assert.equal(canBeginLifecycle(false, true), false);
});

test('WinnerApiClient retries transient failure without changing batch_seq', async () => {
  const calls = [];
  const fetchImpl = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    if (calls.length === 1) {
      throw new Error('network down');
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({ state: 'ALERT', audio_command: 'none' }),
    };
  };
  const client = new WinnerApiClient({
    fetchImpl,
    retryDelayMs: 0,
    maxRetries: 1,
    sleep: async () => {},
  });
  client.sessionId = 'session-1';

  const decision = await client.sendBatch([{ seq: 1, timestamp_ms: 10 }]);
  assert.equal(decision.state, 'ALERT');
  assert.equal(calls.length, 2);
  assert.equal(calls[0].body.batch_seq, 1);
  assert.equal(calls[1].body.batch_seq, 1);
  assert.equal(client.nextBatchSeq, 2);
});

test('WinnerApiClient aborts a stalled request at the configured deadline', async () => {
  const client = new WinnerApiClient({
    fetchImpl: async (_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new Error('aborted')));
    }),
    requestTimeoutMs: 10,
    maxRetries: 0,
  });
  client.sessionId = 'session-1';

  await assert.rejects(() => client.sendBatch([{ seq: 1, timestamp_ms: 10 }]), /timeout/i);
});

test('WinnerApiClient keeps the deadline active while reading the response body', { timeout: 100 }, async () => {
  const client = new WinnerApiClient({
    fetchImpl: async (_url, options) => ({
      ok: true,
      status: 200,
      json: async () => new Promise((_resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new Error('aborted body')));
      }),
    }),
    requestTimeoutMs: 10,
    maxRetries: 0,
  });
  client.sessionId = 'session-1';

  await assert.rejects(() => client.sendBatch([{ seq: 1, timestamp_ms: 10 }]), /timeout/i);
});

test('WinnerApiClient preserves the session across a transient 1.5 second capture stall', async () => {
  const calls = [];
  const resets = [];
  const client = new WinnerApiClient({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith('/api/v1/sessions')) {
        return { ok: true, status: 201, json: async () => ({ session_id: 'session-2' }) };
      }
      return { ok: true, status: 200, json: async () => ({ state: 'ALERT' }) };
    },
    onSessionReset: (event) => resets.push(event.reason),
  });
  client.sessionId = 'session-1';

  await client.sendBatch([{ seq: 1, timestamp_ms: 350 }]);
  const decision = await client.sendBatch([{ seq: 2, timestamp_ms: 1850 }]);

  assert.equal(decision.state, 'ALERT');
  assert.equal(client.sessionId, 'session-1');
  assert.equal(client.nextBatchSeq, 3);
  assert.deepEqual(resets, []);
  assert.equal(calls.filter((call) => call.url.includes('/frames')).length, 2);
});
test('WinnerApiClient resets the server session instead of sending a timestamp discontinuity', async () => {
  const calls = [];
  const resets = [];
  const client = new WinnerApiClient({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (url.endsWith('/api/v1/sessions')) {
        return { ok: true, status: 201, json: async () => ({ session_id: 'session-2' }) };
      }
      return { ok: true, status: 200, json: async () => ({ state: 'ALERT' }) };
    },
    onSessionReset: (event) => resets.push(event.reason),
  });
  client.sessionId = 'session-1';

  await client.sendBatch([{ seq: 1, timestamp_ms: 350 }]);
  const discontinuous = await client.sendBatch([{ seq: 2, timestamp_ms: 3500 }]);

  assert.equal(discontinuous, null);
  assert.equal(client.sessionId, 'session-2');
  assert.equal(client.nextBatchSeq, 1);
  assert.deepEqual(resets, ['gap']);
  assert.equal(calls.filter((call) => call.url.includes('/frames')).length, 1);
  assert.deepEqual(calls.slice(1).map((call) => [call.options.method, call.url]), [
    ['DELETE', '/api/v1/sessions/session-1'],
    ['POST', '/api/v1/sessions'],
  ]);
});

test('WinnerApiClient aborts active lifecycle requests before BFCache suspension', async () => {
  let activeSignal = null;
  let calls = 0;
  const client = new WinnerApiClient({
    fetchImpl: async (_url, options) => new Promise((_resolve, reject) => {
      calls += 1;
      activeSignal = options.signal;
      options.signal.addEventListener('abort', () => reject(new Error('aborted')));
    }),
    requestTimeoutMs: 20,
    retryDelayMs: 0,
  });
  client.sessionId = 'session-1';

  const pending = client.sendBatch([{ seq: 1, timestamp_ms: 10 }]);
  await Promise.resolve();
  client.abortActiveRequests();

  await assert.rejects(() => pending, /cancelled/i);
  assert.equal(activeSignal.aborted, true);
  assert.equal(calls, 1);
});

test('WinnerApiClient cancels a retry while it is waiting in backoff', async () => {
  let calls = 0;
  let releaseSleep;
  let markSleepStarted;
  const sleepStarted = new Promise((resolve) => { markSleepStarted = resolve; });
  const client = new WinnerApiClient({
    fetchImpl: async () => {
      calls += 1;
      if (calls === 1) throw new Error('temporary network failure');
      return { ok: true, status: 200, json: async () => ({ state: 'ALERT' }) };
    },
    sleep: async () => new Promise((resolve) => {
      releaseSleep = resolve;
      markSleepStarted();
    }),
  });
  client.sessionId = 'session-1';

  const pending = client.sendBatch([{ seq: 1, timestamp_ms: 10 }]);
  await sleepStarted;
  client.abortActiveRequests();
  releaseSleep();

  await assert.rejects(() => pending, /cancelled/i);
  assert.equal(calls, 1);
});

test('WinnerApiClient does not replace a gap session when DELETE is unconfirmed', async () => {
  const calls = [];
  const client = new WinnerApiClient({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (options.method === 'DELETE') {
        return { ok: false, status: 503, json: async () => ({ error: 'delete unavailable' }) };
      }
      return { ok: true, status: 200, json: async () => ({ state: 'ALERT' }) };
    },
    maxRetries: 0,
  });
  client.sessionId = 'session-1';

  await client.sendBatch([{ seq: 1, timestamp_ms: 350 }]);
  await assert.rejects(
    () => client.sendBatch([{ seq: 2, timestamp_ms: 3500 }]),
    /delete unavailable/i,
  );

  assert.equal(client.sessionId, 'session-1');
  assert.equal(calls.filter((call) => call.url.endsWith('/api/v1/sessions')).length, 0);
});

test('WinnerApiClient treats DELETE 404 as confirmed absence during gap rollover', async () => {
  const calls = [];
  const client = new WinnerApiClient({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      if (options.method === 'DELETE') {
        return { ok: false, status: 404, json: async () => ({ error: 'already gone' }) };
      }
      if (url.endsWith('/api/v1/sessions')) {
        return { ok: true, status: 201, json: async () => ({ session_id: 'session-2' }) };
      }
      return { ok: true, status: 200, json: async () => ({ state: 'ALERT' }) };
    },
  });
  client.sessionId = 'session-1';

  await client.sendBatch([{ seq: 1, timestamp_ms: 350 }]);
  const decision = await client.sendBatch([{ seq: 2, timestamp_ms: 3500 }]);

  assert.equal(decision, null);
  assert.equal(client.sessionId, 'session-2');
  assert.equal(calls.filter((call) => call.url.endsWith('/api/v1/sessions')).length, 1);
});

test('WinnerApiClient pagehide cleanup uses keepalive and clears only its active session', async () => {
  const calls = [];
  const client = new WinnerApiClient({
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    },
  });
  client.sessionId = 'session-1';

  await client.deleteSessionKeepalive();

  assert.equal(client.sessionId, null);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.method, 'DELETE');
  assert.equal(calls[0].options.keepalive, true);
});

test('WinnerApiClient reset is a no-op when no server session exists', async () => {
  let calls = 0;
  const client = new WinnerApiClient({
    fetchImpl: async () => {
      calls += 1;
      throw new Error('reset must stay local without a session');
    },
  });

  const decision = await client.resetSession();

  assert.equal(decision, null);
  assert.equal(calls, 0);
  assert.equal(client.sessionId, null);
});

test('WinnerApiClient recreates an expired session and reports the reset', async () => {
  const resets = [];
  let calls = 0;
  const client = new WinnerApiClient({
    fetchImpl: async (url) => {
      calls += 1;
      if (url.includes('/frames')) {
        return { ok: false, status: 404, json: async () => ({ error: 'expired' }) };
      }
      return { ok: true, status: 201, json: async () => ({ session_id: 'new-session' }) };
    },
    onSessionReset: (event) => resets.push(event),
  });
  client.sessionId = 'old-session';

  const decision = await client.sendBatch([{ seq: 1, timestamp_ms: 10 }]);

  assert.equal(decision, null);
  assert.equal(client.sessionId, 'new-session');
  assert.equal(client.nextBatchSeq, 1);
  assert.deepEqual(resets.map((event) => event.reason), ['lost']);
  assert.equal(calls, 2);
});
test('RunGenerationGuard rejects responses captured before stop or restart', () => {
  const guard = new RunGenerationGuard();
  const firstRun = guard.capture();
  assert.equal(guard.isCurrent(firstRun), true);

  guard.invalidate();

  assert.equal(guard.isCurrent(firstRun), false);
  assert.equal(guard.isCurrent(guard.capture()), true);
});

test('WinnerApiClient does not retry terminal protocol conflicts', async () => {
  let calls = 0;
  const client = new WinnerApiClient({
    fetchImpl: async () => {
      calls += 1;
      return {
        ok: false,
        status: 409,
        json: async () => ({ error: 'Old or out-of-order frame timestamp' }),
      };
    },
    maxRetries: 2,
    retryDelayMs: 0,
    sleep: async () => {},
  });
  client.sessionId = 'session-1';

  await assert.rejects(() => client.sendBatch([{ seq: 1, timestamp_ms: 10 }]), /out-of-order/);
  assert.equal(calls, 1);
});
test('audio commands are consumed only for fresh server-response renders', () => {
  const decision = { audio_commands: ['double'], audio_command: 'double' };
  assert.deepEqual(audioCommandsForRender(decision, false), []);
  assert.deepEqual(audioCommandsForRender(decision, true), ['double']);
});

test('capture is blocked while reset is in progress', () => {
  assert.equal(canCaptureFrame(true, false), true);
  assert.equal(canCaptureFrame(true, true), false);
  assert.equal(canCaptureFrame(false, false), false);
});

test('page lifecycle reset cannot restore a false-running BFCache state', () => {
  assert.deepEqual(pageLifecycleResetState(), {
    running: false,
    paused: false,
    processing: false,
    resetting: false,
    starting: false,
    sendInFlight: false,
  });
});
test('ActiveSessionClock excludes paused wall time for camera timestamps', () => {
  let now = 1000;
  const clock = new ActiveSessionClock({ now: () => now });

  clock.start();
  now = 1120;
  assert.equal(clock.timestampMs('camera'), 120);
  clock.pause();
  now = 5120;
  assert.equal(clock.timestampMs('camera'), 120);
  clock.resume();
  now = 5165;
  assert.equal(clock.timestampMs('camera'), 165);
});

test('ActiveSessionClock uses media currentTime for file timestamps', () => {
  let now = 1000;
  const clock = new ActiveSessionClock({ now: () => now });
  clock.start();
  now = 9000;

  assert.equal(clock.timestampMs('file', { currentTime: 12.345 }), 12345);
  assert.equal(clock.timestampMs('file', { currentTime: Number.NaN }), 0);
});

test('escapeHtml makes stored history values safe for innerHTML fallbacks', () => {
  assert.equal(
    escapeHtml('<img src=x onerror=alert(1)>"&'),
    '&lt;img src=x onerror=alert(1)&gt;&quot;&amp;',
  );
});

test('history serialization includes winner metadata and exports CSV', () => {
  const entry = serializeHistoryEntry({
    state: 'DROWSY',
    label: 'Drowsy',
    probability: 0.81234,
    threshold: 0.55,
    hybrid_guard: 'visual_only',
    profile: 'recommended',
    model_hash: 'abc123',
    reasons: ['EAR_LOW', 'PERCLOS_HIGH'],
    metrics: { ear: 0.12, perclos: 0.4, perclos_short: 0.8 },
  }, new Date('2026-07-12T01:02:03.000Z'));

  assert.equal(entry.probability, '0.8123');
  assert.equal(entry.guard, 'visual_only');
  assert.equal(entry.profile, 'recommended');
  assert.equal(entry.model_hash, 'abc123');

  const csv = historyToCsv([entry]);
  assert.match(csv, /probability,threshold,guard,profile,model_hash/);
  assert.match(csv, /"0.8123","0.5500","visual_only","recommended","abc123"/);
});
