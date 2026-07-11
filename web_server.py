from __future__ import annotations

from collections import defaultdict, deque
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from runtime.landmark_adapter import REQUIRED_LANDMARKS
from runtime.web_runtime import ProtocolError, SessionLimits, WinnerRuntime, WinnerSession


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, runtime: WinnerRuntime, limits: SessionLimits | None = None) -> None:
        self.runtime = runtime
        self.limits = limits or SessionLimits()
        self.idle_timeout_seconds = float(self.limits.idle_timeout_seconds)
        self.lock = threading.RLock()
        self.sessions: dict[str, WinnerSession] = {}
        self.session_clients: dict[str, str] = {}
        self.create_events: deque[float] = deque()
        self.client_create_events: defaultdict[str, deque[float]] = defaultdict(deque)

    def create(self, source_mode: str, target_fps: int, client_key: str) -> WinnerSession:
        with self.lock:
            self.cleanup()
            now = time.monotonic()
            cutoff = now - 60.0
            while self.create_events and self.create_events[0] <= cutoff:
                self.create_events.popleft()
            key = str(client_key or "unknown")
            client_events = self.client_create_events[key]
            while client_events and client_events[0] <= cutoff:
                client_events.popleft()
            active_for_client = sum(1 for owner in self.session_clients.values() if owner == key)
            if len(self.sessions) >= self.limits.max_active_sessions:
                raise ProtocolError("Active session limit reached", 429)
            if active_for_client >= self.limits.max_active_sessions_per_client:
                raise ProtocolError("Per-client active session limit reached", 429)
            if len(self.create_events) >= self.limits.max_session_creates_per_minute:
                raise ProtocolError("Session create rate limit exceeded", 429)
            if len(client_events) >= self.limits.max_session_creates_per_client_per_minute:
                raise ProtocolError("Per-client session create rate limit exceeded", 429)
            session = self.runtime.create_session(source_mode=source_mode, target_fps=target_fps)
            self.sessions[session.session_id] = session
            self.session_clients[session.session_id] = key
            self.create_events.append(now)
            client_events.append(now)
            return session

    def get(self, session_id: str) -> WinnerSession:
        with self.lock:
            self.cleanup()
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.last_activity = time.monotonic()
            return session

    def delete(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            session = self.sessions.pop(session_id, None)
            self.session_clients.pop(session_id, None)
        if session is None:
            raise KeyError(session_id)
        return session.close()

    def cleanup(self) -> int:
        now = time.monotonic()
        expired: list[WinnerSession] = []
        with self.lock:
            for session_id, session in list(self.sessions.items()):
                if now - session.last_activity > self.idle_timeout_seconds:
                    expired.append(self.sessions.pop(session_id))
                    self.session_clients.pop(session_id, None)
        for session in expired:
            session.close()
        return len(expired)


def _default_runtime() -> tuple[WinnerRuntime | None, str | None]:
    requested = os.getenv("DMS_RUNTIME_PROFILE", "recommended")
    try:
        runtime = WinnerRuntime(ROOT, profile_name=requested)
        LOGGER.info(
            "winner_runtime_ready profile=%s model_hash=%s threshold=%s",
            runtime.profile_name,
            runtime.bundle.sha256,
            runtime.bundle.runtime_threshold,
        )
        return runtime, None
    except Exception as exc:
        LOGGER.exception("winner_runtime_startup_failed")
        return None, str(exc)


def create_app(runtime: WinnerRuntime | None = None, startup_error: str | None = None) -> Flask:
    flask_app = Flask(__name__)
    flask_app.wsgi_app = ProxyFix(flask_app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    flask_app.config["MAX_CONTENT_LENGTH"] = SessionLimits().max_payload_bytes
    flask_app.config["MAX_FORM_MEMORY_SIZE"] = SessionLimits().max_payload_bytes
    active_runtime = runtime
    limits = SessionLimits()
    store = SessionStore(active_runtime, limits=limits) if active_runtime is not None else None

    def require_runtime() -> WinnerRuntime:
        if active_runtime is None:
            raise ProtocolError(startup_error or "Winner runtime is not ready", 503)
        return active_runtime

    def require_store() -> SessionStore:
        require_runtime()
        assert store is not None
        return store

    def request_client_key() -> str:
        return str(request.remote_addr or "unknown")

    def reject_cross_origin() -> None:
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            raise ProtocolError("Cross-origin API requests are not allowed", 403)

    def require_json_object() -> dict[str, Any]:
        reject_cross_origin()
        if not request.is_json:
            raise ProtocolError("Content-Type must be application/json", 415)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ProtocolError("Request body must be a JSON object", 400)
        return payload

    def session_or_404(session_id: str) -> WinnerSession:
        try:
            return require_store().get(session_id)
        except KeyError:
            response = jsonify(
                error="Session not found or expired. Create a new session; calibration will restart.",
                error_code="SESSION_NOT_FOUND",
            )
            response.status_code = 404
            raise ApiResponse(response)

    @flask_app.errorhandler(ProtocolError)
    def handle_protocol_error(exc: ProtocolError):
        return jsonify(error=str(exc), error_code="PROTOCOL_ERROR"), exc.status_code

    @flask_app.errorhandler(ApiResponse)
    def handle_api_response(exc: "ApiResponse"):
        return exc.response

    @flask_app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_exc: RequestEntityTooLarge):
        return jsonify(error="Payload exceeds 64KB", error_code="PAYLOAD_TOO_LARGE"), 413

    @flask_app.after_request
    def add_response_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @flask_app.get("/")
    @flask_app.get("/mobile")
    def index():
        return render_template("mobile.html")

    @flask_app.get("/upload")
    def upload_redirect():
        return redirect(url_for("index", mode="file"))

    @flask_app.get("/api/healthz")
    def healthz():
        if active_runtime is None:
            return jsonify(ready=False, error=startup_error or "runtime unavailable"), 503
        return jsonify(
            ready=True,
            profile=active_runtime.profile_name,
            model_hash=active_runtime.bundle.sha256,
            feature_columns=active_runtime.bundle.feature_columns,
            artifact_threshold=active_runtime.bundle.artifact_threshold,
            runtime_threshold=active_runtime.bundle.runtime_threshold,
        )

    @flask_app.get("/api/v1/runtime-info")
    def runtime_info():
        winner = require_runtime()
        return jsonify(
            profile=winner.profile_name,
            model_hash=winner.bundle.sha256,
            model_name=winner.bundle.manifest["selected_model_name"],
            reference_fps=30,
            allowed_target_fps=[10, 15, 20, 30],
            landmark_indices=list(REQUIRED_LANDMARKS),
            landmark_count=len(REQUIRED_LANDMARKS),
            runtime_threshold=winner.bundle.runtime_threshold,
            cooldown_seconds=winner.config.alerts.drowsy_cooldown_seconds,
            video_upload_enabled=False,
            privacy="Video and images remain in the browser; only normalized landmark JSON is sent.",
            session_idle_timeout_seconds=require_store().idle_timeout_seconds,
            max_active_sessions=limits.max_active_sessions,
            max_active_sessions_per_client=limits.max_active_sessions_per_client,
        )

    @flask_app.post("/api/v1/sessions")
    def create_session():
        payload = require_json_object()
        session = require_store().create(
            source_mode=payload.get("source_mode", "camera"),
            target_fps=payload.get("target_fps", 20),
            client_key=request_client_key(),
        )
        return jsonify(
            session_id=session.session_id,
            profile=session.runtime.profile_name,
            model_hash=session.runtime.bundle.sha256,
            target_fps=session.target_fps,
            source_mode=session.source_mode,
            calibration_reset=True,
        ), 201

    @flask_app.post("/api/v1/sessions/<session_id>/frames")
    def process_frames(session_id: str):
        payload = require_json_object()
        return jsonify(session_or_404(session_id).process_batch(payload))

    @flask_app.post("/api/v1/sessions/<session_id>/reset")
    def reset_session(session_id: str):
        reject_cross_origin()
        return jsonify(session_or_404(session_id).reset())

    @flask_app.delete("/api/v1/sessions/<session_id>")
    def delete_session(session_id: str):
        reject_cross_origin()
        try:
            summary = require_store().delete(session_id)
        except KeyError:
            return jsonify(error="Session not found", error_code="SESSION_NOT_FOUND"), 404
        return jsonify(ok=True, summary=summary)

    def legacy_gone(**_kwargs):
        return jsonify(
            error="This frame/video route was removed. Use /api/v1/sessions with browser landmark JSON.",
            error_code="LEGACY_ROUTE_GONE",
        ), 410

    for rule, endpoint, methods in (
        ("/api/mobile/<path:subpath>", "legacy_mobile", ["GET", "POST"]),
        ("/api/upload", "legacy_upload", ["POST"]),
        ("/api/start", "legacy_start", ["POST"]),
        ("/api/stop", "legacy_stop", ["POST"]),
        ("/api/reset", "legacy_reset", ["POST"]),
        ("/api/status", "legacy_status", ["GET"]),
        ("/api/config", "legacy_config", ["GET"]),
        ("/video_feed", "legacy_video_feed", ["GET"]),
    ):
        flask_app.add_url_rule(rule, endpoint, legacy_gone, methods=methods)

    flask_app.extensions["winner_runtime"] = active_runtime
    flask_app.extensions["winner_session_store"] = store
    return flask_app


class ApiResponse(Exception):
    def __init__(self, response) -> None:
        super().__init__("API response")
        self.response = response


_DEFAULT_RUNTIME, _STARTUP_ERROR = _default_runtime()
app = create_app(_DEFAULT_RUNTIME, _STARTUP_ERROR)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)