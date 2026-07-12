# Canonical Localhost Web App Design

**Date:** 2026-07-12
**Status:** Approved design pending written-spec review
**Repository:** `tridhm/Driver-Drowsiness-Monitor`
**Production baseline:** commit `c8a4954db340e1c24a9d8021ccf13b9dc828aa71`

## 1. Objective

Provide an offline localhost mode that runs the same browser perception bundle, Flask API, winner model, feature pipeline, `camera_hybrid` decision engine, and `recommended` runtime profile as the deployed web application.

The user experience must support:

- one-click startup on Windows through `run_local.cmd`;
- a cross-platform Python launcher;
- camera and local video-file workflows in the existing browser UI;
- no dependency on Render, a CDN, or any external service after the local Python environment has been installed;
- loopback-only access by default, with LAN exposure available only through an explicit flag.

This phase establishes one canonical product code path. It does not tune or replace MediaPipe, the winner model, thresholds, features, or hybrid rules.

## 2. Scope Boundary

### In scope

- A dedicated local launcher that imports the existing Flask application factory.
- A Windows command wrapper that bootstraps the project virtual environment when necessary and starts the launcher.
- Automatic browser opening after the local health endpoint is ready.
- Loopback binding, bounded port fallback, and explicit LAN mode.
- Offline-asset verification for MediaPipe/WASM, JavaScript, audio, model, and configuration files.
- Local startup, API, browser, privacy, and no-external-network tests.
- Replacement of the stale local-run documentation with instructions for the winner web application.

### Out of scope

- Migrating from legacy MediaPipe FaceMesh to MediaPipe Tasks Face Landmarker.
- Changing confidence thresholds, coordinate quantization, target FPS, EMA alpha, calibration values, model threshold, hybrid policy, or alert semantics.
- Retraining or recalibrating the winner.
- Replacing Render deployment or changing `render.yaml`.
- Electron, PyWebView, native desktop packaging, installers, Windows services, background startup, or persistent server-side sessions.
- LAN authentication or Internet exposure. LAN mode is for trusted private networks only.

A later perception-selection phase will compare the current FaceMesh path with shared-model Face Landmarker candidates. That work requires separate train/OOF and held-out/runtime gates.

## 3. Approaches Considered

### A. Run `web_server.py` directly

This reuses current code with almost no implementation work, but it does not provide one-click startup, environment checks, browser opening, bounded port selection, explicit LAN behavior, or focused tests. It also leaves the stale run documentation unresolved.

### B. Python launcher plus Windows wrapper - selected

A focused Python launcher owns local-only concerns while importing the existing Flask application and runtime. A small `run_local.cmd` provides one-click Windows startup and first-run environment setup. Production continues to use `web_server:app` unchanged.

This approach keeps one product implementation while isolating local process management from Flask request handling.

### C. Desktop shell

Electron or PyWebView could hide the browser/server boundary, but would add a second packaging/runtime surface, increase maintenance, and weaken the goal of using one canonical web path. It is rejected for this phase.

## 4. Architecture

```text
Windows run_local.cmd                  Cross-platform terminal
        |                                      |
        +------------> local_app.py <----------+
                           |
                           | creates WinnerRuntime(profile="recommended")
                           | creates Flask app through create_app(...)
                           | binds local HTTP server
                           v
              http://127.0.0.1:<selected-port>/
                           |
                   Existing mobile.html
                           |
          Self-hosted FaceMesh JavaScript/WASM
                           |
                 20 normalized landmarks
                           |
               Same-origin localhost API
                           |
       Same feature/model/hybrid/recommended runtime
```

The local launcher does not fork or copy application logic. It imports `WinnerRuntime` and `create_app` from the production modules. The browser continues to call relative `/api/v1/...` URLs, so no frontend API-base switch is needed.

## 5. Components

### 5.1 `local_app.py`

A root-level Python entrypoint provides the cross-platform CLI and local process lifecycle.

Supported arguments:

```text
--host HOST       Explicit bind host. Defaults to 127.0.0.1.
--port PORT       Preferred port. Defaults to 5000.
--lan             Bind 0.0.0.0 and print a trusted-LAN warning.
--no-browser      Do not open the system browser automatically.
--help            Print usage without loading the model or starting a server.
```

Behavior:

1. Resolve the repository root from `local_app.py`, not the current working directory.
2. Reject contradictory host selection such as `--lan` together with a non-loopback explicit host.
3. Default to `127.0.0.1` and always construct the canonical `recommended` profile. The launcher does not expose a profile selector.
4. When the preferred port is unavailable and the user did not explicitly pass `--port`, try ports `5001` through `5010` in order.
5. When an explicit port is unavailable, fail with a clear message instead of silently choosing another port.
6. Construct `WinnerRuntime(ROOT, profile_name="recommended")` and pass it to `create_app(runtime)`.
7. Start a controllable Werkzeug server without the Flask reloader.
8. Poll the local `/api/healthz` endpoint through loopback.
9. Open the root URL only after health reports `ready=true`.
10. Print the selected URL, profile, model hash prefix, privacy statement, shutdown instruction, and LAN warning when applicable.
11. On `Ctrl+C`, shut down the HTTP server and exit without leaving a background process.

The local launcher must not change the process-global production `app` object and must not write session data, landmarks, images, or video to disk.

### 5.2 `run_local.cmd`

The Windows wrapper operates relative to its own directory and remains usable when launched by double-click.

Behavior:

1. Change to the repository root.
2. Prefer `.venv\Scripts\python.exe` when it exists.
3. If `.venv` is absent, locate Python 3.12 through `py -3.12` or `python`, reject a non-3.12 interpreter, create `.venv`, upgrade pip, and install `requirements.txt`.
4. If setup fails, keep the console visible and print a concise recovery message.
5. Forward all command-line arguments to `local_app.py`.
6. Keep the console visible while the server runs so the user can stop it with `Ctrl+C` or close the window.
7. Return the Python launcher's exit code.

Initial environment creation may require Internet access to download Python packages. After dependencies are installed, the application assets and runtime must operate without external network access.

### 5.3 Existing Flask and browser application

`web_server.py`, `templates/mobile.html`, `static/winner_client.js`, self-hosted MediaPipe assets, model files, and packaged configs remain the canonical application implementation.

The UI may display a small `LOCAL` status derived from `window.location.hostname` when the hostname is `localhost`, `127.0.0.1`, or `::1`. This indicator must not change decision behavior or API payloads.

Production startup remains:

```text
gunicorn web_server:app ...
```

No Render setting depends on `local_app.py` or `run_local.cmd`.

## 6. Network and Security Behavior

### Default local mode

- Bind only `127.0.0.1`.
- Open `http://127.0.0.1:<port>/`.
- Accept requests only from the current machine.
- Browser video and images remain in the browser.
- Only normalized landmark JSON is sent to the local Flask process.

### Explicit LAN mode

- `--lan` binds `0.0.0.0`.
- The launcher opens the browser through loopback and prints a best-effort LAN URL.
- Camera access from another device over a plain `http://<LAN-IP>` origin is not promised because browsers commonly require HTTPS for camera access outside localhost. LAN mode is primarily a trusted-network file/status/debug option unless the user provides a separate HTTPS termination layer.
- The console states that LAN mode has no authentication and is only for a trusted private network.
- LAN mode does not modify Windows Firewall automatically.
- Internet exposure, tunneling, or automatic firewall rules are not provided.

### Offline assets

The active page must not require external fonts, scripts, styles, models, audio, or WASM. Tests will abort all non-local HTTP requests and require successful page and MediaPipe initialization.

## 7. Port Selection and Failure Handling

### Port selection

- Default preferred port: `5000`.
- Automatic fallback range: `5001-5010`.
- Fallback applies only when `--port` was not explicitly supplied.
- Port availability is determined by attempting the actual bind, avoiding a check-then-bind race.

### Startup failures

The launcher exits non-zero with a concise message when:

- no port in the bounded fallback range can be bound;
- the model/config contract fails startup validation;
- the local health endpoint does not become ready within the startup deadline;

Browser-opening failure is non-fatal: the server remains running and the console prints the URL for manual opening.

### Runtime failures

Existing browser behavior remains authoritative: API failure displays `SERVER UNAVAILABLE` and never falls back to the legacy JavaScript FSM.

## 8. Testing Strategy

### Unit tests

Add launcher tests for:

- default loopback host and `recommended` profile;
- default port fallback from `5000` to the first available bounded port;
- explicit busy port failure;
- `--lan` binding and warning metadata;
- `--help` avoiding model load and server startup;
- repository-root resolution independent of current working directory;
- browser-opening failure remaining non-fatal;
- graceful server shutdown.

### Local integration smoke

Start the launcher as a subprocess on a test port and verify:

- `/api/healthz` returns `ready=true`, profile `recommended`, threshold `0.55`, and the expected model hash;
- `/api/v1/runtime-info` returns the current landmark contract and `capture_stall_tolerance_ms=3000`;
- `/`, `static/winner_client.js`, MediaPipe JavaScript/WASM assets, and `static/alert.wav` return HTTP `200`;
- session create/process/delete works on localhost;
- shutdown leaves the port reusable.

### Browser acceptance

Run Playwright against the local launcher with all non-loopback requests blocked:

- desktop and mobile layouts load without console errors or overflow;
- self-hosted MediaPipe initializes;
- fake camera reaches active monitoring;
- file mode uses a local browser object URL;
- network requests contain landmark JSON only and no JPEG/video payload;
- no request targets Render, a CDN, or another external origin;
- the UI identifies local mode without changing runtime profile/model metadata.

### Regression gates

- Full Python suite.
- Full JavaScript suite.
- Dependency check and Python compile check.
- Existing winner evidence, golden parity, resampling, audio, session, and privacy gates.
- GitHub Actions on the feature branch and `main`.

## 9. Documentation

Rewrite `README_RUN.md` for the canonical winner web application:

- Windows one-click startup.
- Cross-platform CLI startup.
- First-run dependency behavior.
- Default local-only privacy boundary.
- `--lan`, `--port`, and `--no-browser` usage.
- Camera/file workflow.
- Shutdown instructions.
- Offline-after-setup limitation.
- Clear statement that desktop OpenCV/Python decision paths are legacy/research paths, not the canonical product runtime.

Add a short localhost section to `README.md` and link to `README_RUN.md`.

## 10. Success Criteria

The phase is complete when:

1. Double-clicking `run_local.cmd` on a configured Windows checkout starts the application and opens the browser.
2. A clean Windows first run can create `.venv` with Python 3.12 and install pinned dependencies with a clear progress/error surface.
3. `python local_app.py` works from any current working directory on supported platforms.
4. Default access is loopback-only; LAN exposure requires `--lan`.
5. Local health identifies the exact packaged winner and `recommended` profile.
6. The app works with external network requests blocked after setup.
7. Local camera and file workflows use the existing browser perception and same-origin winner API.
8. No production Render behavior, model, config, threshold, feature, or hybrid policy changes.
9. All local, browser, Python, JavaScript, parity, privacy, and CI gates pass.

## 11. Follow-up Phase

After canonical localhost packaging is complete, create a separate design and implementation plan for perception selection:

- run current FaceMesh and MediaPipe Tasks Face Landmarker candidates on the same frames;
- share the same `.task` model and option manifest where supported;
- compare face coverage, landmark jitter, EAR/MAR/pitch, calibration, model features, state/onset, alert burden, latency, and mobile CPU;
- select on train/OOF plus real RGB browser clips;
- freeze the perception candidate before using held-out 179 and external/runtime gates;
- promote the selected perception bundle to both online and localhost modes together.
