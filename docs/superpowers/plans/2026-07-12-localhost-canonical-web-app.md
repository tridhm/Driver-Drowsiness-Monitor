# Canonical Localhost Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-click Windows launcher and cross-platform Python launcher that run the exact winner web application on localhost with the `recommended` profile, self-hosted browser perception assets, bounded port fallback, and verified offline behavior.

**Architecture:** `local_app.py` owns only local process concerns: argument parsing, loopback/LAN binding, bounded port selection, health readiness, browser opening, and shutdown. It imports the existing `WinnerRuntime` and Flask `create_app` factory so model, features, hybrid policy, API, templates, and static assets remain one canonical implementation. `run_local.cmd` bootstraps Python 3.12 on Windows, while Playwright proves the browser app works with all non-loopback requests blocked.

**Tech Stack:** Python 3.12, Flask 3.1, Werkzeug, `webbrowser`, Node.js 22+, Node test runner, Playwright 1.61.1, MediaPipe FaceMesh JavaScript/WASM, GitHub Actions.

---

## Source Design

Implement against:

- `docs/superpowers/specs/2026-07-12-localhost-canonical-web-app-design.md`
- baseline commit `c8a4954db340e1c24a9d8021ccf13b9dc828aa71`
- design commit `2eda699`

Do not stage, modify, revert, or commit these unrelated worktree artifacts:

- `verification/load_acceptance_20260712.json`
- `verification/production_bounded_probe_20260712.json`

## File Map

### Create

- `local_app.py` - cross-platform local launcher and server lifecycle.
- `run_local.cmd` - one-click Windows environment bootstrap and launcher wrapper.
- `tests/test_local_app.py` - launcher unit and local subprocess integration tests.
- `tests/browser/local_offline.spec.js` - browser offline/privacy/local-identity acceptance.
- `playwright.config.js` - local server and Chromium configuration.

### Modify

- `static/winner_client.js`
- `templates/mobile.html`
- `tests/js/winner_client.test.js`
- `tests/test_web_api.py`
- `package.json`, `package-lock.json`
- `.github/workflows/ci.yml`
- `README_RUN.md`, `README.md`
- `D:/drowsiness_detection-main/work-notes.md`

### Do not modify

- `models/camera_hybrid_winner.joblib`
- `models/winner_manifest.json`
- `configs/protected.json`
- `configs/recommended.json`
- `render.yaml`
- `runtime/` model, feature, engine, or policy behavior
- source PDF/DOCX/ODT reports

---

### Task 1: Launcher Options and Bounded Port Binding

**Files:**
- Create: `local_app.py`
- Create: `tests/test_local_app.py`

- [ ] **Step 1: Write failing tests for options and bind fallback**

Create `tests/test_local_app.py`:

```python
from __future__ import annotations

import io
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class LocalOptionsTests(unittest.TestCase):
    def test_defaults_are_loopback_recommended_and_auto_browser(self) -> None:
        from local_app import parse_options

        options = parse_options([])
        self.assertEqual(options.host, "127.0.0.1")
        self.assertEqual(options.port, 5000)
        self.assertFalse(options.explicit_port)
        self.assertFalse(options.lan)
        self.assertTrue(options.open_browser)
        self.assertEqual(options.profile_name, "recommended")
        self.assertEqual(options.root, ROOT)

    def test_lan_is_explicit_and_rejects_non_loopback_override(self) -> None:
        from local_app import parse_options

        options = parse_options(["--lan", "--no-browser"])
        self.assertEqual(options.host, "0.0.0.0")
        self.assertTrue(options.lan)
        self.assertFalse(options.open_browser)
        with self.assertRaises(SystemExit):
            parse_options(["--lan", "--host", "192.168.1.20"])
        with self.assertRaises(SystemExit):
            parse_options(["--host", "0.0.0.0"])

    def test_explicit_port_is_recorded_and_validated(self) -> None:
        from local_app import parse_options

        options = parse_options(["--port", "5099"])
        self.assertEqual(options.port, 5099)
        self.assertTrue(options.explicit_port)
        for invalid in ("0", "65536", "abc"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                parse_options(["--port", invalid])

    def test_port_candidates_are_bounded(self) -> None:
        from local_app import port_candidates

        self.assertEqual(list(port_candidates(5000, explicit=False)), list(range(5000, 5011)))
        self.assertEqual(list(port_candidates(5099, explicit=True)), [5099])


class BindServerTests(unittest.TestCase):
    def test_busy_default_port_falls_back(self) -> None:
        from local_app import bind_server

        sentinel = object()
        calls = []

        def fake_factory(host, port, app, threaded):
            calls.append((host, port, app, threaded))
            if port == 5000:
                raise OSError("busy")
            return sentinel

        server, port = bind_server(
            app="app",
            host="127.0.0.1",
            preferred_port=5000,
            explicit_port=False,
            server_factory=fake_factory,
        )
        self.assertIs(server, sentinel)
        self.assertEqual(port, 5001)
        self.assertEqual([call[1] for call in calls], [5000, 5001])

    def test_explicit_busy_port_fails(self) -> None:
        from local_app import LocalLaunchError, bind_server

        with self.assertRaisesRegex(LocalLaunchError, "5099"):
            bind_server(
                app="app",
                host="127.0.0.1",
                preferred_port=5099,
                explicit_port=True,
                server_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")),
            )

    def test_exhausted_default_range_fails(self) -> None:
        from local_app import LocalLaunchError, bind_server

        with self.assertRaisesRegex(LocalLaunchError, "5000-5010"):
            bind_server(
                app="app",
                host="127.0.0.1",
                preferred_port=5000,
                explicit_port=False,
                server_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")),
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'local_app'`.

- [ ] **Step 3: Implement minimal option and binding layer**

Create `local_app.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
LAST_FALLBACK_PORT = 5010
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LocalLaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalOptions:
    root: Path
    host: str
    port: int
    explicit_port: bool
    lan: bool
    open_browser: bool
    profile_name: str = "recommended"


def _valid_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _option_was_explicit(argv: Sequence[str], name: str) -> bool:
    return name in argv or any(item.startswith(name + "=") for item in argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical winner web app on localhost.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_valid_port, default=DEFAULT_PORT)
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    return parser


def parse_options(argv: Sequence[str] | None = None) -> LocalOptions:
    raw = list(argv or [])
    parser = build_parser()
    args = parser.parse_args(raw)
    requested_host = str(args.host).strip().lower()
    if args.lan and requested_host not in LOOPBACK_HOSTS:
        parser.error("--lan cannot be combined with a non-loopback --host")
    if not args.lan and requested_host not in LOOPBACK_HOSTS:
        parser.error("non-loopback binding requires --lan")
    return LocalOptions(
        root=ROOT,
        host="0.0.0.0" if args.lan else str(args.host),
        port=int(args.port),
        explicit_port=_option_was_explicit(raw, "--port"),
        lan=bool(args.lan),
        open_browser=not bool(args.no_browser),
    )


def port_candidates(preferred_port: int, explicit: bool) -> Iterable[int]:
    return (preferred_port,) if explicit else range(preferred_port, LAST_FALLBACK_PORT + 1)


def bind_server(
    *, app: Any, host: str, preferred_port: int, explicit_port: bool,
    server_factory: Callable[..., Any],
) -> tuple[Any, int]:
    last_error: OSError | None = None
    for port in port_candidates(preferred_port, explicit_port):
        try:
            return server_factory(host, port, app, threaded=True), port
        except OSError as exc:
            last_error = exc
    if explicit_port:
        raise LocalLaunchError(f"Port {preferred_port} is unavailable") from last_error
    raise LocalLaunchError("No local port is available in range 5000-5010") from last_error
```

Keep Flask, Werkzeug, and model imports out of module top level so `--help` does not load the model. Standard-library imports such as `webbrowser` are safe.

- [ ] **Step 4: Verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add local_app.py tests/test_local_app.py
git commit -m "Add localhost launcher options and port binding"
```

---

### Task 2: Runtime, Health, Browser Opening, and Shutdown

**Files:**
- Modify: `local_app.py`
- Modify: `tests/test_local_app.py`

- [ ] **Step 1: Add failing lifecycle tests**

Add imports and helpers:

```python
import ipaddress
import json
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

EXPECTED_MODEL_HASH = "8958d2d4dd0a0757b5a922adb11df263144e253873909ac8816cd26c248bc89c"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
```

Add tests:

```python
class LocalLifecycleTests(unittest.TestCase):
    def test_build_local_app_always_uses_recommended(self) -> None:
        from local_app import build_local_app

        app, runtime = build_local_app(ROOT)
        self.assertEqual(runtime.profile_name, "recommended")
        self.assertEqual(runtime.bundle.sha256, EXPECTED_MODEL_HASH)
        self.assertIs(app.extensions["winner_runtime"], runtime)

    def test_runtime_startup_error_is_concise(self) -> None:
        from local_app import LocalLaunchError, build_local_app

        with mock.patch("runtime.web_runtime.WinnerRuntime", side_effect=ValueError("bad model")):
            with self.assertRaisesRegex(LocalLaunchError, "bad model"):
                build_local_app(ROOT)

    def test_open_browser_failure_is_non_fatal(self) -> None:
        from local_app import open_browser

        output = io.StringIO()
        opened = open_browser("http://127.0.0.1:5000/", opener=lambda _url: False, output=output)
        self.assertFalse(opened)
        self.assertIn("Open this URL manually", output.getvalue())

    def test_subprocess_serves_contract_and_releases_port(self) -> None:
        port = free_port()
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "local_app.py"), "--no-browser", "--port", str(port)],
            cwd=str(ROOT.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 30.0
            health = None
            while time.monotonic() < deadline:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/healthz", timeout=2.0) as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(0.1)
            self.assertIsNotNone(health)
            self.assertTrue(health["ready"])
            self.assertEqual(health["profile"], "recommended")
            self.assertEqual(health["model_hash"], EXPECTED_MODEL_HASH)
            self.assertEqual(health["runtime_threshold"], 0.55)
            with urlopen(f"http://127.0.0.1:{port}/api/v1/runtime-info", timeout=2.0) as response:
                info = json.load(response)
            self.assertEqual(info["landmark_count"], 20)
            self.assertEqual(info["capture_stall_tolerance_ms"], 3000)
            self.assertFalse(info["video_upload_enabled"])
        finally:
            process.terminate()
            process.wait(timeout=10)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app.LocalLifecycleTests -v
```

Expected: FAIL because lifecycle functions are missing.

- [ ] **Step 3: Implement lifecycle functions**

Extend `local_app.py`:

```python
import ipaddress
import json
import socket
import sys
import threading
import time
from urllib.request import urlopen
import webbrowser

STARTUP_TIMEOUT_SECONDS = 15.0


def build_local_app(root: Path = ROOT):
    try:
        from runtime.web_runtime import WinnerRuntime
        from web_server import create_app
        runtime = WinnerRuntime(root, profile_name="recommended")
        return create_app(runtime), runtime
    except Exception as exc:
        raise LocalLaunchError(f"Winner runtime is not ready: {exc}") from exc


def wait_for_health(url: str, timeout_seconds: float = STARTUP_TIMEOUT_SECONDS) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() <= deadline:
        try:
            with urlopen(url, timeout=2.0) as response:
                payload = json.load(response)
            if payload.get("ready") is True:
                return payload
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise LocalLaunchError("Local health endpoint did not become ready") from last_error


def open_browser(url: str, *, opener=webbrowser.open, output=sys.stdout) -> bool:
    try:
        opened = bool(opener(url))
    except Exception:
        opened = False
    if not opened:
        print(f"Browser did not open. Open this URL manually: {url}", file=output)
    return opened


def best_effort_lan_ip() -> str | None:
    try:
        values = {str(item[4][0]) for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    except OSError:
        return None
    for value in sorted(values):
        address = ipaddress.ip_address(value)
        if address.version == 4 and address.is_private and not address.is_loopback:
            return value
    return None


def run(options: LocalOptions) -> int:
    from werkzeug.serving import make_server

    app, _runtime = build_local_app(options.root)
    server, selected_port = bind_server(
        app=app, host=options.host, preferred_port=options.port,
        explicit_port=options.explicit_port, server_factory=make_server,
    )
    thread = threading.Thread(target=server.serve_forever, name="dms-local-server", daemon=True)
    thread.start()
    local_url = f"http://127.0.0.1:{selected_port}"
    try:
        health = wait_for_health(local_url + "/api/healthz")
        print(f"Local winner ready: {local_url}/")
        print(f"Profile: {health['profile']}")
        print(f"Model: {health['model_hash'][:12]}...")
        print("Video stays in this browser; only landmark JSON reaches this local process.")
        if options.lan:
            lan_ip = best_effort_lan_ip()
            print("LAN mode has no authentication. Use only on a trusted private network.")
            print("Camera access from another device may require HTTPS.")
            if lan_ip:
                print(f"LAN URL: http://{lan_ip}:{selected_port}/")
        if options.open_browser:
            open_browser(local_url + "/")
        print("Press Ctrl+C to stop.")
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Stopping local winner...")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_options(argv))
    except LocalLaunchError as exc:
        print(f"Local startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Verify Task 2 and full Python suite**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app -v
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add local_app.py tests/test_local_app.py
git commit -m "Run the winner web app on localhost"
```
---

### Task 3: One-Click Windows Bootstrap

**Files:**
- Create: `run_local.cmd`
- Modify: `tests/test_local_app.py`

- [ ] **Step 1: Write failing wrapper contract test**

Append:

```python
class WindowsWrapperTests(unittest.TestCase):
    def test_wrapper_is_relative_pins_python_312_and_forwards_arguments(self) -> None:
        script = (ROOT / "run_local.cmd").read_text(encoding="utf-8")
        self.assertIn('cd /d "%~dp0"', script)
        self.assertIn('.venv\\Scripts\\python.exe', script)
        self.assertIn('py -3.12', script)
        self.assertIn('sys.version_info[:2] == (3, 12)', script)
        self.assertIn('-m venv .venv', script)
        self.assertIn('-m pip install -r requirements.txt', script)
        self.assertIn('local_app.py" %*', script)
        self.assertNotIn('D:\\', script)
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app.WindowsWrapperTests -v
```

Expected: FAIL because `run_local.cmd` is missing.

- [ ] **Step 3: Create `run_local.cmd`**

```bat
@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto run

echo Preparing the local winner environment for the first run...
where py >nul 2>nul
if not errorlevel 1 (
  set "BOOTSTRAP=py -3.12"
  goto verify_python
)
where python >nul 2>nul
if errorlevel 1 goto missing_python
set "BOOTSTRAP=python"

:verify_python
%BOOTSTRAP% -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if errorlevel 1 goto wrong_python
%BOOTSTRAP% -m venv .venv
if errorlevel 1 goto setup_failed
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto setup_failed
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto setup_failed

:run
"%VENV_PY%" "%~dp0local_app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

:missing_python
echo Python 3.12 was not found. Install Python 3.12 and run this file again.
pause
exit /b 1

:wrong_python
echo The available Python is not 3.12. Install Python 3.12 and run this file again.
pause
exit /b 1

:setup_failed
echo Local environment setup failed. Check the network connection and the error above.
pause
exit /b 1
```

Keep the console visible; it is the shutdown surface.

- [ ] **Step 4: Verify wrapper behavior on Windows**

```powershell
cmd /c run_local.cmd --help
```

Expected: usage, exit `0`, no server.

Then run:

```powershell
cmd /c run_local.cmd --no-browser --port 5099
```

Expected: health ready on port 5099; `Ctrl+C` stops and releases the port.

- [ ] **Step 5: Commit Task 3**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app -v
git add run_local.cmd tests/test_local_app.py
git commit -m "Add one-click Windows localhost launcher"
```

---

### Task 4: Local/Online UI Identity

**Files:**
- Modify: `static/winner_client.js`
- Modify: `templates/mobile.html`
- Modify: `tests/js/winner_client.test.js`
- Modify: `tests/test_web_api.py`

- [ ] **Step 1: Write failing JS and HTML tests**

Import `runtimeLocationLabel` in `tests/js/winner_client.test.js`, then add:

```javascript
test('runtimeLocationLabel distinguishes loopback from deployed hosts', () => {
  assert.equal(runtimeLocationLabel('127.0.0.1'), 'LOCAL');
  assert.equal(runtimeLocationLabel('localhost'), 'LOCAL');
  assert.equal(runtimeLocationLabel('LOCALHOST'), 'LOCAL');
  assert.equal(runtimeLocationLabel('::1'), 'LOCAL');
  assert.equal(runtimeLocationLabel('[::1]'), 'LOCAL');
  assert.equal(runtimeLocationLabel('driver-drowsiness-monitor-winner.onrender.com'), 'ONLINE');
});
```

Extend `test_primary_page_labels_probability_and_six_second_perclos_honestly`:

```python
self.assertIn('id="runtimeLocation"', html)
self.assertIn("runtimeLocationLabel(window.location.hostname)", html)
```

- [ ] **Step 2: Verify RED**

```powershell
npm test
.\.venv\Scripts\python.exe -m unittest tests.test_web_api.WinnerWebApiTests.test_primary_page_labels_probability_and_six_second_perclos_honestly -v
```

Expected: FAIL because helper and markup are missing.

- [ ] **Step 3: Add pure helper and compact badge**

Add to `static/winner_client.js` and export it:

```javascript
function runtimeLocationLabel(hostname) {
  const host = String(hostname || '').trim().toLowerCase().replace(/^\[|\]$/g, '');
  return host === 'localhost' || host === '127.0.0.1' || host === '::1' ? 'LOCAL' : 'ONLINE';
}
```

Change the header:

```html
<h1>Driver Drowsiness Monitor <span id="runtimeLocation" class="runtime-location"></span></h1>
```

Add CSS:

```css
.runtime-location { margin-left:6px; color:var(--blue); font-family:Consolas, monospace; font-size:10px; }
```

After `const WC = window.WinnerClient;`:

```javascript
document.getElementById('runtimeLocation').textContent = WC.runtimeLocationLabel(window.location.hostname);
```

Do not modify state, API, session, audio, or decisions.

- [ ] **Step 4: Verify GREEN and commit**

```powershell
npm test
.\.venv\Scripts\python.exe -m unittest tests.test_web_api -v
git add static/winner_client.js templates/mobile.html tests/js/winner_client.test.js tests/test_web_api.py
git commit -m "Identify localhost and online web runtimes"
```

---

### Task 5: Browser Offline and Privacy Acceptance

**Files:**
- Create: `playwright.config.js`
- Create: `tests/browser/local_offline.spec.js`
- Modify: `package.json`
- Modify: `package-lock.json`

- [ ] **Step 1: Install Playwright and create a deliberate RED test**

```powershell
npm install --save-dev @playwright/test@1.61.1
```

Add script:

```json
"test:browser": "playwright test"
```

Create `playwright.config.js`:

```javascript
'use strict';

const fs = require('node:fs');
const { defineConfig } = require('@playwright/test');

const windowsVenv = '.venv\\Scripts\\python.exe';
const python = process.platform === 'win32' && fs.existsSync(windowsVenv) ? windowsVenv : 'python';

module.exports = defineConfig({
  testDir: 'tests/browser',
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  reporter: 'line',
  use: {
    baseURL: 'http://127.0.0.1:5011',
    permissions: ['camera'],
    trace: 'retain-on-failure',
    launchOptions: {
      args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
    },
  },
  webServer: {
    command: `"${python}" local_app.py --no-browser --port 5011`,
    url: 'http://127.0.0.1:5011/api/healthz',
    timeout: 120_000,
    reuseExistingServer: !process.env.CI,
  },
});
```

Create the initial test:

```javascript
'use strict';

const { test, expect } = require('@playwright/test');

test('localhost page exposes local identity', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#runtimeLocation')).toHaveText('LOCAL');
  await expect(page.locator('#offlineAcceptanceReady')).toBeAttached();
});
```

- [ ] **Step 2: Install Chromium and verify intentional RED**

```powershell
npx playwright install chromium
npm run test:browser
```

Expected: server starts, `LOCAL` passes, only `#offlineAcceptanceReady` fails. Remove that deliberate assertion.

- [ ] **Step 3: Replace with complete browser gates**

Use this test file:

```javascript
'use strict';

const { test, expect } = require('@playwright/test');

function isLocalRequest(rawUrl) {
  const url = new URL(rawUrl);
  return url.protocol === 'blob:' || url.protocol === 'data:' ||
    url.hostname === '127.0.0.1' || url.hostname === 'localhost' || url.hostname === '::1';
}

test.beforeEach(async ({ page }) => {
  await page.route('**/*', async (route) => {
    if (isLocalRequest(route.request().url())) await route.continue();
    else await route.abort('blockedbyclient');
  });
});

test('fake camera runs with self-hosted assets and landmark-only requests', async ({ page }) => {
  const consoleErrors = [];
  const externalRequests = [];
  const framePayloads = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('request', (request) => {
    if (!isLocalRequest(request.url())) externalRequests.push(request.url());
    if (request.url().includes('/frames') && request.postData()) {
      framePayloads.push(JSON.parse(request.postData()));
    }
  });

  await page.goto('/');
  await expect(page.locator('#runtimeLocation')).toHaveText('LOCAL');
  await page.locator('#startBtn').click();
  await expect(page.locator('#topStatus')).toContainText('MediaPipe ready', { timeout: 45_000 });
  await expect.poll(() => framePayloads.length, { timeout: 30_000 }).toBeGreaterThan(0);

  for (const payload of framePayloads) {
    expect(Object.keys(payload).sort()).toEqual(['batch_seq', 'frames']);
    for (const frame of payload.frames) {
      const expectedKeys = ['face_detected', 'height', 'seq', 'timestamp_ms', 'width'];
      if (frame.face_detected) expectedKeys.push('landmarks');
      expect(Object.keys(frame).sort()).toEqual(expectedKeys.sort());
      expect(JSON.stringify(frame)).not.toMatch(/jpeg|image\/|video\/|base64/i);
      if (frame.face_detected) expect(Object.keys(frame.landmarks)).toHaveLength(20);
    }
  }
  expect(externalRequests).toEqual([]);
  expect(consoleErrors).toEqual([]);
});

test('file mode creates an object URL and never uploads bytes', async ({ page }) => {
  await page.addInitScript(() => {
    const originalCreate = URL.createObjectURL.bind(URL);
    window.__createdObjectUrls = [];
    URL.createObjectURL = (value) => {
      const url = originalCreate(value);
      window.__createdObjectUrls.push(url);
      return url;
    };
    Object.defineProperty(HTMLMediaElement.prototype, 'readyState', { get: () => 2 });
    Object.defineProperty(HTMLVideoElement.prototype, 'videoWidth', { get: () => 640 });
    Object.defineProperty(HTMLVideoElement.prototype, 'videoHeight', { get: () => 480 });
    HTMLMediaElement.prototype.load = function load() {
      queueMicrotask(() => this.dispatchEvent(new Event('loadedmetadata')));
    };
    HTMLMediaElement.prototype.play = async function play() {};
  });
  const mediaRequests = [];
  page.on('request', (request) => {
    const body = request.postData() || '';
    if (/video\/|base64|AAAA/i.test(body)) mediaRequests.push(request.url());
  });
  await page.goto('/mobile?mode=file');
  await page.locator('#videoFile').setInputFiles({
    name: 'local-test.mp4', mimeType: 'video/mp4',
    buffer: Buffer.from('local-browser-only-fixture'),
  });
  await expect.poll(() => page.evaluate(() => window.__createdObjectUrls.length)).toBeGreaterThan(0);
  expect(mediaRequests).toEqual([]);
});

test('desktop and mobile layouts do not overflow', async ({ page }) => {
  for (const viewport of [{ width: 1280, height: 720 }, { width: 375, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.goto('/');
    const dimensions = await page.evaluate(() => ({
      bodyWidth: document.body.scrollWidth,
      viewportWidth: window.innerWidth,
      controlsWidth: document.querySelector('.controls').scrollWidth,
      controlsClient: document.querySelector('.controls').clientWidth,
    }));
    expect(dimensions.bodyWidth).toBeLessThanOrEqual(dimensions.viewportWidth);
    expect(dimensions.controlsWidth).toBeLessThanOrEqual(dimensions.controlsClient);
  }
});
```

If the synthetic file needs tighter media stubs, change only the test harness; never add an upload/fallback product path.

- [ ] **Step 4: Run browser gates and commit**

```powershell
npm test
npm run test:browser
git add package.json package-lock.json playwright.config.js tests/browser/local_offline.spec.js
git commit -m "Verify localhost browser operation without external network"
```
---

### Task 6: CI Coverage for Browser and Windows Launcher

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Cover all feature branches and browser acceptance**

Change triggers:

```yaml
on:
  push:
    branches: [main, "feat/**"]
  pull_request:
    branches: [main]
```

The browser job is isolated from the Python job, so add Python setup and dependencies before `npm ci`:

```yaml
      - uses: actions/setup-python@v5
        with:
          python-version-file: .python-version
          cache: pip
      - name: Install Python dependencies for localhost server
        run: |
          python -m pip install --upgrade pip
          python -m pip install -r requirements.txt
```

After `npm ci`, use:

```yaml
      - name: Install Chromium
        run: npx playwright install --with-deps chromium
      - name: Run JavaScript tests
        run: npm test
      - name: Run localhost browser acceptance
        run: npm run test:browser
```

Remove any duplicate JavaScript test step.

- [ ] **Step 2: Add Windows wrapper job**

```yaml
  windows-local-launcher:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version-file: .python-version
          cache: pip
      - name: Verify clean one-click bootstrap and help
        shell: cmd
        run: run_local.cmd --help
      - name: Run launcher tests
        shell: pwsh
        run: .\.venv\Scripts\python.exe -m unittest tests.test_local_app -v
```

- [ ] **Step 3: Validate and run CI-equivalent commands**

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; text=Path('.github/workflows/ci.yml').read_text(); assert 'feat/**' in text; assert 'windows-local-launcher' in text; assert 'npm run test:browser' in text"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover tests -v
npm test
npm run test:browser
```

Expected: all exit `0`.

- [ ] **Step 4: Commit Task 6**

```powershell
git add .github/workflows/ci.yml
git commit -m "Test localhost launcher across Linux and Windows"
```

---

### Task 7: Replace Stale Run Documentation

**Files:**
- Modify: `README_RUN.md`
- Modify: `README.md`
- Modify: `tests/test_local_app.py`

- [ ] **Step 1: Add failing documentation test**

```python
class LocalDocumentationTests(unittest.TestCase):
    def test_docs_describe_canonical_local_winner(self) -> None:
        run_readme = (ROOT / "README_RUN.md").read_text(encoding="utf-8")
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for required in (
            "run_local.cmd", "python local_app.py", "127.0.0.1",
            "--lan", "--port", "--no-browser", "recommended",
            "Video stays in the browser", "Python 3.12",
        ):
            self.assertIn(required, run_readme)
        self.assertNotIn("--decision-engine fsm", run_readme)
        self.assertNotIn("Upload & Start", run_readme)
        self.assertIn("README_RUN.md", root_readme)
        self.assertIn("camera_hybrid", root_readme)
```

- [ ] **Step 2: Verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app.LocalDocumentationTests -v
```

Expected: FAIL against current stale docs.

- [ ] **Step 3: Rewrite `README_RUN.md`**

```markdown
# Run the Canonical Winner Web App Locally

The localhost application uses the same browser FaceMesh bundle, 20-landmark API, Flask feature pipeline, winner model, `camera_hybrid` engine, and `recommended` profile as the deployed website.

## Windows: one click

Double-click `run_local.cmd`.

The first run creates `.venv` with Python 3.12 and installs `requirements.txt`. Initial setup can require Internet access. Later runs use self-hosted MediaPipe/WASM, model, config, JavaScript, and audio assets and can operate without external network access.

The launcher opens `http://127.0.0.1:5000/`. If port 5000 is busy, it tries 5001 through 5010.

## Cross-platform CLI

```text
python local_app.py
python local_app.py --no-browser
python local_app.py --port 5099
python local_app.py --lan
```

Default mode binds only `127.0.0.1`. `--lan` has no authentication and is only for a trusted private network. Camera access from another device over plain HTTP may be blocked because non-localhost camera origins commonly require HTTPS.

## Privacy

Video stays in the browser. The browser sends only normalized landmark JSON to the Flask process on the same machine. The server does not store raw landmarks, images, or video.

## Camera and local files

Use **Start camera** for webcam monitoring or **Upload video** to select a local file. The file is processed as a browser object URL and is not uploaded to Flask.

## Stop

Return to the launcher console and press `Ctrl+C`.

## Canonical runtime boundary

The localhost web app is the canonical offline product path. Older OpenCV/Python desktop entrypoints remain research/legacy tools and are not the deployed winner runtime.
```

- [ ] **Step 4: Rewrite root `README.md`**

```markdown
# Driver Drowsiness Monitor Winner Web

Browser MediaPipe perception with a Flask-hosted protected `camera_hybrid` winner. Video remains in the browser; the API receives only 20 normalized landmarks and timing metadata.

## Run locally

See [README_RUN.md](README_RUN.md) or double-click `run_local.cmd` on Windows.

## Deploy

Production uses `web_server:app`; see [README_DEPLOY_RENDER.md](README_DEPLOY_RENDER.md).
```

- [ ] **Step 5: Verify docs and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_local_app.LocalDocumentationTests -v
.\.venv\Scripts\python.exe -c "from pathlib import Path; assert Path('README_RUN.md').exists(); assert Path('README_DEPLOY_RENDER.md').exists()"
git add README.md README_RUN.md tests/test_local_app.py
git commit -m "Document the canonical localhost winner app"
```

---

### Task 8: Full Verification, Review, Merge, and Release Record

**Files:**
- Modify: `D:/drowsiness_detection-main/work-notes.md`
- Optional: `D:/drowsiness_detection-main/docs/final_report_web_winner_corrections_vi_2026-07-12.md`

- [ ] **Step 1: Run static checks**

```powershell
git diff --check
.\.venv\Scripts\python.exe -m compileall -q local_app.py runtime web_server.py tests
.\.venv\Scripts\python.exe -m pip check
```

Expected: clean.

- [ ] **Step 2: Run all functional gates**

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
npm test
npm run test:browser
.\.venv\Scripts\python.exe tools/load_acceptance.py --sessions 3 --duration-seconds 3 --input-fps 10 --batch-size 4 --enforce-production-limits
```

Expected: Python, Node, Playwright, model/parity/privacy, and load gates pass.

- [ ] **Step 3: Run manual Windows one-click acceptance**

Double-click `run_local.cmd` and verify:

1. browser opens localhost;
2. header shows `LOCAL`;
3. health is `recommended`, threshold `0.55`, expected model hash;
4. fake or real camera reaches MediaPipe ready and sends landmark JSON only;
5. local file selection creates a browser object URL and no upload request;
6. `Ctrl+C` stops the process and releases the port.

Record port and startup time. Do not claim pass without direct observation.

- [ ] **Step 4: Review scoped diff**

```powershell
git diff main...HEAD --stat
git diff main...HEAD -- . ':!verification/load_acceptance_20260712.json' ':!verification/production_bounded_probe_20260712.json'
git status --short
```

Confirm no model/config/policy, Render config, media persistence, or unrelated artifact changes.

- [ ] **Step 5: Request code review**

Use `superpowers:requesting-code-review`. Address every blocker/high finding with TDD and rerun affected plus full gates.

- [ ] **Step 6: Push branch and wait for CI**

```powershell
git push -u origin feat/localhost-canonical-web-app
```

Require Linux Python, Linux JavaScript/browser, and Windows launcher jobs to pass.

- [ ] **Step 7: Merge and verify production regression**

After green review/CI, merge to `main`. Verify production:

- `/api/healthz`: ready, `recommended`, threshold `0.55`, expected model hash;
- `/api/v1/runtime-info`: 20 landmarks, `capture_stall_tolerance_ms=3000`;
- page header: `ONLINE`;
- no Render/CDN dependency introduced.

If auto-deploy does not run, use authenticated **Deploy latest commit**. Never request/store passwords or personal API keys.

- [ ] **Step 8: Update running notes**

Record branch/merge commit, launcher behavior, selected profile, test counts, Windows one-click result, external-network-blocked browser result, CI, production regression, and the remaining separate Face Landmarker selection phase in `D:/drowsiness_detection-main/work-notes.md`.

Update the correction report only if it still implies a separate Python/OpenCV offline product path. Keep PDF/DOCX/ODT unchanged.

- [ ] **Step 9: Final completion check**

Rerun local and production health checks, stop all local test servers, and report any unmet gate instead of weakening acceptance criteria.
