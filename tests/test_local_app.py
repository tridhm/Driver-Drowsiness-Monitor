from __future__ import annotations

import builtins
import io
import json
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.request import urlopen

from local_app import (
    LocalLaunchError,
    LocalOptions,
    bind_server,
    build_local_app,
    local_access_host,
    main,
    open_browser,
    parse_options,
    port_candidates,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_HASH = "8958d2d4dd0a0757b5a922adb11df263144e253873909ac8816cd26c248bc89c"


class LocalDocumentationTests(unittest.TestCase):
    def test_docs_describe_canonical_local_winner(self) -> None:
        readme_run = (ROOT / "README_RUN.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for expected in (
            "run_local.cmd",
            "python local_app.py",
            "127.0.0.1",
            "--lan",
            "--port",
            "--no-browser",
            "recommended",
            "Video stays in the browser",
            "Python 3.12",
            "python3.12 -m venv .venv",
            "python -m pip install -r requirements.txt",
            "Raw landmarks, images, and video are not persisted",
            "Flask keeps transient in-memory session state while active",
            "browser may persist derived event history locally",
            "not raw media or landmarks",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, readme_run)

        self.assertNotIn("--decision-engine fsm", readme_run)
        self.assertNotIn("Upload & Start", readme_run)
        self.assertNotIn("does not store raw landmarks", readme_run)
        self.assertIn("README_RUN.md", readme)
        self.assertIn("camera_hybrid", readme)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class WindowsWrapperTests(unittest.TestCase):
    def test_wrapper_is_relative_pins_python_312_and_forwards_arguments(self) -> None:
        script = (ROOT / "run_local.cmd").read_text(encoding="utf-8")

        self.assertIn('cd /d "%~dp0"', script)
        self.assertIn(r".venv\Scripts\python.exe", script)
        self.assertIn("py -3.12", script)
        self.assertIn("sys.version_info[:2] == (3, 12)", script)
        self.assertIn("-m venv .venv", script)
        self.assertIn("-m pip install -r requirements.txt", script)
        self.assertIn('local_app.py" %*', script)
        self.assertNotIn("D:\\", script)

    def test_existing_wrapper_venv_is_version_checked_before_run(self) -> None:
        script = (ROOT / "run_local.cmd").read_text(encoding="utf-8")

        self.assertIn('if exist "%VENV_PY%" goto verify_venv', script)
        self.assertIn(":verify_venv", script)
        self.assertIn(
            '"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"',
            script,
        )
        self.assertIn("if errorlevel 1 goto wrong_python", script)
        self.assertIn("goto run", script)

    def test_existing_wrapper_venv_recovers_missing_dependencies_before_run(self) -> None:
        script = (ROOT / "run_local.cmd").read_text(encoding="utf-8")

        self.assertIn(":install_dependencies", script)
        self.assertIn(
            '"%VENV_PY%" -c "import flask, joblib, numpy, cv2, sklearn"',
            script,
        )
        self.assertIn("if errorlevel 1 goto install_dependencies", script)
        self.assertIn("-m pip install -r requirements.txt", script)

    def test_wrapper_skips_pause_only_for_noninteractive_environment(self) -> None:
        script = (ROOT / "run_local.cmd").read_text(encoding="utf-8")

        self.assertIn(":maybe_pause", script)
        self.assertIn("if defined CI exit /b 0", script)
        self.assertIn("if defined RUN_LOCAL_NO_PAUSE exit /b 0", script)
        self.assertIn("call :maybe_pause", script)
        self.assertIn("pause", script)

    @unittest.skipUnless(os.name == "nt", "Windows wrapper smoke only runs on Windows")
    def test_wrapper_subprocess_serves_contract_and_releases_port(self) -> None:
        port = free_port()
        environment = os.environ.copy()
        environment["RUN_LOCAL_NO_PAUSE"] = "1"
        process = subprocess.Popen(
            [
                "cmd",
                "/c",
                "run_local.cmd",
                "--no-browser",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            health = None
            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1.0)
                    self.fail(f"run_local.cmd exited early with {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/healthz", timeout=2.0) as response:
                        health = json.load(response)
                    if health.get("ready") is True:
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    time.sleep(0.1)

            self.assertIsNotNone(health)
            self.assertTrue(health["ready"])
            self.assertEqual(health["profile"], "recommended")
            self.assertEqual(health["model_hash"], EXPECTED_MODEL_HASH)
        finally:
            if process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
            try:
                process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=1.0)
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.bind(("127.0.0.1", port))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)

    @unittest.skipUnless(os.name == "nt", "Windows wrapper smoke only runs on Windows")
    def test_wrapper_invalid_arguments_exit_without_pause_when_disabled(self) -> None:
        environment = os.environ.copy()
        environment["RUN_LOCAL_NO_PAUSE"] = "1"

        result = subprocess.run(
            ["cmd", "/c", "run_local.cmd", "--definitely-invalid-option"],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        combined_output = result.stdout + result.stderr
        self.assertIn("--definitely-invalid-option", combined_output)
        self.assertNotIn("Press any key", combined_output)


class LocalOptionsTests(unittest.TestCase):
    def test_defaults_are_loopback_recommended_and_auto_browser(self) -> None:
        options = parse_options([])

        self.assertEqual(options.root, ROOT)
        self.assertEqual(options.host, "127.0.0.1")
        self.assertEqual(options.port, 5000)
        self.assertFalse(options.explicit_port)
        self.assertFalse(options.lan)
        self.assertTrue(options.open_browser)
        self.assertEqual(options.profile_name, "recommended")

    def test_lan_is_explicit_and_rejects_non_loopback_override(self) -> None:
        options = parse_options(["--lan", "--no-browser"])

        self.assertEqual(options.host, "0.0.0.0")
        self.assertTrue(options.lan)
        self.assertFalse(options.open_browser)

        with self.assertRaises(SystemExit):
            parse_options(["--lan", "--host", "192.168.1.20"])
        with self.assertRaises(SystemExit):
            parse_options(["--host", "0.0.0.0"])

    def test_loopback_host_is_normalized(self) -> None:
        options = parse_options(["--host", " LOCALHOST "])

        self.assertEqual(options.host, "localhost")
        self.assertFalse(options.lan)

    def test_explicit_port_is_recorded_and_validated(self) -> None:
        options = parse_options(["--port=5099"])

        self.assertEqual(options.port, 5099)
        self.assertTrue(options.explicit_port)

        for port in ("0", "65536", "abc"):
            with self.subTest(port=port):
                with self.assertRaises(SystemExit):
                    parse_options(["--port", port])

    def test_port_candidates_are_bounded(self) -> None:
        self.assertEqual(list(port_candidates(5000, explicit=False)), list(range(5000, 5011)))
        self.assertEqual(list(port_candidates(5099, explicit=True)), [5099])


class BindServerTests(unittest.TestCase):
    def test_busy_default_port_falls_back(self) -> None:
        app = object()
        calls: list[tuple[str, int, object, bool]] = []
        expected_server = object()

        def factory(host: str, port: int, server_app: object, *, threaded: bool) -> object:
            calls.append((host, port, server_app, threaded))
            if port == 5000:
                raise OSError("busy")
            return expected_server

        server, selected_port = bind_server(
            app=app,
            host="127.0.0.1",
            preferred_port=5000,
            explicit_port=False,
            server_factory=factory,
        )

        self.assertIs(server, expected_server)
        self.assertEqual(selected_port, 5001)
        self.assertEqual(calls, [
            ("127.0.0.1", 5000, app, True),
            ("127.0.0.1", 5001, app, True),
        ])

    def test_explicit_busy_port_fails(self) -> None:
        def factory(host: str, port: int, app: object, *, threaded: bool) -> object:
            raise OSError("busy")

        with self.assertRaises(LocalLaunchError) as context:
            bind_server(
                app=object(),
                host="127.0.0.1",
                preferred_port=5099,
                explicit_port=True,
                server_factory=factory,
            )

        self.assertIn("5099", str(context.exception))

    def test_exhausted_default_range_fails(self) -> None:
        def factory(host: str, port: int, app: object, *, threaded: bool) -> object:
            raise OSError("busy")

        with self.assertRaises(LocalLaunchError) as context:
            bind_server(
                app=object(),
                host="127.0.0.1",
                preferred_port=5000,
                explicit_port=False,
                server_factory=factory,
            )

        self.assertIn("5000-5010", str(context.exception))


class LocalLifecycleTests(unittest.TestCase):
    def test_local_access_host_matches_bind_host(self) -> None:
        self.assertEqual(local_access_host("::1"), "[::1]")
        self.assertEqual(local_access_host("0.0.0.0"), "127.0.0.1")
        self.assertEqual(local_access_host("127.0.0.1"), "127.0.0.1")
        self.assertEqual(local_access_host("localhost"), "localhost")

    def test_build_local_app_always_uses_recommended(self) -> None:
        app, runtime = build_local_app(ROOT)

        self.assertEqual(runtime.profile_name, "recommended")
        self.assertEqual(runtime.bundle.sha256, EXPECTED_MODEL_HASH)
        self.assertIs(app.extensions["winner_runtime"], runtime)

    def test_runtime_startup_error_is_concise(self) -> None:
        with mock.patch("runtime.web_runtime.WinnerRuntime", side_effect=ValueError("bad model")):
            with self.assertRaises(LocalLaunchError) as context:
                build_local_app(ROOT)

        self.assertIn("bad model", str(context.exception))

    def test_open_browser_failure_is_non_fatal(self) -> None:
        output = io.StringIO()

        result = open_browser("http://127.0.0.1:5000/", opener=lambda _url: False, output=output)

        self.assertFalse(result)
        self.assertIn("Open this URL manually", output.getvalue())

    def test_run_prints_shutdown_instruction_and_lan_warnings_without_lan_ip(self) -> None:
        class FakeServer:
            def serve_forever(self) -> None:
                return None

            def shutdown(self) -> None:
                return None

            def server_close(self) -> None:
                return None

        options = LocalOptions(
            root=ROOT,
            host="0.0.0.0",
            port=5099,
            explicit_port=True,
            lan=True,
            open_browser=False,
        )
        output = io.StringIO()

        with mock.patch("local_app.build_local_app", return_value=(object(), object())), \
                mock.patch("local_app.bind_server", return_value=(FakeServer(), 5099)), \
                mock.patch("local_app.wait_for_health", return_value={
                    "ready": True,
                    "profile": "recommended",
                    "model_hash": EXPECTED_MODEL_HASH,
                }), \
                mock.patch("local_app.best_effort_lan_ip", return_value=None), \
                mock.patch("sys.stdout", output):
            self.assertEqual(run(options), 0)

        text = output.getvalue()
        self.assertIn("Press Ctrl+C to stop.", text)
        self.assertIn("LAN mode has no authentication", text)
        self.assertIn("trusted private network only", text)
        self.assertIn("camera access from another device may require HTTPS", text)
        self.assertIn("LAN URL: unavailable", text)

    def test_main_reports_missing_werkzeug_concisely(self) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "werkzeug.serving":
                raise ModuleNotFoundError("No module named 'werkzeug'")
            return real_import(name, *args, **kwargs)

        stderr = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=fake_import), \
                mock.patch("sys.stderr", stderr):
            result = main(["--no-browser"])

        self.assertEqual(result, 1)
        self.assertIn("Local startup failed:", stderr.getvalue())
        self.assertIn("werkzeug", stderr.getvalue())

    def test_subprocess_serves_contract_and_releases_port(self) -> None:
        port = free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "local_app.py"),
                "--no-browser",
                "--port",
                str(port),
            ],
            cwd=ROOT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            health = None
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1.0)
                    self.fail(f"local_app.py exited early with {process.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/api/healthz", timeout=2.0) as response:
                        health = json.load(response)
                    if health.get("ready") is True:
                        break
                except (OSError, ValueError, json.JSONDecodeError):
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
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
            try:
                process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=1.0)
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.bind(("127.0.0.1", port))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
