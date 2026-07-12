from __future__ import annotations

import unittest
from pathlib import Path

from local_app import (
    LocalLaunchError,
    bind_server,
    parse_options,
    port_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
