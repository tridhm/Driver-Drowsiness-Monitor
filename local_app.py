from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
LAST_FALLBACK_PORT = 5010
LOOPBACK_HOSTS = frozenset(("127.0.0.1", "localhost", "::1"))
STARTUP_TIMEOUT_SECONDS = 15.0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local driver drowsiness web app.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=_valid_port, default=DEFAULT_PORT)
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false", default=True)
    return parser


def parse_options(argv: Sequence[str] | None = None) -> LocalOptions:
    args_list = list(argv) if argv is not None else None
    parser = build_parser()
    args = parser.parse_args(args_list)
    explicit_port = bool(args_list and any(item == "--port" or item.startswith("--port=") for item in args_list))
    explicit_host = bool(args_list and any(item == "--host" or item.startswith("--host=") for item in args_list))
    requested_host = args.host.strip().lower()

    if args.lan:
        if explicit_host and requested_host not in LOOPBACK_HOSTS:
            parser.error("--lan cannot be combined with a non-loopback --host")
        host = "0.0.0.0"
    else:
        if requested_host not in LOOPBACK_HOSTS:
            parser.error("non-loopback hosts require --lan")
        host = requested_host

    return LocalOptions(
        root=ROOT,
        host=host,
        port=args.port,
        explicit_port=explicit_port,
        lan=args.lan,
        open_browser=args.open_browser,
    )


def port_candidates(preferred_port: int, explicit: bool) -> Iterable[int]:
    if explicit:
        return (preferred_port,)
    return range(preferred_port, LAST_FALLBACK_PORT + 1)


def bind_server(
    *,
    app: object,
    host: str,
    preferred_port: int,
    explicit_port: bool,
    server_factory: Callable[..., object],
) -> tuple[object, int]:
    last_error: OSError | None = None
    for port in port_candidates(preferred_port, explicit=explicit_port):
        try:
            return server_factory(host, port, app, threaded=True), port
        except OSError as exc:
            last_error = exc
            if explicit_port:
                raise LocalLaunchError(f"Port {port} is not available.") from exc

    message = f"No available localhost port in {DEFAULT_PORT}-{LAST_FALLBACK_PORT}."
    raise LocalLaunchError(message) from last_error


def build_local_app(root: Path = ROOT) -> tuple[object, object]:
    try:
        from runtime.web_runtime import WinnerRuntime
        from web_server import create_app

        runtime = WinnerRuntime(root, profile_name="recommended")
        app = create_app(runtime)
        return app, runtime
    except Exception as exc:
        raise LocalLaunchError(f"Winner runtime is not ready: {exc}") from exc


def wait_for_health(url: str, timeout_seconds: float = STARTUP_TIMEOUT_SECONDS) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2.0) as response:
                body = json.load(response)
            if body.get("ready") is True:
                return body
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise LocalLaunchError("Local health endpoint did not become ready.") from last_error


def open_browser(
    url: str,
    opener: Callable[[str], bool] = webbrowser.open,
    output: Any = sys.stdout,
) -> bool:
    try:
        opened = bool(opener(url))
    except Exception:
        opened = False
    if not opened:
        print(f"Open this URL manually: {url}", file=output)
        return False
    return True


def best_effort_lan_ip() -> str | None:
    try:
        candidates = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
    except OSError:
        return None
    for candidate in candidates:
        address = candidate[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.version == 4 and ip.is_private and not ip.is_loopback:
            return address
    return None


def run(options: LocalOptions) -> int:
    from werkzeug.serving import make_server

    app, runtime = build_local_app(options.root)
    server, selected_port = bind_server(
        app=app,
        host=options.host,
        preferred_port=options.port,
        explicit_port=options.explicit_port,
        server_factory=make_server,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        local_url = f"http://127.0.0.1:{selected_port}/"
        health = wait_for_health(local_url + "api/healthz")
        print(f"Local URL: {local_url}")
        print(f"Profile: {health['profile']}")
        print(f"Model: {health['model_hash'][:12]}...")
        print("Privacy: Video and images remain in the browser; only normalized landmark JSON is sent.")
        if options.lan:
            print("LAN mode has no authentication; trusted private network only.")
            print("camera access from another device may require HTTPS.")
            lan_ip = best_effort_lan_ip()
            if lan_ip is None:
                print("LAN URL: unavailable; open the local URL on this computer.")
            else:
                print(f"LAN URL: http://{lan_ip}:{selected_port}/")
        if options.open_browser:
            open_browser(local_url)
        print("Press Ctrl+C to stop.")
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            print("Local server stopped.")
        return 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_options(argv))
    except LocalLaunchError as exc:
        print(f"Local startup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
