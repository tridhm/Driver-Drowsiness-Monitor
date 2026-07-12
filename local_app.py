from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
LAST_FALLBACK_PORT = 5010
LOOPBACK_HOSTS = frozenset(("127.0.0.1", "localhost", "::1"))


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


ServerFactory = Callable[[str, int, object], object]


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

    if args.lan:
        if explicit_host and args.host not in LOOPBACK_HOSTS:
            parser.error("--lan cannot be combined with a non-loopback --host")
        host = "0.0.0.0"
    else:
        if args.host not in LOOPBACK_HOSTS:
            parser.error("non-loopback hosts require --lan")
        host = args.host

    return LocalOptions(
        root=ROOT,
        host=host,
        port=args.port,
        explicit_port=explicit_port,
        lan=args.lan,
        open_browser=args.open_browser,
    )


def port_candidates(options: LocalOptions) -> Iterable[int]:
    if options.explicit_port:
        return (options.port,)
    return range(DEFAULT_PORT, LAST_FALLBACK_PORT + 1)


def _default_server_factory(host: str, port: int, app: object, *, threaded: bool) -> object:
    from werkzeug.serving import make_server

    return make_server(host, port, app, threaded=threaded)


def bind_server(
    options: LocalOptions,
    app: object,
    *,
    server_factory: Callable[..., object] = _default_server_factory,
) -> object:
    last_error: OSError | None = None
    for port in port_candidates(options):
        try:
            return server_factory(options.host, port, app, threaded=True)
        except OSError as exc:
            last_error = exc
            if options.explicit_port:
                raise LocalLaunchError(f"Port {port} is not available.") from exc

    message = f"No available localhost port in {DEFAULT_PORT}-{LAST_FALLBACK_PORT}."
    raise LocalLaunchError(message) from last_error
