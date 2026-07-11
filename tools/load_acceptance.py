from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.web_runtime import WinnerRuntime
from web_server import create_app


def nearest_rank_percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(len(ordered) * quantile)))
    return ordered[rank - 1]


def run(
    root: Path,
    sessions: int,
    duration_seconds: int,
    input_fps: int,
    batch_size: int,
    enforce_production_limits: bool = False,
) -> dict:
    runtime = WinnerRuntime(root, profile_name="recommended")
    app = create_app(runtime)
    app.testing = True
    store = app.extensions["winner_session_store"]

    def worker(worker_id: int) -> dict:
        client = app.test_client()
        created = client.post("/api/v1/sessions", json={"source_mode": "file", "target_fps": input_fps})
        if created.status_code != 201:
            raise RuntimeError(created.get_data(as_text=True))
        session_id = created.get_json()["session_id"]
        session = store.sessions[session_id]
        if not enforce_production_limits:
            session.batch_limiter.limit = 10**9
            session.frame_limiter.limit = 10**9
            session.virtual_frame_limiter.limit = 10**9
        timings: list[float] = []
        statuses: list[int] = []
        frames: list[dict] = []
        batch_seq = 0
        total_inputs = duration_seconds * input_fps
        for index in range(total_inputs):
            frames.append({
                "seq": index + 1,
                "timestamp_ms": index * (1000.0 / input_fps),
                "width": 640,
                "height": 480,
                "face_detected": False,
            })
            if len(frames) == batch_size or index == total_inputs - 1:
                batch_seq += 1
                started = time.perf_counter()
                response = client.post(
                    f"/api/v1/sessions/{session_id}/frames",
                    json={"batch_seq": batch_seq, "frames": frames},
                )
                timings.append(time.perf_counter() - started)
                statuses.append(response.status_code)
                if response.status_code != 200:
                    raise RuntimeError(response.get_data(as_text=True))
                frames = []
        summary = client.delete(f"/api/v1/sessions/{session_id}").get_json()["summary"]
        p95 = nearest_rank_percentile(timings, 0.95)
        return {
            "worker": worker_id,
            "requests": len(timings),
            "p95_seconds": p95,
            "max_seconds": max(timings),
            "http_statuses": sorted(set(statuses)),
            "summary": summary,
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=sessions) as pool:
        results = list(pool.map(worker, range(sessions)))
    return {
        "profile": "recommended",
        "sessions": sessions,
        "logical_duration_seconds_per_session": duration_seconds,
        "input_fps": input_fps,
        "batch_size": batch_size,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "rate_limit_note": (
            "Production session rate limits enforced."
            if enforce_production_limits
            else "Raised only for accelerated logical-time replay; API tests and CI smoke cover production limits."
        ),
        "results": results,
        "acceptance": {
            "no_5xx": all(result["http_statuses"] == [200] for result in results),
            "session_isolation": len({result["summary"]["session_id"] for result in results}) == sessions,
            "p95_under_one_second": all(result["p95_seconds"] < 1.0 for result in results),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Accelerated logical-time winner web load acceptance.")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--input-fps", type=int, default=10, choices=[10, 15, 20, 30])
    parser.add_argument("--batch-size", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--enforce-production-limits", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = ROOT
    result = run(
        root,
        args.sessions,
        args.duration_seconds,
        args.input_fps,
        args.batch_size,
        enforce_production_limits=args.enforce_production_limits,
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(result["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
