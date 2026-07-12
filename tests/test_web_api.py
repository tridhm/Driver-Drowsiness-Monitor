from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def landmark_fixture() -> dict[str, list[float]]:
    return {
        "33": [0.30, 0.40], "160": [0.32, 0.38], "158": [0.36, 0.38],
        "133": [0.40, 0.40], "153": [0.36, 0.42], "144": [0.32, 0.42],
        "362": [0.60, 0.40], "385": [0.62, 0.38], "387": [0.66, 0.38],
        "263": [0.70, 0.40], "373": [0.66, 0.42], "380": [0.62, 0.42],
        "61": [0.40, 0.62], "291": [0.60, 0.62],
        "13": [0.50, 0.60], "14": [0.50, 0.66],
        "1": [0.50, 0.48], "152": [0.50, 0.78],
        "468": [0.35, 0.40], "473": [0.65, 0.40],
    }


class WinnerWebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from runtime.web_runtime import WinnerRuntime
        from web_server import create_app

        cls.runtime = WinnerRuntime(ROOT, profile_name="recommended")
        cls.app = create_app(cls.runtime)
        cls.app.testing = True

    def setUp(self) -> None:
        from web_server import create_app

        self.app = create_app(self.runtime)
        self.app.testing = True
        self.client = self.app.test_client()

    def _create_session(self, source_mode: str = "camera", target_fps: int = 20) -> str:
        response = self.client.post(
            "/api/v1/sessions",
            json={"source_mode": source_mode, "target_fps": target_fps},
        )
        self.assertEqual(response.status_code, 201)
        return response.get_json()["session_id"]

    def test_health_and_runtime_info_expose_validated_contract(self) -> None:
        health = self.client.get("/api/healthz")
        self.assertEqual(health.status_code, 200)
        body = health.get_json()
        self.assertTrue(body["ready"])
        self.assertEqual(body["profile"], "recommended")
        self.assertEqual(body["model_hash"], self.runtime.bundle.sha256)
        self.assertEqual(body["runtime_threshold"], 0.55)

        info = self.client.get("/api/v1/runtime-info").get_json()
        self.assertEqual(info["reference_fps"], 30)
        self.assertEqual(info["landmark_count"], 20)
        self.assertFalse(info["video_upload_enabled"])
        self.assertEqual(info["session_idle_timeout_seconds"], self.app.extensions["winner_session_store"].idle_timeout_seconds)
        self.assertEqual(info["capture_stall_tolerance_ms"], 3000)

    def test_create_process_reset_and_delete_session(self) -> None:
        session_id = self._create_session()
        packet = {
            "batch_seq": 1,
            "frames": [{
                "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                "face_detected": True, "landmarks": landmark_fixture(),
            }],
        }
        decision = self.client.post(f"/api/v1/sessions/{session_id}/frames", json=packet)
        self.assertEqual(decision.status_code, 200)
        body = decision.get_json()
        self.assertEqual(body["session_id"], session_id)
        self.assertEqual(body["profile"], "recommended")
        self.assertIn("audio_command", body)
        self.assertIn("ear", body["metrics"])
        self.assertIn("mar", body["metrics"])
        self.assertIn("pitch", body["metrics"])
        self.assertEqual(body["metrics"]["mar_threshold"], 0.5)
        self.assertIn("head_nod_detected", body["metrics"])

        reset = self.client.post(f"/api/v1/sessions/{session_id}/reset")
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.get_json()["calibration"]["valid_face_frames"], 0)

        deleted = self.client.delete(f"/api/v1/sessions/{session_id}")
        self.assertEqual(deleted.status_code, 200)
        missing = self.client.post(f"/api/v1/sessions/{session_id}/reset")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.get_json()["error_code"], "SESSION_NOT_FOUND")

    def test_landmark_contract_rejects_extra_or_out_of_range_points(self) -> None:
        session_id = self._create_session()
        extra = landmark_fixture()
        extra["2"] = [0.5, 0.5]
        response = self.client.post(f"/api/v1/sessions/{session_id}/frames", json={
            "batch_seq": 1,
            "frames": [{
                "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                "face_detected": True, "landmarks": extra,
            }],
        })
        self.assertEqual(response.status_code, 400)

        invalid = landmark_fixture()
        invalid["1"] = [1.2, 0.5]
        response = self.client.post(f"/api/v1/sessions/{session_id}/frames", json={
            "batch_seq": 1,
            "frames": [{
                "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                "face_detected": True, "landmarks": invalid,
            }],
        })
        self.assertEqual(response.status_code, 400)
    def test_payload_and_batch_limits_are_enforced(self) -> None:
        session_id = self._create_session()
        oversized = json.dumps({"padding": "x" * (64 * 1024)}).encode("utf-8")
        response = self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            data=oversized,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)

        too_many = {
            "batch_seq": 1,
            "frames": [
                {"seq": index, "timestamp_ms": float(index), "width": 1, "height": 1, "face_detected": False}
                for index in range(1, 6)
            ],
        }
        response = self.client.post(f"/api/v1/sessions/{session_id}/frames", json=too_many)
        self.assertEqual(response.status_code, 400)

    def test_create_requires_json_rejects_cross_origin_and_validates_target_fps(self) -> None:
        form = self.client.post("/api/v1/sessions", data={"source_mode": "camera", "target_fps": "20"})
        self.assertEqual(form.status_code, 415)

        cross_origin = self.client.post(
            "/api/v1/sessions",
            json={"source_mode": "camera", "target_fps": 20},
            headers={"Origin": "https://attacker.example"},
        )
        self.assertEqual(cross_origin.status_code, 403)

        for invalid in (None, {}, "20", 12.5):
            response = self.client.post(
                "/api/v1/sessions",
                json={"source_mode": "camera", "target_fps": invalid},
            )
            self.assertEqual(response.status_code, 400, invalid)

    def test_privacy_schema_rejects_unknown_batch_and_frame_fields(self) -> None:
        session_id = self._create_session()
        frame = {
            "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
            "face_detected": True, "landmarks": landmark_fixture(),
        }
        unknown_batch = self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            json={"batch_seq": 1, "frames": [frame], "video": "data:video/mp4;base64,AAAA"},
        )
        self.assertEqual(unknown_batch.status_code, 400)

        unknown_frame = self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            json={"batch_seq": 1, "frames": [{**frame, "jpeg": "data:image/jpeg;base64,AAAA"}]},
        )
        self.assertEqual(unknown_frame.status_code, 400)

        no_face_media = self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            json={
                "batch_seq": 1,
                "frames": [{
                    "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                    "face_detected": False, "landmarks": "data:image/jpeg;base64,AAAA",
                }],
            },
        )
        self.assertEqual(no_face_media.status_code, 400)

    def test_request_body_limit_applies_to_session_creation_before_json_parsing(self) -> None:
        oversized = json.dumps({"padding": "x" * (64 * 1024)}).encode("utf-8")
        response = self.client.post("/api/v1/sessions", data=oversized, content_type="application/json")
        self.assertEqual(response.status_code, 413)

    def test_per_client_session_cap_does_not_block_other_clients(self) -> None:
        from web_server import create_app

        app = create_app(self.runtime)
        app.testing = True
        client = app.test_client()
        for _ in range(4):
            response = client.post(
                "/api/v1/sessions",
                json={"source_mode": "camera", "target_fps": 20},
                environ_base={"REMOTE_ADDR": "10.0.0.1"},
            )
            self.assertEqual(response.status_code, 201)
        rejected = client.post(
            "/api/v1/sessions",
            json={"source_mode": "camera", "target_fps": 20},
            environ_base={"REMOTE_ADDR": "10.0.0.1"},
        )
        self.assertEqual(rejected.status_code, 429)

        other_client = client.post(
            "/api/v1/sessions",
            json={"source_mode": "camera", "target_fps": 20},
            environ_base={"REMOTE_ADDR": "10.0.0.2"},
        )
        self.assertEqual(other_client.status_code, 201)

    def test_sessions_are_isolated_expire_and_batch_rate_is_limited(self) -> None:
        first = self._create_session()
        second = self._create_session()
        base = {
            "frames": [{
                "seq": 1, "timestamp_ms": 0.0, "width": 1000, "height": 800,
                "face_detected": True, "landmarks": landmark_fixture(),
            }],
        }
        self.assertEqual(
            self.client.post(f"/api/v1/sessions/{first}/frames", json={"batch_seq": 1, **base}).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(f"/api/v1/sessions/{second}/frames", json={"batch_seq": 1, **base}).status_code,
            200,
        )

        limited = self._create_session()
        status_codes = []
        for index in range(1, 14):
            status_codes.append(self.client.post(
                f"/api/v1/sessions/{limited}/frames",
                json={
                    "batch_seq": index,
                    "frames": [{
                        "seq": index, "timestamp_ms": float(index * 40), "width": 2, "height": 2,
                        "face_detected": False,
                    }],
                },
            ).status_code)
        self.assertEqual(status_codes[-1], 429)

        store = self.app.extensions["winner_session_store"]
        store.sessions[first].last_activity -= 301.0
        expired = self.client.post(f"/api/v1/sessions/{first}/reset")
        self.assertEqual(expired.status_code, 404)
    def test_http_batches_resample_jitter_and_keep_retries_idempotent(self) -> None:
        session_id = self._create_session(target_fps=20)
        first_payload = {
            "batch_seq": 1,
            "frames": [
                {
                    "seq": index + 1,
                    "timestamp_ms": timestamp_ms,
                    "width": 1000,
                    "height": 800,
                    "face_detected": True,
                    "landmarks": landmark_fixture(),
                }
                for index, timestamp_ms in enumerate((0.0, 52.0, 101.0, 151.0))
            ],
        }
        first = self.client.post(f"/api/v1/sessions/{session_id}/frames", json=first_payload)
        self.assertEqual(first.status_code, 200)
        duplicate = self.client.post(f"/api/v1/sessions/{session_id}/frames", json=first_payload)
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(duplicate.get_json(), first.get_json())

        changed = json.loads(json.dumps(first_payload))
        changed["frames"][0]["timestamp_ms"] = 1.0
        conflict = self.client.post(f"/api/v1/sessions/{session_id}/frames", json=changed)
        self.assertEqual(conflict.status_code, 409)

        second_payload = {
            "batch_seq": 2,
            "frames": [
                {
                    "seq": index + 5,
                    "timestamp_ms": timestamp_ms,
                    "width": 1000,
                    "height": 800,
                    "face_detected": True,
                    "landmarks": landmark_fixture(),
                }
                for index, timestamp_ms in enumerate((201.0, 252.0, 600.0, 650.0))
            ],
        }
        second = self.client.post(f"/api/v1/sessions/{session_id}/frames", json=second_payload)
        self.assertEqual(second.status_code, 200)
        body = second.get_json()
        self.assertEqual(body["input_frames"], 8)
        self.assertGreater(body["virtual_frames"], body["input_frames"])

    def test_http_session_survives_transient_1500ms_capture_stall(self) -> None:
        session_id = self._create_session(target_fps=20)
        first = self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            json={
                "batch_seq": 1,
                "frames": [{
                    "seq": 1,
                    "timestamp_ms": 0.0,
                    "width": 1000,
                    "height": 800,
                    "face_detected": False,
                    "landmarks": {},
                }],
            },
        )
        self.assertEqual(first.status_code, 200)

        stalled = self.client.post(
            f"/api/v1/sessions/{session_id}/frames",
            json={
                "batch_seq": 2,
                "frames": [{
                    "seq": 2,
                    "timestamp_ms": 1500.0,
                    "width": 1000,
                    "height": 800,
                    "face_detected": False,
                    "landmarks": {},
                }],
            },
        )

        self.assertEqual(stalled.status_code, 200)
        self.assertEqual(stalled.get_json()["session_id"], session_id)
        self.assertGreater(stalled.get_json()["virtual_frames"], 32)

    def test_session_store_caps_active_sessions(self) -> None:
        from web_server import create_app

        app = create_app(self.runtime)
        app.testing = True
        client = app.test_client()
        for client_index in range(4):
            for _ in range(4):
                response = client.post(
                    "/api/v1/sessions",
                    json={"source_mode": "camera", "target_fps": 20},
                    environ_base={"REMOTE_ADDR": f"10.0.0.{client_index + 1}"},
                )
                self.assertEqual(response.status_code, 201)
        rejected = client.post(
            "/api/v1/sessions",
            json={"source_mode": "camera", "target_fps": 20},
            environ_base={"REMOTE_ADDR": "10.0.0.99"},
        )
        self.assertEqual(rejected.status_code, 429)
        self.assertIn("active session limit", rejected.get_json()["error"].lower())
    def test_primary_page_labels_probability_and_six_second_perclos_honestly(self) -> None:
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("Sleepy probability", html)
        self.assertIn("PERCLOS 6s", html)
        self.assertIn('id="runtimeLocation"', html)
        self.assertIn("runtimeLocationLabel(window.location.hostname)", html)
        self.assertIn("dynamic_total_progress", html)
        self.assertIn("FPS xử lý thay đổi ngay từ frame tiếp theo", html)
        self.assertNotIn("sau reset", html)
        self.assertNotIn("Evidence score", html)
        self.assertIn("/static/vendor/face_mesh/face_mesh.js", html)
        self.assertNotIn("cdn.jsdelivr.net", html)
        self.assertIn("if(running) sessionClock.start();", html)
        self.assertIn("runGuard.isCurrent", html)
        self.assertIn("canFlush:()=>!sendInFlight", html)
        self.assertIn("if(!batcher.canAccept()) return;", html)
        self.assertNotIn("pendingSend = pendingSend.then", html)
        self.assertIn("await api.requestSession()", html)
        self.assertIn("audioState.consume", html)
        self.assertIn("if(audioUnlocked || audioContinuous || audioState.continuousRequested) return;", html)
        self.assertIn("alarmAudio.volume=1;\n    if(generation!==audioGeneration", html)
        self.assertEqual(html.count("const stopPromise=fullStop(false, false); unlockAudio(); await stopPromise;"), 2)
        self.assertIn("window.addEventListener('pagehide', ()=>{", html)
        self.assertIn("api.abortActiveRequests();", html)
        self.assertIn("void api.deleteSessionKeepalive();", html)
        self.assertNotIn("activeSendPromise=Promise.resolve();", html)
        self.assertIn("window.addEventListener('pageshow', event=>{", html)
        self.assertIn("if(!event.persisted) return;", html)
    def test_upload_redirects_to_local_file_mode_and_legacy_routes_are_gone(self) -> None:
        upload = self.client.get("/upload")
        self.assertEqual(upload.status_code, 302)
        self.assertIn("mode=file", upload.headers["Location"])

        for route in ("/api/mobile/frame", "/api/upload", "/api/start", "/video_feed"):
            response = self.client.post(route) if route != "/video_feed" else self.client.get(route)
            self.assertEqual(response.status_code, 410, route)
            self.assertEqual(response.get_json()["error_code"], "LEGACY_ROUTE_GONE")


if __name__ == "__main__":
    unittest.main()
