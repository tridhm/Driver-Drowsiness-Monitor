from __future__ import annotations

import time
from typing import Any


class AudioAlertController:
    """Telemetry-only compatibility controller.

    Production audio is commanded explicitly to the browser by the landmark API.
    This class preserves the old runtime event contract without loading or playing
    audio on the server.
    """

    def __init__(self, sound_file: str, drowsy_cooldown_seconds: float = 5.0):
        self.sound_file = sound_file
        self.drowsy_cooldown_seconds = max(0.0, float(drowsy_cooldown_seconds))
        self._continuous_active = False
        self._last_double_alert: float | None = None

    def update(self, sound_type: str, now: float | None = None, clock_source: str = "wall_clock") -> dict[str, Any]:
        requested_sound = str(sound_type or "none")
        clock_time = time.time() if now is None else float(now)
        event = self._event(requested_sound, clock_time=clock_time, clock_source=clock_source)

        if requested_sound == "continuous":
            event["audio_action"] = "continuous_active" if self._continuous_active else "continuous_started"
            self._continuous_active = True
            return event

        if self._continuous_active:
            self._continuous_active = False
            event["continuous_stopped"] = True

        if requested_sound == "double":
            previous = self._last_double_alert
            seconds_since_last = None if previous is None else max(0.0, clock_time - previous)
            event["seconds_since_last_double"] = seconds_since_last
            if previous is None or float(seconds_since_last) >= self.drowsy_cooldown_seconds:
                self._last_double_alert = clock_time
                event["audio_action"] = "double_fired"
                event["double_alert_fired"] = True
            else:
                event["audio_action"] = "double_cooldown_suppressed"
                event["double_alert_suppressed"] = True
        return event

    def close(self) -> None:
        self._continuous_active = False

    def _event(self, requested_sound: str, clock_time: float, clock_source: str) -> dict[str, Any]:
        return {
            "requested_sound": requested_sound,
            "audio_action": "none",
            "double_alert_fired": False,
            "double_alert_suppressed": False,
            "seconds_since_last_double": None,
            "cooldown_seconds": self.drowsy_cooldown_seconds,
            "continuous_stopped": False,
            "clock_time": float(clock_time),
            "clock_source": str(clock_source or "wall_clock"),
        }