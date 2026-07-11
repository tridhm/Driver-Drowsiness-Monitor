from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RawPerception:
    face_detected: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_center: Optional[tuple[float, float]] = None


class PerceptionExtractor:
    """Compatibility guard for the removed server-side MediaPipe path."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "Server-side MediaPipe is not available. Use browser landmark packets through /api/v1/sessions."
        )