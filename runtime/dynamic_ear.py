from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


EWMA_LAMBDA = 0.20
SQAD_P = 0.6827
SQAD_K = 2.5
SQAD_WINDOW = 150
SQAD_FLOOR = 0.18
SQAD_GAP = 0.013
WARMUP_FRAMES = 30
CALIB_FRAMES = 270


@dataclass(frozen=True)
class FrameStatus:
    phase: str
    is_closed: bool
    T_low: Optional[float]
    mu: float
    sigma: float
    progress: float
    n: int

    @property
    def locked(self) -> bool:
        return self.phase == "LOCKED"


class DynamicEAR:
    def __init__(self) -> None:
        self._window: deque[float] = deque(maxlen=SQAD_WINDOW)
        self._mu: float | None = None
        self._sigma = 0.02
        self._sqad_counter = 0
        self._closed = False
        self._phase = "WARMUP"
        self._frame = 0
        self._locked_t_low: float | None = None

    def update(self, ear_raw: float) -> FrameStatus:
        ear = float(ear_raw)
        self._frame += 1
        self._mu = ear if self._mu is None else EWMA_LAMBDA * ear + (1.0 - EWMA_LAMBDA) * self._mu

        if self._phase == "LOCKED":
            threshold = float(self._locked_t_low)
            if not self._closed and ear < threshold:
                self._closed = True
            elif self._closed and ear > threshold + SQAD_GAP:
                self._closed = False
            return self._status(self._closed, 1.0)

        self._window.append(ear)
        if self._phase == "WARMUP":
            if self._frame >= WARMUP_FRAMES:
                self._phase = "CALIBRATING"
            return self._status(False, min(1.0, self._frame / WARMUP_FRAMES), phase="WARMUP")

        self._sigma = self._compute_sqad()
        frame_in_calib = self._frame - WARMUP_FRAMES
        if frame_in_calib >= CALIB_FRAMES:
            self._locked_t_low = max(float(self._mu) - SQAD_K * self._sigma, SQAD_FLOOR)
            self._phase = "LOCKED"
        return self._status(
            False,
            min(1.0, frame_in_calib / CALIB_FRAMES),
            phase="CALIBRATING",
            expose_threshold=False,
        )

    @property
    def locked(self) -> bool:
        return self._phase == "LOCKED"

    @property
    def T_low(self) -> float | None:
        return self._locked_t_low

    def reset(self) -> None:
        self.__init__()

    def _compute_sqad(self) -> float:
        if len(self._window) < 10:
            return self._sigma
        self._sqad_counter += 1
        if self._sqad_counter < 5:
            return self._sigma
        self._sqad_counter = 0
        values = np.asarray(self._window, dtype=float)
        deviations = np.abs(values - np.median(values))
        return max(float(np.quantile(deviations, SQAD_P)), 0.003)

    def _status(
        self,
        is_closed: bool,
        progress: float,
        phase: str | None = None,
        *,
        expose_threshold: bool = True,
    ) -> FrameStatus:
        return FrameStatus(
            phase=phase or self._phase,
            is_closed=is_closed,
            T_low=self._locked_t_low if expose_threshold else None,
            mu=float(self._mu or 0.0),
            sigma=float(self._sigma),
            progress=float(progress),
            n=len(self._window),
        )
