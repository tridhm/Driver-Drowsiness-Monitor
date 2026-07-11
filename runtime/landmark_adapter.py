from __future__ import annotations

import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from runtime.perception import RawPerception


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (61, 291, 13, 14)
HEAD_POSE = (1, 152, 33, 263, 61, 291)
IRIS = (468, 473)
REQUIRED_LANDMARKS = tuple(sorted(set(LEFT_EYE + RIGHT_EYE + MOUTH + HEAD_POSE + IRIS)))



def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(points: Sequence[tuple[float, float]]) -> float:
    denominator = 2.0 * _distance(points[0], points[3])
    return (_distance(points[1], points[5]) + _distance(points[2], points[4])) / denominator if denominator > 1e-9 else 0.0


def _mar(points: Sequence[tuple[float, float]]) -> float:
    denominator = _distance(points[0], points[1])
    return _distance(points[2], points[3]) / denominator if denominator > 1e-9 else 0.0


class LandmarkPacketAdapter:
    required_indices = REQUIRED_LANDMARKS

    def from_normalized(
        self,
        landmarks: Mapping[int | str, Sequence[float]],
        *,
        width: int,
        height: int,
        face_detected: bool,
    ) -> RawPerception:
        if not face_detected:
            return RawPerception(face_detected=False)
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")

        normalized = {int(key): value for key, value in landmarks.items()}
        missing = [index for index in REQUIRED_LANDMARKS if index not in normalized]
        if missing:
            raise ValueError(f"Missing required landmarks: {missing}")

        def point(index: int) -> tuple[float, float]:
            value = normalized[index]
            if len(value) < 2:
                raise ValueError(f"Landmark {index} must contain x and y")
            x, y = float(value[0]), float(value[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Landmark {index} must be finite")
            return x * width, y * height

        left = [point(index) for index in LEFT_EYE]
        right = [point(index) for index in RIGHT_EYE]
        mouth = [point(index) for index in MOUTH]
        pitch, yaw, roll = self._head_pose([point(index) for index in HEAD_POSE], width, height)
        left_iris, right_iris = [point(index) for index in IRIS]

        return RawPerception(
            face_detected=True,
            ear=(_ear(left) + _ear(right)) / 2.0,
            mar=_mar(mouth),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            gaze_center=((left_iris[0] + right_iris[0]) / 2.0, (left_iris[1] + right_iris[1]) / 2.0),
        )

    @staticmethod
    def _head_pose(points: Sequence[tuple[float, float]], width: int, height: int) -> tuple[float, float, float]:
        image_points = np.asarray(points, dtype="double")
        model_points = np.asarray(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype="double",
        )
        focal_length = float(width)
        camera_matrix = np.asarray(
            [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0.0, 0.0, 1.0]],
            dtype="double",
        )
        ok, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch = math.degrees(math.atan2(rotation_matrix[2][1], rotation_matrix[2][2]))
        yaw = math.degrees(math.atan2(-rotation_matrix[2][0], math.hypot(rotation_matrix[2][1], rotation_matrix[2][2])))
        roll = math.degrees(math.atan2(rotation_matrix[1][0], rotation_matrix[0][0]))
        return pitch, yaw, roll
