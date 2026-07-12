from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load


@lru_cache(maxsize=4)
def load_model_artifact(model_path: str) -> dict[str, Any]:
    artifact = load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Winner model artifact must be a dictionary")
    return artifact


@dataclass(frozen=True)
class _IsotonicFold:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray


class FastIsotonicBinaryPredictor:
    """Validated inference fast path for the packaged calibrated logistic model."""

    def __init__(self, folds: tuple[_IsotonicFold, ...], feature_count: int) -> None:
        self.folds = folds
        self.feature_count = int(feature_count)

    @classmethod
    def from_model(cls, model: Any) -> "FastIsotonicBinaryPredictor":
        classes = [int(value) for value in getattr(model, "classes_", [])]
        calibrated = list(getattr(model, "calibrated_classifiers_", []))
        if classes != [0, 1] or not calibrated:
            raise ValueError("Fast predictor requires a fitted binary calibrated classifier")

        folds: list[_IsotonicFold] = []
        for classifier in calibrated:
            if str(getattr(classifier, "method", "")) != "isotonic":
                raise ValueError("Fast predictor requires isotonic calibration")
            estimator = getattr(classifier, "estimator", None)
            named_steps = getattr(estimator, "named_steps", {})
            scaler = named_steps.get("standardscaler")
            logistic = named_steps.get("logisticregression")
            calibrators = list(getattr(classifier, "calibrators", []))
            if scaler is None or logistic is None or len(calibrators) != 1:
                raise ValueError("Fast predictor requires StandardScaler + LogisticRegression")

            coefficients = np.asarray(logistic.coef_, dtype=np.float64)
            intercept = np.asarray(logistic.intercept_, dtype=np.float64)
            mean = np.asarray(scaler.mean_, dtype=np.float64)
            scale = np.asarray(scaler.scale_, dtype=np.float64)
            calibrator = calibrators[0]
            if coefficients.shape != (1, mean.size) or intercept.shape != (1,):
                raise ValueError("Fast predictor received an unexpected logistic shape")
            if scale.shape != mean.shape or np.any(scale == 0.0):
                raise ValueError("Fast predictor received an invalid scaler contract")

            folds.append(_IsotonicFold(
                mean=mean.copy(),
                scale=scale.copy(),
                coefficients=coefficients[0].copy(),
                intercept=float(intercept[0]),
                x_thresholds=np.asarray(calibrator.X_thresholds_, dtype=np.float64).copy(),
                y_thresholds=np.asarray(calibrator.y_thresholds_, dtype=np.float64).copy(),
            ))

        predictor = cls(
            tuple(folds),
            feature_count=int(getattr(model, "n_features_in_", folds[0].mean.size)),
        )
        predictor._validate_against(model)
        return predictor

    def predict_proba(self, rows: Any) -> np.ndarray:
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} model features")

        positive = np.zeros(matrix.shape[0], dtype=np.float64)
        for fold in self.folds:
            scores = ((matrix - fold.mean) / fold.scale) @ fold.coefficients + fold.intercept
            positive += np.interp(scores, fold.x_thresholds, fold.y_thresholds)
        positive /= len(self.folds)
        return np.column_stack((1.0 - positive, positive))

    def _validate_against(self, model: Any) -> None:
        probes = [np.zeros(self.feature_count, dtype=np.float64)]
        for fold in self.folds:
            probes.extend((fold.mean, fold.mean - fold.scale, fold.mean + fold.scale))
        matrix = np.vstack(probes)
        expected = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        actual = self.predict_proba(matrix)
        if expected.shape != actual.shape or not np.allclose(expected, actual, rtol=0.0, atol=1e-12):
            raise ValueError("Fast predictor does not match the packaged sklearn model")


@dataclass(frozen=True)
class WinnerModelBundle:
    model_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    artifact: dict[str, Any]

    @classmethod
    def load(cls, model_path: Path, manifest_path: Path) -> "WinnerModelBundle":
        model_path = Path(model_path)
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != str(manifest.get("sha256", "")).lower():
            raise ValueError(f"Winner model SHA-256 mismatch: {digest}")

        artifact = load_model_artifact(str(model_path.resolve()))
        if "model" not in artifact or "feature_columns" not in artifact:
            raise ValueError("Winner artifact must contain model and feature_columns")
        columns = [str(value) for value in artifact["feature_columns"]]
        if columns != [str(value) for value in manifest.get("feature_columns", [])]:
            raise ValueError("Winner feature contract does not match manifest")
        if float(artifact.get("probability_threshold", 0.0)) != float(manifest["artifact_threshold"]):
            raise ValueError("Winner artifact threshold does not match manifest")
        return cls(model_path, manifest_path, manifest, artifact)

    @property
    def model(self):
        return self.artifact["model"]

    @property
    def feature_columns(self) -> list[str]:
        return [str(value) for value in self.artifact["feature_columns"]]

    @property
    def artifact_threshold(self) -> float:
        return float(self.artifact["probability_threshold"])

    @property
    def runtime_threshold(self) -> float:
        return float(self.manifest["runtime_threshold"])

    @property
    def sha256(self) -> str:
        return str(self.manifest["sha256"])