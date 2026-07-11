from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from joblib import load


@lru_cache(maxsize=4)
def load_model_artifact(model_path: str) -> dict[str, Any]:
    artifact = load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("Winner model artifact must be a dictionary")
    return artifact


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