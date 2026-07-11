from __future__ import annotations

from runtime.config import RuntimeConfig
from runtime.engines.camera_hybrid_engine import CameraHybridDecisionEngine
from runtime.engines.base import DecisionEngine
from runtime.engines.camera_model_engine import CameraModelDecisionEngine
from runtime.engines.fsm_engine import FSMDecisionEngine
from runtime.engines.legacy_engine import LegacyRuleEngine


ENGINE_REGISTRY: dict[str, type[DecisionEngine]] = {
    "camera_hybrid": CameraHybridDecisionEngine,
    "camera_model": CameraModelDecisionEngine,
    "fsm": FSMDecisionEngine,
    "legacy": LegacyRuleEngine,
}


def available_engines() -> list[str]:
    return sorted(ENGINE_REGISTRY.keys())


def create_engine(name: str, config: RuntimeConfig) -> DecisionEngine:
    if name not in ENGINE_REGISTRY:
        available = ", ".join(available_engines())
        raise ValueError(f"Unsupported decision engine '{name}'. Available: {available}")
    return ENGINE_REGISTRY[name](config)
