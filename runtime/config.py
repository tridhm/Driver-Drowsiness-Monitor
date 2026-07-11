from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InputConfig:
    source: str = "webcam"
    video_path: str = "Video Database\\Sub 03.avi"
    loop_file: bool = True


@dataclass
class RuntimeTuning:
    fps: float = 30.0
    use_file_video_fps: bool = False
    warmup_seconds: float = 3.0
    calibration_frames: int = 60
    max_frames: int = 0


@dataclass
class ThresholdConfig:
    ear_default: float = 0.23
    ear_calibration_factor: float = 0.65
    ear_min: float = 0.15
    ear_max: float = 0.35
    mar: float = 0.50
    pitch: float = 20.0
    head_yaw: float = 30.0
    head_nod_frames: int = 30
    blink_freq: int = 10
    yawn_frames: int = 20
    max_yawns_window: int = 3
    gaze_move: float = 1.5


@dataclass
class WindowConfig:
    perclos_seconds: float = 60.0
    perclos_short_seconds: float = 6.0
    yawn_window_seconds: float = 60.0
    blink_window_seconds: float = 10.0


@dataclass
class AlertPolicy:
    drowsy_cooldown_seconds: float = 5.0


@dataclass
class PerceptionConfig:
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.60
    min_tracking_confidence: float = 0.60


@dataclass
class CameraModelConfig:
    @dataclass
    class QualityGuardConfig:
        enabled: bool = False
        min_valid_face_ratio: float = 0.50

    model_path: str = "reports/exhaustive_best_model_2026_06_26/deploy_best_179_policy_winner/uldd_17978727_camera_model.joblib"
    window_seconds: float = 60.0
    min_window_seconds: float = 5.0
    min_frames: int = 5
    probability_threshold: float | None = 0.55
    suppress_warmup_alerts: bool = False
    quality_guard: QualityGuardConfig = field(default_factory=QualityGuardConfig)


@dataclass
class HybridPolicyConfig:
    @dataclass
    class LowProbabilityReleaseConfig:
        enabled: bool = False
        probability_max: float = 0.10
        clean_streak: int = 3
        max_base_fsm_evidence: float = 0.20
        max_perclos_short: float = 0.15
        max_perclos_long: float | None = None
        max_eyes_closed_consecutive: int = 1
        max_yawn_count: float = 0.0
        max_head_drop_count: float = 0.0
        min_head_drop_count: float = 0.0
        max_max_eye_closed_duration_sec: float | None = None
        max_max_fsm_evidence: float | None = None
        target_state: str = "SUSPICIOUS"

    @dataclass
    class SubtleRescueConfig:
        enabled: bool = True
        probability_low: float = 0.45
        probability_high: float = 0.52
        ear_std_min: float = 0.055
        ear_p10_max: float = 0.10
        head_drop_max: int = 0

    @dataclass
    class FPSuppressionConfig:
        enabled: bool = True
        probability_high: float = 0.45
        head_drop_min: int = 500
        maxeye_max: float = 1.0

    @dataclass
    class SeededMacroeventBridgeConfig:
        enabled: bool = False
        probability_min: float = 0.42
        head_drop_min: float = 400.0
        max_mar_min: float = 0.15
        min_seed_hits: int = 2
        bridge_perclos_max: float = 0.30
        bridge_maxeye_max: float = 1.0
        bridge_mean_mar_min: float = 0.005
        bridge_min_max_fsm_evidence: float | None = None
        target_state: str = "DROWSY"
        alert_sound_override: str | None = None
        active_seconds: float = 240.0

    @dataclass
    class GuardedSevereRescueConfig:
        enabled: bool = False
        min_max_fsm_evidence: float = 0.85
        min_max_eye_closed_duration_sec: float = 5.0
        max_mean_ear: float = 0.20
        target_state: str = "DROWSY"

    @dataclass
    class HoldDecayConfig:
        enabled: bool = False
        probability_max: float = 0.10
        hold_streak: int = 3
        max_base_state: str = "SUSPICIOUS"
        max_base_fsm_evidence: float = 0.35
        max_perclos_short: float = 0.30
        max_perclos_long: float | None = None
        max_eyes_closed_consecutive: int = 3
        max_yawn_count: float = 0.0
        max_head_drop_count: float = 0.0
        max_max_eye_closed_duration_sec: float | None = None
        max_max_fsm_evidence: float | None = None
        target_state: str = "SUSPICIOUS"

    @dataclass
    class DwellReliefConfig:
        enabled: bool = False
        probability_max: float = 0.10
        min_credit: float = 2.0
        alert_gain: float = 1.0
        suspicious_gain: float = 0.5
        decay: float = 0.5
        max_base_state: str = "SUSPICIOUS"
        max_base_fsm_evidence: float = 0.35
        max_perclos_short: float = 0.30
        max_perclos_long: float | None = None
        max_eyes_closed_consecutive: int = 3
        max_yawn_count: float = 0.0
        max_head_drop_count: float = 0.0
        max_max_eye_closed_duration_sec: float | None = None
        max_max_fsm_evidence: float | None = None
        target_state: str = "SUSPICIOUS"

    fsm_safety_min_state: str = "CRITICAL"
    fsm_safety_cap_state: str = "SUSPICIOUS"
    critical_probability_threshold: float = 0.85
    critical_severe_streak: int = 2
    elevated_streak_threshold: int = 3
    model_plus_recent_drowsy_probability_min: float | None = None
    model_plus_recent_max_eye_closed_duration_sec: float | None = None
    recovery_clean_streak: int = 3
    post_rule_order: str = "rescue_then_suppress"

    clean_max_base_fsm_evidence: float = 0.20
    clean_max_perclos_short: float = 0.15
    clean_max_eyes_closed_consecutive: int = 1
    clean_max_yawn_count: float = 0.0
    clean_max_head_drop_count: float = 0.0
    clean_max_eye_closed_duration_sec: float = 0.25

    recent_min_base_fsm_evidence: float = 0.35
    recent_min_max_fsm_evidence: float = 0.35
    recent_min_perclos_short: float = 0.20
    recent_min_eyes_closed_consecutive: int = 3
    recent_min_max_eye_closed_duration_sec: float = 0.50

    severe_min_perclos_short: float = 0.60
    severe_min_eyes_closed_consecutive: int = 10
    severe_min_max_eye_closed_duration_sec: float = 2.0
    severe_min_max_fsm_evidence: float = 0.60
    severe_combined_min_base_fsm_evidence: float = 0.60
    severe_combined_min_perclos_long: float = 0.35
    low_probability_release: LowProbabilityReleaseConfig = field(default_factory=LowProbabilityReleaseConfig)
    subtle_rescue: SubtleRescueConfig = field(default_factory=SubtleRescueConfig)
    fp_suppression: FPSuppressionConfig = field(default_factory=FPSuppressionConfig)
    seeded_macroevent_bridge: SeededMacroeventBridgeConfig = field(default_factory=SeededMacroeventBridgeConfig)
    guarded_severe_rescue: GuardedSevereRescueConfig = field(default_factory=GuardedSevereRescueConfig)
    hold_decay: HoldDecayConfig = field(default_factory=HoldDecayConfig)
    dwell_relief: DwellReliefConfig = field(default_factory=DwellReliefConfig)


@dataclass
class FSMConfig:
    perclos_threshold: float = 0.20
    pitch_velocity_evidence_threshold: float = 5.0
    yawn_frequency_threshold: int = 3
    blink_frequency_threshold: int = 10

    hysteresis_alert_to_suspicious: float = 0.45
    hysteresis_suspicious_to_alert: float = 0.25
    hysteresis_suspicious_to_drowsy: float = 0.60
    hysteresis_drowsy_to_suspicious: float = 0.40
    hysteresis_drowsy_to_critical: float = 0.75
    hysteresis_critical_to_drowsy: float = 0.55

    extreme_critical_perclos_short_drowsy: float = 0.60
    extreme_critical_perclos_long: float = 0.70
    extreme_drowsy_perclos_short: float = 0.60
    extreme_drowsy_perclos_long: float = 0.50
    extreme_drowsy_ear: float = 0.10
    extreme_drowsy_eye_closed_seconds: float = 1.0
    extreme_suspicious_perclos_long: float = 0.35
    extreme_suspicious_pitch_velocity: float = 10.0

    frames_to_suspicious_seconds: float = 0.50
    frames_to_recovery_seconds: float = 1.0
    sustained_evidence_seconds: float = 1.0
    min_dwell_alert_seconds: float = 5.0
    min_dwell_suspicious_seconds: float = 3.0
    recovery_grace_period_seconds: float = 10.0
    extreme_signal_seconds: float = 1.0
    recovery_perclos_short_threshold: float = 0.40
    drowsy_recovery_seconds: float = 10.0


@dataclass
class RuntimeLogConfig:
    enabled: bool = True
    path: str = ""


@dataclass
class RuntimeConfig:
    input: InputConfig = field(default_factory=InputConfig)
    runtime: RuntimeTuning = field(default_factory=RuntimeTuning)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    alerts: AlertPolicy = field(default_factory=AlertPolicy)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    camera_model: CameraModelConfig = field(default_factory=CameraModelConfig)
    hybrid_policy: HybridPolicyConfig = field(default_factory=HybridPolicyConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    runtime_log: RuntimeLogConfig = field(default_factory=RuntimeLogConfig)
    feature_backend: str = "phuong"
    decision_engine: str = "fsm"
    enable_legacy_feature_overlay: bool = False
    display_window: bool = True
    log_every_n_frames: int = 100
    runtime_profile: str = ""
    config_path: str = ""


RUNTIME_PROFILE_CONFIGS = {
    "protected": "configs/protected.json",
    "recommended": "configs/recommended.json",
}


CLI_OVERRIDE_MAP = {
    "source": "input.source",
    "video_path": "input.video_path",
    "loop_file": "input.loop_file",
    "max_frames": "runtime.max_frames",
    "use_file_video_fps": "runtime.use_file_video_fps",
    "decision_engine": "decision_engine",
    "camera_model_path": "camera_model.model_path",
    "feature_backend": "feature_backend",
    "runtime_log_enabled": "runtime_log.enabled",
    "runtime_log_path": "runtime_log.path",
    "log_every_n_frames": "log_every_n_frames",
    "drowsy_cooldown_seconds": "alerts.drowsy_cooldown_seconds",
    "enable_legacy_feature_overlay": "enable_legacy_feature_overlay",
    "display_window": "display_window",
    "runtime_profile": "runtime_profile",
    "config_path": "config_path",
}


def default_runtime_config() -> RuntimeConfig:
    return RuntimeConfig()


def available_runtime_profiles() -> tuple[str, ...]:
    return tuple(RUNTIME_PROFILE_CONFIGS.keys())


def runtime_profile_config_path(profile: str) -> Path:
    key = str(profile or "").strip().lower()
    if key not in RUNTIME_PROFILE_CONFIGS:
        choices = ", ".join(available_runtime_profiles())
        raise ValueError(f"Unknown runtime profile '{profile}'. Choose one of: {choices}.")
    return Path(RUNTIME_PROFILE_CONFIGS[key])


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _load_config_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON root must be an object.")
    return data


def _to_runtime_config(data: dict[str, Any]) -> RuntimeConfig:
    input_cfg = InputConfig(**data.get("input", {}))
    runtime_cfg = RuntimeTuning(**data.get("runtime", {}))
    threshold_cfg = ThresholdConfig(**data.get("thresholds", {}))
    window_cfg = WindowConfig(**data.get("windows", {}))
    alert_cfg = AlertPolicy(**data.get("alerts", {}))
    perception_cfg = PerceptionConfig(**data.get("perception", {}))
    camera_model_data = dict(data.get("camera_model", {}))
    quality_guard_cfg = CameraModelConfig.QualityGuardConfig(**camera_model_data.pop("quality_guard", {}))
    camera_model_cfg = CameraModelConfig(**camera_model_data, quality_guard=quality_guard_cfg)
    hybrid_policy_data = dict(data.get("hybrid_policy", {}))
    low_probability_release_cfg = HybridPolicyConfig.LowProbabilityReleaseConfig(
        **hybrid_policy_data.pop("low_probability_release", {})
    )
    subtle_rescue_cfg = HybridPolicyConfig.SubtleRescueConfig(**hybrid_policy_data.pop("subtle_rescue", {}))
    fp_suppression_cfg = HybridPolicyConfig.FPSuppressionConfig(**hybrid_policy_data.pop("fp_suppression", {}))
    seeded_macroevent_bridge_cfg = HybridPolicyConfig.SeededMacroeventBridgeConfig(
        **hybrid_policy_data.pop("seeded_macroevent_bridge", {})
    )
    guarded_severe_rescue_cfg = HybridPolicyConfig.GuardedSevereRescueConfig(
        **hybrid_policy_data.pop("guarded_severe_rescue", {})
    )
    hold_decay_cfg = HybridPolicyConfig.HoldDecayConfig(**hybrid_policy_data.pop("hold_decay", {}))
    dwell_relief_cfg = HybridPolicyConfig.DwellReliefConfig(**hybrid_policy_data.pop("dwell_relief", {}))
    hybrid_policy_cfg = HybridPolicyConfig(
        **hybrid_policy_data,
        low_probability_release=low_probability_release_cfg,
        subtle_rescue=subtle_rescue_cfg,
        fp_suppression=fp_suppression_cfg,
        seeded_macroevent_bridge=seeded_macroevent_bridge_cfg,
        guarded_severe_rescue=guarded_severe_rescue_cfg,
        hold_decay=hold_decay_cfg,
        dwell_relief=dwell_relief_cfg,
    )
    fsm_cfg = FSMConfig(**data.get("fsm", {}))
    runtime_log_cfg = RuntimeLogConfig(**data.get("runtime_log", {}))
    feature_backend = str(data.get("feature_backend", "phuong")).strip().lower()
    if feature_backend not in {"legacy", "phuong"}:
        raise ValueError("feature_backend must be 'legacy' or 'phuong'.")

    return RuntimeConfig(
        input=input_cfg,
        runtime=runtime_cfg,
        thresholds=threshold_cfg,
        windows=window_cfg,
        alerts=alert_cfg,
        perception=perception_cfg,
        camera_model=camera_model_cfg,
        hybrid_policy=hybrid_policy_cfg,
        fsm=fsm_cfg,
        runtime_log=runtime_log_cfg,
        feature_backend=feature_backend,
        decision_engine=data.get("decision_engine", "fsm"),
        enable_legacy_feature_overlay=bool(data.get("enable_legacy_feature_overlay", False)),
        display_window=bool(data.get("display_window", True)),
        log_every_n_frames=int(data.get("log_every_n_frames", 100)),
        runtime_profile=str(data.get("runtime_profile", "") or "").strip(),
        config_path=str(data.get("config_path", "") or "").strip(),
    )


def load_runtime_config(config_path: str | None, cli_overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    base = asdict(default_runtime_config())
    json_cfg = _load_config_json(config_path)
    _deep_merge_dict(base, json_cfg)

    if cli_overrides:
        for cli_key, cli_value in cli_overrides.items():
            if cli_key not in CLI_OVERRIDE_MAP:
                continue
            if cli_value is None:
                continue
            _set_nested(base, CLI_OVERRIDE_MAP[cli_key], cli_value)

    return _to_runtime_config(base)


def config_to_dict(config: RuntimeConfig) -> dict[str, Any]:
    return asdict(config)
