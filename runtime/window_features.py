from __future__ import annotations

import csv
import math
import os
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable


FRAME_FEATURE_COLUMNS = [
    "subject_id",
    "session_id",
    "video_id",
    "frame_index",
    "timestamp_sec",
    "face_detected",
    "ear",
    "mar",
    "pitch",
    "yaw",
    "roll",
    "eye_closed",
    "mouth_open",
    "head_nod_detected",
    "perclos_60s",
    "perclos_5s",
    "blink_frequency",
    "yawn_frequency",
    "pitch_velocity",
    "gaze_stable",
    "fsm_state",
    "fsm_evidence",
    "fsm_reasons",
]

WINDOW_FEATURE_COLUMNS = [
    "subject_id",
    "session_id",
    "video_id",
    "window_start_sec",
    "window_end_sec",
    "frame_count",
    "valid_face_ratio",
    "mean_ear",
    "min_ear",
    "ear_std",
    "ear_p10",
    "ear_p25",
    "ear_end_minus_start",
    "ear_abs_diff_mean",
    "ear_linear_slope",
    "ear_mean_over_calibration",
    "ear_p10_over_calibration",
    "ear_min_over_calibration",
    "ear_below_calibration90_ratio",
    "ear_below_calibration90_run_p90_sec",
    "ear_below_calibration90_run_ge_05_count",
    "mean_mar",
    "max_mar",
    "perclos_60s",
    "perclos_5s",
    "eye_closed_ratio",
    "eye_closed_transition_count",
    "max_eye_closed_duration_sec",
    "mean_eye_closed_run_sec",
    "eye_closed_run_p50_sec",
    "eye_closed_run_p90_sec",
    "eye_closed_run_count",
    "eye_closed_run_ge_02_count",
    "eye_closed_run_ge_03_count",
    "eye_closed_run_ge_05_count",
    "eye_closed_run_ge_10_count",
    "closed_ear_mean",
    "closed_ear_p10",
    "blink_rate_per_min",
    "yawn_count",
    "head_drop_count",
    "max_pitch_velocity",
    "mean_fsm_evidence",
    "max_fsm_evidence",
    "fsm_alert_ratio",
    "fsm_suspicious_ratio",
    "fsm_drowsy_ratio",
    "fsm_critical_ratio",
    "fsm_state_mode",
]

KSS_COLUMNS = [
    "subject_id",
    "session_id",
    "video_id",
    "start_time_sec",
    "end_time_sec",
    "kss_score",
    "kss_band",
    "notes",
]

EYE_FEATURES = [
    "mean_ear",
    "min_ear",
    "perclos_60s",
    "perclos_5s",
    "max_eye_closed_duration_sec",
    "blink_rate_per_min",
]
PERCLOS_FEATURES = ["perclos_60s", "perclos_5s"]
YAWN_FEATURES = ["mean_mar", "max_mar", "yawn_count"]
HEAD_POSE_FEATURES = ["head_drop_count", "max_pitch_velocity"]
FSM_FEATURES = ["mean_fsm_evidence", "max_fsm_evidence", "fsm_state_mode"]
FULL_FEATURES = EYE_FEATURES + YAWN_FEATURES + HEAD_POSE_FEATURES + FSM_FEATURES

FSM_STATE_ENCODING = {
    "ALERT": 0.0,
    "SUSPICIOUS": 1.0,
    "DROWSY": 2.0,
    "CRITICAL": 3.0,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key, "")) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return value


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(round(as_float(row, key, float(default))))


def as_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key, "")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def capped_n_jobs(requested: int) -> int:
    cap = max(1, (os.cpu_count() or 2) // 2)
    if requested <= 0:
        return cap
    return min(requested, cap)


def kss_is_sleepy(row: dict[str, Any]) -> bool:
    explicit_target = str(row.get("target_sleepy", "")).strip()
    if explicit_target != "":
        return as_int(row, "target_sleepy") >= 1
    band = str(row.get("kss_band", "")).strip().lower()
    if band:
        return band in {"sleepy", "drowsy"}
    return as_float(row, "kss_score", 0.0) >= 7.0


def fsm_state_is_sleepy(value: Any) -> bool:
    state = str(value).strip().upper()
    return state in {"DROWSY", "CRITICAL"}


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": total,
    }


def aggregate_frame_rows(
    frame_rows: list[dict[str, Any]],
    window_seconds: float,
    stride_seconds: float,
    calibration_frames: int = 60,
    calibration_soft_factor: float = 0.90,
) -> list[dict[str, Any]]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be > 0")
    if not frame_rows:
        return []

    rows = sorted(frame_rows, key=lambda row: as_float(row, "timestamp_sec"))
    timestamps = [as_float(row, "timestamp_sec") for row in rows]
    start = timestamps[0]
    end = timestamps[-1]
    sample_interval = _estimate_sample_interval(timestamps)
    calibration_ear = _calibration_ear_baseline(rows, calibration_frames)
    windows: list[dict[str, Any]] = []

    window_start = start
    while window_start <= end + 1e-9:
        window_end = window_start + window_seconds
        members = [row for row in rows if window_start <= as_float(row, "timestamp_sec") < window_end]
        if len(members) >= 2:
            windows.append(
                _aggregate_one_window(
                    members,
                    window_start,
                    window_end,
                    sample_interval,
                    calibration_ear=calibration_ear,
                    calibration_soft_factor=calibration_soft_factor,
                )
            )
        window_start += stride_seconds
    return windows


def aggregate_runtime_window(
    frame_rows: list[dict[str, Any]],
    window_seconds: float,
    *,
    rows_are_ordered: bool = False,
) -> dict[str, Any] | None:
    """Aggregate only the features consumed by the production camera hybrid."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if not frame_rows:
        return None
    if rows_are_ordered:
        return _aggregate_ordered_runtime_window(frame_rows, window_seconds)

    rows = sorted(frame_rows, key=lambda row: as_float(row, "timestamp_sec"))
    window_end = as_float(rows[0], "timestamp_sec") + window_seconds
    members = [row for row in rows if as_float(row, "timestamp_sec") < window_end]
    if len(members) < 2:
        return None

    timestamps = [as_float(row, "timestamp_sec") for row in members]
    sample_interval = _estimate_sample_interval(timestamps)
    ear_values = [as_float(row, "ear") for row in members]
    mar_values = [as_float(row, "mar") for row in members]
    evidence_values = [as_float(row, "fsm_evidence") for row in members]
    states = [str(row.get("fsm_state", "")) for row in members]
    face_flags = [as_bool(row, "face_detected") for row in members]
    eye_closed_flags = [as_bool(row, "eye_closed") for row in members]
    head_drop_flags = [as_bool(row, "head_nod_detected") for row in members]
    yawn_values = [as_int(row, "yawn_frequency") for row in members]

    longest_closed = 0
    current_closed = 0
    for closed in eye_closed_flags:
        if closed:
            current_closed += 1
            longest_closed = max(longest_closed, current_closed)
        else:
            current_closed = 0
    if sample_interval <= 0:
        sample_interval = 1.0

    return {
        "frame_count": len(members),
        "valid_face_ratio": sum(1 for value in face_flags if value) / len(members),
        "mean_ear": _mean(ear_values),
        "min_ear": min(ear_values) if ear_values else 0.0,
        "ear_std": _std(ear_values),
        "ear_p10": _quantile(ear_values, 0.10),
        "mean_mar": _mean(mar_values),
        "max_mar": max(mar_values) if mar_values else 0.0,
        "perclos_60s": as_float(members[-1], "perclos_60s"),
        "perclos_5s": as_float(members[-1], "perclos_5s"),
        "max_eye_closed_duration_sec": longest_closed * sample_interval,
        "yawn_count": max(0, max(yawn_values) - min(yawn_values)) if yawn_values else 0,
        "head_drop_count": sum(1 for value in head_drop_flags if value),
        "mean_fsm_evidence": _mean(evidence_values),
        "max_fsm_evidence": max(evidence_values) if evidence_values else 0.0,
        "fsm_state_mode": _mode(states),
    }


def _aggregate_ordered_runtime_window(
    rows: list[dict[str, Any]],
    window_seconds: float,
) -> dict[str, Any] | None:
    """Specialized exact path for the engine's already ordered numeric rows."""
    window_end = float(rows[0].get("timestamp_sec", 0.0) or 0.0) + window_seconds
    members: list[dict[str, Any]] = []
    for row in rows:
        if float(row.get("timestamp_sec", 0.0) or 0.0) >= window_end:
            break
        members.append(row)
    if len(members) < 2:
        return None

    timestamps = [float(row.get("timestamp_sec", 0.0) or 0.0) for row in members]
    sample_interval = _estimate_sample_interval(timestamps)
    ear_values = [float(row.get("ear", 0.0) or 0.0) for row in members]
    mar_values = [float(row.get("mar", 0.0) or 0.0) for row in members]
    evidence_values = [float(row.get("fsm_evidence", 0.0) or 0.0) for row in members]
    states = [str(row.get("fsm_state", "")) for row in members]
    face_flags = [bool(row.get("face_detected", False)) for row in members]
    eye_closed_flags = [bool(row.get("eye_closed", False)) for row in members]
    head_drop_flags = [bool(row.get("head_nod_detected", False)) for row in members]
    yawn_values = [int(round(float(row.get("yawn_frequency", 0) or 0))) for row in members]

    longest_closed = 0
    current_closed = 0
    for closed in eye_closed_flags:
        if closed:
            current_closed += 1
            longest_closed = max(longest_closed, current_closed)
        else:
            current_closed = 0
    if sample_interval <= 0:
        sample_interval = 1.0

    return {
        "frame_count": len(members),
        "valid_face_ratio": sum(1 for value in face_flags if value) / len(members),
        "mean_ear": _mean(ear_values),
        "min_ear": min(ear_values) if ear_values else 0.0,
        "ear_std": _std(ear_values),
        "ear_p10": _quantile(ear_values, 0.10),
        "mean_mar": _mean(mar_values),
        "max_mar": max(mar_values) if mar_values else 0.0,
        "perclos_60s": float(members[-1].get("perclos_60s", 0.0) or 0.0),
        "perclos_5s": float(members[-1].get("perclos_5s", 0.0) or 0.0),
        "max_eye_closed_duration_sec": longest_closed * sample_interval,
        "yawn_count": max(0, max(yawn_values) - min(yawn_values)) if yawn_values else 0,
        "head_drop_count": sum(1 for value in head_drop_flags if value),
        "mean_fsm_evidence": _mean(evidence_values),
        "max_fsm_evidence": max(evidence_values) if evidence_values else 0.0,
        "fsm_state_mode": _mode(states),
    }


def _estimate_sample_interval(timestamps: list[float]) -> float:
    diffs = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not diffs:
        return 0.0
    return float(median(diffs))


def _calibration_ear_baseline(rows: list[dict[str, Any]], calibration_frames: int) -> float:
    detected_ears = [as_float(row, "ear") for row in rows if as_bool(row, "face_detected")]
    if not detected_ears:
        detected_ears = [as_float(row, "ear") for row in rows]
    if not detected_ears:
        return 0.0
    count = len(detected_ears) if calibration_frames <= 0 else min(len(detected_ears), calibration_frames)
    return _mean(detected_ears[:count])


def _aggregate_one_window(
    rows: list[dict[str, Any]],
    window_start: float,
    window_end: float,
    sample_interval: float,
    calibration_ear: float,
    calibration_soft_factor: float,
) -> dict[str, Any]:
    first = rows[0]
    timestamps = [as_float(row, "timestamp_sec") for row in rows]
    observed_seconds = max(timestamps[-1] - timestamps[0], sample_interval, 1e-9)

    ear_values = [as_float(row, "ear") for row in rows]
    mar_values = [as_float(row, "mar") for row in rows]
    evidence_values = [as_float(row, "fsm_evidence") for row in rows]
    pitch_velocity_values = [abs(as_float(row, "pitch_velocity")) for row in rows]
    states = [str(row.get("fsm_state", "")) for row in rows]
    eye_closed_flags = [as_bool(row, "eye_closed") for row in rows]
    eye_closed_run_lengths = _true_run_lengths_seconds(rows, "eye_closed", sample_interval)
    detected_rows = [row for row in rows if as_bool(row, "face_detected")] or rows
    detected_ear_values = [as_float(row, "ear") for row in detected_rows]
    soft_ear_threshold = calibration_ear * calibration_soft_factor if calibration_ear > 0.0 else 0.0
    soft_ear_run_lengths = _value_below_threshold_run_lengths_seconds(detected_rows, "ear", soft_ear_threshold, sample_interval)

    blink_count = _rolling_count_delta(rows, "blink_frequency")
    yawn_count = _rolling_count_delta(rows, "yawn_frequency")

    return {
        "subject_id": first.get("subject_id", ""),
        "session_id": first.get("session_id", ""),
        "video_id": first.get("video_id", ""),
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "frame_count": len(rows),
        "valid_face_ratio": sum(1 for row in rows if as_bool(row, "face_detected")) / len(rows),
        "mean_ear": _mean(ear_values),
        "min_ear": min(ear_values) if ear_values else 0.0,
        "ear_std": _std(ear_values),
        "ear_p10": _quantile(ear_values, 0.10),
        "ear_p25": _quantile(ear_values, 0.25),
        "ear_end_minus_start": (ear_values[-1] - ear_values[0]) if len(ear_values) >= 2 else 0.0,
        "ear_abs_diff_mean": _mean_abs_diff(ear_values),
        "ear_linear_slope": _linear_slope(timestamps, ear_values),
        "ear_mean_over_calibration": _safe_divide(_mean(detected_ear_values), calibration_ear),
        "ear_p10_over_calibration": _safe_divide(_quantile(detected_ear_values, 0.10), calibration_ear),
        "ear_min_over_calibration": _safe_divide(min(detected_ear_values) if detected_ear_values else 0.0, calibration_ear),
        "ear_below_calibration90_ratio": _value_below_threshold_ratio(detected_ear_values, soft_ear_threshold),
        "ear_below_calibration90_run_p90_sec": _quantile(soft_ear_run_lengths, 0.90),
        "ear_below_calibration90_run_ge_05_count": _count_values_at_least(soft_ear_run_lengths, 0.5),
        "mean_mar": _mean(mar_values),
        "max_mar": max(mar_values) if mar_values else 0.0,
        "perclos_60s": as_float(rows[-1], "perclos_60s"),
        "perclos_5s": as_float(rows[-1], "perclos_5s"),
        "eye_closed_ratio": sum(1 for value in eye_closed_flags if value) / len(rows),
        "eye_closed_transition_count": _transition_count(eye_closed_flags),
        "max_eye_closed_duration_sec": _max_consecutive_true_seconds(rows, "eye_closed", sample_interval),
        "mean_eye_closed_run_sec": _mean(eye_closed_run_lengths),
        "eye_closed_run_p50_sec": _quantile(eye_closed_run_lengths, 0.50),
        "eye_closed_run_p90_sec": _quantile(eye_closed_run_lengths, 0.90),
        "eye_closed_run_count": len(eye_closed_run_lengths),
        "eye_closed_run_ge_02_count": _count_values_at_least(eye_closed_run_lengths, 0.2),
        "eye_closed_run_ge_03_count": _count_values_at_least(eye_closed_run_lengths, 0.3),
        "eye_closed_run_ge_05_count": _count_values_at_least(eye_closed_run_lengths, 0.5),
        "eye_closed_run_ge_10_count": _count_values_at_least(eye_closed_run_lengths, 1.0),
        "closed_ear_mean": _mean([value for value, closed in zip(ear_values, eye_closed_flags) if closed]),
        "closed_ear_p10": _quantile([value for value, closed in zip(ear_values, eye_closed_flags) if closed], 0.10),
        "blink_rate_per_min": (blink_count / observed_seconds) * 60.0,
        "yawn_count": yawn_count,
        "head_drop_count": sum(1 for row in rows if as_bool(row, "head_nod_detected")),
        "max_pitch_velocity": max(pitch_velocity_values) if pitch_velocity_values else 0.0,
        "mean_fsm_evidence": _mean(evidence_values),
        "max_fsm_evidence": max(evidence_values) if evidence_values else 0.0,
        "fsm_alert_ratio": _value_equal_ratio(states, "ALERT"),
        "fsm_suspicious_ratio": _value_equal_ratio(states, "SUSPICIOUS"),
        "fsm_drowsy_ratio": _value_equal_ratio(states, "DROWSY"),
        "fsm_critical_ratio": _value_equal_ratio(states, "CRITICAL"),
        "fsm_state_mode": _mode(states),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _count_values_at_least(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value >= threshold)


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _value_below_threshold_ratio(values: list[float], threshold: float) -> float:
    if not values or threshold <= 0.0:
        return 0.0
    return sum(1 for value in values if value < threshold) / len(values)


def _value_equal_ratio(values: list[str], target: str) -> float:
    if not values:
        return 0.0
    target_upper = target.upper()
    return sum(1 for value in values if str(value).strip().upper() == target_upper) / len(values)


def _mean_abs_diff(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    diffs = [abs(current - previous) for previous, current in zip(values, values[1:])]
    return _mean(diffs)


def _linear_slope(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) <= 1:
        return 0.0
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x_values, y_values))
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _transition_count(values: list[bool]) -> int:
    if len(values) <= 1:
        return 0
    return sum(1 for previous, current in zip(values, values[1:]) if previous != current)


def _mode(values: list[str]) -> str:
    clean_values = [value for value in values if value]
    if not clean_values:
        return ""
    counts = Counter(clean_values)
    severity = {"ALERT": 0, "SUSPICIOUS": 1, "DROWSY": 2, "CRITICAL": 3}
    return max(counts, key=lambda value: (counts[value], severity.get(value.upper(), 0)))


def _rolling_count_delta(rows: list[dict[str, Any]], key: str) -> int:
    values = [as_int(row, key) for row in rows]
    if not values:
        return 0
    return max(0, max(values) - min(values))


def _max_consecutive_true_seconds(rows: list[dict[str, Any]], key: str, sample_interval: float) -> float:
    if sample_interval <= 0:
        sample_interval = 1.0
    longest = 0
    current = 0
    for row in rows:
        if as_bool(row, key):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest * sample_interval


def _true_run_lengths_seconds(rows: list[dict[str, Any]], key: str, sample_interval: float) -> list[float]:
    if sample_interval <= 0:
        sample_interval = 1.0
    run_lengths: list[float] = []
    current = 0
    for row in rows:
        if as_bool(row, key):
            current += 1
        elif current > 0:
            run_lengths.append(current * sample_interval)
            current = 0
    if current > 0:
        run_lengths.append(current * sample_interval)
    return run_lengths


def _value_below_threshold_run_lengths_seconds(
    rows: list[dict[str, Any]],
    key: str,
    threshold: float,
    sample_interval: float,
) -> list[float]:
    if sample_interval <= 0:
        sample_interval = 1.0
    if threshold <= 0.0:
        return []
    run_lengths: list[float] = []
    current = 0
    for row in rows:
        if as_float(row, key) < threshold:
            current += 1
        elif current > 0:
            run_lengths.append(current * sample_interval)
            current = 0
    if current > 0:
        run_lengths.append(current * sample_interval)
    return run_lengths


def numeric_feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([numeric_feature_value(row, feature) for feature in feature_names])
    return matrix


def numeric_feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "fsm_state_mode":
        return FSM_STATE_ENCODING.get(str(row.get(feature, "")).strip().upper(), 0.0)
    return as_float(row, feature)


def available_features(rows: list[dict[str, Any]], requested: list[str]) -> list[str]:
    if not rows:
        return []
    keys = set(rows[0].keys())
    return [feature for feature in requested if feature in keys]
