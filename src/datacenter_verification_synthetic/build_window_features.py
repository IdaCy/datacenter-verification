"""Build window feature rows from normalized synthetic raw records."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

try:
    from .common import (
        BASE_REQUIRED_FEATURE_COLUMNS,
        FEATURE_PIPELINE_VERSION,
        HARDWARE_NORMALIZATION_VERSION,
        POLICY_THRESHOLD_VERSION,
        REQUIRED_FEATURE_COLUMNS,
        WINDOW_SPECS,
        derive_label,
        normalize_feature_row,
        optional_write_parquet,
        parse_utc,
        raw_payload_hash,
        read_jsonl,
        stable_hash,
        utc_iso,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        BASE_REQUIRED_FEATURE_COLUMNS,
        FEATURE_PIPELINE_VERSION,
        HARDWARE_NORMALIZATION_VERSION,
        POLICY_THRESHOLD_VERSION,
        REQUIRED_FEATURE_COLUMNS,
        WINDOW_SPECS,
        derive_label,
        normalize_feature_row,
        optional_write_parquet,
        parse_utc,
        raw_payload_hash,
        read_jsonl,
        stable_hash,
        utc_iso,
        write_csv,
    )


def load_episode_summaries(raw_dir: Path) -> list[dict[str, Any]]:
    events_path = raw_dir / "event_records.jsonl"
    summaries: list[dict[str, Any]] = []
    for event in read_jsonl(events_path):
        if event.get("event_type") != "synthetic_episode_summary":
            continue
        attrs = json.loads(event["attributes_json"])
        attrs["episode_id"] = event["episode_id"]
        attrs["site_id"] = event["site_id"]
        attrs["event_time"] = event["event_time"]
        attrs["event_end_time"] = event["event_end_time"]
        summaries.append(attrs)
    if not summaries:
        raise ValueError(f"no synthetic_episode_summary events found in {events_path}")
    return summaries


def _window_starts(summary: dict[str, Any], length_seconds: int) -> list[Any]:
    start = parse_utc(summary["episode_start"])
    end = parse_utc(summary["episode_end"])
    duration_seconds = max(1, int((end - start).total_seconds()))
    max_windows = int(summary.get("max_windows_per_episode_per_length") or 8)
    if duration_seconds <= length_seconds:
        return [start]

    available = max(1, duration_seconds - length_seconds)
    full_count = available // length_seconds + 1
    if full_count <= max_windows:
        return [start + timedelta(seconds=length_seconds * idx) for idx in range(full_count)]

    starts = []
    for idx in range(max_windows):
        offset = round(idx * available / max(1, max_windows - 1))
        starts.append(start + timedelta(seconds=offset))
    return starts


def _float_value(value: Any, default: float = 0.0) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scale_numeric(row: dict[str, Any], columns: list[str], factor: float, low: float = 0.0, high: float | None = None) -> None:
    for column in columns:
        if row.get(column) in {None, ""}:
            continue
        value = _float_value(row[column]) * factor
        if high is not None:
            value = min(high, value)
        row[column] = max(low, value)


def _phase_for_v1(summary: dict[str, Any], window_start: Any, length_seconds: int) -> str:
    start = parse_utc(summary["episode_start"])
    end = parse_utc(summary["episode_end"])
    duration_seconds = max(1.0, (end - start).total_seconds())
    position = max(0.0, min(1.0, (window_start - start).total_seconds() / duration_seconds))
    family = str(summary.get("scenario_family") or summary.get("latent_workload_class") or "")
    variant = str(summary.get("scenario_variant") or "")
    data_quality = str(summary.get("feature_template", {}).get("data_quality_regime") or "")
    phase_seed = int(stable_hash(summary["episode_id"], utc_iso(window_start), length_seconds, length=8), 16)

    if position < 0.08:
        return "ramp_up"
    if position > 0.92:
        return "cooldown"
    if data_quality in {"collector_gap_short", "collector_gap_long", "maintenance_observability_loss"} and 0.38 <= position <= 0.62:
        return "collector_gap"
    if "preempt" in family or "preempt" in variant or "elastic" in family:
        if 0.30 <= position <= 0.42:
            return "restart_or_recovery"
        if 0.52 <= position <= 0.68:
            return "scale_resize"
    if "multi_stage" in family:
        if 0.18 <= position <= 0.32:
            return "checkpoint_burst"
        if 0.68 <= position <= 0.82:
            return "evaluation_phase"
    if "delayed_logs" in family and position < 0.55:
        return "steady_state"
    if phase_seed % 11 in {0, 1} and length_seconds <= 6 * 60 * 60:
        return "checkpoint_burst"
    if phase_seed % 17 == 0 and family in {"pretraining_standard", "large_fine_tune_standard", "sparse_or_moe_bursty_training"}:
        return "evaluation_phase"
    return "steady_state"


def _apply_v1_window_dynamics(row: dict[str, Any], summary: dict[str, Any], window_start: Any, length_seconds: int) -> None:
    if not row.get("scenario_family"):
        return
    phase = _phase_for_v1(summary, window_start, length_seconds)
    row["temporal_phase"] = phase
    family = str(row.get("scenario_family") or "")

    if phase == "ramp_up":
        _scale_numeric(
            row,
            [
                "o4_gpu_util_p50",
                "o4_gpu_util_p95",
                "o4_gpu_util_duty_gt_70",
                "o4_sm_tensor_active_p95",
                "o6_nvlink_util_p95",
                "o7_scaleout_port_util_p95",
                "o7_collective_periodicity_score",
                "o8_rack_power_fraction_p95",
            ],
            0.68,
            high=100.0,
        )
        row["o7_synchronized_fabric_footprint"] = int(_float_value(row.get("o7_synchronized_fabric_footprint")) * 0.72)
        _scale_numeric(row, ["o11_checkpoint_periodicity_score", "o11_artifact_write_pattern_score"], 0.35, high=1.0)
    elif phase == "cooldown":
        _scale_numeric(
            row,
            [
                "o4_gpu_util_p50",
                "o4_gpu_util_p95",
                "o4_gpu_util_duty_gt_70",
                "o4_sm_tensor_active_p95",
                "o6_nvlink_util_p95",
                "o7_scaleout_port_util_p95",
                "o7_collective_periodicity_score",
            ],
            0.52,
            high=100.0,
        )
        row["o7_synchronized_fabric_footprint"] = int(_float_value(row.get("o7_synchronized_fabric_footprint")) * 0.55)
        _scale_numeric(row, ["o8_rack_power_fraction_p95"], 0.72, high=1.0)
    elif phase == "checkpoint_burst":
        row["o11_checkpoint_periodicity_score"] = min(1.0, _float_value(row.get("o11_checkpoint_periodicity_score")) + 0.18)
        row["o11_artifact_write_pattern_score"] = min(1.0, _float_value(row.get("o11_artifact_write_pattern_score")) + 0.18)
        row["o11_checkpoint_write_tb_per_event"] = _float_value(row.get("o11_checkpoint_write_tb_per_event")) * 1.35
        row["o11_storage_cotraffic_score"] = min(1.0, _float_value(row.get("o11_storage_cotraffic_score")) + 0.22)
        _scale_numeric(row, ["o7_scaleout_port_util_p95", "o7_collective_periodicity_score"], 0.92, high=1.0)
    elif phase == "evaluation_phase":
        _scale_numeric(row, ["o4_sm_tensor_active_p95", "o7_collective_periodicity_score", "o6_nvlink_periodicity_score"], 0.70, high=100.0)
        row["o7_inference_fanout_score"] = min(1.0, _float_value(row.get("o7_inference_fanout_score")) + 0.22)
        row["o11_artifact_write_pattern_score"] = min(1.0, _float_value(row.get("o11_artifact_write_pattern_score")) + 0.12)
    elif phase == "restart_or_recovery":
        _scale_numeric(row, ["o4_gpu_util_p95", "o4_gpu_util_duty_gt_70", "o7_collective_periodicity_score"], 0.58, high=100.0)
        row["o2_preemption_restart_count"] = int(_float_value(row.get("o2_preemption_restart_count")) + 1)
        row["o11_checkpoint_jitter_score"] = min(1.0, _float_value(row.get("o11_checkpoint_jitter_score")) + 0.24)
    elif phase == "scale_resize":
        row["o2_elastic_resize_count"] = int(_float_value(row.get("o2_elastic_resize_count")) + 1)
        _scale_numeric(row, ["o2_max_concurrent_normalized_gpus", "o7_synchronized_fabric_footprint"], 0.82)
        row["o2_gpu_hours_policy_ratio"] = _float_value(row.get("o2_max_concurrent_normalized_gpus")) * _float_value(row.get("o2_allocation_duration_hours")) / (512.0 * 24.0)
        row["policy_compute_ratio"] = row["o2_gpu_hours_policy_ratio"]
    elif phase == "collector_gap":
        for obs in ["o4", "o7", "o8", "o14"]:
            coverage_col = f"{obs}_coverage_fraction"
            row[coverage_col] = min(_float_value(row.get(coverage_col), 1.0), 0.62)
            row[f"{obs}_missing_reason"] = "collector_gap"
        row["o14_min_critical_coverage"] = min(_float_value(row.get("o14_min_critical_coverage"), 1.0), 0.62)
        row["o14_gap_fraction_critical"] = max(_float_value(row.get("o14_gap_fraction_critical")), 0.38)

    if length_seconds == 15 * 60:
        _scale_numeric(row, ["o7_collective_periodicity_score", "o6_nvlink_periodicity_score"], 0.86, high=1.0)
        row["o7_collective_jitter_score"] = min(1.0, _float_value(row.get("o7_collective_jitter_score")) + 0.08)
    elif length_seconds >= 24 * 60 * 60 and family in {
        "underclocked_energy_capped_training",
        "fragmented_training_linked",
        "training_without_semantic_logs",
    }:
        row["o11_checkpoint_periodicity_score"] = min(1.0, _float_value(row.get("o11_checkpoint_periodicity_score")) + 0.08)
        row["o7_account_flow_linkage_confidence"] = min(1.0, _float_value(row.get("o7_account_flow_linkage_confidence")) + 0.10)


def build_row(summary: dict[str, Any], window_name: str, window_start: Any, length_seconds: int) -> dict[str, Any]:
    template = dict(summary["feature_template"])
    window_end = window_start + timedelta(seconds=length_seconds)
    window_hours = length_seconds / 3600.0
    row_id = "feat_" + stable_hash(
        summary["dataset_id"],
        summary["episode_id"],
        window_name,
        utc_iso(window_start),
        template.get("scope_type"),
        template.get("scope_id_hash"),
        length=24,
    )

    rack_power_fraction = float(template.get("o8_rack_power_fraction_p95") or 0.0)
    facility_mw = float(template.get("o8_facility_it_power_mw") or 0.0)
    baseline_mw = float(summary.get("site_baseline_it_mw") or max(0.1, facility_mw * 0.22))
    baseline_subtracted = max(0.0, (facility_mw * rack_power_fraction - baseline_mw) * 1000.0 * window_hours)
    duration_hours = float(template.get("o2_allocation_duration_hours") or summary.get("duration_hours") or 0.0)

    row = {
        **template,
        "feature_row_id": row_id,
        "dataset_id": summary["dataset_id"],
        "seed": summary["seed"],
        "site_id": summary["site_id"],
        "window_start": utc_iso(window_start),
        "window_end": utc_iso(window_end),
        "window_length_seconds": length_seconds,
        "episode_id": summary["episode_id"],
        "latent_workload_class": summary["latent_workload_class"],
        "o8_baseline_subtracted_energy_kwh": baseline_subtracted,
        "o8_power_continuity_days": min(duration_hours / 24.0, length_seconds / 86400.0 if duration_hours == 0 else duration_hours / 24.0),
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "policy_threshold_version": POLICY_THRESHOLD_VERSION,
        "hardware_normalization_version": HARDWARE_NORMALIZATION_VERSION,
    }
    _apply_v1_window_dynamics(row, summary, window_start, length_seconds)
    row["raw_input_manifest_hash"] = raw_payload_hash(
        {
            "episode_id": summary["episode_id"],
            "window_name": window_name,
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "raw_event_hash": summary.get("raw_event_hash"),
        }
    )
    label, confidence, reason = derive_label(row)
    row["label_0_to_4"] = label
    row["label_confidence"] = confidence
    row["label_reason"] = reason
    row["label_source"] = "synthetic_latent_truth_plus_composite_rules"

    profile = str(row.get("synthetic_evidence_profile") or "")
    row["capacity_evidence_only"] = profile == "capacity_only"
    row["integrity_evidence_only"] = profile == "integrity_only"
    row["physical_evidence_only"] = profile == "physical_only"
    return normalize_feature_row(row)


def build_features(raw_dir: Path, output_dir: Path, seed: int | None = None) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_episode_summaries(raw_dir)
    if seed is not None:
        for summary in summaries:
            summary.setdefault("seed", seed)

    rows_by_window: dict[str, list[dict[str, Any]]] = {name: [] for name in WINDOW_SPECS}
    for summary in summaries:
        for window_name, length_seconds in WINDOW_SPECS.items():
            for window_start in _window_starts(summary, length_seconds):
                rows_by_window[window_name].append(build_row(summary, window_name, window_start, length_seconds))

    counts: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []
    hard_profile = any(row.get("scenario_family") for rows in rows_by_window.values() for row in rows)
    feature_columns = REQUIRED_FEATURE_COLUMNS if hard_profile else BASE_REQUIRED_FEATURE_COLUMNS
    for window_name, rows in rows_by_window.items():
        rows.sort(key=lambda row: (row["window_start"], row["site_id"], row["feature_row_id"]))
        path = output_dir / f"window_features_{window_name}.csv"
        counts[path.name] = write_csv(path, rows, feature_columns)
        optional_write_parquet(path)
        all_rows.extend(rows)

    all_rows.sort(key=lambda row: (row["window_start"], row["window_length_seconds"], row["site_id"], row["feature_row_id"]))
    all_path = output_dir / "window_features_all.csv"
    counts[all_path.name] = write_csv(all_path, all_rows, feature_columns)
    optional_write_parquet(all_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing raw_normalized JSONL files.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for feature CSV files.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    counts = build_features(args.input, args.output, seed=args.seed)
    for file_name, count in counts.items():
        print(f"{file_name}: {count} rows")


if __name__ == "__main__":
    main()
