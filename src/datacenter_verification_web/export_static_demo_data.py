"""Export compact JSON for the GitHub Pages training-run verification demo."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = [
    {
        "code": 0,
        "name": "No training likely",
        "meaning": "Window is inconsistent with a large training run under adequate coverage.",
    },
    {
        "code": 1,
        "name": "Training possible",
        "meaning": "Capacity or weak activity exists, but not enough evidence for a large run.",
    },
    {
        "code": 2,
        "name": "Elevated probability",
        "meaning": "A primary signal, supportive anomaly, or integrity problem creates a candidate window.",
    },
    {
        "code": 3,
        "name": "Training likely",
        "meaning": "Sustained primary activity is corroborated by independent evidence.",
    },
    {
        "code": 4,
        "name": "Highest warning / definite",
        "meaning": "Authenticated ML evidence or coherent multi-layer evidence crosses the policy scale.",
    },
]

NUMERIC_FEATURES = [
    "policy_compute_ratio",
    "o1_normalized_h100e_capacity",
    "o1_largest_contiguous_domain_gpus",
    "o1_homogeneous_high_end_fraction",
    "o1_non_partitioned_fraction",
    "o2_max_concurrent_normalized_gpus",
    "o2_allocation_duration_hours",
    "o2_gpu_hours_policy_ratio",
    "o2_concurrency_fraction_domain",
    "o2_topology_contiguity_score",
    "o3_batch_provisioned_gpus",
    "o3_capacity_reservation_duration_hours",
    "o3_training_sku_fraction",
    "o3_billing_continuity_score",
    "o4_gpu_util_p50",
    "o4_gpu_util_p95",
    "o4_gpu_util_duty_gt_70",
    "o4_sm_tensor_active_p95",
    "o4_hbm_used_fraction_p50",
    "o4_hbm_bandwidth_active_p95",
    "o4_gpu_power_fraction_p95",
    "o5_kernel_training_motif_score",
    "o5_tensor_throughput_ratio",
    "o6_nvlink_util_p95",
    "o6_nvlink_periodicity_score",
    "o7_scaleout_port_util_p95",
    "o7_synchronized_fabric_footprint",
    "o7_collective_periodicity_score",
    "o7_burst_duty_cycle",
    "o7_rdma_congestion_score",
    "o7_job_to_port_mapping_coverage",
    "o8_rack_power_fraction_p95",
    "o8_facility_it_power_mw",
    "o8_baseline_subtracted_energy_kwh",
    "o8_power_continuity_days",
    "o8_power_cv",
    "o8_power_to_gpu_residual",
    "o9_gpu_hbm_temp_score",
    "o9_thermal_delta_t_score",
    "o9_cooling_flow_duty",
    "o10_world_size",
    "o10_rank_stability_score",
    "o10_same_image_gpu_count",
    "o11_data_staging_tb",
    "o11_checkpoint_write_tb_per_event",
    "o11_checkpoint_periodicity_score",
    "o11_read_write_training_pattern_score",
    "o12_coverage_fraction",
    "o12_declared_parameter_count_b",
    "o12_training_tokens_b",
    "o12_step_count",
    "o13_attestation_valid_fraction",
    "o13_confidential_compute_mode_fraction",
    "o14_min_critical_coverage",
    "o14_gap_fraction_critical",
    "o14_clock_drift_max_ms",
    "o14_counter_reset_count",
    "o16_probe_throughput_ratio_min",
    "o16_probe_latency_inflation_max",
    "o16_vram_residency_conflict_score",
    "o1_coverage_fraction",
    "o2_coverage_fraction",
    "o4_coverage_fraction",
    "o7_coverage_fraction",
    "o8_coverage_fraction",
    "o14_coverage_fraction",
]

CATEGORICAL_FEATURES = [
    "o2_declared_workload_class",
    "o10_runtime_framework_class",
    "o1_missing_reason",
    "o2_missing_reason",
    "o4_missing_reason",
    "o7_missing_reason",
    "o8_missing_reason",
    "o14_missing_reason",
    "o12_missing_reason",
    "scheduler_signature_status",
    "gpu_telemetry_trust_level",
    "fabric_telemetry_trust_level",
    "power_meter_trust_level",
]

BOOLEAN_FEATURES = [
    "capacity_possible",
    "o2_reservation_exclusive_flag",
    "o5_profiler_available",
    "o10_rendezvous_present",
    "o12_signed_ml_logs_present",
    "o12_loss_curve_present",
    "o12_optimizer_state_present",
    "o13_collector_measurement_valid",
    "o15_unapproved_physical_change_near_window",
    "o15_firmware_bmc_change_near_window",
]

PREDICTION_FIELDS = [
    "split",
    "episode_id",
    "feature_row_id",
    "site_id",
    "scope_type",
    "scope_id_hash",
    "window_start",
    "window_end",
    "window_length_seconds",
    "latent_workload_class",
    "label_0_to_4",
    "predicted_label",
    "p_large_training",
    "severity_score",
    "negative_certification_confidence",
    "integrity_warning",
    "critical_missing_layers",
    "top_evidence",
    "min_critical_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/synthetic_v0/features/window_features_all.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/model_runs/synthetic_v0_baseline/predictions_all.csv"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/synthetic_v0/manifest.json"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/model_runs/synthetic_v0_baseline/metrics.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--jsonp-output",
        type=Path,
        help="Optional JavaScript payload path for local file:// demos. Defaults to output with .js suffix.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_by_id(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "null"}:
        return None
    return text


def as_float(value: Any) -> float | None:
    value = clean_value(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def as_bool(value: Any) -> bool:
    value = clean_value(value)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "t", "1", "yes", "y"}


def quantile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    index = round((len(sorted_values) - 1) * fraction)
    index = min(len(sorted_values) - 1, max(0, int(index)))
    return sorted_values[index]


def numeric_ranges(feature_rows: dict[str, dict[str, str]]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for column in NUMERIC_FEATURES:
        values = sorted(
            number
            for row in feature_rows.values()
            if (number := as_float(row.get(column))) is not None
        )
        if not values:
            continue
        out[column] = {
            "min": values[0],
            "p05": quantile(values, 0.05),
            "median": quantile(values, 0.50),
            "p95": quantile(values, 0.95),
            "max": values[-1],
        }
    return out


def categorical_values(feature_rows: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for column in CATEGORICAL_FEATURES:
        values = sorted(
            str(value)
            for row in feature_rows.values()
            if (value := clean_value(row.get(column))) is not None
        )
        out[column] = sorted(set(values))
    return out


def compact_feature_frame(row: dict[str, str]) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        features[column] = as_float(row.get(column))
    for column in CATEGORICAL_FEATURES:
        features[column] = clean_value(row.get(column))
    for column in BOOLEAN_FEATURES:
        features[column] = as_bool(row.get(column))
    return features


def compact_prediction_frame(prediction: dict[str, str], features: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for column in PREDICTION_FIELDS:
        row[column] = clean_value(prediction.get(column))
    row["label_0_to_4"] = int(as_float(row["label_0_to_4"]) or 0)
    row["predicted_label"] = int(as_float(row["predicted_label"]) or 0)
    row["window_length_seconds"] = int(as_float(row["window_length_seconds"]) or 0)
    row["p_large_training"] = as_float(row["p_large_training"]) or 0.0
    row["severity_score"] = as_float(row["severity_score"]) or 0.0
    row["negative_certification_confidence"] = as_float(row["negative_certification_confidence"]) or 0.0
    row["min_critical_coverage"] = as_float(row["min_critical_coverage"]) or 0.0
    row["integrity_warning"] = as_bool(row["integrity_warning"])
    row["p_labels"] = [
        as_float(prediction.get(f"p_label_{label}")) or 0.0
        for label in range(5)
    ]
    row["features"] = compact_feature_frame(features)
    return row


def scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["latent_workload_class"])].append(row)

    out: list[dict[str, Any]] = []
    for scenario, scenario_rows in grouped.items():
        labels = Counter(str(row["label_0_to_4"]) for row in scenario_rows)
        p_large = [float(row["p_large_training"]) for row in scenario_rows]
        out.append(
            {
                "scenario": scenario,
                "rows": len(scenario_rows),
                "label_distribution": dict(sorted(labels.items())),
                "mean_p_large_training": round(sum(p_large) / len(p_large), 6),
                "max_p_large_training": round(max(p_large), 6),
            }
        )
    return sorted(out, key=lambda item: (-int(item["rows"]), item["scenario"]))


def site_summary(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_sites = {site["site_id"]: site for site in manifest.get("sites", [])}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["site_id"])].append(row)

    out: list[dict[str, Any]] = []
    for site_id, site_rows in grouped.items():
        labels = Counter(str(row["label_0_to_4"]) for row in site_rows)
        scenarios = Counter(str(row["latent_workload_class"]) for row in site_rows)
        site = dict(manifest_sites.get(site_id, {"site_id": site_id}))
        site["rows"] = len(site_rows)
        site["label_distribution"] = dict(sorted(labels.items()))
        site["top_scenarios"] = [
            {"scenario": scenario, "rows": count}
            for scenario, count in scenarios.most_common(6)
        ]
        out.append(site)
    return sorted(out, key=lambda item: item["site_id"])


def example_rows(rows: list[dict[str, Any]]) -> dict[str, str]:
    examples: dict[str, str] = {}
    for label in range(5):
        label_rows = [row for row in rows if row["label_0_to_4"] == label]
        preferred = [
            row
            for row in label_rows
            if row["window_length_seconds"] == 3600 and row["split"] in {"test", "validation"}
        ]
        chosen = (preferred or label_rows)[0]
        examples[str(label)] = str(chosen["feature_row_id"])
    return examples


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest)
    metrics = read_json(args.metrics)
    feature_rows = read_csv_by_id(args.features, "feature_row_id")
    prediction_rows = read_csv_rows(args.predictions)

    rows: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        row_id = prediction["feature_row_id"]
        if row_id not in feature_rows:
            raise ValueError(f"prediction row missing from features: {row_id}")
        rows.append(compact_prediction_frame(prediction, feature_rows[row_id]))

    return {
        "metadata": {
            "title": "Datacenter training-run verification synthetic v0",
            "generated_from": {
                "features": str(args.features),
                "predictions": str(args.predictions),
                "manifest": str(args.manifest),
                "metrics": str(args.metrics),
            },
            "dataset_id": manifest.get("dataset_id"),
            "seed": manifest.get("seed"),
            "generator_version": manifest.get("generator_version"),
            "feature_pipeline_version": manifest.get("feature_pipeline_version"),
            "hardware_normalization_version": manifest.get("hardware_normalization_version"),
            "policy_threshold_version": manifest.get("policy_threshold_version"),
            "synthetic_notice": "All site, account, workload, and telemetry records are synthetic.",
            "row_count": len(rows),
            "site_count": manifest.get("site_count"),
            "episode_count": manifest.get("episode_count"),
            "raw_record_counts": manifest.get("raw_record_counts"),
            "model": metrics.get("model", {}),
            "calibration": metrics.get("calibration", {}),
            "governance": metrics.get("governance", {}),
        },
        "labels": LABELS,
        "sites": site_summary(rows, manifest),
        "scenarios": scenario_summary(rows),
        "feature_ranges": numeric_ranges(feature_rows),
        "categorical_values": categorical_values(feature_rows),
        "example_rows": example_rows(rows),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compact_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    args.output.write_text(compact_json + "\n", encoding="utf-8")
    jsonp_output = args.jsonp_output or args.output.with_suffix(".js")
    jsonp_output.parent.mkdir(parents=True, exist_ok=True)
    jsonp_output.write_text(f"window.DCVDemoData={compact_json};\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {jsonp_output}")
    print(f"rows {payload['metadata']['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
