"""Deterministic evidence-rule baseline for synthetic v0.

The rule baseline: not intended to replace the supervised model; it provides a leakage-free point of comparison and encodes the study's governance constraints around capacity, missingness, and multi-layer evidence
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import LABELS, PROB_COLUMNS, as_bool, load_feature_table, normalize_probability_array, row_bool, row_float
except ImportError:  # pragma: no cover - direct script execution
    from common import LABELS, PROB_COLUMNS, as_bool, load_feature_table, normalize_probability_array, row_bool, row_float


FALSE_POSITIVE_RUNTIME_MARKERS = [
    "inference",
    "hpc",
    "mpi",
    "nccl",
    "benchmark",
    "burn_in",
    "storage",
    "etl",
    "synthetic_data",
]

FALSE_POSITIVE_DECLARED_CLASSES = {
    "inference",
    "hpc",
    "benchmark",
    "burn_in",
    "data",
    "reserved",
    "none",
}


def evidence_flags(row: pd.Series) -> dict[str, bool]:
    capacity = row_bool(row, "capacity_possible")
    external_conflict = row_float(row, "o17_external_capacity_conflict_score")
    allocation_gpus = row_float(row, "o2_max_concurrent_normalized_gpus")
    allocation_hours = row_float(row, "o2_allocation_duration_hours")
    gpu_hours_ratio = row_float(row, "o2_gpu_hours_policy_ratio")
    policy_ratio = row_float(row, "policy_compute_ratio")
    gpu_util = row_float(row, "o4_gpu_util_p95")
    gpu_duty = row_float(row, "o4_gpu_util_duty_gt_70")
    tensor = row_float(row, "o4_sm_tensor_active_p95")
    fabric_footprint = row_float(row, "o7_synchronized_fabric_footprint")
    fabric_periodicity = row_float(row, "o7_collective_periodicity_score")
    fabric_util = row_float(row, "o7_scaleout_port_util_p95")
    rack_power = row_float(row, "o8_rack_power_fraction_p95")
    checkpoint = row_float(row, "o11_checkpoint_periodicity_score")
    checkpoint_tb = row_float(row, "o11_checkpoint_write_tb_per_event")
    storage_pattern = row_float(row, "o11_read_write_training_pattern_score")
    signed_logs = row_bool(row, "o12_signed_ml_logs_present")
    declared_params = row_float(row, "o12_declared_parameter_count_b")
    training_tokens = row_float(row, "o12_training_tokens_b")
    optimizer_state = row_bool(row, "o12_optimizer_state_present")
    min_coverage = row_float(row, "o14_min_critical_coverage", 1.0)
    gap_fraction = row_float(row, "o14_gap_fraction_critical")
    cc_fraction = row_float(row, "o13_confidential_compute_mode_fraction")
    counter_resets = row_float(row, "o14_counter_reset_count")
    runtime = str(row["o10_runtime_framework_class"]).lower() if "o10_runtime_framework_class" in row else ""
    declared_class = str(row["o2_declared_workload_class"]).lower() if "o2_declared_workload_class" in row else ""

    allocation = allocation_gpus >= 512 or gpu_hours_ratio >= 1.0 or policy_ratio >= 1.0 or (
        allocation_gpus >= 256 and allocation_hours >= 12
    )
    gpu_activity = gpu_util >= 70 or gpu_duty >= 0.45 or tensor >= 0.60
    fabric_sync = fabric_footprint >= 256 or fabric_periodicity >= 0.55 or fabric_util >= 0.65
    physical_support = rack_power >= 0.55
    storage_semantic = checkpoint >= 0.55 or checkpoint_tb >= 0.25 or storage_pattern >= 0.60
    runtime_semantic = "training" in runtime or "fine_tune" in runtime or "pytorch_distributed" in runtime
    ml_semantic = signed_logs or declared_params >= 50 or training_tokens >= 100 or optimizer_state
    integrity = (
        gap_fraction > 0.05
        or min_coverage < 0.80
        or cc_fraction > 0.50
        or counter_resets > 0
        or row_bool(row, "o15_unapproved_physical_change_near_window")
        or str(row.get("o4_missing_reason", "")) == "counter_disabled_by_cc_mode"
    )
    false_positive_pattern = any(marker in runtime for marker in FALSE_POSITIVE_RUNTIME_MARKERS) or declared_class in (
        FALSE_POSITIVE_DECLARED_CLASSES
    )
    reserved_without_activity = (
        allocation_gpus >= 512
        and gpu_util < 30
        and fabric_footprint < 64
        and fabric_periodicity < 0.20
        and not storage_semantic
        and not ml_semantic
    )
    return {
        "capacity": capacity,
        "external_conflict": external_conflict >= 0.5,
        "allocation": allocation,
        "gpu_activity": gpu_activity,
        "fabric_sync": fabric_sync,
        "physical_support": physical_support,
        "storage_semantic": storage_semantic,
        "runtime_semantic": runtime_semantic,
        "ml_semantic": ml_semantic,
        "signed_logs": signed_logs,
        "integrity": integrity,
        "false_positive_pattern": false_positive_pattern,
        "reserved_without_activity": reserved_without_activity,
        "strong_coverage": min_coverage >= 0.90 and gap_fraction <= 0.05,
        "policy_scale": policy_ratio >= 1.0 or gpu_hours_ratio >= 1.0 or allocation_gpus >= 512,
    }


def predict_rule_label(row: pd.Series) -> int:
    flags = evidence_flags(row)
    primary_count = int(flags["allocation"]) + int(flags["gpu_activity"]) + int(flags["fabric_sync"])
    semantic_count = int(flags["runtime_semantic"]) + int(flags["storage_semantic"]) + int(flags["ml_semantic"])

    if not flags["capacity"] and not flags["external_conflict"]:
        return 0 if flags["strong_coverage"] else 1

    if flags["signed_logs"] and flags["policy_scale"] and primary_count >= 2:
        return 4

    if (
        flags["allocation"]
        and flags["gpu_activity"]
        and flags["fabric_sync"]
        and flags["physical_support"]
        and semantic_count >= 2
        and flags["policy_scale"]
    ):
        return 4

    if flags["reserved_without_activity"]:
        return 1

    if flags["false_positive_pattern"] and (flags["gpu_activity"] or flags["fabric_sync"] or flags["physical_support"]):
        return 2

    if flags["integrity"] and not flags["ml_semantic"] and (primary_count >= 1 or flags["physical_support"]):
        return 2

    if primary_count >= 2 and (flags["physical_support"] or semantic_count >= 1):
        return 3

    if primary_count >= 1 or flags["physical_support"] or semantic_count >= 1:
        return 2

    return 1 if flags["capacity"] else 0


def predict_rule_labels(df: pd.DataFrame) -> pd.Series:
    return df.apply(predict_rule_label, axis=1).astype(int)


def predict_rule_probabilities(df: pd.DataFrame, confidence: float = 0.82) -> pd.DataFrame:
    labels = predict_rule_labels(df)
    off_probability = (1.0 - confidence) / (len(LABELS) - 1)
    probabilities = np.full((len(df), len(LABELS)), off_probability, dtype=float)
    for row_index, label in enumerate(labels):
        probabilities[row_index, LABELS.index(int(label))] = confidence
    probabilities = normalize_probability_array(probabilities)
    return pd.DataFrame(probabilities, columns=PROB_COLUMNS, index=df.index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    df = load_feature_table(args.features)
    labels = predict_rule_labels(df)
    out = df[["feature_row_id", "episode_id", "latent_workload_class", "label_0_to_4"]].copy()
    out["rule_predicted_label"] = labels
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
    else:
        print(out.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

