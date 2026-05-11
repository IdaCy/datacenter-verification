"""Validate hard synthetic-v1 evidence structure.

This validator complements the schema/range validators. It checks whether a
generated hard-profile dataset contains the scenario families, counterfactual
groups, missingness regimes, overlap, and leakage exclusions needed for the
study benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .common import OBSERVABLE_IDS, V1_HARD_NEGATIVE_FAMILIES, V1_HARD_POSITIVE_FAMILIES, write_csv
except ImportError:  # pragma: no cover - direct script execution
    from common import OBSERVABLE_IDS, V1_HARD_NEGATIVE_FAMILIES, V1_HARD_POSITIVE_FAMILIES, write_csv

try:
    from src.datacenter_verification_modeling.common import BASE_EXCLUDED_COLUMNS, determine_feature_columns
except ImportError:  # pragma: no cover
    from datacenter_verification_modeling.common import BASE_EXCLUDED_COLUMNS, determine_feature_columns


TARGET_LABEL_RATIOS = {
    0: (0.25, 0.40),
    1: (0.15, 0.25),
    2: (0.20, 0.35),
    3: (0.12, 0.22),
    4: (0.04, 0.10),
}

AUDIT_LEAKAGE_COLUMNS = {
    "scenario_family",
    "scenario_variant",
    "evidence_recipe_id",
    "counterfactual_group_id",
    "synthetic_counterfactual_role",
    "data_quality_regime",
    "privacy_tier",
    "collector_profile",
    "topology_class",
    "temporal_phase",
    "synthetic_hard_case_tags",
}

HARD_POSITIVE_CHECKS = {
    "underclocked": ["underclocked_energy_capped_training"],
    "redacted": ["cloud_training_redacted_runtime"],
    "elastic": ["elastic_preempted_training"],
    "fragmented": ["fragmented_training_linked"],
    "sparse_or_bursty": ["sparse_or_moe_bursty_training"],
    "delayed_logs": ["training_with_delayed_logs"],
}

HARD_NEGATIVE_CHECKS = {
    "hpc": ["hpc_mpi_collective"],
    "nccl": ["nccl_extended_benchmark"],
    "model_parallel_inference": ["model_parallel_inference"],
    "storage_replication": ["storage_rebuild_or_replication"],
    "burn_in": ["hardware_burn_in_or_thermal_soak"],
    "maintenance_gaps": ["maintenance_with_collector_gaps"],
    "fragmented_nontraining": ["multi_tenant_fragmented_nontraining"],
}

OVERLAP_FEATURES = [
    "o4_gpu_util_p95",
    "o6_nvlink_util_p95",
    "o7_collective_periodicity_score",
    "o7_synchronized_fabric_footprint",
    "o8_rack_power_fraction_p95",
]


@dataclass
class Finding:
    severity: str
    code: str
    message: str


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _ratio(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def _counter_csv(path: Path, counter: Counter[Any], key_name: str) -> None:
    rows = [{key_name: str(key), "row_count": int(value)} for key, value in sorted(counter.items(), key=lambda item: str(item[0]))]
    write_csv(path, rows, [key_name, "row_count"])


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _overlap(a: pd.Series, b: pd.Series) -> bool:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 10 or len(b) < 10:
        return False
    a_low, a_high = float(a.quantile(0.10)), float(a.quantile(0.90))
    b_low, b_high = float(b.quantile(0.10)), float(b.quantile(0.90))
    return max(a_low, b_low) <= min(a_high, b_high)


def _hard_load_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (_numeric(df, "o4_gpu_util_p95") >= 70)
        | (_numeric(df, "o7_collective_periodicity_score") >= 0.55)
        | (_numeric(df, "o7_synchronized_fabric_footprint") >= 256)
        | (_numeric(df, "o8_rack_power_fraction_p95") >= 0.55)
        | (_numeric(df, "o11_backup_or_replication_pattern_score") >= 0.55)
    )


def _boundary_mask(df: pd.DataFrame) -> pd.Series:
    tags = df.get("synthetic_hard_case_tags", pd.Series([""] * len(df), index=df.index)).astype(str)
    return (
        tags.str.contains("boundary|overlap|missingness_edge|fragmented|underclocked|delayed", regex=True)
        | (_numeric(df, "o10_declared_vs_observed_mismatch_score") >= 0.5)
        | ((_numeric(df, "o7_inference_fanout_score") >= 0.55) & (_numeric(df, "o4_gpu_util_p95") >= 65))
        | ((_numeric(df, "o14_min_critical_coverage") < 0.85) & (_numeric(df, "o8_rack_power_fraction_p95") >= 0.45))
    )


def validate_hardness(dataset_dir: Path) -> tuple[bool, list[Finding], dict[str, Any]]:
    findings: list[Finding] = []

    def error(code: str, message: str) -> None:
        findings.append(Finding("ERROR", code, message))

    def warn(code: str, message: str) -> None:
        findings.append(Finding("WARNING", code, message))

    def info(code: str, message: str) -> None:
        findings.append(Finding("INFO", code, message))

    validation_dir = dataset_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    features_path = dataset_dir / "features" / "window_features_all.csv"
    if not features_path.exists():
        error("features_missing", f"Missing feature table: {features_path}")
        return False, findings, {}

    df = pd.read_csv(features_path)
    if "label_0_to_4" not in df.columns:
        error("label_missing", "Feature table is missing label_0_to_4")
        return False, findings, {}
    df["label_0_to_4"] = df["label_0_to_4"].astype(int)

    required_columns = set(AUDIT_LEAKAGE_COLUMNS)
    for obs_id in OBSERVABLE_IDS:
        key = obs_id.lower()
        required_columns.add(f"{key}_coverage_fraction")
        required_columns.add(f"{key}_missing_reason")
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        error("hard_columns_missing", f"Missing v1 hard/audit columns: {missing_columns[:30]}")

    feature_columns, exclusion_meta = determine_feature_columns(df.copy())
    supervised_leaks = sorted(AUDIT_LEAKAGE_COLUMNS & set(feature_columns))
    missing_model_exclusions = sorted(AUDIT_LEAKAGE_COLUMNS - set(BASE_EXCLUDED_COLUMNS))
    if supervised_leaks:
        error("leakage_columns_in_model_features", f"Synthetic audit columns are present in supervised feature columns: {supervised_leaks}")
    if missing_model_exclusions:
        error("leakage_columns_not_excluded_by_modeling", f"Modeling exclusions do not list: {missing_model_exclusions}")

    label_counts = Counter(int(value) for value in df["label_0_to_4"])
    total_rows = len(df)
    for label, (low, high) in TARGET_LABEL_RATIOS.items():
        ratio = label_counts.get(label, 0) / max(total_rows, 1)
        if label_counts.get(label, 0) == 0:
            error("label_absent", f"Label {label} has no rows")
        elif low <= ratio <= high:
            info("label_distribution_target", f"Label {label} ratio {ratio:.3f} is within target {low:.2f}-{high:.2f}")
        elif low - 0.04 <= ratio <= high + 0.04:
            warn("label_distribution_near_target", f"Label {label} ratio {ratio:.3f} is near target {low:.2f}-{high:.2f}")
        else:
            error("label_distribution_far_from_target", f"Label {label} ratio {ratio:.3f} is far from target {low:.2f}-{high:.2f}")

    families = set(df["scenario_family"].dropna().astype(str)) if "scenario_family" in df.columns else set()
    missing_positive = sorted(set(V1_HARD_POSITIVE_FAMILIES) - families)
    missing_negative = sorted(set(V1_HARD_NEGATIVE_FAMILIES) - families)
    if missing_positive:
        error("hard_positive_families_missing", f"Missing hard positive families: {missing_positive}")
    if missing_negative:
        error("hard_negative_families_missing", f"Missing hard negative families: {missing_negative}")

    label34 = df["label_0_to_4"] >= 3
    for name, check_families in HARD_POSITIVE_CHECKS.items():
        part = df[df["scenario_family"].isin(check_families)]
        count = int((part["label_0_to_4"] >= 3).sum()) if len(part) else 0
        if count == 0:
            error("hard_positive_rows_missing", f"No label 3/4 rows for hard positive category `{name}`")
        else:
            info("hard_positive_rows", f"{name}: {count} label 3/4 rows")

    for name, check_families in HARD_NEGATIVE_CHECKS.items():
        part = df[df["scenario_family"].isin(check_families)]
        hard_count = int((_hard_load_mask(part) & (part["label_0_to_4"] < 3)).sum()) if len(part) else 0
        if hard_count == 0:
            error("hard_negative_rows_missing", f"No high-load label 0/1/2 rows for hard negative category `{name}`")
        else:
            info("hard_negative_rows", f"{name}: {hard_count} high-load non-training rows")

    boundary_ratio = _ratio(_boundary_mask(df))
    if boundary_ratio < 0.10:
        error("boundary_rows_too_low", f"Boundary/conflicting rows are only {boundary_ratio:.1%}")
    elif not 0.20 <= boundary_ratio <= 0.35:
        warn("boundary_rows_outside_target", f"Boundary/conflicting rows are {boundary_ratio:.1%}; target is 20-35%")
    else:
        info("boundary_rows_target", f"Boundary/conflicting rows are {boundary_ratio:.1%}")

    group_col = df.get("counterfactual_group_id", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    grouped_rows = df[group_col != ""]
    group_count = int(grouped_rows["counterfactual_group_id"].nunique()) if len(grouped_rows) else 0
    grouped_episode_count = int(grouped_rows["episode_id"].nunique()) if len(grouped_rows) else 0
    total_episode_count = int(df["episode_id"].nunique()) if "episode_id" in df.columns else 0
    multi_label_groups = 0
    single_label_groups: list[str] = []
    for group_id, part in grouped_rows.groupby("counterfactual_group_id"):
        labels = set(int(value) for value in part["label_0_to_4"])
        if len(labels) >= 2:
            multi_label_groups += 1
        elif len(single_label_groups) < 10:
            single_label_groups.append(str(group_id))
    if group_count < 100:
        error("counterfactual_groups_too_few", f"Counterfactual group count is {group_count}; expected at least 100")
    if total_episode_count and grouped_episode_count / total_episode_count < 0.15:
        error("counterfactual_episode_share_too_low", f"Only {grouped_episode_count / total_episode_count:.1%} of episodes belong to counterfactual groups")
    if group_count and multi_label_groups / group_count < 0.95:
        error("counterfactual_groups_single_label", f"{group_count - multi_label_groups} groups do not contain at least two labels; examples: {single_label_groups}")
    else:
        info("counterfactual_groups", f"{group_count} groups; {multi_label_groups} contain multiple labels")

    signed_logs = _as_bool(df.get("o12_signed_ml_logs_present", pd.Series(["false"] * len(df), index=df.index)))
    label34_count = int(label34.sum())
    unsigned_label34_ratio = float(((label34) & (~signed_logs)).sum() / label34_count) if label34_count else 0.0
    if unsigned_label34_ratio < 0.10:
        error("label34_without_signed_logs_too_low", f"Only {unsigned_label34_ratio:.1%} of label 3/4 rows lack signed O12 logs")
    elif not 0.20 <= unsigned_label34_ratio <= 0.40:
        warn("label34_without_signed_logs_outside_target", f"{unsigned_label34_ratio:.1%} of label 3/4 rows lack signed O12 logs; target is 20-40%")
    else:
        info("label34_without_signed_logs_target", f"{unsigned_label34_ratio:.1%} of label 3/4 rows lack signed O12 logs")

    missing_reasons = pd.concat(
        [df[f"{obs.lower()}_missing_reason"].astype(str) for obs in OBSERVABLE_IDS if f"{obs.lower()}_missing_reason" in df.columns],
        axis=0,
    )
    rows_with_missingness = pd.Series(False, index=df.index)
    routine_reasons = {"", "observed", "nan", "not_applicable", "routine_profiler_disabled", "not_scheduled"}
    for obs in OBSERVABLE_IDS:
        reason_col = f"{obs.lower()}_missing_reason"
        coverage_col = f"{obs.lower()}_coverage_fraction"
        reasons = df[reason_col].astype(str) if reason_col in df.columns else pd.Series([""] * len(df), index=df.index)
        obs_routine_reasons = set(routine_reasons)
        if obs == "O12":
            # Missing signed ML logs is tracked separately for label 3/4 rows; for ordinary negatives,
            # O12 privacy redaction is not counted as a non-trivial coverage issue by itself.
            obs_routine_reasons.add("privacy_redacted")
        if reason_col in df.columns:
            rows_with_missingness |= ~reasons.isin(obs_routine_reasons)
        if coverage_col in df.columns:
            rows_with_missingness |= (_numeric(df, coverage_col) < 0.99) & (~reasons.isin(obs_routine_reasons))
    missingness_ratio = _ratio(rows_with_missingness)
    low_critical_ratio = _ratio(_numeric(df, "o14_min_critical_coverage") < 0.85)
    if not 0.25 <= missingness_ratio <= 0.45:
        warn("missingness_share_outside_target", f"{missingness_ratio:.1%} of rows have non-trivial missingness; target is 25-45%")
    if low_critical_ratio < 0.08:
        error("critical_low_coverage_too_low", f"Only {low_critical_ratio:.1%} of rows have critical coverage below 0.85")
    elif not 0.10 <= low_critical_ratio <= 0.20:
        warn("critical_low_coverage_outside_target", f"{low_critical_ratio:.1%} of rows have critical coverage below 0.85; target is 10-20%")

    redacted_label34_ratio = (
        float((label34 & ((_numeric(df, "o10_runtime_metadata_confidence") < 0.55) | (df.get("o10_missing_reason", "").astype(str) == "privacy_redacted"))).sum() / label34_count)
        if label34_count
        else 0.0
    )
    if not 0.15 <= redacted_label34_ratio <= 0.30:
        warn("label34_redacted_runtime_outside_target", f"{redacted_label34_ratio:.1%} of label 3/4 rows have redacted/low-confidence runtime; target is 15-30%")

    variant_label34_mask = label34 & df["scenario_family"].isin(
        ["underclocked_energy_capped_training", "elastic_preempted_training", "fragmented_training_linked"]
    )
    variant_label34_ratio = float(variant_label34_mask.sum() / label34_count) if label34_count else 0.0
    if not 0.10 <= variant_label34_ratio <= 0.22:
        warn("hard_variant_label34_outside_target", f"{variant_label34_ratio:.1%} of label 3/4 rows are underclocked/elastic/fragmented; target is 10-20%")

    label2 = df["label_0_to_4"] == 2
    close_label2 = label2 & _boundary_mask(df) & ((_numeric(df, "o4_gpu_util_p95") >= 65) | (_numeric(df, "o7_collective_periodicity_score") >= 0.45))
    close_label2_ratio = float(close_label2.sum() / max(1, label2.sum()))
    if close_label2_ratio < 0.20:
        warn("label2_close_to_label3_low", f"Only {close_label2_ratio:.1%} of label 2 rows look close to label 3")

    for family, part in df.groupby("scenario_family"):
        if len(part) < 20:
            continue
        labels = set(int(value) for value in part["label_0_to_4"])
        if labels and all(value >= 3 for value in labels):
            if family in V1_HARD_NEGATIVE_FAMILIES:
                error("negative_family_exclusively_positive", f"Negative family `{family}` maps exclusively to labels 3/4")
            else:
                warn("family_exclusively_label34", f"Family `{family}` maps exclusively to labels 3/4; document if defensible")

    for feature in OVERLAP_FEATURES:
        label2_values = df.loc[df["label_0_to_4"] == 2, feature] if feature in df.columns else pd.Series(dtype=float)
        label3_values = df.loc[df["label_0_to_4"] == 3, feature] if feature in df.columns else pd.Series(dtype=float)
        if not _overlap(label2_values, label3_values):
            warn("label2_label3_overlap_weak", f"Label 2 and label 3 distributions do not overlap enough for `{feature}`")

    low_labels = df["label_0_to_4"].isin([0, 1])
    if not ((_numeric(df.loc[low_labels], "o1_normalized_h100e_capacity") >= 512).any() and _hard_load_mask(df.loc[df["label_0_to_4"] == 2]).any()):
        warn("low_and_elevated_capacity_overlap_weak", "Label 0/1 and label 2 capacity/evidence overlap is weak")

    moderate_gpu_label34 = label34 & (_numeric(df, "o4_gpu_util_p95").between(50, 78)) & (
        (_numeric(df, "o11_checkpoint_periodicity_score") >= 0.5) | (_numeric(df, "o2_allocation_duration_hours") >= 72)
    )
    if int(moderate_gpu_label34.sum()) == 0:
        error("moderate_gpu_label34_missing", "No label 3/4 rows with moderate GPU utilization and strong duration/storage evidence")

    hard_negative_high_signal = (df["scenario_family"].isin(V1_HARD_NEGATIVE_FAMILIES)) & (_hard_load_mask(df)) & (df["label_0_to_4"] < 3)
    if int(hard_negative_high_signal.sum()) == 0:
        error("hard_negative_high_signal_missing", "No hard negatives with high fabric or high GPU but low training semantics")

    _counter_csv(validation_dir / "scenario_family_distribution.csv", Counter(df["scenario_family"].fillna("")), "scenario_family")
    _counter_csv(validation_dir / "counterfactual_group_distribution.csv", Counter(grouped_rows["counterfactual_group_id"]) if len(grouped_rows) else Counter(), "counterfactual_group_id")
    _counter_csv(validation_dir / "data_quality_regime_distribution.csv", Counter(df["data_quality_regime"].fillna("")), "data_quality_regime")

    summary = {
        "row_count": int(total_rows),
        "label_distribution": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "scenario_family_count": int(len(families)),
        "scenario_variant_count": int(df["scenario_variant"].nunique()) if "scenario_variant" in df.columns else 0,
        "counterfactual_group_count": group_count,
        "counterfactual_episode_share": grouped_episode_count / total_episode_count if total_episode_count else 0.0,
        "boundary_conflicting_ratio": boundary_ratio,
        "missingness_ratio": missingness_ratio,
        "low_critical_coverage_ratio": low_critical_ratio,
        "label34_without_signed_logs_ratio": unsigned_label34_ratio,
        "label34_redacted_runtime_ratio": redacted_label34_ratio,
        "model_exclusion_metadata": exclusion_meta,
        "missing_reason_distribution": {str(key): int(value) for key, value in Counter(missing_reasons).most_common()},
    }

    errors = [finding for finding in findings if finding.severity == "ERROR"]
    warnings = [finding for finding in findings if finding.severity == "WARNING"]
    lines = [
        "# Synthetic v1 Hardness Report",
        "",
        f"Dataset: `{dataset_dir}`",
        f"Status: {'PASS' if not errors else 'FAIL'}",
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['row_count']}",
        f"- Labels: {summary['label_distribution']}",
        f"- Scenario families: {summary['scenario_family_count']}",
        f"- Scenario variants: {summary['scenario_variant_count']}",
        f"- Counterfactual groups: {summary['counterfactual_group_count']}",
        f"- Boundary/conflicting rows: {boundary_ratio:.1%}",
        f"- Rows with missingness: {missingness_ratio:.1%}",
        f"- Rows with critical coverage below 0.85: {low_critical_ratio:.1%}",
        f"- Label 3/4 rows without signed O12 logs: {unsigned_label34_ratio:.1%}",
        "",
        "## Findings",
        "",
    ]
    if findings:
        for severity in ["ERROR", "WARNING", "INFO"]:
            group = [finding for finding in findings if finding.severity == severity]
            if not group:
                continue
            lines.append(f"### {severity}")
            for finding in group:
                lines.append(f"- `{finding.code}`: {finding.message}")
            lines.append("")
    else:
        lines.append("- No findings.")
    (validation_dir / "hardness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return not errors, findings, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    ok, findings, summary = validate_hardness(args.dataset)
    errors = [finding for finding in findings if finding.severity == "ERROR"]
    warnings = [finding for finding in findings if finding.severity == "WARNING"]
    print("PASS" if ok else "FAIL")
    print(f"rows: {summary.get('row_count', 0)}")
    print(f"label_distribution: {summary.get('label_distribution', {})}")
    print(f"counterfactual_groups: {summary.get('counterfactual_group_count', 0)}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    if errors:
        for finding in errors[:20]:
            print(f"ERROR: {finding.code}: {finding.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
