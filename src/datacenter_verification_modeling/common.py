"""Shared utilities for synthetic datacenter verification modeling baselines."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_SEED = 20260510
LABELS = [0, 1, 2, 3, 4]
SPLITS = ["train", "validation", "test"]
PROB_COLUMNS = [f"p_label_{label}" for label in LABELS]
RAW_PROB_COLUMNS = [f"raw_p_label_{label}" for label in LABELS]

BASE_EXCLUDED_COLUMNS = [
    "feature_row_id",
    "dataset_id",
    "seed",
    "site_id",
    "scope_id_hash",
    "window_start",
    "window_end",
    "episode_id",
    "label_0_to_4",
    "label_confidence",
    "label_reason",
    "label_source",
    "raw_input_manifest_hash",
    "latent_workload_class",
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
    "synthetic_evidence_profile",
    "capacity_evidence_only",
    "integrity_evidence_only",
    "physical_evidence_only",
]

VERSION_COLUMNS_EXCLUDED = [
    "feature_pipeline_version",
    "policy_threshold_version",
    "hardware_normalization_version",
]

CRITICAL_COVERAGE_COLUMNS = [
    "o1_coverage_fraction",
    "o2_coverage_fraction",
    "o4_coverage_fraction",
    "o7_coverage_fraction",
    "o8_coverage_fraction",
    "o14_coverage_fraction",
]

CRITICAL_MISSING_REASON_COLUMNS = {
    "o1_coverage_fraction": "o1_missing_reason",
    "o2_coverage_fraction": "o2_missing_reason",
    "o4_coverage_fraction": "o4_missing_reason",
    "o7_coverage_fraction": "o7_missing_reason",
    "o8_coverage_fraction": "o8_missing_reason",
    "o14_coverage_fraction": "o14_missing_reason",
}

SELECTED_AUDIT_FEATURES = [
    "policy_compute_ratio",
    "o2_max_concurrent_normalized_gpus",
    "o2_allocation_duration_hours",
    "o2_gpu_hours_policy_ratio",
    "o4_gpu_util_p95",
    "o4_gpu_util_duty_gt_70",
    "o7_synchronized_fabric_footprint",
    "o7_collective_periodicity_score",
    "o8_rack_power_fraction_p95",
    "o10_runtime_framework_class",
    "o11_checkpoint_periodicity_score",
    "o12_signed_ml_logs_present",
    "o12_declared_parameter_count_b",
    "o14_min_critical_coverage",
    "o14_gap_fraction_critical",
    "o13_confidential_compute_mode_fraction",
    "o2_elastic_resize_count",
    "o2_preemption_restart_count",
    "o2_account_linkage_confidence",
    "o4_hbm_pressure_duration_fraction",
    "o4_power_cap_active_fraction",
    "o7_account_flow_linkage_confidence",
    "o10_runtime_metadata_confidence",
    "o11_artifact_write_pattern_score",
    "o11_dataloader_read_pattern_score",
    "o12_log_delivery_delay_hours",
    "o12_log_completeness_fraction",
    "o4_missing_reason",
    "o7_missing_reason",
    "o12_missing_reason",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=to_jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    return as_float(row[column], default) if column in row else default


def row_bool(row: pd.Series, column: str, default: bool = False) -> bool:
    return as_bool(row[column], default) if column in row else default


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "feature_row_id",
        "episode_id",
        "latent_workload_class",
        "label_0_to_4",
        "capacity_possible",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"feature table missing required columns: {missing}")
    df["label_0_to_4"] = df["label_0_to_4"].astype(int)
    return df


def derive_dataset_dir(features_path: Path) -> Path:
    path = features_path.resolve()
    if path.parent.name == "features":
        return path.parent.parent
    return path.parent


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        summary[split] = {
            "rows": int(len(part)),
            "episodes": int(part["episode_id"].nunique()),
            "label_distribution": {
                str(label): int(count)
                for label, count in part["label_0_to_4"].value_counts().sort_index().items()
            },
            "scenario_distribution": {
                str(name): int(count)
                for name, count in part["latent_workload_class"].value_counts().sort_index().items()
            },
        }
    return summary


def make_episode_split(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a deterministic scenario-stratified grouped split by episode_id."""
    episode_df = (
        df.groupby("episode_id", as_index=False)
        .agg(
            latent_workload_class=("latent_workload_class", "first"),
            label_mode=("label_0_to_4", lambda values: int(values.mode().iloc[0])),
            label_min=("label_0_to_4", "min"),
            label_max=("label_0_to_4", "max"),
            row_count=("feature_row_id", "count"),
            site_count=("site_id", "nunique") if "site_id" in df.columns else ("episode_id", "count"),
        )
        .sort_values(["latent_workload_class", "episode_id"])
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    scenario_allocations: dict[str, dict[str, int]] = {}
    for scenario, scenario_rows in episode_df.groupby("latent_workload_class", sort=True):
        episode_ids = scenario_rows["episode_id"].to_numpy().copy()
        rng.shuffle(episode_ids)
        count = len(episode_ids)
        if count >= 3:
            validation_count = max(1, round(0.20 * count))
            test_count = max(1, round(0.20 * count))
            train_count = count - validation_count - test_count
            if train_count < 1:
                train_count = 1
                if validation_count > test_count:
                    validation_count -= 1
                else:
                    test_count -= 1
        elif count == 2:
            train_count, validation_count, test_count = 1, 1, 0
        else:
            train_count, validation_count, test_count = 1, 0, 0

        for episode_id in episode_ids[:train_count]:
            assignments[str(episode_id)] = "train"
        for episode_id in episode_ids[train_count : train_count + validation_count]:
            assignments[str(episode_id)] = "validation"
        for episode_id in episode_ids[train_count + validation_count :]:
            assignments[str(episode_id)] = "test"
        scenario_allocations[str(scenario)] = {
            "episodes": int(count),
            "train": int(train_count),
            "validation": int(validation_count),
            "test": int(test_count),
        }

    split_df = df.copy()
    split_df["split"] = split_df["episode_id"].map(assignments)
    if split_df["split"].isna().any():
        raise ValueError("episode split assignment failed for some rows")

    episode_assignments = []
    split_lookup = dict(zip(episode_df["episode_id"], episode_df.index, strict=False))
    for episode_id, split in sorted(assignments.items()):
        row = episode_df.iloc[split_lookup[episode_id]]
        episode_assignments.append(
            {
                "episode_id": episode_id,
                "split": split,
                "latent_workload_class": row["latent_workload_class"],
                "label_mode": int(row["label_mode"]),
                "label_min": int(row["label_min"]),
                "label_max": int(row["label_max"]),
                "row_count": int(row["row_count"]),
            }
        )

    manifest = {
        "seed": int(seed),
        "method": "scenario_stratified_grouped_by_episode_id",
        "split_fractions_requested": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "leakage_prevention": "All rows from the same episode_id are assigned to exactly one split.",
        "scenario_allocations": scenario_allocations,
        "summary": split_summary(split_df),
        "episode_assignments": episode_assignments,
    }
    return split_df, manifest


def apply_split_manifest(df: pd.DataFrame, split_manifest: dict[str, Any]) -> pd.DataFrame:
    assignments = {
        item["episode_id"]: item["split"]
        for item in split_manifest.get("episode_assignments", [])
        if "episode_id" in item and "split" in item
    }
    out = df.copy()
    out["split"] = out["episode_id"].map(assignments).fillna("unassigned")
    return out


def determine_feature_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    requested_exclusions = BASE_EXCLUDED_COLUMNS + VERSION_COLUMNS_EXCLUDED
    present_exclusions = [column for column in requested_exclusions if column in df.columns]
    missing_requested = [column for column in requested_exclusions if column not in df.columns]
    feature_columns = [column for column in df.columns if column not in set(present_exclusions + ["split"])]
    feature_columns = [column for column in feature_columns if not column.startswith("raw_")]
    metadata = {
        "requested_excluded_columns": requested_exclusions,
        "present_excluded_columns": present_exclusions,
        "missing_requested_excluded_columns": missing_requested,
        "notes": [
            "Identifier, label, direct leakage, site, and synthetic-only audit columns are excluded.",
            "scope_type and window_length_seconds are retained because they are valid deployment-time context.",
            "Constant version metadata columns are excluded from the baseline.",
        ],
    }
    return feature_columns, metadata


def categorize_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in feature_columns:
        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return {"numeric": numeric_columns, "categorical": categorical_columns}


def make_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    types = categorize_feature_columns(df, feature_columns)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if types["numeric"]:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ]
                ),
                types["numeric"],
            )
        )
    if types["categorical"]:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        except TypeError:  # pragma: no cover - compatibility for older scikit-learn
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", encoder),
                    ]
                ),
                types["categorical"],
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def model_input_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"feature table missing model feature columns: {missing}")
    return df.loc[:, feature_columns].copy()


def probability_frame(model: Any, transformed_features: Any, prefix: str = "p_label") -> pd.DataFrame:
    raw = model.predict_proba(transformed_features)
    classes = [int(label) for label in model.classes_]
    data = np.zeros((raw.shape[0], len(LABELS)), dtype=float)
    for source_index, label in enumerate(classes):
        if label in LABELS:
            data[:, LABELS.index(label)] = raw[:, source_index]
    data = normalize_probability_array(data)
    return pd.DataFrame(data, columns=[f"{prefix}_{label}" for label in LABELS])


def normalize_probability_array(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    row_sums = probs.sum(axis=1)
    empty = row_sums <= 0
    if np.any(empty):
        probs[empty, :] = 1.0 / len(LABELS)
        row_sums = probs.sum(axis=1)
    return probs / row_sums[:, None]


def minimum_critical_coverage(df: pd.DataFrame) -> pd.Series:
    coverage_values = []
    for column in CRITICAL_COVERAGE_COLUMNS:
        if column in df.columns:
            coverage_values.append(pd.to_numeric(df[column], errors="coerce").fillna(0.0).clip(0.0, 1.0))
        else:
            coverage_values.append(pd.Series(np.zeros(len(df)), index=df.index))
    return pd.concat(coverage_values, axis=1).min(axis=1)


def integrity_warning_series(df: pd.DataFrame) -> pd.Series:
    warnings = pd.Series(False, index=df.index)
    checks = {
        "o14_gap_fraction_critical": (0.05, "gt"),
        "o14_min_critical_coverage": (0.80, "lt"),
        "o13_attestation_valid_fraction": (0.90, "lt"),
        "o13_confidential_compute_mode_fraction": (0.50, "gt"),
        "o14_counter_reset_count": (0.0, "gt"),
    }
    for column, (threshold, direction) in checks.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        if direction == "gt":
            warnings |= values > threshold
        else:
            warnings |= values < threshold
    if "o15_unapproved_physical_change_near_window" in df.columns:
        warnings |= df["o15_unapproved_physical_change_near_window"].map(as_bool)
    return warnings.astype(bool)


def apply_capacity_gate(df: pd.DataFrame, probabilities: pd.DataFrame, high_label_cap: float = 0.02) -> pd.DataFrame:
    probs = probabilities.loc[:, PROB_COLUMNS].to_numpy(copy=True)
    if "capacity_possible" not in df.columns:
        return pd.DataFrame(normalize_probability_array(probs), columns=PROB_COLUMNS, index=probabilities.index)
    capacity_possible = df["capacity_possible"].map(as_bool).to_numpy()
    if "o17_external_capacity_conflict_score" in df.columns:
        external_conflict = pd.to_numeric(df["o17_external_capacity_conflict_score"], errors="coerce").fillna(0.0).to_numpy()
    else:
        external_conflict = np.zeros(len(df))
    gate_mask = (~capacity_possible) & (external_conflict < 0.5)
    for row_index in np.where(gate_mask)[0]:
        high = probs[row_index, 2:5].sum()
        capped_high = min(float(high), high_label_cap)
        if high > 0:
            probs[row_index, 2:5] *= capped_high / high
        low_target = 1.0 - capped_high
        low = probs[row_index, 0:2].sum()
        if low > 0:
            probs[row_index, 0:2] *= low_target / low
        else:
            probs[row_index, 0] = low_target
            probs[row_index, 1] = 0.0
    return pd.DataFrame(normalize_probability_array(probs), columns=PROB_COLUMNS, index=probabilities.index)


def critical_missing_layers_for_row(row: pd.Series) -> str:
    layers: list[str] = []
    for coverage_column, missing_column in CRITICAL_MISSING_REASON_COLUMNS.items():
        coverage = row_float(row, coverage_column, 0.0)
        reason = str(row[missing_column]) if missing_column in row and not pd.isna(row[missing_column]) else "unknown"
        if coverage < 0.80 or reason not in {"observed", ""}:
            observable = coverage_column.split("_", 1)[0].upper()
            layers.append(f"{observable}:{reason}:coverage={coverage:.2f}")
    return "; ".join(layers)


def top_evidence_for_row(row: pd.Series) -> str:
    evidence: list[str] = []
    capacity = row_bool(row, "capacity_possible")
    external_conflict = row_float(row, "o17_external_capacity_conflict_score", 0.0)
    allocation_gpus = row_float(row, "o2_max_concurrent_normalized_gpus")
    allocation_hours = row_float(row, "o2_allocation_duration_hours")
    gpu_hours_ratio = row_float(row, "o2_gpu_hours_policy_ratio")
    gpu_util = row_float(row, "o4_gpu_util_p95")
    tensor = row_float(row, "o4_sm_tensor_active_p95")
    fabric_footprint = row_float(row, "o7_synchronized_fabric_footprint")
    fabric_periodicity = row_float(row, "o7_collective_periodicity_score")
    rack_power = row_float(row, "o8_rack_power_fraction_p95")
    checkpoint = row_float(row, "o11_checkpoint_periodicity_score")
    signed_logs = row_bool(row, "o12_signed_ml_logs_present")
    min_coverage = row_float(row, "o14_min_critical_coverage", 1.0)
    gap_fraction = row_float(row, "o14_gap_fraction_critical")
    cc_fraction = row_float(row, "o13_confidential_compute_mode_fraction")
    runtime = str(row["o10_runtime_framework_class"]) if "o10_runtime_framework_class" in row else ""

    if not capacity and external_conflict < 0.5:
        evidence.append("capacity below policy threshold")
    if external_conflict >= 0.5:
        evidence.append("external capacity conflict")
    if allocation_gpus >= 512 or gpu_hours_ratio >= 1.0:
        evidence.append("large allocation")
    elif allocation_gpus >= 128:
        evidence.append("moderate allocation")
    if allocation_hours >= 24:
        evidence.append("long allocation duration")
    if gpu_util >= 70 or tensor >= 0.60:
        evidence.append("high GPU activity")
    if fabric_footprint >= 512 or fabric_periodicity >= 0.60:
        evidence.append("synchronized scale-out fabric")
    if rack_power >= 0.60:
        evidence.append("power corroboration")
    if "training" in runtime or "fine_tune" in runtime:
        evidence.append("training runtime metadata")
    if checkpoint >= 0.55:
        evidence.append("checkpoint cadence")
    if signed_logs:
        evidence.append("signed ML logs")
    if min_coverage < 0.80 or gap_fraction > 0.05:
        evidence.append("low critical coverage")
    if cc_fraction > 0.50 or str(row.get("o4_missing_reason", "")) == "counter_disabled_by_cc_mode":
        evidence.append("counter disabled by CC mode")
    if not evidence:
        evidence.append("no strong positive evidence")
    return "; ".join(evidence[:8])


def add_governance_outputs(df: pd.DataFrame, raw_probabilities: pd.DataFrame) -> pd.DataFrame:
    post_probs = apply_capacity_gate(df, raw_probabilities.loc[:, PROB_COLUMNS])
    out = post_probs.copy()
    prob_values = out.loc[:, PROB_COLUMNS].to_numpy()
    predicted = np.asarray(LABELS)[np.argmax(prob_values, axis=1)]
    out["predicted_label"] = predicted.astype(int)
    out["p_large_training"] = out["p_label_3"] + out["p_label_4"]
    out["severity_score"] = sum(label * out[f"p_label_{label}"] for label in LABELS)
    min_coverage = minimum_critical_coverage(df)
    out["min_critical_coverage"] = min_coverage
    out["negative_certification_confidence"] = out["p_label_0"] * min_coverage
    out["capacity_possible"] = df["capacity_possible"].map(as_bool) if "capacity_possible" in df.columns else False
    out["integrity_warning"] = integrity_warning_series(df)
    out["critical_missing_layers"] = df.apply(critical_missing_layers_for_row, axis=1)
    out["top_evidence"] = df.apply(top_evidence_for_row, axis=1)
    return out


def build_prediction_frame(
    df: pd.DataFrame,
    raw_probabilities: pd.DataFrame,
    governance_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
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
        "scenario_family",
        "scenario_variant",
        "data_quality_regime",
        "counterfactual_group_id",
        "synthetic_hard_case_tags",
        "label_0_to_4",
    ]
    present_base_columns = [column for column in base_columns if column in df.columns]
    out = df.loc[:, present_base_columns].copy()
    raw = raw_probabilities.loc[:, PROB_COLUMNS].copy()
    raw.columns = RAW_PROB_COLUMNS
    out = pd.concat([out.reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    governance_columns = PROB_COLUMNS + [
        "predicted_label",
        "p_large_training",
        "severity_score",
        "capacity_possible",
        "negative_certification_confidence",
        "integrity_warning",
        "critical_missing_layers",
        "top_evidence",
        "min_critical_coverage",
    ]
    out = pd.concat([out, governance_probabilities.loc[:, governance_columns].reset_index(drop=True)], axis=1)
    for column in SELECTED_AUDIT_FEATURES:
        if column in df.columns and column not in out.columns:
            out[column] = df[column].to_numpy()
    return out
