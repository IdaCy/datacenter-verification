"""Evaluate a trained synthetic v0 modeling run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

try:
    from .common import (
        DEFAULT_SEED,
        LABELS,
        PROB_COLUMNS,
        RAW_PROB_COLUMNS,
        SELECTED_AUDIT_FEATURES,
        add_governance_outputs,
        apply_split_manifest,
        build_prediction_frame,
        load_feature_table,
        minimum_critical_coverage,
        model_input_frame,
        probability_frame,
        read_json,
        sha256_file,
        utc_now_iso,
        write_json,
    )
    from .rule_baseline import predict_rule_labels
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        DEFAULT_SEED,
        LABELS,
        PROB_COLUMNS,
        RAW_PROB_COLUMNS,
        SELECTED_AUDIT_FEATURES,
        add_governance_outputs,
        apply_split_manifest,
        build_prediction_frame,
        load_feature_table,
        minimum_critical_coverage,
        model_input_frame,
        probability_frame,
        read_json,
        sha256_file,
        utc_now_iso,
        write_json,
    )
    from rule_baseline import predict_rule_labels


HARD_FALSE_POSITIVE_SCENARIOS = {
    "large_batch_inference",
    "synthetic_data_generation",
    "hpc_mpi_simulation",
    "nccl_benchmark",
    "hardware_burn_in",
    "storage_rebuild",
    "large_etl_data_movement",
    "reserved_but_unused_capacity",
    "maintenance_window",
}

TRAINING_SCENARIOS = {
    "large_fine_tune",
    "pretraining",
    "cloud_reservation_used_for_training",
    "adversarial_fragmented_training",
    "underclocked_long_duration_training",
}

REQUIRED_OUTPUTS = [
    "README.md",
    "manifest.json",
    "model.joblib",
    "preprocessing.joblib",
    "feature_columns.json",
    "excluded_columns.json",
    "split_manifest.json",
    "metrics.json",
    "calibration_metrics.json",
    "confusion_matrix.csv",
    "classification_report.json",
    "predictions_test.csv",
    "predictions_all.csv",
    "feature_importance.csv",
    "evidence_audit_sample.csv",
    "validation_summary.md",
]


def predict_for_model_run(model_run_dir: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_feature_table(features_path)
    split_manifest_path = model_run_dir / "split_manifest.json"
    if split_manifest_path.exists():
        df = apply_split_manifest(df, read_json(split_manifest_path))
    else:
        df = df.copy()
        df["split"] = "unassigned"

    feature_columns = read_json(model_run_dir / "feature_columns.json")
    preprocessor = joblib.load(model_run_dir / "preprocessing.joblib")
    model = joblib.load(model_run_dir / "model.joblib")

    model_frame = model_input_frame(df, feature_columns)
    transformed = preprocessor.transform(model_frame)
    raw_probabilities = probability_frame(model, transformed)
    governance = add_governance_outputs(df, raw_probabilities)
    predictions = build_prediction_frame(df, raw_probabilities, governance)
    return df, predictions


def binary_prf(true_binary: pd.Series | np.ndarray, predicted_binary: pd.Series | np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_binary.astype(bool),
        predicted_binary.astype(bool),
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def scenario_counts(frame: pd.DataFrame) -> dict[str, int]:
    if len(frame) == 0:
        return {}
    return {str(key): int(value) for key, value in frame["latent_workload_class"].value_counts().sort_values(ascending=False).items()}


def calibration_by_label(test_predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = test_predictions["label_0_to_4"].astype(int).to_numpy()
    out: dict[str, Any] = {}
    for label in LABELS:
        observed = (y_true == label).astype(int)
        predicted = test_predictions[f"p_label_{label}"].to_numpy()
        out[str(label)] = {
            "rows": int(len(test_predictions)),
            "observed_fraction": float(observed.mean()) if len(observed) else 0.0,
            "mean_predicted_probability": float(predicted.mean()) if len(predicted) else 0.0,
            "brier": float(brier_score_loss(observed, predicted)) if len(np.unique(observed)) > 1 else float(np.mean((predicted - observed) ** 2)),
        }
    return out


def reliability_bins(probabilities: np.ndarray, observed: np.ndarray, bin_count: int = 10) -> tuple[list[dict[str, Any]], float]:
    bins: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    total = len(probabilities)
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    for index in range(bin_count):
        lower = edges[index]
        upper = edges[index + 1]
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count:
            mean_predicted = float(probabilities[mask].mean())
            observed_fraction = float(observed[mask].mean())
            expected_calibration_error += (count / total) * abs(mean_predicted - observed_fraction)
        else:
            mean_predicted = 0.0
            observed_fraction = 0.0
        bins.append(
            {
                "bin_index": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_predicted_probability": mean_predicted,
                "observed_fraction": observed_fraction,
            }
        )
    return bins, float(expected_calibration_error)


def compute_calibration_metrics(test_predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = test_predictions["label_0_to_4"].astype(int).to_numpy()
    probabilities = test_predictions[PROB_COLUMNS].to_numpy()
    large_observed = (y_true >= 3).astype(int)
    large_probability = test_predictions["p_large_training"].to_numpy()
    bins, ece = reliability_bins(large_probability, large_observed)
    return {
        "log_loss": float(log_loss(y_true, probabilities, labels=LABELS)),
        "brier_large_training": float(brier_score_loss(large_observed, large_probability)),
        "expected_calibration_error_large_training": ece,
        "reliability_bins_large_training": bins,
        "calibration_by_label": calibration_by_label(test_predictions),
    }


def subgroup_metrics(test_predictions: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    if group_column not in test_predictions.columns:
        return []
    rows: list[dict[str, Any]] = []
    for value, part in test_predictions.groupby(group_column, dropna=False):
        y_true = part["label_0_to_4"].astype(int)
        y_pred = part["predicted_label"].astype(int)
        large_true = y_true >= 3
        large_pred = part["p_large_training"] >= 0.5
        prf = binary_prf(large_true.to_numpy(), large_pred.to_numpy())
        rows.append(
            {
                "group": str(value),
                "rows": int(len(part)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
                "large_training_precision_at_0_5": prf["precision"],
                "large_training_recall_at_0_5": prf["recall"],
                "large_training_false_positives": int(((~large_true) & large_pred).sum()),
                "large_training_false_negatives": int((large_true & (~large_pred)).sum()),
            }
        )
    return sorted(rows, key=lambda item: (-item["rows"], item["group"]))


def compute_metrics(test_predictions: pd.DataFrame, test_features: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    y_true = test_predictions["label_0_to_4"].astype(int)
    y_pred = test_predictions["predicted_label"].astype(int)
    probabilities = test_predictions[PROB_COLUMNS].to_numpy()
    large_true = y_true >= 3
    large_pred_by_label = y_pred >= 3
    large_pred_by_threshold = test_predictions["p_large_training"] >= 0.5
    large_label_prf = binary_prf(large_true.to_numpy(), large_pred_by_label.to_numpy())
    large_threshold_prf = binary_prf(large_true.to_numpy(), large_pred_by_threshold.to_numpy())

    y_one_hot = np.zeros_like(probabilities)
    y_one_hot[np.arange(len(y_true)), y_true.to_numpy()] = 1.0

    rule_labels = predict_rule_labels(test_features)
    rule_large = rule_labels >= 3
    rule_prf = binary_prf(large_true.to_numpy(), rule_large.to_numpy())

    false_positives = test_predictions[(y_true < 3) & large_pred_by_threshold]
    false_negatives = test_predictions[(y_true >= 3) & (~large_pred_by_threshold)]
    high_coverage_label0 = test_predictions[
        (y_true == 0) & (pd.to_numeric(test_predictions["min_critical_coverage"], errors="coerce").fillna(0.0) >= 0.95)
    ]
    label0_missed = high_coverage_label0[high_coverage_label0["predicted_label"] != 0]

    metrics: dict[str, Any] = {
        "dataset": {
            "test_rows": int(len(test_predictions)),
            "test_episodes": int(test_predictions["episode_id"].nunique()),
            "test_label_distribution": {
                str(label): int(count) for label, count in y_true.value_counts().sort_index().items()
            },
        },
        "model": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
            "log_loss": float(log_loss(y_true, probabilities, labels=LABELS)),
            "brier_multiclass_mean": float(np.mean((probabilities - y_one_hot) ** 2)),
        },
        "governance": {
            "label_3_4_predicted_label": large_label_prf,
            "p_large_training_threshold_0_5": large_threshold_prf,
            "false_positive_scenarios_at_0_5": scenario_counts(false_positives),
            "false_negative_scenarios_at_0_5": scenario_counts(false_negatives),
            "label_0_missed_under_high_coverage": {
                "rows": int(len(high_coverage_label0)),
                "missed_rows": int(len(label0_missed)),
                "missed_rate": float(len(label0_missed) / len(high_coverage_label0)) if len(high_coverage_label0) else 0.0,
            },
        },
        "rule_baseline": {
            "accuracy": float(accuracy_score(y_true, rule_labels)),
            "macro_f1": float(f1_score(y_true, rule_labels, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, rule_labels, labels=LABELS, average="weighted", zero_division=0)),
            "label_3_4_predicted_label": rule_prf,
        },
        "subgroup_metrics": {
            column: subgroup_metrics(test_predictions, column)
            for column in [
                "latent_workload_class",
                "site_id",
                "window_length_seconds",
                "o4_missing_reason",
                "o7_missing_reason",
                "o12_missing_reason",
                "capacity_possible",
                "integrity_warning",
            ]
        },
    }
    calibration = compute_calibration_metrics(test_predictions)
    metrics["calibration"] = {
        "log_loss": calibration["log_loss"],
        "brier_large_training": calibration["brier_large_training"],
        "expected_calibration_error_large_training": calibration["expected_calibration_error_large_training"],
    }
    return metrics, calibration


def write_confusion_matrix(path: Path, test_predictions: pd.DataFrame) -> None:
    y_true = test_predictions["label_0_to_4"].astype(int)
    y_pred = test_predictions["predicted_label"].astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    frame = pd.DataFrame(matrix, index=[f"true_{label}" for label in LABELS], columns=[f"pred_{label}" for label in LABELS])
    frame.to_csv(path, index_label="true_label")


def write_classification_report(path: Path, test_predictions: pd.DataFrame) -> None:
    report = classification_report(
        test_predictions["label_0_to_4"].astype(int),
        test_predictions["predicted_label"].astype(int),
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    write_json(path, report)


def compute_feature_importance(
    model_run_dir: Path,
    test_features: pd.DataFrame,
    test_predictions: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    repeats: int = 3,
) -> pd.DataFrame:
    feature_columns = read_json(model_run_dir / "feature_columns.json")
    preprocessor = joblib.load(model_run_dir / "preprocessing.joblib")
    model = joblib.load(model_run_dir / "model.joblib")
    x_test = model_input_frame(test_features, feature_columns)
    y_true = test_predictions["label_0_to_4"].astype(int)
    baseline_pred = np.asarray(LABELS)[
        np.argmax(probability_frame(model, preprocessor.transform(x_test))[PROB_COLUMNS].to_numpy(), axis=1)
    ]
    baseline_score = f1_score(y_true, baseline_pred, labels=LABELS, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        scores: list[float] = []
        if x_test[column].nunique(dropna=False) <= 1:
            importances = [0.0] * repeats
        else:
            importances = []
            for _ in range(repeats):
                permuted = x_test.copy()
                values = permuted[column].to_numpy(copy=True)
                rng.shuffle(values)
                permuted[column] = values
                permuted_pred = np.asarray(LABELS)[
                    np.argmax(
                        probability_frame(model, preprocessor.transform(permuted))[PROB_COLUMNS].to_numpy(),
                        axis=1,
                    )
                ]
                score = f1_score(y_true, permuted_pred, labels=LABELS, average="macro", zero_division=0)
                scores.append(float(score))
                importances.append(float(baseline_score - score))
        rows.append(
            {
                "feature": column,
                "importance_mean": float(np.mean(importances)),
                "importance_std": float(np.std(importances)),
                "baseline_macro_f1": float(baseline_score),
                "permuted_macro_f1_mean": float(np.mean(scores)) if scores else float(baseline_score),
            }
        )
    return pd.DataFrame(rows).sort_values(["importance_mean", "feature"], ascending=[False, True])


def evidence_audit_sample(test_predictions: pd.DataFrame, min_rows: int = 50) -> pd.DataFrame:
    selected_indices: set[int] = set()
    samples: list[pd.DataFrame] = []

    def add_sample(name: str, mask: pd.Series, count: int, sort_column: str | None = None, ascending: bool = False) -> None:
        candidates = test_predictions[mask & (~test_predictions.index.isin(selected_indices))].copy()
        if sort_column and sort_column in candidates.columns:
            candidates = candidates.sort_values(sort_column, ascending=ascending)
        candidates = candidates.head(count)
        if len(candidates):
            selected_indices.update(int(index) for index in candidates.index)
            candidates["audit_category"] = name
            samples.append(candidates)

    true_label = test_predictions["label_0_to_4"].astype(int)
    predicted_label = test_predictions["predicted_label"].astype(int)
    large_pred = test_predictions["p_large_training"] >= 0.5
    add_sample("correct_label_0", (true_label == 0) & (predicted_label == 0), 10, "negative_certification_confidence", False)
    add_sample(
        "correct_label_2_hard_false_positive",
        (true_label == 2)
        & (predicted_label == 2)
        & test_predictions["latent_workload_class"].isin(HARD_FALSE_POSITIVE_SCENARIOS),
        10,
        "p_large_training",
        False,
    )
    add_sample("correct_label_3_4_likely_training", (true_label >= 3) & (predicted_label >= 3), 10, "p_large_training", False)
    add_sample("false_positive_large_training", (true_label < 3) & large_pred, 10, "p_large_training", False)
    add_sample("false_negative_large_training", (true_label >= 3) & (~large_pred), 10, "p_large_training", True)

    if samples:
        sample = pd.concat(samples, axis=0)
    else:
        sample = test_predictions.iloc[0:0].copy()
    if len(sample) < min_rows:
        filler = test_predictions[~test_predictions.index.isin(selected_indices)].copy()
        filler["uncertainty_distance"] = (filler["p_large_training"] - 0.5).abs()
        filler = filler.sort_values(["uncertainty_distance", "severity_score"], ascending=[True, False]).head(min_rows - len(sample))
        filler["audit_category"] = "high_uncertainty_or_coverage_edge"
        sample = pd.concat([sample, filler], axis=0)

    columns = [
        "audit_category",
        "feature_row_id",
        "split",
        "latent_workload_class",
        "label_0_to_4",
        "predicted_label",
        "p_large_training",
        "severity_score",
        "top_evidence",
        "critical_missing_layers",
        "integrity_warning",
    ]
    columns += [column for column in SELECTED_AUDIT_FEATURES if column in sample.columns]
    return sample.loc[:, [column for column in columns if column in sample.columns]].rename(columns={"label_0_to_4": "true_label"})


def top_scenarios_text(scenarios: dict[str, int], limit: int = 5) -> str:
    if not scenarios:
        return "none"
    return ", ".join(f"{name}: {count}" for name, count in list(scenarios.items())[:limit])


def write_run_readme(model_run_dir: Path, features_path: Path, metrics: dict[str, Any], calibration: dict[str, Any]) -> None:
    model_metrics = metrics["model"]
    governance = metrics["governance"]
    split_manifest = read_json(model_run_dir / "split_manifest.json")
    readme = f"""# synthetic v0 baseline model run

This directory contains the first public runnable baseline for the synthetic v0 datacenter training-run verification dataset.

## Dataset

- Feature table: `{features_path}`
- Rows: {sum(split['rows'] for split in split_manifest['summary'].values())}
- Episode split: grouped by `episode_id`, scenario-stratified, seed `{split_manifest['seed']}`
- Split rows: train {split_manifest['summary']['train']['rows']}, validation {split_manifest['summary']['validation']['rows']}, test {split_manifest['summary']['test']['rows']}
- Split episodes: train {split_manifest['summary']['train']['episodes']}, validation {split_manifest['summary']['validation']['episodes']}, test {split_manifest['summary']['test']['episodes']}

## Model

- Supervised model: calibrated scikit-learn histogram gradient boosting classifier
- Calibration: validation split only, held-out test evaluated once
- Rule baseline: deterministic evidence rules in `src/datacenter_verification_modeling/rule_baseline.py`
- Leakage exclusions: identifiers, labels, site id, episode id, raw manifest hash, and synthetic-only audit columns

## Headline Test Metrics

- Accuracy: {model_metrics['accuracy']:.4f}
- Macro F1: {model_metrics['macro_f1']:.4f}
- Weighted F1: {model_metrics['weighted_f1']:.4f}
- Log loss: {model_metrics['log_loss']:.4f}
- Label 3/4 precision by predicted label: {governance['label_3_4_predicted_label']['precision']:.4f}
- Label 3/4 recall by predicted label: {governance['label_3_4_predicted_label']['recall']:.4f}
- `p_large_training >= 0.5` precision: {governance['p_large_training_threshold_0_5']['precision']:.4f}
- `p_large_training >= 0.5` recall: {governance['p_large_training_threshold_0_5']['recall']:.4f}
- Rule baseline macro F1: {metrics['rule_baseline']['macro_f1']:.4f}

## Error Scenarios

- Largest false-positive scenarios at `p_large_training >= 0.5`: {top_scenarios_text(governance['false_positive_scenarios_at_0_5'])}
- Largest false-negative scenarios at `p_large_training >= 0.5`: {top_scenarios_text(governance['false_negative_scenarios_at_0_5'])}

## Calibration

- Brier score for `p_large_training`: {calibration['brier_large_training']:.4f}
- Expected calibration error for `p_large_training`: {calibration['expected_calibration_error_large_training']:.4f}

## Reproduce

```bash
python src/datacenter_verification_modeling/train_model.py \\
  --features data/synthetic_v0/features/window_features_all.csv \\
  --output data/model_runs/synthetic_v0_baseline \\
  --seed 20260510
```

```bash
python src/datacenter_verification_modeling/evaluate_model.py \\
  --model-run data/model_runs/synthetic_v0_baseline \\
  --features data/synthetic_v0/features/window_features_all.csv
```

```bash
python src/datacenter_verification_modeling/predict.py \\
  --model-run data/model_runs/synthetic_v0_baseline \\
  --features data/synthetic_v0/features/window_features_all.csv \\
  --output data/model_runs/synthetic_v0_baseline/predictions_all.csv
```

## Limitations

- This model is trained on synthetic data only.
- Performance numbers are not real-world deployment claims.
- Adjacent windows are correlated, so group splitting by `episode_id` is mandatory.
- Synthetic labels are generated from rules and latent scenarios, so the model may learn generator assumptions.
- Real datacenter deployment would require calibration on real telemetry and controlled drills.
- The model should assist audit triage; it should not be treated as sole proof of a violation.
"""
    (model_run_dir / "README.md").write_text(readme, encoding="utf-8")


def write_validation_summary(
    model_run_dir: Path,
    features_path: Path,
    metrics: dict[str, Any],
    validation_status: list[dict[str, Any]] | None,
) -> None:
    if validation_status is None and (model_run_dir / "manifest.json").exists():
        existing_manifest = read_json(model_run_dir / "manifest.json")
        validation_status = existing_manifest.get("validation_status") or None
    required_rows = []
    for filename in REQUIRED_OUTPUTS:
        path = model_run_dir / filename
        exists = path.exists() or filename == "validation_summary.md"
        required_rows.append(f"- `{filename}`: {'present' if exists else 'missing'}")
    validation_lines = []
    if validation_status:
        for item in validation_status:
            validation_lines.append(
                f"- `{item['name']}`: return code {item['returncode']}; "
                f"{'PASS' if item['returncode'] == 0 else 'FAIL'}"
            )
    else:
        validation_lines.append("- Dataset validation was not rerun by this evaluation command; see `manifest.json` for any recorded training validation status.")

    model_metrics = metrics["model"]
    governance = metrics["governance"]
    text = f"""# Validation Summary

Feature table: `{features_path}`

## Dataset Validation Before Training

{chr(10).join(validation_lines)}

## Required Artifacts

{chr(10).join(required_rows)}

## Test Metrics

- Accuracy: {model_metrics['accuracy']:.4f}
- Macro F1: {model_metrics['macro_f1']:.4f}
- Label 3/4 precision: {governance['label_3_4_predicted_label']['precision']:.4f}
- Label 3/4 recall: {governance['label_3_4_predicted_label']['recall']:.4f}
- `p_large_training >= 0.5` precision: {governance['p_large_training_threshold_0_5']['precision']:.4f}
- `p_large_training >= 0.5` recall: {governance['p_large_training_threshold_0_5']['recall']:.4f}

## Governance Checks

- Capacity gate applied to post-processed probabilities.
- Negative certification confidence is `p_label_0 * min_critical_coverage`.
- Integrity warnings are reported separately from positive training evidence.
- Raw model probabilities are retained as `raw_p_label_0` through `raw_p_label_4`.
"""
    (model_run_dir / "validation_summary.md").write_text(text, encoding="utf-8")


def write_manifest(
    model_run_dir: Path,
    features_path: Path,
    metrics: dict[str, Any],
    calibration: dict[str, Any],
    validation_status: list[dict[str, Any]] | None,
    training_metadata: dict[str, Any] | None,
) -> None:
    existing: dict[str, Any] = {}
    manifest_path = model_run_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
    artifact_hashes = {
        filename: sha256_file(model_run_dir / filename)
        for filename in REQUIRED_OUTPUTS
        if (model_run_dir / filename).exists() and filename != "manifest.json"
    }
    manifest = {
        **existing,
        "created_or_updated_at": utc_now_iso(),
        "model_run_id": "synthetic_v0_baseline",
        "features_path": str(features_path),
        "model_type": "CalibratedClassifierCV over HistGradientBoostingClassifier",
        "calibration_method": existing.get("calibration_method", "sigmoid_on_validation_split"),
        "metrics_summary": {
            "accuracy": metrics["model"]["accuracy"],
            "macro_f1": metrics["model"]["macro_f1"],
            "weighted_f1": metrics["model"]["weighted_f1"],
            "label_3_4_precision": metrics["governance"]["label_3_4_predicted_label"]["precision"],
            "label_3_4_recall": metrics["governance"]["label_3_4_predicted_label"]["recall"],
            "p_large_training_precision_at_0_5": metrics["governance"]["p_large_training_threshold_0_5"]["precision"],
            "p_large_training_recall_at_0_5": metrics["governance"]["p_large_training_threshold_0_5"]["recall"],
            "brier_large_training": calibration["brier_large_training"],
            "ece_large_training": calibration["expected_calibration_error_large_training"],
        },
        "validation_status": validation_status or existing.get("validation_status", []),
        "training_metadata": training_metadata or existing.get("training_metadata", {}),
        "required_outputs": REQUIRED_OUTPUTS,
        "artifact_hashes": artifact_hashes,
        "limitations": [
            "Trained on synthetic data only.",
            "Performance numbers are not real-world deployment claims.",
            "Episode-level group splitting is required because adjacent windows are correlated.",
            "Synthetic labels encode generator assumptions.",
            "Real deployment requires real telemetry calibration and controlled drills.",
        ],
    }
    write_json(manifest_path, manifest)


def evaluate_model_run(
    model_run_dir: Path,
    features_path: Path,
    validation_status: list[dict[str, Any]] | None = None,
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_run_dir.mkdir(parents=True, exist_ok=True)
    feature_df, predictions = predict_for_model_run(model_run_dir, features_path)
    predictions.to_csv(model_run_dir / "predictions_all.csv", index=False)
    test_predictions = predictions[predictions["split"] == "test"].copy()
    test_features = feature_df[feature_df["split"] == "test"].copy()
    if test_predictions.empty:
        raise ValueError("no test split rows available for evaluation")
    test_predictions.to_csv(model_run_dir / "predictions_test.csv", index=False)

    metrics, calibration = compute_metrics(test_predictions, test_features)
    write_json(model_run_dir / "metrics.json", metrics)
    write_json(model_run_dir / "calibration_metrics.json", calibration)
    write_confusion_matrix(model_run_dir / "confusion_matrix.csv", test_predictions)
    write_classification_report(model_run_dir / "classification_report.json", test_predictions)
    importance = compute_feature_importance(model_run_dir, test_features, test_predictions)
    importance.to_csv(model_run_dir / "feature_importance.csv", index=False)
    audit = evidence_audit_sample(test_predictions)
    audit.to_csv(model_run_dir / "evidence_audit_sample.csv", index=False)
    write_run_readme(model_run_dir, features_path, metrics, calibration)
    write_manifest(model_run_dir, features_path, metrics, calibration, validation_status, training_metadata)
    write_validation_summary(model_run_dir, features_path, metrics, validation_status)
    write_manifest(model_run_dir, features_path, metrics, calibration, validation_status, training_metadata)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = evaluate_model_run(args.model_run, args.features)
    print(f"accuracy: {metrics['model']['accuracy']:.4f}")
    print(f"macro_f1: {metrics['model']['macro_f1']:.4f}")
    print(f"label_3_4_precision: {metrics['governance']['label_3_4_predicted_label']['precision']:.4f}")
    print(f"label_3_4_recall: {metrics['governance']['label_3_4_predicted_label']['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
