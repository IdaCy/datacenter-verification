"""Evaluate stress splits for synthetic datacenter verification datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_recall_fscore_support

try:
    from .common import (
        DEFAULT_SEED,
        LABELS,
        PROB_COLUMNS,
        determine_feature_columns,
        ensure_dir,
        load_feature_table,
        make_episode_split,
        make_preprocessor,
        model_input_frame,
        probability_frame,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        DEFAULT_SEED,
        LABELS,
        PROB_COLUMNS,
        determine_feature_columns,
        ensure_dir,
        load_feature_table,
        make_episode_split,
        make_preprocessor,
        model_input_frame,
        probability_frame,
        write_json,
    )


FAMILY_HOLDOUTS = [
    "underclocked_energy_capped_training",
    "fragmented_training_linked",
    "multi_tenant_fragmented_nontraining",
    "model_parallel_inference",
    "hpc_mpi_collective",
]

SOURCE_ABLATIONS = {
    "source_ablation_drop_fabric": ["o6_", "o7_", "fabric_telemetry_trust_level"],
    "source_ablation_drop_runtime_and_ml_logs": ["o10_", "o12_"],
    "source_ablation_drop_gpu_telemetry": ["o4_", "o5_", "gpu_telemetry_trust_level"],
    "source_ablation_drop_power": ["o8_", "o9_", "power_meter_trust_level"],
    "source_ablation_drop_storage": ["o11_"],
}


def _base_classifier(seed: int, max_iter: int) -> HistGradientBoostingClassifier:
    kwargs: dict[str, Any] = {
        "learning_rate": 0.06,
        "max_iter": max_iter,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.04,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": seed,
    }
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **kwargs)
    except TypeError:  # pragma: no cover
        return HistGradientBoostingClassifier(**kwargs)


def _binary_prf(y_true: np.ndarray, p_large: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true >= 3,
        p_large >= 0.5,
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _scenario_counts(df: pd.DataFrame, mask: pd.Series, scenario_column: str) -> dict[str, int]:
    if not mask.any():
        return {}
    return {str(key): int(value) for key, value in df.loc[mask, scenario_column].value_counts().sort_values(ascending=False).items()}


def _middle_probability_bins(p_large: np.ndarray) -> dict[str, int]:
    return {
        "0.1_to_0.3": int(((p_large >= 0.1) & (p_large < 0.3)).sum()),
        "0.3_to_0.7": int(((p_large >= 0.3) & (p_large < 0.7)).sum()),
        "0.7_to_0.9": int(((p_large >= 0.7) & (p_large < 0.9)).sum()),
    }


def _fit_predict_split(
    df: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    feature_columns: list[str],
    seed: int,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("stress split has empty train or test set")
    if train_df["label_0_to_4"].nunique() < 2:
        raise ValueError("stress split train set has fewer than two labels")

    preprocessor = make_preprocessor(train_df, feature_columns)
    x_train = preprocessor.fit_transform(model_input_frame(train_df, feature_columns))
    y_train = train_df["label_0_to_4"].astype(int).to_numpy()
    x_test = preprocessor.transform(model_input_frame(test_df, feature_columns))
    y_test = test_df["label_0_to_4"].astype(int).to_numpy()

    model = _base_classifier(seed, max_iter)
    model.fit(x_train, y_train)
    probabilities = probability_frame(model, x_test)
    p_values = probabilities[PROB_COLUMNS].to_numpy()
    y_pred = np.asarray(LABELS)[np.argmax(p_values, axis=1)]
    p_large = probabilities["p_label_3"].to_numpy() + probabilities["p_label_4"].to_numpy()
    out = test_df[["feature_row_id", "episode_id", "label_0_to_4"]].copy()
    for column in ["scenario_family", "latent_workload_class", "site_id", "window_start"]:
        if column in test_df.columns:
            out[column] = test_df[column].to_numpy()
    out["predicted_label"] = y_pred
    out["p_large_training"] = p_large
    for column in PROB_COLUMNS:
        out[column] = probabilities[column].to_numpy()

    large_true = y_test >= 3
    large_pred = p_large >= 0.5
    scenario_column = "scenario_family" if "scenario_family" in test_df.columns else "latent_workload_class"
    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_episodes": int(train_df["episode_id"].nunique()),
        "test_episodes": int(test_df["episode_id"].nunique()),
        "test_label_distribution": {str(k): int(v) for k, v in Counter(y_test).items()},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "large_training": _binary_prf(y_test, p_large),
        "large_training_false_positive_count": int(((~large_true) & large_pred).sum()),
        "large_training_false_negative_count": int((large_true & (~large_pred)).sum()),
        "false_positive_scenarios": _scenario_counts(test_df, pd.Series((~large_true) & large_pred, index=test_df.index), scenario_column),
        "false_negative_scenarios": _scenario_counts(test_df, pd.Series(large_true & (~large_pred), index=test_df.index), scenario_column),
        "middle_probability_bins": _middle_probability_bins(p_large),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_test, p_values, labels=LABELS))
    except ValueError:
        metrics["log_loss"] = None
    return out, metrics


def _drop_features(feature_columns: list[str], prefixes: list[str]) -> list[str]:
    dropped = []
    for column in feature_columns:
        if any(column == prefix or column.startswith(prefix) for prefix in prefixes):
            continue
        dropped.append(column)
    return dropped


def evaluate_stress_splits(features_path: Path, output_dir: Path, seed: int = DEFAULT_SEED, max_iter: int = 120) -> dict[str, Any]:
    ensure_dir(output_dir)
    df = load_feature_table(features_path)
    scenario_column = "scenario_family" if "scenario_family" in df.columns else "latent_workload_class"
    feature_columns, exclusion_meta = determine_feature_columns(df.copy())
    metrics: dict[str, Any] = {
        "features_path": str(features_path),
        "seed": int(seed),
        "max_iter": int(max_iter),
        "scenario_column": scenario_column,
        "model_feature_count": int(len(feature_columns)),
        "exclusion_metadata": exclusion_meta,
        "splits": {},
    }

    split_df, split_manifest = make_episode_split(df, seed=seed)
    random_train = split_df["split"].isin(["train", "validation"])
    random_test = split_df["split"] == "test"
    _, metrics["splits"]["episode_grouped_random"] = _fit_predict_split(
        split_df, random_train, random_test, feature_columns, seed, max_iter
    )

    sorted_times = pd.to_datetime(df["window_start"], utc=True)
    cutoff = sorted_times.quantile(0.80)
    _, metrics["splits"]["time_holdout"] = _fit_predict_split(
        df, sorted_times < cutoff, sorted_times >= cutoff, feature_columns, seed + 1, max_iter
    )

    site_results = {}
    for idx, site_id in enumerate(sorted(df["site_id"].dropna().unique())):
        test_mask = df["site_id"] == site_id
        train_mask = ~test_mask
        try:
            _, site_results[str(site_id)] = _fit_predict_split(df, train_mask, test_mask, feature_columns, seed + 10 + idx, max_iter)
        except ValueError as exc:
            site_results[str(site_id)] = {"error": str(exc)}
    metrics["splits"]["site_holdout"] = site_results

    family_results = {}
    for idx, family in enumerate(FAMILY_HOLDOUTS):
        if family not in set(df[scenario_column].astype(str)):
            family_results[family] = {"error": "family not present"}
            continue
        test_mask = df[scenario_column].astype(str) == family
        train_mask = ~test_mask
        try:
            _, family_results[family] = _fit_predict_split(df, train_mask, test_mask, feature_columns, seed + 30 + idx, max_iter)
        except ValueError as exc:
            family_results[family] = {"error": str(exc)}
    metrics["splits"]["scenario_family_holdout"] = family_results

    ablation_results = {}
    for idx, (name, prefixes) in enumerate(SOURCE_ABLATIONS.items()):
        ablated_features = _drop_features(feature_columns, prefixes)
        try:
            _, ablation_results[name] = _fit_predict_split(
                split_df, random_train, random_test, ablated_features, seed + 50 + idx, max_iter
            )
            ablation_results[name]["feature_count"] = int(len(ablated_features))
            ablation_results[name]["dropped_prefixes"] = prefixes
        except ValueError as exc:
            ablation_results[name] = {"error": str(exc)}
    metrics["splits"]["source_ablations"] = ablation_results

    write_json(output_dir / "stress_metrics.json", metrics)
    write_stress_summary(output_dir / "stress_summary.md", metrics)
    update_model_run_readme_with_stress(output_dir, metrics)
    return metrics


def _split_line(name: str, item: dict[str, Any]) -> str:
    if "error" in item:
        return f"- {name}: ERROR: {item['error']}"
    return (
        f"- {name}: rows={item['test_rows']}, macro F1={item['macro_f1']:.3f}, "
        f"large precision={item['large_training']['precision']:.3f}, "
        f"large recall={item['large_training']['recall']:.3f}, "
        f"FP={item['large_training_false_positive_count']}, FN={item['large_training_false_negative_count']}, "
        f"middle p bins={item['middle_probability_bins']}"
    )


def write_stress_summary(path: Path, metrics: dict[str, Any]) -> None:
    splits = metrics["splits"]
    lines = [
        "# Synthetic v1 Stress Summary",
        "",
        f"Feature table: `{metrics['features_path']}`",
        f"Seed: `{metrics['seed']}`",
        f"Lightweight stress model max_iter: `{metrics['max_iter']}`",
        "",
        "## Core Splits",
        "",
        _split_line("episode_grouped_random", splits["episode_grouped_random"]),
        _split_line("time_holdout", splits["time_holdout"]),
        "",
        "## Site Holdout",
        "",
    ]
    for site_id, item in splits["site_holdout"].items():
        lines.append(_split_line(site_id, item))
    lines.extend(["", "## Scenario Family Holdout", ""])
    for family, item in splits["scenario_family_holdout"].items():
        lines.append(_split_line(family, item))
        if "false_negative_scenarios" in item and item["false_negative_scenarios"]:
            lines.append(f"  false negatives: {item['false_negative_scenarios']}")
        if "false_positive_scenarios" in item and item["false_positive_scenarios"]:
            lines.append(f"  false positives: {item['false_positive_scenarios']}")
    lines.extend(["", "## Source Ablations", ""])
    baseline = splits["episode_grouped_random"]
    baseline_f1 = baseline.get("macro_f1") if "error" not in baseline else None
    for name, item in splits["source_ablations"].items():
        lines.append(_split_line(name, item))
        if baseline_f1 is not None and "macro_f1" in item:
            lines.append(f"  macro F1 delta vs random baseline: {item['macro_f1'] - baseline_f1:+.3f}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These are stress diagnostics, not tuned headline metrics.",
            "- Family holdouts identify hard families that still fail under scenario generalization.",
            "- Source ablations should degrade performance when the dropped layer carries independent evidence.",
            "- Middle probability bins indicate whether v1 has less all-or-nothing large-training probability mass than v0.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_model_run_readme_with_stress(model_run_dir: Path, metrics: dict[str, Any]) -> None:
    readme_path = model_run_dir / "README.md"
    if not readme_path.exists():
        return
    splits = metrics["splits"]
    baseline = splits["episode_grouped_random"]
    fragmented = splits["scenario_family_holdout"].get("fragmented_training_linked", {})
    underclocked = splits["scenario_family_holdout"].get("underclocked_energy_capped_training", {})
    runtime_drop = splits["source_ablations"].get("source_ablation_drop_runtime_and_ml_logs", {})
    baseline_f1 = baseline.get("macro_f1", 0.0)
    runtime_delta = runtime_drop.get("macro_f1", baseline_f1) - baseline_f1 if "macro_f1" in runtime_drop else 0.0
    section = f"""## Stress Splits

Stress outputs are in:

- `stress_metrics.json`
- `stress_summary.md`

Key stress checks from the lightweight retraining script:

- Episode-grouped random split macro F1: {baseline.get('macro_f1', 0.0):.4f}; large-training recall: {baseline.get('large_training', {}).get('recall', 0.0):.4f}
- Time holdout macro F1: {splits['time_holdout'].get('macro_f1', 0.0):.4f}; large-training recall: {splits['time_holdout'].get('large_training', {}).get('recall', 0.0):.4f}
- Underclocked holdout large-training recall: {underclocked.get('large_training', {}).get('recall', 0.0):.4f}; false negatives: {underclocked.get('large_training_false_negative_count', 0)}
- Fragmented linked training holdout remains hard: {fragmented.get('large_training_false_negative_count', 0)} large-training false negatives
- Source ablation dropping runtime and ML logs changed macro F1 by {runtime_delta:+.3f} versus the random stress baseline
- Middle `p_large_training` bins are populated in stress checks: {baseline.get('middle_probability_bins', {})}

"""
    readme = readme_path.read_text(encoding="utf-8")
    start = readme.find("## Stress Splits\n")
    if start != -1:
        end = readme.find("## Reproduce\n", start)
        if end == -1:
            end = readme.find("## Limitations\n", start)
        if end == -1:
            readme = readme[:start].rstrip() + "\n\n" + section
        else:
            readme = readme[:start].rstrip() + "\n\n" + section + readme[end:]
    else:
        marker = "## Reproduce\n"
        idx = readme.find(marker)
        if idx == -1:
            readme = readme.rstrip() + "\n\n" + section
        else:
            readme = readme[:idx].rstrip() + "\n\n" + section + readme[idx:]
    readme_path.write_text(readme, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model-run", type=Path, required=True, help="Directory to write stress outputs into.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args(argv)
    metrics = evaluate_stress_splits(args.features, args.model_run, seed=args.seed, max_iter=args.max_iter)
    random_metrics = metrics["splits"]["episode_grouped_random"]
    print(f"episode_grouped_random_macro_f1: {random_metrics['macro_f1']:.4f}")
    print(f"episode_grouped_random_large_recall: {random_metrics['large_training']['recall']:.4f}")
    print(f"stress_metrics: {args.model_run / 'stress_metrics.json'}")
    print(f"stress_summary: {args.model_run / 'stress_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
