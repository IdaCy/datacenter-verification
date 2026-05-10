"""Train the calibrated supervised baseline for synthetic v0."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from .common import (
        DEFAULT_SEED,
        determine_feature_columns,
        derive_dataset_dir,
        ensure_dir,
        load_feature_table,
        make_episode_split,
        make_preprocessor,
        model_input_frame,
        utc_now_iso,
        write_json,
    )
    from .evaluate_model import evaluate_model_run
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        DEFAULT_SEED,
        determine_feature_columns,
        derive_dataset_dir,
        ensure_dir,
        load_feature_table,
        make_episode_split,
        make_preprocessor,
        model_input_frame,
        utc_now_iso,
        write_json,
    )
    from evaluate_model import evaluate_model_run


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_dataset_validations(features_path: Path) -> list[dict[str, Any]]:
    root = repo_root()
    dataset_dir = derive_dataset_dir(features_path)
    try:
        dataset_arg = dataset_dir.relative_to(root)
    except ValueError:
        dataset_arg = dataset_dir
    commands = [
        {
            "name": "synthetic_dataset_validator",
            "command": [
                sys.executable,
                str(root / "src/datacenter_verification_synthetic/validate_synthetic_dataset.py"),
                "--dataset",
                str(dataset_arg),
            ],
        },
        {
            "name": "public_dataset_validator",
            "command": [
                sys.executable,
                "-m",
                "src.datacenter_verification_validators",
                "--dataset",
                str(dataset_arg),
            ],
        },
    ]
    results: list[dict[str, Any]] = []
    for item in commands:
        completed = subprocess.run(
            item["command"],
            cwd=root,
            check=False,
            text=True,
            capture_output=True,
        )
        result = {
            "name": item["name"],
            "command": " ".join(item["command"]),
            "returncode": int(completed.returncode),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"dataset validation failed for {item['name']} with return code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )
    return results


def make_base_classifier(seed: int) -> HistGradientBoostingClassifier:
    kwargs: dict[str, Any] = {
        "learning_rate": 0.05,
        "max_iter": 350,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.03,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "random_state": seed,
    }
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **kwargs)
    except TypeError:  # pragma: no cover - older scikit-learn fallback
        return HistGradientBoostingClassifier(**kwargs)


def calibrate_prefit_model(base_model: Any, x_validation: np.ndarray, y_validation: np.ndarray) -> tuple[Any, str]:
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
        method = "sigmoid_on_validation_split_frozen_estimator"
    except Exception:  # pragma: no cover - older scikit-learn fallback
        calibrator = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
        method = "sigmoid_on_validation_split_prefit"
    calibrator.fit(x_validation, y_validation)
    return calibrator, method


def train_model(features_path: Path, output_dir: Path, seed: int = DEFAULT_SEED, skip_dataset_validation: bool = False) -> dict[str, Any]:
    ensure_dir(output_dir)
    validation_status = [] if skip_dataset_validation else run_dataset_validations(features_path)

    df = load_feature_table(features_path)
    split_df, split_manifest = make_episode_split(df, seed=seed)
    feature_columns, excluded_columns = determine_feature_columns(split_df)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "feature_columns.json", feature_columns)
    write_json(output_dir / "excluded_columns.json", excluded_columns)

    train_df = split_df[split_df["split"] == "train"].copy()
    validation_df = split_df[split_df["split"] == "validation"].copy()
    if train_df.empty or validation_df.empty:
        raise ValueError("train and validation splits must both be non-empty")

    preprocessor = make_preprocessor(train_df, feature_columns)
    x_train = preprocessor.fit_transform(model_input_frame(train_df, feature_columns))
    y_train = train_df["label_0_to_4"].astype(int).to_numpy()
    x_validation = preprocessor.transform(model_input_frame(validation_df, feature_columns))
    y_validation = validation_df["label_0_to_4"].astype(int).to_numpy()

    base_model = make_base_classifier(seed)
    base_model.fit(x_train, y_train)
    calibrated_model, calibration_method = calibrate_prefit_model(base_model, x_validation, y_validation)

    joblib.dump(calibrated_model, output_dir / "model.joblib")
    joblib.dump(preprocessor, output_dir / "preprocessing.joblib")

    training_metadata = {
        "trained_at": utc_now_iso(),
        "seed": int(seed),
        "features_path": str(features_path),
        "model_type": "HistGradientBoostingClassifier",
        "model_parameters": base_model.get_params(),
        "calibration_method": calibration_method,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "train_episodes": int(train_df["episode_id"].nunique()),
        "validation_episodes": int(validation_df["episode_id"].nunique()),
        "feature_count": int(len(feature_columns)),
        "dataset_validation_status": validation_status,
    }
    write_json(
        output_dir / "manifest.json",
        {
            "model_run_id": "synthetic_v0_baseline",
            "created_or_updated_at": utc_now_iso(),
            "features_path": str(features_path),
            "model_type": "CalibratedClassifierCV over HistGradientBoostingClassifier",
            "calibration_method": calibration_method,
            "training_metadata": training_metadata,
            "validation_status": validation_status,
        },
    )

    metrics = evaluate_model_run(
        output_dir,
        features_path,
        validation_status=validation_status,
        training_metadata=training_metadata,
    )
    return {
        "metrics": metrics,
        "split_summary": split_manifest["summary"],
        "validation_status": validation_status,
        "training_metadata": training_metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--skip-dataset-validation",
        action="store_true",
        help="Skip pre-training dataset validation. Intended only for development.",
    )
    args = parser.parse_args(argv)
    result = train_model(args.features, args.output, seed=args.seed, skip_dataset_validation=args.skip_dataset_validation)
    metrics = result["metrics"]
    print(f"train_rows: {result['split_summary']['train']['rows']}")
    print(f"validation_rows: {result['split_summary']['validation']['rows']}")
    print(f"test_rows: {result['split_summary']['test']['rows']}")
    print(f"accuracy: {metrics['model']['accuracy']:.4f}")
    print(f"macro_f1: {metrics['model']['macro_f1']:.4f}")
    print(f"label_3_4_precision: {metrics['governance']['label_3_4_predicted_label']['precision']:.4f}")
    print(f"label_3_4_recall: {metrics['governance']['label_3_4_predicted_label']['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
