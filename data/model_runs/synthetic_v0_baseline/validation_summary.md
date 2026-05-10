# Validation Summary

Feature table: `data/synthetic_v0/features/window_features_all.csv`

## Dataset Validation Before Training

- `synthetic_dataset_validator`: return code 0; PASS
- `public_dataset_validator`: return code 0; PASS

## Required Artifacts

- `README.md`: present
- `manifest.json`: present
- `model.joblib`: present
- `preprocessing.joblib`: present
- `feature_columns.json`: present
- `excluded_columns.json`: present
- `split_manifest.json`: present
- `metrics.json`: present
- `calibration_metrics.json`: present
- `confusion_matrix.csv`: present
- `classification_report.json`: present
- `predictions_test.csv`: present
- `predictions_all.csv`: present
- `feature_importance.csv`: present
- `evidence_audit_sample.csv`: present
- `validation_summary.md`: present

## Test Metrics

- Accuracy: 0.9906
- Macro F1: 0.9920
- Label 3/4 precision: 1.0000
- Label 3/4 recall: 1.0000
- `p_large_training >= 0.5` precision: 1.0000
- `p_large_training >= 0.5` recall: 1.0000

## Governance Checks

- Capacity gate applied to post-processed probabilities.
- Negative certification confidence is `p_label_0 * min_critical_coverage`.
- Integrity warnings are reported separately from positive training evidence.
- Raw model probabilities are retained as `raw_p_label_0` through `raw_p_label_4`.
