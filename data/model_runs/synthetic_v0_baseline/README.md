# synthetic v0 baseline model run

This directory contains the first public runnable baseline for the synthetic v0 datacenter training-run verification dataset.

## Dataset

- Feature table: `data/synthetic_v0/features/window_features_all.csv`
- Rows: 4380
- Episode split: grouped by `episode_id`, scenario-stratified, seed `20260510`
- Split rows: train 2671, validation 862, test 847
- Split episodes: train 126, validation 42, test 42

## Model

- Supervised model: calibrated scikit-learn histogram gradient boosting classifier
- Calibration: validation split only, held-out test evaluated once
- Rule baseline: deterministic evidence rules in `src/datacenter_verification_modeling/rule_baseline.py`
- Leakage exclusions: identifiers, labels, site id, episode id, raw manifest hash, and synthetic-only audit columns

## Headline Test Metrics

- Accuracy: 0.9906
- Macro F1: 0.9920
- Weighted F1: 0.9905
- Log loss: 0.1876
- Label 3/4 precision by predicted label: 1.0000
- Label 3/4 recall by predicted label: 1.0000
- `p_large_training >= 0.5` precision: 1.0000
- `p_large_training >= 0.5` recall: 1.0000
- Rule baseline macro F1: 0.6045

## Error Scenarios

- Largest false-positive scenarios at `p_large_training >= 0.5`: none
- Largest false-negative scenarios at `p_large_training >= 0.5`: none

## Calibration

- Brier score for `p_large_training`: 0.0015
- Expected calibration error for `p_large_training`: 0.0237

## Reproduce

```bash
python src/datacenter_verification_modeling/train_model.py \
  --features data/synthetic_v0/features/window_features_all.csv \
  --output data/model_runs/synthetic_v0_baseline \
  --seed 20260510
```

```bash
python src/datacenter_verification_modeling/evaluate_model.py \
  --model-run data/model_runs/synthetic_v0_baseline \
  --features data/synthetic_v0/features/window_features_all.csv
```

```bash
python src/datacenter_verification_modeling/predict.py \
  --model-run data/model_runs/synthetic_v0_baseline \
  --features data/synthetic_v0/features/window_features_all.csv \
  --output data/model_runs/synthetic_v0_baseline/predictions_all.csv
```

## Limitations

- This model is trained on synthetic data only.
- Performance numbers are not real-world deployment claims.
- Adjacent windows are correlated, so group splitting by `episode_id` is mandatory.
- Synthetic labels are generated from rules and latent scenarios, so the model may learn generator assumptions.
- Real datacenter deployment would require calibration on real telemetry and controlled drills.
- The model should assist audit triage; it should not be treated as sole proof of a violation.
