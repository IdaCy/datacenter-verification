# synthetic_v1_baseline model run

This directory contains a public runnable baseline for the `synthetic_v1` datacenter training-run verification dataset.

## Dataset

- Feature table: `data/synthetic_v1/features/window_features_all.csv`
- Rows: 14712
- Episode split: grouped by `episode_id`, scenario-stratified, seed `20260510`
- Split rows: train 8870, validation 2937, test 2905
- Split episodes: train 458, validation 152, test 150

## Model

- Supervised model: calibrated scikit-learn histogram gradient boosting classifier
- Calibration: validation split only, held-out test evaluated once
- Rule baseline: deterministic evidence rules in `src/datacenter_verification_modeling/rule_baseline.py`
- Leakage exclusions: identifiers, labels, site id, episode id, raw manifest hash, scenario metadata, counterfactual metadata, and synthetic-only audit columns

## Headline Test Metrics

- Accuracy: 0.9725
- Macro F1: 0.9781
- Weighted F1: 0.9723
- Log loss: 0.1057
- Label 3/4 precision by predicted label: 0.9935
- Label 3/4 recall by predicted label: 0.9967
- `p_large_training >= 0.5` precision: 0.9935
- `p_large_training >= 0.5` recall: 0.9967
- Rule baseline macro F1: 0.3636

## Error Scenarios

- Largest false-positive scenarios at `p_large_training >= 0.5`: underclocked_energy_capped_training: 4
- Largest false-negative scenarios at `p_large_training >= 0.5`: fragmented_training_linked: 2

## Calibration

- Brier score for `p_large_training`: 0.0018
- Expected calibration error for `p_large_training`: 0.0105

## Stress Splits

Stress outputs are in:

- `stress_metrics.json`
- `stress_summary.md`

Key stress checks from the lightweight retraining script:

- Episode-grouped random split macro F1: 0.9820; large-training recall: 0.9967
- Time holdout macro F1: 0.9863; large-training recall: 1.0000
- Underclocked holdout large-training recall: 1.0000; false negatives: 0
- Fragmented linked training holdout remains hard: 91 large-training false negatives
- Source ablation dropping runtime and ML logs changed macro F1 by -0.089 versus the random stress baseline
- Middle `p_large_training` bins are populated in stress checks: {'0.1_to_0.3': 18, '0.3_to_0.7': 2, '0.7_to_0.9': 0}

## Reproduce

```bash
python src/datacenter_verification_modeling/train_model.py \
  --features data/synthetic_v1/features/window_features_all.csv \
  --output data/model_runs/synthetic_v1_baseline \
  --seed 20260510
```

```bash
python src/datacenter_verification_modeling/evaluate_model.py \
  --model-run data/model_runs/synthetic_v1_baseline \
  --features data/synthetic_v1/features/window_features_all.csv
```

```bash
python src/datacenter_verification_modeling/predict.py \
  --model-run data/model_runs/synthetic_v1_baseline \
  --features data/synthetic_v1/features/window_features_all.csv \
  --output data/model_runs/synthetic_v1_baseline/predictions_all.csv
```

## Limitations

- This model is trained on synthetic data only.
- Performance numbers are not real-world deployment claims.
- Adjacent windows are correlated, so group splitting by `episode_id` is mandatory.
- Synthetic labels are generated from rules and latent scenarios, so the model may learn generator assumptions.
- Real datacenter deployment would require calibration on real telemetry and controlled drills.
- The model should assist audit triage; it should not be treated as sole proof of a violation.
