# datacenter verification modeling

this package trains and evaluates the first public baseline model for the synthetic v0 datacenter training-run verification dataset.

the training unit is one row from:

```text
data/synthetic_v0/features/window_features_all.csv
```

the model does not read raw telemetry samples, events, or snapshots directly

## outputs

the default model run is written to:

```text
data/model_runs/synthetic_v0_baseline/
```

it contains the fitted calibrated model, preprocessing pipeline, split manifest, feature metadata, predictions, metrics, calibration diagnostics, feature importance, and an evidence audit sample

## leakage controls

rows are split by `episode_id`, not randomly. This is required because the same latent episode appears in multiple adjacent windows. The default split is scenario-stratified at the episode level with seed `20260510`:

```text
train: 60%
validation/calibration: 20%
test: 20%
```

the supervised model excludes identifiers, direct labels, site id, episode id, raw manifest hashes, and synthetic-only audit columns such as
`latent_workload_class` and `synthetic_evidence_profile`

## model

the supervised baseline is:

```text
SimpleImputer + OneHotEncoder preprocessing
HistGradientBoostingClassifier
CalibratedClassifierCV on the validation split
```

the package also includes `rule_baseline.py`, a deterministic evidence-rule baseline that encodes broad study logic:

- capacity below threshold with strong coverage is negative evidence
- capacity alone does not prove training
- missing data is not zero activity
- integrity anomalies are warnings, not positive proof
- labels 3 and 4 require coherent multi-layer evidence

## commands

train, evaluate, and generate all default artifacts:

```bash
python src/datacenter_verification_modeling/train_model.py \
  --features data/synthetic_v0/features/window_features_all.csv \
  --output data/model_runs/synthetic_v0_baseline \
  --seed 20260510
```

Evaluate an existing run:

```bash
python src/datacenter_verification_modeling/evaluate_model.py \
  --model-run data/model_runs/synthetic_v0_baseline \
  --features data/synthetic_v0/features/window_features_all.csv
```

Generate predictions from an existing run:

```bash
python src/datacenter_verification_modeling/predict.py \
  --model-run data/model_runs/synthetic_v0_baseline \
  --features data/synthetic_v0/features/window_features_all.csv \
  --output data/model_runs/synthetic_v0_baseline/predictions_all.csv
```

Prepare split and feature metadata without training:

```bash
python src/datacenter_verification_modeling/prepare_features.py \
  --features data/synthetic_v0/features/window_features_all.csv \
  --output data/model_runs/synthetic_v0_baseline \
  --seed 20260510
```

## governance outputs

prediction files include:

- `p_label_0` through `p_label_4`
- `raw_p_label_0` through `raw_p_label_4`
- `p_large_training`
- `severity_score`
- `capacity_possible`
- `negative_certification_confidence`
- `integrity_warning`
- `critical_missing_layers`
- `top_evidence`

the `p_label_*` columns are post-processed with the capacity gate; the `raw_p_label_*` columns preserve the calibrated model probabilities before that
gate

## limitations

This is a synthetic v0 prototype ! Metrics are useful for testing the pipeline, not for deployment claims. Real datacenter use would require real telemetry, controlled drills, operational calibration, and independent review

