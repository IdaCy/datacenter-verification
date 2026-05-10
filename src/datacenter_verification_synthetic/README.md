# datacenter verification synthetic data

This package generates a compact synthetic dataset for training-run verification
experiments. It emits normalized raw records, workbook-derived rule exports,
windowed feature rows, JSON schemas, examples, and validation reports.

the primary command is:

```bash
python src/datacenter_verification_synthetic/generate_synthetic_dataset.py \
  --output data/synthetic_v0 \
  --scale v0 \
  --seed 20260510
```

Feature rows are one datapoint per `(site_id, scope_type, scope_id_hash,
window_start, window_end)`. They include engineered values, per-observable
coverage and missingness columns, trust fields, and a `label_0_to_4` derived
from latent workload truth plus workbook-inspired composite rules.

Validation:

```bash
python src/datacenter_verification_synthetic/validate_synthetic_dataset.py \
  --dataset data/synthetic_v0
```

Dependencies are intentionally small: Python 3.10+, `numpy`, `pandas`, and
`openpyxl`. If `pyarrow` is present, feature CSVs are also written as Parquet.

