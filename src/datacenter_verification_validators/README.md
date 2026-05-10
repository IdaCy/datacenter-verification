# datacenter verification dataset validators

This package validates synthetic/study datasets generated for the datacenter
training-run verification project.

It checks that generated data has the structure and patterns needed for the
study:

- required files exist under `data/synthetic_*`;
- raw normalized JSONL files parse and contain required fields;
- model-ready feature rows contain required columns;
- labels are in the workbook's `0-4` range;
- timestamps are UTC ISO-8601 strings ending in `Z`;
- coverage and missingness fields exist for O1-O17;
- missing telemetry is represented as missingness, not silently as zero;
- capacity-only evidence is capped at label 1;
- physical-only and integrity-only evidence do not become training proof;
- label 3/4 rows contain coherent multi-layer training evidence;
- required scenario classes appear;
- hard false positives are present;
- cross-feature dependencies are plausible, such as GPU utilization moving with
  power and large training labels moving with fabric synchronization.

Run:

```bash
python -m src.datacenter_verification_validators --dataset data/synthetic_v0
```

or:

```bash
python src/datacenter_verification_validators/validate_dataset.py \
  --dataset data/synthetic_v0
```

The validator writes:

```text
data/synthetic_v0/validation/validation_report.md
```

Use `--strict` to return a non-zero exit code for warnings as well as errors.
