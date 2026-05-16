# Observable Rule Layer

These files are algorithm-facing rule specifications derived from
`observables/observables.yaml.

`observables/observables.yaml` remains the raw feature inventory. The files in
this directory define how those source-emitted values can be combined into
capacity bounds, training evidence, support/counterevidence, and discrepancy
checks.

## Evidence Split

- `capacity`: capability rules for whether a threshold-scale run was physically
  or computationally possible in a monitored scope and window.
- `training_core`: narrow training-evidence rules that can drive an algorithmic
  candidate label. These rules focus on aligned allocation/running context,
  sustained accelerator activity, collective-like fabric cadence, checkpoint
  signatures, achieved-operation counters, and serving counterevidence.
- `training_support`: weaker or contextual training signals. These can
  strengthen, weaken, explain, or route review of a core finding but should not
  be treated as primary training evidence by themselves.
- `discrepancy`: cross-layer checks for missing telemetry, manipulated or
  inconsistent records, incorrect attribution, or benign operational
  explanations such as maintenance, storage rebuilds, throttling, or topology
  changes.

Training evidence is intentionally modeled as multiple pathways rather than one
universal detector. Large compute and high fabric bandwidth are not sufficient
by themselves; strong evidence comes from time-aligned capacity, accelerator
activity, workload-shape signals, and false-positive checks.

## Concrete Defaults And Portability

The rule layer now contains executable trigger defaults. Treat them as initial
review defaults for unseen datacenters, not final operational thresholds. Exact
positive training-identity thresholds are not portable across accelerator
families, topology generations, framework parallelism, telemetry aggregation,
storage systems, and local workload mixes.

Each concrete trigger default records an `evidence_status`:

- `source_backed`: directly supported by a cited source or policy value.
- `source_informed`: source-backed mechanism or measurement surface, but the
  numeric threshold is still an implementation choice.
- `mechanism_inferred`: derived from distributed-training, telemetry, or
  physical mechanism rather than an exact source threshold.
- `calibration_default`: executable starting point that must be replaced with
  local data before strong operational use.
- `project_assumption`: verification-design choice used to keep the algorithm
  runnable when no portable source threshold exists.

Normalized scores are preferred over raw thresholds: fractions of peak/capacity,
per-participant fabric volumes, overlap windows, coverage fractions, local
baseline percentiles, and topology-aware participant sets. Raw bytes, advertised
peak rates, allocation size, and power alone do not classify training.

`capacity_upper_bound_flop` is an upper-bound calculation. It can rule out a
threshold-scale run for a monitored scope/window when coverage is sufficient,
and it can flag claimed/achieved compute above capacity. It does not prove
training occurred. Similarly, `achieved_operation_integral >= 1e25` creates a
large-compute candidate; a training label still requires independent
training-identity support such as collective cadence, checkpoint signatures, or
non-serving model-development shape.

Before production use, replace calibration defaults with local distributions
from known training, inference, HPC/MPI, NCCL benchmark, burn-in, storage
maintenance, ETL, backup/restore, and serving workloads. Also calibrate
telemetry coverage, sampling intervals, clock drift, source delivery delays,
operation-unit normalization, and hardware-specific peak/cap/power behavior.

## Files

- `feature_rule_index.yaml`: coverage map from every feature ID in
  `observables/observables.yaml` to rule IDs.
- `capability_rules.yaml`: capacity and capability rules for upper bounds,
  topology constraints, cloud quota/reservation limits, and physical-service
  timelines.
- `training_core_rules_comprehensive.yaml`: comprehensive core
  training-evidence rule families that combine multiple aligned channels and
  suppressors.
- `training_core_rules_minimal.yaml`: sparse-telemetry rule families that make
  the strongest honest candidate, weak-evidence, capacity, or suppression
  statement possible from one to three raw observable features.
- `training_support_rules.yaml`: supportive training signals and explicit
  false-positive filters.
- `discrepancy_rules.yaml`: integrity, evasion, attribution, and benign
  explanation checks.
- `derived_signals.yaml`: reusable computations referenced by multiple rule
  files.
