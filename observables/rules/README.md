# Observable Rule Layer

These files are algorithm-facing rule specifications derived from
`observables/observables.yaml`.

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

Sparse rules are the operational entry point when only one, two, or three raw
features exist for a scope/window. They emit narrow facts, screens, support,
suppressors, explanations, missingness warnings, or contradictions.
`aggregation_rules.yaml` is the default staged operational layer above those
sparse outputs.

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

## Staged Aggregation Flow

`aggregation_rules.yaml` implements the algorithm-facing staged workflow:

1. `A_capacity_gate`: split the selected audit window into capacity-validity
   segments and evaluate capacity first.
2. `B_training_candidate_detection`: run only for capacity-possible,
   capacity-limited, or capacity-unknown segments. Promote warning-height
   candidates only from aligned accelerator activity plus independent
   identity-shape evidence.
3. `C_discrepancy_and_explanation_review`: run targeted discrepancy,
   suppressor, and missingness checks only when a live candidate, capacity
   conflict, or decision-blocking gap requires adjudication.
4. `final_claim_routing`: emit one final route per monitored scope/window
   segment, including warning height and caveats.

C has two targeted modes. The negative-screen integrity mode runs when B emits
no candidate in a live capacity segment and checks whether primary activity,
identity-shape, scope-mapping, and clock-alignment coverage are sufficient to
trust the absence of evidence. Candidate adjudication mode runs when B emits a
candidate, conflict, suppressor, explanation, or decision-blocking missingness.
Neither mode is a full global discrepancy sweep after a clean capacity rule-out
or a quiet, well-covered segment.

Candidate-window derived signals are stage-specific. B rules consume
`aggregation_candidate_seed_window`; C rules consume
`aggregation_candidate_review_window`, which contains B/support candidate state
but no C outcomes; final routing consumes `aggregation_candidate_final_window`,
which contains C suppressor, explanation, discrepancy, and missingness outcomes.

The capacity gate can short-circuit a segment only when a conservative
high-coverage upper bound emits `capacity_ruled_out_for_scope`. In that case B
candidate detection and general C review do not run for the ruled-out segment.
Capacity-unknown, capacity-limited, and sparse capacity-possible outputs remain
live.

Aggregation rules normally consume sparse rule categories, labels, and derived
signals rather than raw features. They therefore are not listed under every raw
feature in `feature_rule_index.yaml`; the index maps raw features to the rules
that interpret the raw values.

Before production use, replace calibration defaults with local distributions
from known training, inference, HPC/MPI, NCCL benchmark, burn-in, storage
maintenance, ETL, backup/restore, and serving workloads. Also calibrate
telemetry coverage, sampling intervals, clock drift, source delivery delays,
operation-unit normalization, and hardware-specific peak/cap/power behavior.

## Files

- `feature_rule_index.yaml`: coverage map from every feature ID in
  `observables/observables.yaml` to rule IDs.
- `aggregation_rules.yaml`: staged A/B/C/final aggregation over sparse rule
  outputs, including capacity short-circuiting, B warning-height promotion,
  suppressor/explanation demotion, targeted C routing, and final claim routing.
- `capability_rules.yaml`: sparse capacity screens for accelerator
  count/shape, peak rate, memory, topology, availability, power service,
  installation, quota, reservation, running, health, and maintenance features.
- `training_core_rules.yaml`: sparse-telemetry rule families that make
  the strongest honest candidate, weak-evidence, capacity, or suppression
  statement possible from one to three raw observable features.
- `training_support_rules.yaml`: sparse support, counterevidence, and
  explanation screens for physical load, memory residency, data lifecycle,
  topology, serving-like traffic, storage operations, benchmark/HPC
  alternatives, and lifecycle context.
- `discrepancy_rules.yaml`: sparse pair/triplet conflicts and
  missingness checks for activity attribution, capacity claims, physical
  timelines, power/activity, fabric mapping, storage/activity, telemetry gaps,
  health/throttle states, and route/topology changes.
- `derived_signals.yaml`: reusable computations referenced by multiple rule
  files.
- `source_ledger.yaml`: public source ledger for `Sxx` source references used by
  the rule files.
