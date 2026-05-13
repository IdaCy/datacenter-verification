# datacenter verification

work in progress.

this repo is for a study on what datacenter telemetry could help detect, rule out, and explain large AI training runs on monitored GPU clusters

basic idea: access to the right internal datacenter data - then a large training run should leave a pattern including several layers;

- enough accelerator capacity to make the run possible
- a large allocation or reservation lasting long enough to matter
- sustained GPU activity
- synchronized fabric or network traffic
- rack or facility power that agrees with the compute load
- storage, checkpoint, runtime, or ML-log evidence when available
- monitoring coverage strong enough that missing evidence actually means something

the current direction is the algorithmic evidence catalog

## current status

this is early. the repo currently contains public scaffolding for:

- the v2 evidence catalog under `catalog/`
- source-backed observable and feature definitions
- qualitative feature effects for large-training evidence
- dependency rules between observables
- plausibility checks across evidence layers
- discrepancy and evasion rules

private planning notes, paper drafts, and working documents live under `xx_private/`. that directory is intentionally gitignored and is not part of the public repo

## what one audit window means

the framework should not reason from one raw telemetry sample

a raw sample might be one GPU utilization measurement, one rack power reading, one scheduler event, one fabric counter, or one storage log entry. those are inputs

one audit window is a structured evidence row:

```text
site + scope + time window -> features -> evidence level
```

for example:

```text
site_a, topology_domain_03, 2026-05-10 12:00:00Z to 13:00:00Z
```

that row should summarize what was true in that window:

- how much capacity existed
- how many normalized accelerators were allocated
- how long the allocation lasted
- how high utilization, fabric traffic, and power were
- whether checkpoint, runtime, declaration, or ML-log evidence appeared
- which observability layers were missing or delayed
- how much trust to place in the telemetry
- the evidence level, from `0` to `4`

## labels

the study uses five evidence levels:

```text
0 = no training likely
1 = training possible
2 = elevated training probability
3 = training likely happening
4 = highest warning or definite
```

no big truth, but they're a structured way to test whether evidence from many datacenter systems points toward a large training run

then:

- capacity alone can only make training possible
- missing data is not the same as zero activity
- integrity problems are not proof of training by themselves
- high power alone is not training proof
- labels `3` and `4` should require coherent evidence across independent layers or authenticated semantic evidence

## public structure

```text
.
├── catalog/
│   ├── catalog_index.v2.yaml
│   ├── observable_feature_catalog.v2.yaml
│   ├── training_probability_effects.v2.yaml
│   ├── feature_dependencies.v2.yaml
│   ├── conditional_plausibility_windows.v2.yaml
│   ├── discrepancy_evasion_rules.v2.yaml
│   └── validate_v2_catalog.py
├── README.md
└── READMEsuggested.md
```

validate:

```bash
python3 catalog/validate_v2_catalog.py
```

## catalog

the catalog is not a trained detector

it records what evidence exists, what it can support, where it is ambiguous, and what combinations should raise a discrepancy

the main files are:

- `observable_feature_catalog.v2.yaml` for observables and features
- `training_probability_effects.v2.yaml` for evidence direction and caps
- `feature_dependencies.v2.yaml` for feature relationships
- `conditional_plausibility_windows.v2.yaml` for cross-layer plausibility checks
- `discrepancy_evasion_rules.v2.yaml` for anomaly and evasion hypotheses

## intended evidence patterns

the evidence should not be random columns sampled independently

a large training run should create dependent telemetry. for example:

- high GPU utilization should usually raise rack power
- sustained training should usually create synchronized fabric patterns
- checkpoint-like writes should align with long training windows
- confidential-compute or collector outages should create missingness, not zeros
- large reserved capacity with no GPU load should not look like active training
- high power with missing GPU telemetry should become an integrity warning

hard false positives need to stay first-class, such as HPC, batch inference, benchmarks, burn-in, storage rebuilds, and reserved but unused capacity

this dependence structure is the point of the study. simple thresholds are not enough

## why this exists

public information about datacenters is too incomplete for strong verification. if governance depends on knowing whether very large training runs happened, then the useful evidence is likely inside monitored clusters: scheduler logs, GPU telemetry, fabric counters, power data, runtime metadata, storage logs, declarations, and monitoring-integrity records

this repo is an attempt to make that evidence model concrete enough to inspect, test, criticize, and eventually compare against real datacenter telemetry

## not ready yet

this repo is not a finished compliance tool

it does not provide enforcement-grade claims. it is a study scaffold for specifying what observable datacenter evidence could support detection, negative certification, attribution, and discrepancy review

expect the catalog, thresholds, label rules, and examples to change as the study gets sharper
