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

## labels

the study uses preliminarily five evidence levels:

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
python3 catalog/validate_v*_catalog.py
```

hard false positives need to stay first-class, such as HPC, batch inference, benchmarks, burn-in, storage rebuilds, and reserved but unused capacity

this dependence structure is the point of the study. simple thresholds are not enough

## why we're doing this

public information about datacenters is too incomplete for strong verification. if governance depends on knowing whether very large training runs happened, then the useful evidence is likely inside monitored clusters: scheduler logs, GPU telemetry, fabric counters, power data, runtime metadata, storage logs, declarations, and monitoring-integrity records

this repo is an attempt to make that evidence model concrete enough to inspect, test, criticize, and eventually compare against real datacenter telemetry

! this repo is not a finished tool

it does not provide enforcement-grade claims. it is a study scaffold for specifying what observable datacenter evidence could support detection, negative certification, attribution, and discrepancy review

expect the catalog, thresholds, label rules, and examples to change as the study goes on
