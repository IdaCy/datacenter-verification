# datacenter verification

work in progress.

this repo is for a study on whether rich datacenter telemetry can help detect, rule out, and explain large AI training runs on monitored GPU clusters

[try-out interface](https://idacy.github.io/datacenter-verification/index.html)

basic idea: access to the right internal datacenter data - then a large training run should leave a pattern including several layers;

- enough accelerator capacity to make the run possible
- a large allocation or reservation lasting long enough to matter
- sustained GPU activity
- synchronized fabric or network traffic
- rack or facility power that agrees with the compute load
- storage, checkpoint, runtime, or ML-log evidence when available
- monitoring coverage strong enough that missing evidence actually means something

we're doing data and modeling pipeline right now

## interactive demo

An unlisted static demo is hosted here:

https://idacy.github.io/datacenter-verification/

It replays the synthetic v0 model outputs and includes a browser-side rule sandbox for changing evidence inputs. Selecting a datapoint shows the trained tabular model's exported prediction for that synthetic window; moving an evidence control switches to the local rule sandbox so the page can respond instantly without server-side model inference. All data shown there is synthetic and fictional.

Regenerate the hosted demo payload from this repo with:

```bash
python src/datacenter_verification_web/export_static_demo_data.py \
  --output ../IdaCy.github.io/datacenter-verification/data/demo-data.json
```

This writes both `demo-data.json` and a local-file-friendly `demo-data.js` payload for the static page.

## current status

this is early. the repo currently contains public scaffolding for:

- synthetic datacenter data generation work under `src/datacenter_verification_synthetic/`
- dataset validators under `src/datacenter_verification_validators/`
- public generated or study data under `data/`

the synthetic generator is still being built. the validator package is present and can already check whether a generated dataset has the structure and patterns the study expects.

private planning notes, paper drafts, and working documents live under `xx_private/`. that directory is intentionally gitignored and is not part of the public repo.

## what one datapoint means (see more: data/)

the model should not train on one raw telemetry sample.

a raw sample might be one GPU utilization measurement, one rack power reading, one scheduler event, or one fabric counter. those are inputs.

one ML datapoint is a windowed feature row:

```text
site + scope + time window -> features -> label
```

for example:

```text
site_a, topology_domain_03, 2026-05-10 12:00:00Z to 13:00:00Z
```

that row should summarize what was true in that window:

- how much capacity existed
- how many normalized GPUs were allocated
- how long the allocation lasted
- how high utilization and power were
- whether fabric traffic looked synchronized
- whether checkpoint or runtime evidence appeared
- which observability layers were missing or delayed
- how much trust to place in the telemetry
- the label, from `0` to `4`

## labels

the study uses five evidence levels:

```text
0 = no training likely
1 = training possible
2 = elevated training probability
3 = training likely happening
4 = highest warning or definite
```

no big truth, but they're a structured way to train and test whether evidence from many datacenter systems points toward a large training run

then:

- capacity alone can only make training possible  
- missing data is not the same as zero activity  
- integrity problems are not proof of training by themselves  
- high power alone is not training proof  
- labels `3` and `4` should require coherent evidence across independent layers  

## planned public structure

```text
.
├── data/
│   └── synthetic_v0/
│       ├── raw_normalized/
│       ├── features/
│       ├── examples/
│       ├── schemas/
│       └── validation/
├── src/
│   ├── datacenter_verification_synthetic/
│   └── datacenter_verification_validators/
└── README.md
```

planned data files:

```text
data/synthetic_v0/raw_normalized/metric_samples.jsonl
data/synthetic_v0/raw_normalized/event_records.jsonl
data/synthetic_v0/raw_normalized/snapshot_records.jsonl
data/synthetic_v0/features/window_features_all.csv
data/synthetic_v0/examples/one_datapoint_label0.json
data/synthetic_v0/examples/one_datapoint_label1.json
data/synthetic_v0/examples/one_datapoint_label2.json
data/synthetic_v0/examples/one_datapoint_label3.json
data/synthetic_v0/examples/one_datapoint_label4.json
```

## validators

the validator checks whether generated data has the patterns we actually want

run:

```bash
python -m src.datacenter_verification_validators --dataset data/synthetic_v0
```

it checks things like:

- all required files exist
- raw `JSONL` records parse
- feature rows have required columns
- labels are in the `0-4` range
- timestamps use `UTC`
- every observable family `O1-O17` has coverage and missingness columns
- missing telemetry is explicit
- capacity-only rows do not become positive training labels
- physical-only rows do not exceed supportive evidence
- integrity-only rows do not become training proof
- likely training rows contain allocation, GPU, fabric, power, storage, or semantic coherence
- hard false positives are present, such as HPC, batch inference, benchmarks, burn-in, and storage rebuilds

the validator writes:

```text
data/synthetic_v0/validation/validation_report.md
```

## intended data patterns

the generated data should not be random columns sampled independently.

it should be generated from latent episodes, such as:

- idle
- normal inference
- large batch inference
- synthetic data generation
- small fine-tune
- large fine-tune
- pretraining
- HPC simulation
- NCCL benchmark
- hardware burn-in
- storage rebuild
- reserved but unused capacity
- maintenance window
- fragmented training
- counter-suppressed candidate window

those episodes should create dependent telemetry. for example:

- high GPU utilization should usually raise rack power
- sustained training should usually create synchronized fabric patterns
- checkpoint-like writes should align with long training windows
- confidential-compute or collector outages should create missingness, not zeros
- large reserved capacity with no GPU load should not look like active training
- high power with missing GPU telemetry should become an integrity warning

this dependence structure is the point of the study. if the synthetic data does not preserve these relationships, a model trained on it will learn the wrong thing.

## modeling direction

the planned detector is a small auditable tabular model trained on windowed feature rows.

the current default direction is a calibrated gradient-boosted tree model, not a language model. a language model may be useful later for summarizing audit evidence or messy free text, but it should not be the core detector.

the model output should stay multi-part:

- probability for each label `0-4`
- probability of large training
- negative-certification confidence
- capacity possible
- integrity warning
- critical missing layers
- top evidence

those outputs should not be collapsed into one opaque score.

## why this exists

public information about datacenters is too incomplete for strong verification. if governance depends on knowing whether very large training runs happened, then the useful evidence is likely inside monitored clusters: scheduler logs, GPU telemetry, fabric counters, power data, runtime metadata, storage logs, declarations, and monitoring-integrity records.

this repo is an attempt to make that evidence model concrete enough to simulate, test, criticize, and eventually compare against real datacenter telemetry.

## not ready yet

this repo is not a finished compliance tool.

it does not yet provide a trained model. it does not yet provide enforcement-grade claims. it is a study scaffold for building synthetic data, validating that the data has the right evidence structure, and later training/testing a detector under clear assumptions.

expect the schemas, labels, generator, and validators to change as the study gets sharper.
