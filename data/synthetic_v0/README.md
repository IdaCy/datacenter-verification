# synthetic datacenter verification dataset v0

this directory is generated synthetic study data; it contains fictional raw-like datacenter telemetry, windowed feature rows, workbook-derived rule exports, schemas, examples, and validation artifacts

dataset ID: `synthetic_v0_seed_20260510`  
scale: `v0`  
seed: `20260510`  
generator: `synthetic-generator-v0.1.0`  

regenerate:

```bash
python src/datacenter_verification_synthetic/generate_synthetic_dataset.py \
  --output data/synthetic_v0 \
  --scale v0 \
  --seed 20260510
```

validate:

```bash
python src/datacenter_verification_synthetic/validate_synthetic_dataset.py \
  --dataset data/synthetic_v0
```

the model training unit is one row in `features/window_features_all.csv`, not an individual raw metric sample or event record


# the dataset has two layers:

1. raw-like datacenter records
2. model-ready datapoints

the raw records are meant to look like telemetry coming from a datacenter. the model-ready datapoints are what a classifier
would actually train on.

one datapoint is:

site + scope + time window -> summarized evidence -> label

example:

site_b
linked_job_group_x
2026-05-18 12:00Z to 13:00Z

that row says: during this one-hour window, how much capacity existed, how many GPUs were allocated, how busy they were,
whether the network looked synchronized, whether power agreed, whether checkpoints/logs existed, whether telemetry was
missing, and what evidence label applies.

## raw data types

there are three raw record types.

1. metric samples

these are repeated measurements over time.

examples:

gpu utilization
tensor activity
gpu power fraction
nvlink utilization
scale-out fabric utilization
rack power
thermal delta
attestation-valid fraction
monitoring coverage

file:

data/synthetic_v0/raw_normalized/metric_samples.jsonl

these are time-series data. they have event_time, value_num, unit, entity_type, observable_id, and coverage/trust metadata.

2. event records

these are things that happen at a point or over an interval.

examples:

scheduler allocation interval
cloud reservation
runtime metadata event
checkpoint write event
signed ML declaration
collector gap
maintenance event
active probe result

file:

data/synthetic_v0/raw_normalized/event_records.jsonl

these often have event_time and event_end_time, because many important things are intervals, not points.

3. snapshot records

these are facts valid over a period.

examples:

hardware inventory
topology
attestation state
fabric mapping

file:

data/synthetic_v0/raw_normalized/snapshot_records.jsonl

these have valid_from and valid_to.

## model-ready data

the actual ML rows are here:

data/synthetic_v0/features/window_features_all.csv

there are:

4,380 feature rows
137 columns
5 labels
18 scenario classes

each row is one windowed datapoint.

the labels are:

0 = no training likely
1 = training possible
2 = elevated training probability
3 = training likely happening
4 = highest warning or definite

## the main data families

the fields are grouped by observable family O1 to O17.

capacity data

examples:

O1 hardware inventory
O17 external capacity reconciliation

this answers:

could a policy-scale run happen here?

it matters a lot for ruling things out, but it does not prove training.

if a site has too little capacity, label should be 0.
if a site has enough capacity, that only gets you to 1: training possible.

capacity alone must not produce labels 2, 3, or 4.

allocation and provisioning data

examples:

O2 scheduler allocation
O3 cloud billing / reservation

this answers:

were many GPUs reserved or assigned to one job/account?

this matters a lot. a huge training run normally needs many GPUs allocated together for long enough.

important fields:

o2_max_concurrent_normalized_gpus
o2_allocation_duration_hours
o2_gpu_hours_policy_ratio
o2_concurrency_fraction_domain
o2_topology_contiguity_score

but allocation alone is still not proof. reserved-but-unused capacity exists.

gpu activity data

examples:

O4 GPU utilization
O5 profiler / kernel counters

this answers:

were the GPUs actually doing heavy compute?

important fields:

o4_gpu_util_p95
o4_gpu_util_duty_gt_70
o4_sm_tensor_active_p95
o4_hbm_used_fraction_p50
o4_gpu_power_fraction_p95

high gpu activity is strong evidence of heavy compute, but not necessarily training. batch inference, burn-in, benchmarks, and
HPC can also be high.

fabric / network data

examples:

O6 nvlink / nvswitch
O7 scale-out fabric

this answers:

were many nodes acting like one synchronized distributed job?

this is one of the strongest non-semantic signals.

important fields:

o7_synchronized_fabric_footprint
o7_collective_periodicity_score
o7_scaleout_port_util_p95
o7_burst_duty_cycle

training often creates repeated collective communication patterns: all-reduce, all-gather, reduce-scatter, etc. inference
usually does not create the same large synchronized periodic footprint.

but HPC and NCCL benchmarks can mimic this, so fabric is strong but not definitive.

power and cooling data

examples:

O8 rack/facility power
O9 cooling/thermal

this answers:

does the physical site load agree with the claimed compute activity?

important fields:

o8_rack_power_fraction_p95
o8_baseline_subtracted_energy_kwh
o8_power_continuity_days
o9_thermal_delta_t_score

power is hard to fake and useful for corroboration. but high power alone is not semantic evidence. it can be training, HPC,
burn-in, storage work, cooling effects, or other load.

runtime / semantic data

examples:

O10 runtime metadata
O11 storage/checkpoints
O12 ML logs/declarations

this answers:

what kind of workload was it?

important fields:

o10_world_size
o10_runtime_framework_class
o10_rank_stability_score
o11_checkpoint_periodicity_score
o11_checkpoint_write_tb_per_event
o12_signed_ml_logs_present
o12_declared_parameter_count_b
o12_training_tokens_b
o12_step_count
o12_optimizer_state_present

this is the best way to separate training from false positives.

for example:

large allocation + high GPU + fabric sync = maybe training or HPC
large allocation + high GPU + fabric sync + pytorch distributed + checkpoints = likely training
signed ML logs crossing threshold = highest warning / definite

integrity and missingness data

examples:

O13 attestation
O14 monitoring integrity
O15 physical/security changes
O16 challenge probes

this answers:

can we trust the telemetry?

important fields:

o13_attestation_valid_fraction
o13_confidential_compute_mode_fraction
o14_min_critical_coverage
o14_gap_fraction_critical
o14_clock_drift_max_ms
o15_unapproved_physical_change_near_window
o16_probe_throughput_ratio_min

this is crucial. missing data is not zero.

if gpu telemetry is missing because the collector is down, you cannot say “no training”. you say “we do not know” or
“integrity warning”.

## do fields just have ranges?

no. ranges matter, but they are not enough.

bad synthetic data would do this:

gpu utilization = random number
rack power = random number
fabric traffic = random number
label = random label

that would be useless.

good synthetic data has dependencies.

examples:

high gpu utilization -> usually higher rack power
high rack power -> cooling load rises
large distributed training -> synchronized fabric bursts
training runtime -> stable ranks / world size
training over time -> checkpoint writes
signed ML logs -> parameters, tokens, steps, optimizer state
counter suppression -> missing gpu telemetry + high power + integrity warning
reserved-but-unused capacity -> high allocation but low gpu activity

so the model should learn patterns, not isolated thresholds.

## examples of patterns

no training likely

capacity below threshold
or
capacity exists but no allocation, no gpu activity, no fabric sync, low power, strong monitoring coverage

label: 0

training possible

capacity exists
small workload or normal inference
not enough evidence for large training

label: 1

elevated probability

large batch inference
HPC job
NCCL benchmark
hardware burn-in
counter-suppressed window

label: 2

these are important because they teach the model caution.

training likely

large allocation
high gpu utilization
fabric synchronization
checkpoint cadence
training runtime metadata
but no signed ML logs

label: 3

highest warning / definite

signed ML logs
large model / tokens / steps
policy compute threshold crossed
plus scheduler, gpu, fabric, power, storage coherence

label: 4

## how much each matters

roughly:

capacity: essential gate, weak positive evidence
scheduler/allocation: very important activity anchor
gpu telemetry: strong activity evidence
fabric/network: strongest non-semantic distributed-run signal
power/cooling: strong physical corroboration
runtime/storage/ML logs: strongest semantic evidence
integrity/missingness: determines whether absence of evidence means anything
external data: discovery/capacity sanity check, weak activity evidence

the best training signal is not one field. it is coherence:

capacity
+ allocation
+ duration
+ gpu activity
+ fabric synchronization
+ power
+ storage/runtime/ML evidence
+ good integrity coverage

## what the current data shows

the current dataset has the intended broad structure:

raw records: 42,961
feature rows: 4,380
labels: 0-4 all represented
scenarios: 18 represented

label severity rises with the right things:

higher labels have more GPUs allocated
higher labels have larger fabric footprint
higher labels have stronger checkpoint cadence
label 4 usually has signed ML logs

some direct checks:

gpu utilization vs rack power correlation: 0.720
fabric periodicity vs label correlation: 0.774
checkpoint periodicity vs label correlation: 0.765

that means the current synthetic data is not just ranges. it has the intended dependency structure.

## what to keep in mind

this is a v0 synthetic dataset. it is good for testing the pipeline and training a first prototype.

it is not yet enough for a serious final study model. for that, we need more sites, more episodes, more hardware/topology variation, more failure modes, and eventually real or controlled-drill telemetry.

