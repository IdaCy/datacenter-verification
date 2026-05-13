
Footprints of Activity: Datacenter Telemetry for Training Run Verification

## overview

As artificial intelligence (AI) systems become more capable, governments may want to verify when major AI models are trained, by whom, and on what scale. Because large training runs require accelerator clusters with substantial compute, power, cooling, storage, and networking, they are very distinct entities and can be the source of verification evidence.

Public information about AI datacenters is currently limited, especially for determiing how installed accelerator capacity is used. Epoch AI (https://epoch.ai/data/data-centers) provides the most comprehensive open-source effort on AI compute infrastructure. It uses satellite imagery, permit filings, and public disclosures to estimate properties of individual datacenters. This kind of public evidence is useful for discovering and sizing infrastructures, but is usually cannot show whether a particular monitored cluster was used for a particular training run. This paper therefore assumes on top of public discovery a regulated auditor access: it asks what non-public datacenter telemetry governments could usefully request to be reported, checked, or inspected under a monitoring regime.

The central question is which datacenter observables are useful for detecting, ruling out, and attributing large training runs to times, workloads, accounts, customers, and operators, and how much confidence auditors can place in those observables when telemetry may be incomplete or manipulated.

Prior work has surveyes compute-covernance mechanisms, training-run verification methods, and possible evasion strategies at a higher level. Other work develops institutional verification framewoks and hardware-level taxonomies for monitoring compute use. Separately, current systems documentation shows that rich telemetry already exists in schedulers, cloud control planes, GPU monitoring stacks, fabric monitors, power systems, storage logs, and runtime or ML logging tools, although these records are usually not public. What is still missing is a system that shows what can be detected, rules out, and attributed on known monitored accelerator clusters under realistic missingness, privacy, constraints, and evasion behavior.

This paper contributes a taxonomy of datacenter observables and what they are informing auditors of, an evidence model for detection and negative certification, and a rule-based catalog of feature effects, dependencies, plausibility checks, and discrepancy rules. Synthetic examples and a demonstration tools are used to stress-test the framework and illustrate edge-cases, and open sourced.


## The verification target and threat model

We use the definition of a "large training run" to be verified as measured by total training compute, measured in FLOPs, using the EU AI Act systemic-risk compute threshold as the clearest current legal anchor (see Appendix A). The AI Act presumes a general-purpose AI model to have high-impact capabilities when the cumulative amount of computation used for training is greater than 10^25 FLOP. The AI Act also notes that this threshold may need to be updated over time as technology changes. This is the threshold-scale training-compute event we aim to detect. As datacenter telemetry does not include a FLOP threshold, we approach it with allocations, devices, power, fabric traffic, runtime metadata, logs, and other operational traces.

This definition of a target event is not limited to one kind of pretraining run. Pretraining is the clearst case, but governance may also care about other large model-development work that updates or materially changes model weights, as well as training-adjacent large inference if those activities substantially contribute to frontier model development. These boundary cases should not automatically count as prohibited training, but should be visible in the framework to be inspected for the occurence of an event of interest.

There are three main verification tasks:
  - Detection: is there evidence consistent with a threshold-scale training or model-development run?
  - Negative certification: can we say that no such run occurred on the monitored capacity during the relevant time window?
  - Attribution: what workload, account, project, customer, or operator behavior best explains the evidence?

The framework does not assume that every anomaly is evasion. Many suspicious patterns have benign explanations: unused reservations, failed jobs, scheduler mapping errors, storage rebuilds, HPC collectives, NCCL benchmarks, thermal events, firmware updates, collector outages, billing delays, or privacy redaction. The aim is to separate positive training evidence from discrepancy evidence and to require an explanation when the layers do not agree.

The datacenters we are concerned about are not all datacenters in the broad industry sense. "Data center" is not a homogenous category: public directories such as Data Center Map list more than 11,000 facilities worldwide, and Synergy Research reported 1,360 hyperscale data centers at the end of Q4 2025. Most of these facilities are not individually relevant to detecting large training runs. The relevant unit for this paper is an accelerator cluster or provider-controlled compute pool: GPUs, TPUs, Trainium chips, or comparable accelerators with enough power, cooling, storage, and low-latency fabric to support threshold-scale model-development workloads.

Publicly discussed examples, with very different levels of site visibility and evidentiary confidence, include xAI's Colossus cluster in Memphis/Southaven, OpenAI/Oracle/Crusoe infrastructure in Abilene, Microsoft’s Fairwater AI datacenters, AWS Project Rainier, Google TPU training infrastructure, Meta’s H100 clusters and Louisiana buildout, and AI-specialized cloud providers such as CoreWeave, Lambda, Fluidstack, Nebius, and nScale. These examples are very different from one another. Some are single-campus clusters, others are multi-site or provider-controlled pools. Some disclose accelerator counts, others disclose only power, spending, architectural claims, or customer relationships. A verification regime should therefore maintain a monitored-cluster registry with confidence levels for capacity, topology, ownership, operator, customer, and telemetry access.

The claim made by negative certification is also limited. It is not "no large training run happened anywhere". It is "given the monitored boundary, the time window, the policy threshold, and the available telemetry coverage, no threshold-scale run occurred on this monitored capacity."

[may want to write about trust assumptions - cooperative operator, semi-cooperative operator with privacy concerns, untrusted...]

[may want to write about the about the adversary model - operator may try to conceal a run, make training look like something else, manipulate logs or counters, slopt the run across time or providers, exploit the boundary between training and other workloads, just stay below the limit to not be caught, but use some tiny edge (unseen connection/GPU/...) to still do the trun]

Research points:
- EU AI Act Article 51 uses >10^25 FLOP cumulative training compute as the presumption for systemic-risk GPAI models, and says
  thresholds can be amended as technology evolves: https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-51
- EU AI Act Article 52 says the compute threshold triggers notification without delay and within two weeks:
  https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-52
- Commission GPAI guidance treats >10^23 FLOP as an indicative GPAI criterion, and clarifies the one-third original-training-compute boundary for modifications/fine-tuning:
  https://digital-strategy.ec.europa.eu/en/faqs/guidelines-obligations-general-purpose-ai-providers
- AI Act Annex XI includes computational resources, training time, and energy consumption in GPAI technical documentation:
  https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-11
- Heim & Koessler argue training compute thresholds are useful but imperfect initial filters, not standalone risk decisions:
  https://arxiv.org/abs/2405.10799
- Schuett et al. discuss legal loopholes around fine-tuning, model reuse, model expansion, and inference compute:
  https://arxiv.org/abs/2502.00003
- Data Center Map global directory: https://www.datacentermap.com/datacenters/
- Synergy reported 1,360 hyperscale data centers at end Q4 2025:
  https://www.srgresearch.com/articles/hyperscale-operators-to-account-for-67-of-all-data-center-capacity-by-2031

## Evidence model

We treat verification as a multi-source evidence problem. Renting a large set of GPUs is not sufficient evidence for a large training run. It may become evidence in relation to other observables. Scale-out bandwidth is a special case: in some bandwidth-limit regimes, verified absence of enough cross-pod bandwidth can individually support ruling out efficient threshold-scale distributed training. Observed high scale-out bandwidth, however, is strongest as coordinated-workload evidence when footprint, duration, cadence, and mapping are known; it still needs false-positive checks for HPC, benchmarks, distributed inference, or storage traffic. Therefore, we combine the main evidence points for a strong detection.

The evidence has several roles.

Capacity evidence tells us if a large training run could happen on the monitored cluster. If we can rule out capacity, the monitoring system can be spared. Capacity monitoring includes accelerator inventory, usable contiguous topology, hardware generation, partitioning, and external capacity context. Capacity evidence is necessary for many claims, but is not activity evidence by itself.

Activity evidence tells us if substantial compute was used. This includes scheduler allocations, cloud provisioning records, GPU utilization, tensor activity, memory pressure, fabric traffic, and storage movement. Activity evidence is stronger when it is sustained over time and linked to the same job, account, topology domain, or provider-controlled compute pool.

Attribution evidence shows what kind of workload best explains the activity. This includes declared workload class, runtime metadata, container or image information, framework traces, checkpoint metadata, ML logs, and signed declarations. This evidence is especially useful for differentiating training from close alternatives such as HPC collectives, NCCL benchmarks, batch inference, model-parallel inference, synthetic data generation, storage rebuilds, and hardware burn-in. It is also often the most sensitive evidence, so it may only be available after an escalation trigger.

Integrity evidence tells us if the other evidence can be trusted. This includes telemetry coverage, collector gaps, counter reseats, clock drift, attestation state, confidential-compute modes, firmware changes, physical access records, and active probe results where available. A telemetry gap is not proof of training, but it can make a no-run claim much weaker.

Single evidence approaches have ceilings, so we combine any information provided to get the highest confidence possible. This is the case both for positive and for negative certification. The auditor should receive either strong evidence that the monitored capacity was below the relevant threshold, or strong coverage across the important activity layers during the relevant time window.

The outputs of the framework should therefore have several parts. A audit resuld includes:
- a training evidence level
- if capacity was sufficient for a large training run
- the main evidence layers supporting the result
- the confidence of negative certification
- missing or weak telemetry layers
- discrepancy or evasion warnings
- plausible benign explanations that still need to be checked [check if we include this in the end!!]



## Observables 

An observable is a data source from a datacenter; each observable has features that are quantifiable information on the datacenter operations. All observables can be found in Table 1. Not all of these observable features exist in the same form at all relevant datacenters. A self-managed cluster, a hyperscale cloud region, a single-tenant AI cloud development, and a multi-site provider-controlled training pool keep different records. The framework includes the possibilities for what they can expose, and uses existing ones for the monitoring, while tracking missing, delayed, redacted, or less trusted information.

Table 1. Datacenter observables used in the framework.

| Observable name | Importance | Example features |
|---|---|---|
| Hardware inventory and accelerator capacity | Capacity gate; negative-certification prerequisite | Accelerator count, GPU/accelerator SKU, normalized training compute capacity, largest low-latency topology domain, partitioning/MIG/vGPU fraction, inventory delta rate |
| Scheduler, job, reservation, and allocation metadata | Primary detection signal; attribution signal | Allocated accelerator count, allocation duration, GPU-hours, concurrency fraction, topology contiguity, declared workload class, account/job linkage, reservation and preemption pattern |
| Cloud control-plane, reservation, and billing records | Primary detection signal for cloud settings; attribution signal | Batch provisioning size, instance type, capacity reservation duration, billing continuity, region/AZ/placement group, account or organization linkage, egress/inter-region movement |
| On-device GPU telemetry | Primary activity signal | GPU utilization, SM or tensor-core activity, HBM memory used, HBM bandwidth, GPU power draw, per-process accounting, GPU health/error counts |
| GPU profiling and kernel/counter telemetry | Strong but often unavailable activity signal; workload-shape evidence | Kernel family or hashed kernel sequence, achieved tensor-throughput ratio, profiler availability state |
| Intra-node GPU fabric telemetry | Primary/supporting activity signal for local parallelism | NVLink/NVSwitch utilization, local fabric periodicity, local link error or recovery events |
| Scale-out network/fabric telemetry | Primary detection signal for distributed training; fragmentation and attribution signal | Scale-out port utilization, synchronized fabric footprint, collective periodicity or step cadence, RDMA congestion/retry/drops, job-to-port mapping coverage |
| Rack/server/facility power and energy telemetry | Secondary support; physical consistency check; discrepancy detection | Rack IT power, facility IT power, power continuity, power-to-telemetry consistency, baseline-subtracted energy |
| Cooling and thermal telemetry | Secondary support; physical plausibility check | GPU/HBM temperature, liquid-cooling delta-T, cooling flow or fan speed, thermal-throttle support |
| Host, VM, container, process, and distributed-runtime metadata | Attribution and workload-type evidence; escalation evidence | Distributed world size/rank count, runtime framework class, rendezvous/rank mapping stability, container or VM image digest recurrence |
| Storage, object-store, filesystem, and data-movement logs | Supporting activity and workload-type evidence | Initial data staging volume, checkpoint write size, checkpoint period, read/write operation pattern, artifact export pattern |
| Workload declarations, experiment trackers, and ML training logs | Strong workload-type evidence when authenticated; highest-warning support | Declared model parameter count, training tokens/examples, step count and step time, loss curves, optimizer state, checkpoint metadata, log completeness |
| Attestation, trusted computing, and telemetry provenance | Trust/integrity evidence; attempted evasion detection | Attestation validity, confidential-compute mode, telemetry collector measurement, device or collector signatures |
| Monitoring-pipeline integrity, coverage, and time synchronization | Negative-certification foundation; attempted evasion detection | Telemetry coverage fraction by layer, missed scrapes, clock drift, counter resets, collector config changes |
| Physical security, maintenance, and change-management records | Integrity support; attempted evasion and benign-explanation evidence | Rack-door or badge events, maintenance tickets, firmware/BMC changes, approved physical changes, hardware replacement records |
| Active challenge probes and weak-trust telemetry tests | Auxiliary attempted evasion detection under weak trust | Probe throughput ratio, probe latency inflation, VRAM residency/free-memory test |
| External and out-of-band evidence | Capacity discovery and checking against other records; weak activity evidence | External IT power estimate, construction or commissioning timeline, chip shipment/procurement indicators, public disclosures, permit or utility evidence |

First we identified how useful each data source is for determining the presence of training runs. The primary observables for detecting candidate large training runs are scale-out fabric telemetry, scheduler/allocation metadata, cloud provisioning or reservation records where relevant, GPU telemetry, and runtime or ML-log evidence where available. They can either show large compute allocation, actual accelerator activity, distributed synchronization, or workload purpose: what kind of workload was running.

Secondary observables do not usually identify training by themselves, but they are important for checking the primary evidence. Storage and checkpoint patterns can support a training interpretation when they indicate the same usage as allocation, GPU activity, runtime metadata, or ML logs. Power and cooling data can show if physical load shows the same patterns as claimed compute activity. External evidence can support capacity discovery and help check whether the monitored inventory matches public, utility, procurement, and timeline records, but it is usually too coarse to prove that a specific training run happened in a specific time window.

Some obervables are mainly used for monitoring-integrity and attempted-evasion detection. Attestation, monitoring coverate, counter resets, clock drift, physical access records, firmware changes, and active probes do not prove training by themselves. They instead tell us how much to trust other evidence, if a no-run claim is credibe, and if missing or inconsistent telemetry needs explanation.

Not all primary observables can individually determine a reliable probability for a large training run - a large allocation cam be idle - and their data may be missing. Therefore, each of them is checked to verify a training possibility. In addition, high GPU usage can be inference, HPC, benchmarking, burn-in, or data processing. High fabric traffic can be training, but also HPC, NCCL tests, storage replication, or distributed inference. High power can show load, but not the workload type. This is why the framework combines observables.

We create a catalog of observables, and group features into capacity evidence, activity evidence, attribution evidence, physical checks, and integrity evidence. We then create a framework with rules about their interdependence. Some features become stronger only when another feature is also present: for example, high fabric synchronization is more informative when there is a large scheduler allocation or runtime records showing that the devices were working together. Other combinations create disrepancy warnings: for example, high rack power with low GPU telemetry, or high fabric activity with no correspondinv scheduler allocation.

The existence of many measurements is externally supported, but the exact value at which a feature becomes suspicious depends on hardware generation, topology, metering point, policy threshold, workload type, and telemetry coverage.


## Training-run signatures

We use datacenter records for claims about how much compute is used. The target quantity is cumulative training compute during an audit window `W`, measured in FLOP. `W` is the time period being evaluated. `T` is the policy threshold being tested.

The thresholds used here are summarized in Table 2.

Table 2. Policy thresholds and defined review tiers.

| Threshold | Value | Role |
|---|---:|---|
| `T_gpai_indicator` | `10^23 FLOP` | Commission guideline indicator for GPAI qualification; not a systemic-risk threshold. |
| `T_review` | `10^24 FLOP` | Paper-defined review tier, unless later tied to a specific policy source. |
| `T_downstream` | `original_model_training_compute / 3` | Commission guideline boundary for significant modification/fine-tuning; not generally `T_sys / 3`. |
| `T_sys` | `10^25 FLOP` | EU AI Act systemic-risk presumption. |

The current systemic-risk reference is `T_sys = 10^25 FLOP`. To estimate this level of compute being reached at a datacenter, we check observable sources. For each source s, we check what range of compute is consistent with its values. The lower end is the smallest amount of compute the source can support, the upper end is the largest amount of compute that could still be possible given that source and its coverage:

`C_s^- <= C_true <= C_s^+`

- `C_true`: the real training compute used by the workload.
- `C_s^-`: the lower bound supported by source `s`. If the source is valid, at least this much compute is supported by the record.
- `C_s^+`: the upper bound supported by source `s`. Given that source and its monitored scope, the workload could not have used more than this amount.

For a threshold T, the source interval allows one of three outcomes:

1. Threshold supported: C_s^- >= T
   The source supports at least T compute.
2. Threshold possible: C_s^- < T <= C_s^+
   The source does not prove T, but T is still possible.
3. Threshold ruled out for this source scope: C_s^+ < T
   The source’s maximum possible compute is below T. If coverage is trusted, this rules out T within that monitored scope.

Some observables only require an upper or a lower bound. For example, capacity is used to find if a training is physically possible; only an upper bound of C_capacity^+ is needed to rule out training runs. Achieved-throughput telemetry and authenticated workload logs can support a positive threshold finding when their lower bound reaches the threshold, so C_gpu^- >= T or C_declared^- >= T.

### Conversion hardware-time into FLOP

We convert how long the hardware was used into how many FLOP could have been done.

- h = hardware type, e.g. H100 SXM, H100 PCIe, B200, TPU, Trainium.
- p = precision/training mode, e.g. BF16 dense, FP16 sparse, FP8.
- R_peak(h,p) = the vendor’s advertised maximum FLOP/s for that hardware and mode.
- R_sustained(h,p) = the realistic sustained FLOP/s we use for auditing.
- sustained_efficiency = the fraction of peak actually sustained by the workload. For example, if peak is 1,000 TFLOP/s and the workload sustains 40%, then R_sustained = 0.4 * 1,000 = 400 TFLOP/s.
- n_h(t) = number of accelerators of type h active at time t.
- C(W) = total compute over window W.

For each accelerator SKU `h` and precision or training mode `p`:

`R_peak(h,p) = vendor-documented peak FLOP/s`

`R_sustained(h,p) = sustained_efficiency(h,p,site,workload) * R_peak(h,p)`

`C(W) = integral_W sum_h n_h(t) * R_sustained(h,p) dt`

For exclusion, use an upper-bound rate `R_max` consistent with the hardware, precision, power caps, topology, and monitored scope. For positive lower-bound claims, use measured achieved throughput or a conservative lower-bound rate `R_min` justified by activity telemetry. `R_sustained` is a calibrated point estimate, not automatically the right value for both upper and lower bounds.

So if 2,000 H100s are active for 30 days, we multiply:
2,000 * sustained FLOP/s per H100 * 30 days in seconds. That gives the estimated FLOP for that window.

The sustained-efficiency value is estimated from site benchmarks, audited training windows, or conservative audit defaults. If no site calibration exists, the result should be reported as an interval using `sustained_efficiency_min` and `sustained_efficiency_max`.

Epoch AI's compute-estimation documentation uses two main routes. The first is hardware-based: chip-time, hardware type, numerical format, and utilization. The second is operation-count based: model architecture and training data. For hardware-based estimates, Epoch's worked examples multiply chip-days by peak hardware FLOP/s and by a utilization multiplier.

### H100 reference calculation

For H100 SXM, NVIDIA documents BF16/FP16 Tensor Core performance around `1000 TFLOP/s` dense and about `2000 TFLOP/s` with sparsity in the Hopper architecture documentation. NVIDIA's public H100 product table lists `1,979 TFLOP/s` for BF16/FP16 Tensor Core with a sparsity footnote. Therefore the hardware-normalization table must store:

- hardware SKU;
- precision mode;
- dense or sparse mode;
- whether the workload can use the advertised mode;
- sustained-efficiency assumptions.

At `sustained_efficiency = 1`, the required H100-hours are shown in Table 3.

Table 3. H100-hours required at two peak-reference rates.

| Reference rate | `10^23 FLOP` | `10^24 FLOP` | `3.3 * 10^24 FLOP` | `10^25 FLOP` |
|---:|---:|---:|---:|---:|
| `1.0e15 FLOP/s` | `2.78e4` H100-hours | `2.78e5` H100-hours | `9.26e5` H100-hours | `2.78e6` H100-hours |
| `1.979e15 FLOP/s` | `1.40e4` H100-hours | `1.40e5` H100-hours | `4.68e5` H100-hours | `1.40e6` H100-hours |

At `sustained_efficiency = 0.4`, each value in Table 3 would be multiplied by `1 / 0.4 = 2.5`.

A `512` H100 allocation for `24` hours is:

`512 * 24 * 3600 * 1.0e15 = 4.42 * 10^22 FLOP`

Using the `1.979e15 FLOP/s` reference rate:

`512 * 24 * 3600 * 1.979e15 = 8.75 * 10^22 FLOP`

So `512` H100s for one day is below `10^23 FLOP` at peak and far below `10^25 FLOP`. This is why GPU count alone is not a threshold rule. The threshold depends on hardware rate, duration, sustained efficiency, and whether the devices were actually used for training.

## Source-specific threshold logic

### O1: Hardware inventory and accelerator capacity

O1 shows what capacity exists. It includes accelerator count, SKU, memory, topology domain, partitioning state, and the largest usable low-latency accelerator domain.

Upper bound:

`C_capacity^+(W) = integral_W sum_h N_h,usable(t) * R_max(h,p,site,scope) dt`

`N_h,usable` is the number of accelerators that are actually usable for threshold-scale training. It should account for SKU, memory, topology, partitioning, and monitored-boundary scope.

O1 can rule out threshold-scale training inside the monitored boundary when `C_capacity^+(W) < T` and inventory coverage is trusted.

### O2: Scheduler, reservation, and allocation metadata

O2 tells us which jobs or linked job groups were allocated accelerators, for how long, under which account/project/user, and on which nodes or partitions. Slurm-style records support allocated TRES/GRES, elapsed time, job state, node fields, accounts, and related accounting fields.

The allocation compute upper bound is:

`C_alloc^+(W) = sum_j,h N_j,h * D_j(W) * R_max(h,p,site,scope)`

where:

- `N_j,h` is the allocated accelerator count for job or linked job group `j`;
- `D_j` is the elapsed allocation time;
- `R_max(h,p,site,scope)` is the upper-bound throughput for the hardware, precision mode, site, and monitored scope.

O2 can identify a threshold-capable candidate window. It upper-bounds possible compute, but it does not by itself establish a lower bound on compute actually performed. It can support L3 only when job/account/workflow linkage is stable and the allocation is joined to activity evidence such as O4, O7, O8/O9, O10, or O12 for the same window.

O2 needs O4 or O8 to find if there was active use or allocated-but-idle capacity. It needs O7, O10, O11, or O12 to interpret the workload as training rather than HPC, inference, benchmark, or burn-in. Allocation evidence alone is capped below L4.

### O3: Cloud control plane, reservation, and billing

O3 is the cloud equivalent of allocation and reservation evidence. It covers instance type, accelerator count, reservation duration, launch/stop events, placement, account, and billing continuity. AWS Capacity Blocks support reservation size, duration, instance type, start time, region, and placement concepts. CloudTrail supports API event records. Cost and Usage Reports support usage and billing exports.

Depending on whether the cloud record shows reserved capacity only or running/use evidence, we calculate:

`C_cloud_reserved^+(W) = reserved_accelerator_hours(W) * R_max(h,p,site,scope)`

`C_cloud_running^+(W) = running_accelerator_hours(W) * R_max(h,p,site,scope)`

`C_cloud_activity^-(W) = integral_covered_W N_running,mapped(t) * R_min_activity(t) dt`

`C_cloud_reserved^+` is capacity commitment. `C_cloud_running^+` is a stronger candidate upper bound because instances were running. Neither is a positive training-compute lower bound without mapped activity evidence. `C_cloud_activity^-` requires utilization, achieved-throughput, fabric, runtime, storage, or authenticated workload evidence mapped to the running instances.

Running instance-hours can create a candidate threshold window because they show accelerator capacity active for a measured duration. They still need activity and workload-type evidence to show training. Reservation-only evidence shows capacity held for possible use, not compute performed. O3 requires instance-to-accelerator mapping, account or organization aggregation, placement and region scope, and alignment with O4/O7/O10/O11 activity evidence.

### O4: On-device GPU telemetry

O4 tells us if the accelerators were active. It can include usage, tensor activity, memory use, memory bandwidth, power draw, process/accounting fields, and health/error counters. DCGM, nvidia-smi, and cloud GPU monitoring support these surfaces.

If O4 only contains usage, memory, and power, it converts O2/O3 from allocation evidence into active-use evidence. If O4 contains achieved tensor throughput or equivalent counters, compute can be integrated directly:

`C_gpu^-(W) = integral_{covered W} sum_i achieved_FLOP/s_i(t) dt`

`C_gpu^+(W) = C_gpu^-(W) + integral_{uncovered W} sum_i R_sustained,max(h,p) dt`

Achieved-throughput telemetry can cross `T` directly when it is mapped to devices, job/account, and time. Generic GPU busy or power cannot produce a training threshold result by itself, because many non-training workloads can keep GPUs busy.

O4 is usable only when GPU telemetry is mapped to the relevant devices, jobs, accounts, and time window. It can show active accelerator use, but not training by itself. L4 requires matching fabric, runtime, storage/checkpoint, or authenticated workload-log evidence (O7/O10/O11/O12) for the same window.

### O5: Profiler and kernel counters

O5 can provide lower-level evidence: achieved tensor throughput, kernel-family motifs, hashed kernel sequences, or sampled traces.

If profiler data exposes achieved tensor throughput, we use the O4 throughput integral. If it only exposes kernel-family motifs or hashes, it supplies workload-shape evidence rather than a FLOP total.

Achieved-throughput counters can support a threshold result. Kernel motifs alone cannot. Motif evidence needs authentication, sampling coverage, and alignment with O2/O4/O7/O10/O11/O12. Profiler absence is interpreted through O13/O14, because confidential-compute or security modes can restrict counters.

### O6: Intra-node GPU fabric

O6 covers NVLink/NVSwitch usage, correlation, and error events inside a node or local GPU domain. It helps identify tensor parallelism, model parallelism, or local collective communication.

O6 does not define training compute without active devices and duration. It can support a threshold result only when combined with O4 achieved compute or O10/O12 runtime/log evidence.

O6 strengthens O4/O10 when local tensor/model parallelism is visible. It is also used with O7 to check whether local and scale-out communication patterns are compatible with one coordinated workload.

### O7: Scale-out network and fabric telemetry

O7 is scale-out fabric evidence: port bytes, line rate, synchronized footprint, cadence, congestion/retry/drop counters, and job-port mapping. UFM, InfiniBand, EFA, NCCL, and related sources support bandwidth, port counters, congestion, retries, errors, and distributed communication context.

To get a lower-bound compute estimate from fabric telemetry, the active ports must first be mapped to accelerators, jobs, ranks, or accounts. The calculation then uses a lower sustained FLOP/s value per accelerator and counts only the fraction of the window where the fabric trace supports active coordinated work:

`C_fabric^-(W) = integral_W N_mapped(t) * R_min_sustained(h,p) * q_active(t) dt`

`C_fabric^+(W) = integral_W N_possible(t) * R_max_sustained(h,p) * q_possible(t) dt`

where:

- `N_mapped(t)` is the synchronized accelerator membership inferred from ports or ranks;
- `N_possible(t)` is the maximum participant set consistent with port, rank, job, or account mapping;
- `R_min_sustained(h,p)` is the conservative sustained compute rate for the mapped hardware;
- `R_max_sustained(h,p)` is the upper sustained compute rate consistent with the mapped hardware and precision mode;
- `q_active(t)` is a calibrated activity duty factor justified by the fabric trace;
- `q_possible(t)` is the maximum activity duty factor consistent with the fabric trace and coverage.

O7 can support a threshold result only if mapped synchronized membership, duration, and conservative hardware-rate assumptions give `C_fabric^- >= T`. If the fabric trace shows synchronization but does not support conservative compute reconstruction, it remains coordinated-run evidence rather than a threshold result.

In bandwidth-limit regimes, O7 can also be used negatively: verified absence of enough cross-pod bandwidth can rule out efficient multi-pod training under explicit assumptions.

O7 requires job-to-port, rank-to-port, pod-to-port, or account-to-port mapping for attribution. It also needs O10/O12 or workload-class evidence to distinguish training from HPC, storage, or benchmark traffic. Missing fabric telemetry prevents strong no-run claims for workloads that would normally be fabric-visible.

### O8 and O9: Power, cooling, and thermal telemetry

O8 measures power and energy. O9 measures cooling and thermal response. These sources show physical load after baseline subtraction. They are used to check it sustained accelerator activity appears in power and, with site-specific lag, in cooling or thermal records.

A physical upper-bound estimator is possible:

`C_power^+(W) = integral_W P_accel_domain(t) / joules_per_FLOP_min dt`

`joules_per_FLOP_min` is a hardware/site-calibrated value. Power does not identify workload class. A high-power window can be training, HPC, inference, burn-in, or other dense load.

O8/O9 normally do not create a positive training threshold result. They check whether the physical load is consistent with the recorded accelerator activity, or expose inconsistency. They are used with O4 to check GPU-power/rack-power consistency. If power is incompatible with visible telemetry, the result is a discrepancy rather than a clean training finding.

### O10: Runtime metadata

O10 includes runtime fields such as distributed world size, rank mapping, framework class, rendezvous metadata, container or VM image identity, and duration. PyTorch and torchrun sources support ranks, world size, rendezvous, process groups, and distributed APIs.

Runtime metadata can produce an upper-bound compute estimate:

`C_runtime^+(W) = mapped_accelerators * runtime_duration * R_sustained(h,p)`

`C_runtime^-(W) = mapped_accelerators * confirmed_active_duration_min * R_min_sustained(h,p)`

O10 can support L3 when authenticated runtime, mapped hardware, duration, and activity evidence align. It can support L4 only with workload-type/log evidence or authenticated training runtime plus primary activity alignment.

WORLD_SIZE or rank count alone is not enough. O10 requires stable rank-device mapping, scheduler/cloud linkage, and O4/O7 activity. Runtime framework class is interpreted together with O2 declarations and O11/O12 artifacts.

### O11: Storage, checkpoints, and data movement

O11 includes initial data staging, shard reads, checkpoint writes, checkpoint cadence, and post-run artifacts. PyTorch distributed checkpoint, object-store logs, MLflow artifacts, and scheduler I/O records can support these surfaces.

If checkpoint format is known, model-state size can be bounded:

`P_state^-(W) = B_ckpt / b_state_max`

`P_state^+(W) = B_ckpt / b_state_min`

where:

- `B_ckpt` is checkpoint event size;
- `b_state_min` and `b_state_max` are the minimum and maximum plausible bytes per parameter-state element;
- the state-size bounds depend on whether the checkpoint contains weights only, optimizer state, gradients, mixed-precision state, full-precision state, or sharded state.

O11 does not by itself produce FLOP. It can support model-scale interpretation, but FLOP requires tokens, steps, hardware-time, or achieved throughput.

O11 needs periodicity, shard fanout, job/account linkage, and alignment with O2/O4/O10/O12. Backup and replication explanations must be excluded before storage evidence contributes to L3.

### O12: Workload declarations, experiment trackers, and ML logs

O12 can be the most direct workload-type evidence when logs are authenticated. It can include declared model size, tokens/examples, step count, step time, loss curves, optimizer metadata, and checkpoint metadata. MLflow and W&B support run metadata, parameters, metrics, artifacts, and checkpoints. The catalog treats authenticity separately from content.

For dense transformer language-model training:

`C_declared^-(W) = 6 * P_active^- * D_tokens^-`

`C_declared^+(W) = 6 * P_active^+ * D_tokens^+`

where:

- `P_active^-` and `P_active^+` bound the active parameter count;
- `D_tokens^-` and `D_tokens^+` bound the training tokens;
- the factor `6` is the standard dense-transformer training-compute approximation.

If logs provide steps:

`D_tokens = steps * global_batch_tokens_per_step`

The formula must be adjusted for architecture, MoE active parameters, repeated epochs, multimodal components, and the fine-tuning/pretraining boundary.

Authenticated O12 can support L4 when `C_declared^- >= T` and the run aligns with O2/O4/O7/O11. It requires signature/completeness, stable run identity, and activity alignment. Unsigned or partial logs are capped unless O2/O4/O7/O11 records support the same run window. MoE and non-transformer formulas must be policy-defined and versioned.

### O13-O17: Trust, integrity, probes, and external checks

O13-O17 do not estimate training FLOP. They determine how much to trust the other sources, whether missing evidence can support a no-run claim, whether hidden activity is plausible, and whether monitored inventory matches external records.

O13 covers attestation and telemetry provenance: device attestation, collector measurement, telemetry provenance, and confidential-compute state. It controls trust caps on O4/O5/O10/O12/O14. O14 covers monitoring coverage and time synchronization: missed scrapes, gaps, counter resets, collector changes, and clock drift. It determines if absence of evidence gives useful information and if cross-source timing comparisons are valid. O15 covers physical security, maintenance, and change records that can explain or support telemetry gaps or inventory changes. O16 covers active probes for hidden contention under weak trust, such as throughput, latency, or VRAM-residency tests; probe thresholds need hardware/site calibration and safety limits. O17 covers public and external evidence: power capacity, construction, commissioning, procurement, chip shipments, public disclosures, permits, and imagery. It supports capacity discovery and checking monitored records against external records, not active-run attribution.


## How sources are combined

Source-specific formulas do not always provide detection results. Some give evidence only in dependency with other sources. These dependencies are recorded in a dependency graph. It records which observables produce compute quantities, which observables constrain other observables, which observables check consistency, and which observables modify trust instead of estimating FLOP.

Each observable contributes one of four things:

- Maximum compute consistent with the source: O1 capacity, O2 allocation, O3 reservation/running instance-hours, O8/O9 power, O10 runtime without confirmed activity.
- Minimum compute supported by the source: O4 achieved throughput, O7 mapped fabric, O10 runtime with confirmed activity, O12 authenticated logs.
- Workload evidence that helps identify training but does not by itself estimate FLOP: O5 motifs, O6 local fabric, O11 checkpoint/storage pattern.
- Trust or discrepancy evidence: O13 attestation, O14 coverage/time sync, O15 physical changes, O16 probes, O17 external checks.

A source raises the result only when its dependencies are satisfied: allocation needs activity evidence, activity needs workload evidence to become training evidence, and all positive or negative claims need enough telemetry trust and coverage.

Process for combining sources:

1. Capacity defines the monitored scope.  
   O1/O17 define what hardware could have been available during `W`. If `C_capacity^+(W) < T`, threshold-scale training can be ruled out inside the monitored boundary, assuming trusted inventory coverage. If `C_capacity^+(W) >= T`, training is possible but not shown.

2. Allocation and cloud records create candidate windows.  
   O2/O3 show assigned, reserved, provisioned, or running accelerator capacity. Their quantities can reach a threshold, but they do not prove training without activity and workload evidence.

3. Activity evidence separates use from idle capacity.  
   O4, O5 achieved throughput, O7, O8/O9, and O10 with confirmed activity show whether hardware was active. If O2/O3 show threshold-scale capacity but O4 and O8/O9 show idle load under trusted O14 coverage, the result remains a maximum possible compute estimate, not evidence of compute performed.

4. Fabric and runtime evidence connect activity to one workload.  
   O7 and O10 test whether active devices belong to the same coordinated job/account/window. Agreement supports one coordinated workload; disagreement creates a mapping, coverage, federation, or off-ledger-capacity discrepancy.

5. Workload evidence identifies training.  
   O10 runtime class, O11 checkpoint/storage behavior, and O12 authenticated logs distinguish training from inference, HPC, benchmarks, burn-in, backup, or data movement. O12 can support L4 when `C_declared^- >= T` and the same window is supported by activity evidence.

6. Trust and integrity decide whether evidence is usable.  
   O13-O17 do not estimate training compute. They decide whether telemetry is authentic, complete, time-aligned, physically explainable, or externally plausible. Low O14 coverage prevents strong no-run certification; invalid O13 attestation weakens claims; O17 mismatch raises off-ledger capacity suspicion, not active-run proof.

The combined result is a label: an evidence level for a training run. In the synthetic examples it may be displayed with a probability-like score, but use on existing datacenters would require calibration against known real workloads. Table 4 summarizes the result categories.

Table 4. Result categories.

| Result | Required pattern |
|---|---|
| Rule out threshold run | Maximum possible compute from capacity or activity evidence falls below `T` with trusted coverage. |
| Candidate window | Maximum possible compute or running/provisioned capacity reaches a review threshold, but no minimum-supported compute estimate proves `T_sys`. |
| Strong activity finding | A primary activity source supports at least `T_sys`, or independent activity intervals overlap for the same job/account/window. |
| Strong training finding | Authenticated workload compute reaches `T_sys` and aligns with activity evidence, or a full aligned evidence stack supports the same run. |
| Discrepancy | Sources that should agree do not agree, or trust/coverage is too weak to interpret missing evidence. |


Table 5 gives the more technical label definitions used by the framework.

Table 5. Evidence levels used by the framework.
| Level | Meaning | Technical condition |
|---|---|---|
| L0 | Threshold-scale training is ruled out for the monitored boundary. | `C_capacity^+ < T`, or all primary activity estimators have `C_s^+ < T` under strong O14 coverage and trusted inventory. |
| L1 | Capacity exists, but activity evidence is weak, unavailable, or unattributed. | `C_capacity^+ >= T`, but all available activity estimators are below review thresholds, unavailable, or not attributable. |
| L2 | Candidate evidence exists, but no source lower-bound crosses `T_sys`. | At least one primary estimator has `C_s^+ >= T_gpai_indicator` or `C_s^+ >= T_review`, but no source has `C_s^- >= T_sys`; or a physical/integrity discrepancy prevents L0. |
| L3 | Strong threshold-scale activity evidence exists. | At least one primary activity source has `C_s^- >= T_sys`, or at least two independent activity estimators have overlapping threshold-scale intervals for the same job/account/window. |
| L4 | Threshold-scale training is supported by authenticated workload-type evidence or by a full aligned evidence stack. | Authenticated O12 compute has `C_declared^- >= T_sys` and aligns with primary activity evidence, or L3 is supported by scheduler/cloud, GPU activity, fabric/runtime or storage, physical load, and O14 coverage above the certification threshold. |

Primary activity sources are:

- O2 scheduler/allocation;
- O3 cloud records when they show running instance/use evidence;
- O4 achieved GPU activity;
- O5 achieved throughput;
- O7 mapped synchronized fabric;
- O10/O12 when authenticated and mapped.

Capacity sources O1/O17 establish feasibility or exclusion. Integrity sources O13-O15 modify trust and caps.

### Source basis

The catalog source map supports measurement surfaces. It does not supply universal training thresholds for utilization, fabric periodicity, power fractions, checkpoint sizes, or coverage cutoffs. Those thresholds come from policy definitions, hardware conversion, and calibration.

- EU AI Act Q&A / Article 51: `10^25 FLOP` systemic-risk training-compute threshold and `10^23 FLOP` GPAI context.
- Epoch AI model-estimation documentation: hardware-based compute estimation from chip-time, hardware type, numerical format, and utilization; architecture/data-based estimation including transformer operation counting.
- NVIDIA H100 and Hopper documentation: H100 SXM Tensor Core peak rates, dense/sparse distinction, NVLink/NVSwitch and training/HPC hardware context.
- Slurm `sacct`, TRES, and GRES docs: job accounting, elapsed time, allocated TRES/GRES, typed GPU resources, states, accounts, nodes, energy, and I/O fields.
- Kubernetes device-plugin docs: device discovery/allocation surface for GPU-style accelerators.
- AWS Capacity Blocks, CloudTrail EC2, RunInstances, CUR, and EFA docs: reservation size/duration/start time, EC2 API events, launch parameters, billing/usage exports, and low-latency fabric context.
- Google Cloud GPU monitoring and audit-log docs: cloud GPU telemetry and administrative/data-access event records.
- NVIDIA DCGM/nvidia-smi docs: GPU telemetry, utilization, memory, power, temperature, process/accounting, health/error counters, and profiling surfaces.
- NVIDIA UFM/InfiniBand docs: bandwidth, port counters, congestion, XmitWait, retry/drop/error, latency, and port-based monitoring.
- PyTorch distributed and torchrun docs: process groups, collectives, ranks, world size, rendezvous, rank changes, and distributed launch metadata.
- PyTorch distributed checkpoint docs: distributed checkpoint state and checkpoint APIs.
- NCCL docs: collective communication, topology, networking, and debug/environment surfaces.
- S3 server access logs, MLflow, and W&B docs: object access logging, run metadata, parameters, metrics, artifacts, checkpoints, and experiment records.
- Prometheus/PromQL and time-synchronization docs: coverage, missingness, aggregation windows, and clock-error monitoring.
- Redfish/NIST/change-management sources: physical security, maintenance, firmware/software inventory, audit/accountability, and configuration-management context.
- Active-probe research source in the catalog: timing, memory, GEMM, proof-of-work-like, and VRAM-residency probe concepts under weak trust.
- Epoch frontier datacenter dataset: public discovery and capacity checks against external records, not active-run attribution.

Useful links:

- EU AI Act FAQ: https://digital-strategy.ec.europa.eu/en/faqs/general-purpose-ai-models-ai-act-questions-answers
- Epoch AI estimation docs: https://epoch.ai/data/ai-models-documentation/estimation
- NVIDIA H100 specs: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA Hopper architecture: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Slurm sacct: https://slurm.schedmd.com/sacct.html
- Slurm TRES: https://slurm.schedmd.com/tres.html
- Slurm GRES: https://slurm.schedmd.com/gres.html
- AWS Capacity Blocks: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html
- AWS CloudTrail EC2: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitor-with-cloudtrail.html
- AWS CUR: https://docs.aws.amazon.com/cur/latest/userguide/what-is-cur.html
- NVIDIA DCGM Exporter: https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html
- NVIDIA UFM Telemetry: https://docs.nvidia.com/networking/display/ufmsdnappumv4100/telemetry
- PyTorch distributed: https://docs.pytorch.org/docs/2.11/distributed.html
- torchrun: https://docs.pytorch.org/docs/2.11/elastic/run.html
- PyTorch distributed checkpoint: https://docs.pytorch.org/docs/2.11/distributed.checkpoint.html
- MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking/
- W&B Experiments: https://docs.wandb.ai/models/track


## discrepancy scoring

We separately estimate in a window shows evidence of evasion or telemetry inconsistency. 

A ruleset in the catalog defines expected relationships between sources, such as when power should match GPU activity or when fabric activity should match scheduler records. Each rule has a trigger condition, required coverage, benign explanations to check, possible evasion hypotheses, a severity level, and a label effect. For audit window `W`, each rule `r` contributes only if its trigger is present and the required coverage is available:

`s_r(W) = trigger_r(W) * coverage_r(W) * weight(severity_r) * (1 - benign_r(W))`

where `trigger_r(W)` marks if the discrepancy occurs, `coverage_r(W)` marks if the relevant sources were collected, mapped, authenticated, and time-aligned, `weight(severity_r)` is larger for high-severity discrepancies, and `benign_r(W)` measures if an authenticated benign explanation accounts for the conflict.

The window-level evasion score is:

`S_evasion(W) = 1 - product_r (1 - s_r(W))`

The rule families include activity without allocation, high physical load with low visible telemetry, synchronized fabric without scheduler/account linkage, cloud/scheduler mismatch, counts exceeding monitored capacity, telemetry gaps during candidate windows, counter resets or collector changes near activity, unexplained profiler/counter absence, workload declarations conflicting with activity, checkpoint patterns without training context, external capacity conflicts, active-probe hidden-load signals, and physical or firmware changes near telemetry gaps.


## Limitations

- Public evidence alone is usually not enough to identify training activity inside a specific cluster.
- Many numeric cutoffs need site-specific calibration, including utilization, fabric, power, telemetry coverage, and active-probe thresholds.
- Joining scheduler, GPU, fabric, power, storage, runtime, and account records into one view is hard in real deployment
- Privacy and commercial sensitivity may limit which records can be reported routinely and which require escalation.
- Fragmented runs across accounts, regions, providers, or unmonitored capacity remain more difficult to rule out.
- The current catalog defines evidence relationships and example rules; it does not yet claim empirical deployment performance.
- check and note here any simplifications we assume


## Appendix

### Appendix A: Policy thresholds and boundary cases

This paper separates legal thresholds, guideline indicators, and paper-defined review tiers.

The clearest current legal anchor is the EU AI Act Article 51 presumption that a general-purpose AI model has high-impact capabilities when cumulative training computation exceeds `10^25` FLOP. Article 51 also allows the threshold to be amended as technology evolves, including in light of algorithmic improvements or increased hardware efficiency. Article 52 requires notification without delay and within two weeks after the threshold is met, or after it becomes known that it will be met.

The Commission guidelines discuss `10^23` FLOP as an indicative criterion for GPAI qualification for models capable of generating language, text-to-image, or text-to-video, while noting that exceptions can apply. This is not the same as the Article 51 systemic-risk threshold.

For modifications and fine-tuning, the Commission guidance says actors modifying or fine-tuning a GPAI model are not automatically providers. They become providers only in exceptional circumstances, specifically when the modification or fine-tuning uses more than one-third of the original model's training compute. Therefore a downstream threshold is model-relative, not generally `10^25 / 3`.

Any additional review tier used by this paper, such as `10^24` FLOP, is an internal audit or monitoring tier unless adopted by a policy instrument. It should be versioned and separated from legal thresholds.

### Appendix B: False-positive-rate calibration

False-positive rate means the rate at which a rule incorrectly fires on windows that were not threshold-scale training.

For rule `r` and label level `k`:

`FPR(r,k) = count(non-threshold-run windows with required coverage that trigger r at level >= k) / count(non-threshold-run windows with required coverage for r)`

The denominator includes only windows where the rule could have been evaluated. It excludes windows where:

- the source was not collected;
- the collector was unauthenticated;
- job/account mapping was unavailable;
- clock alignment was outside the rule's tolerance.

Missing telemetry is therefore not counted as negative evidence.

For non-FLOP thresholds, the threshold value is chosen from calibration windows. These include utilization bins, tensor-active bins, fabric-periodicity scores, power-continuity scores, checkpoint-cadence scores, probe-degradation scores, and coverage cutoffs.

The selection rule is:

`theta(r,k) = min theta such that FPR(r,k,theta) <= alpha_k`

`alpha_k` is the allowed false-positive rate for label level `k`.

Calibration must include named negative cohorts:

- idle reservations;
- failed or cancelled allocations;
- batch inference;
- distributed inference;
- HPC/MPI jobs;
- NCCL benchmarks;
- GEMM or accelerator burn-in benchmarks;
- storage replication;
- ETL/data processing;
- backup/export jobs;
- cooling/power tests;
- planned maintenance windows;
- monitoring outages and counter-reset windows.

Thresholds are stored as versioned calibration parameters. The calibration record must include hardware generation, precision mode, topology, metering point, telemetry access tier, aggregation window, and the negative-cohort set used to select the value.

Combined rules fire only when the dependency graph is satisfied:

- Allocation/provisioning counts must be bounded by O1 inventory and O17 external-capacity checks.
- GPU activity must be compatible with O8/O9 physical load after baseline subtraction.
- Scale-out fabric footprint must be compatible with O10 rank/world-size or with job-port/account mapping.
- Checkpoint and storage cadence must align with O2/O10 windows.
- O12 workload-type declarations must align with O2/O4/O7/O11 activity evidence.
- O13/O14/O15 integrity failures modify trust and caps; they do not create FLOP estimates.

### Appendix C: False-positive handling

The following rules describe how common false-positive routes are handled.

- O2 threshold-scale allocation with low O4 and low O8 under trusted O14 coverage is classified as reserved, idle, failed, or non-active until contrary evidence appears.
- O2 threshold-scale allocation with missing O4/O8 during a candidate window is an integrity or coverage issue. It is not treated as idle.
- O4 threshold-scale achieved activity with inference runtime, no checkpoint pattern, and no synchronized scale-out footprint remains inference/HPC/benchmark-candidate evidence unless O12 or training runtime evidence conflicts with that explanation.
- O5 kernel motifs without achieved-throughput integration and without runtime/log alignment do not cross `T`.
- O7 synchronized fabric with MPI/HPC runtime, NCCL benchmark class, or storage-replication O11 pattern remains an attributed non-training collective unless O12 or authenticated training runtime evidence conflicts with that explanation.
- O8/O9 physical load without O2/O4/O7 activity becomes a power-to-telemetry discrepancy.
- O11 checkpoint-like writes without O2/O4/O10/O12 alignment remain storage ambiguity.
- O12 signed logs that cross `T` but lack O2/O4/O7/O11 alignment produce a workload-type/telemetry discrepancy rather than automatic L4.
- O13-O15 integrity failures cap no-run certification and trigger review; they do not create training-compute estimates.

### Appendix D: Source-specific false-positive routes

- O1 hardware inventory and accelerator capacity: idle installed capacity; heterogeneous SKUs summed as if they were homogeneous; MIG/vGPU/MPS partitions counted as full devices; off-ledger capacity outside the monitored boundary.
- O2 scheduler, reservation, and allocation metadata: idle reservations; failed jobs; long-running HPC/MPI; batch inference; NCCL benchmark; hardware burn-in.
- O3 cloud control plane, reservation, and billing: unused Capacity Blocks; billing delay or aggregation; account attribution error; capacity held for inference, HPC, benchmark, or availability planning.
- O4 on-device GPU telemetry: dense inference; HPC linear algebra; GEMM benchmark; burn-in; data processing; synthetic-data generation.
- O5 profiler and kernel counters: GEMM loops; vendor benchmarks; HPC dense linear algebra; fused inference kernels; profiler sampling bias.
- O6 intra-node GPU fabric: tensor-parallel inference; local NCCL tests; HPC collectives; topology tests.
- O7 scale-out network and fabric telemetry: MPI/HPC collectives; NCCL benchmark; storage replication; distributed inference; fabric stress tests; routing or congestion incidents.
- O8/O9 power, cooling, and thermal telemetry: non-GPU IT load; meter mapping error; storage/network load; cooling tests; burn-in; baseline drift.
- O10 runtime metadata: MPI/HPC ranks; distributed inference; evaluation sweeps; homogeneous serving fleet; elastic jobs with rank changes.
- O11 storage, checkpoints, and data movement: backup; ETL; storage rebuild; object replication; analytics export; model export without training.
- O12 workload declarations, experiment trackers, and ML logs: fabricated logs; partial logs; eval-only or benchmark loops; toy runs with inflated declarations; logs for a different run/window.
- O13 attestation and telemetry provenance: expired certificates; unavailable attestation; security policy changes; legitimate confidential-compute operation.
- O14 monitoring integrity and time synchronization: collector outage; scrape misconfiguration; time-sync fault; benign counter reset; logging retention boundary.
- O15 physical security, maintenance, and change records: planned maintenance; RMA; firmware upgrade; access-control system error.
- O16 active probes: power caps; thermal throttling; co-tenancy; legitimate background load; probe noise; cache/memory fragmentation.
- O17 external and out-of-band evidence: public estimate error; non-AI load; confidential leasing; partial commissioning; delayed public records.

### Appendix E: Assumptions and calibration choices
- todo
