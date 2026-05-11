# Synthetic v1 Stress Summary

Feature table: `data/synthetic_v1/features/window_features_all.csv`
Seed: `20260510`
Lightweight stress model max_iter: `90`

## Core Splits

- episode_grouped_random: rows=2905, macro F1=0.982, large precision=0.994, large recall=0.997, FP=4, FN=2, middle p bins={'0.1_to_0.3': 18, '0.3_to_0.7': 2, '0.7_to_0.9': 0}
- time_holdout: rows=2943, macro F1=0.986, large precision=1.000, large recall=1.000, FP=0, FN=0, middle p bins={'0.1_to_0.3': 4, '0.3_to_0.7': 0, '0.7_to_0.9': 0}

## Site Holdout

- site_a: rows=2379, macro F1=0.970, large precision=1.000, large recall=0.992, FP=0, FN=5, middle p bins={'0.1_to_0.3': 3, '0.3_to_0.7': 6, '0.7_to_0.9': 0}
- site_b: rows=2796, macro F1=0.982, large precision=1.000, large recall=1.000, FP=0, FN=0, middle p bins={'0.1_to_0.3': 59, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
- site_c: rows=1596, macro F1=0.981, large precision=0.987, large recall=1.000, FP=4, FN=0, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
- site_d: rows=2585, macro F1=0.976, large precision=1.000, large recall=1.000, FP=0, FN=0, middle p bins={'0.1_to_0.3': 20, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
- site_e: rows=1986, macro F1=0.967, large precision=1.000, large recall=0.991, FP=0, FN=4, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
- site_f: rows=1809, macro F1=0.980, large precision=1.000, large recall=0.990, FP=0, FN=4, middle p bins={'0.1_to_0.3': 2, '0.3_to_0.7': 4, '0.7_to_0.9': 0}
- site_g: rows=1561, macro F1=0.972, large precision=1.000, large recall=0.993, FP=0, FN=3, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 7, '0.7_to_0.9': 5}

## Scenario Family Holdout

- underclocked_energy_capped_training: rows=391, macro F1=0.387, large precision=0.975, large recall=1.000, FP=8, FN=0, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
  false positives: {'underclocked_energy_capped_training': 8}
- fragmented_training_linked: rows=762, macro F1=0.350, large precision=1.000, large recall=0.734, FP=0, FN=91, middle p bins={'0.1_to_0.3': 6, '0.3_to_0.7': 81, '0.7_to_0.9': 0}
  false negatives: {'fragmented_training_linked': 91}
- multi_tenant_fragmented_nontraining: rows=461, macro F1=0.200, large precision=0.000, large recall=0.000, FP=0, FN=0, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
- model_parallel_inference: rows=464, macro F1=0.329, large precision=0.000, large recall=0.000, FP=0, FN=0, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
- hpc_mpi_collective: rows=852, macro F1=0.197, large precision=0.000, large recall=0.000, FP=0, FN=0, middle p bins={'0.1_to_0.3': 58, '0.3_to_0.7': 0, '0.7_to_0.9': 0}

## Source Ablations

- source_ablation_drop_fabric: rows=2905, macro F1=0.989, large precision=0.994, large recall=1.000, FP=4, FN=0, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
  macro F1 delta vs random baseline: +0.007
- source_ablation_drop_runtime_and_ml_logs: rows=2905, macro F1=0.893, large precision=0.987, large recall=1.000, FP=8, FN=0, middle p bins={'0.1_to_0.3': 4, '0.3_to_0.7': 4, '0.7_to_0.9': 44}
  macro F1 delta vs random baseline: -0.089
- source_ablation_drop_gpu_telemetry: rows=2905, macro F1=0.972, large precision=0.994, large recall=0.997, FP=4, FN=2, middle p bins={'0.1_to_0.3': 0, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
  macro F1 delta vs random baseline: -0.010
- source_ablation_drop_power: rows=2905, macro F1=0.972, large precision=0.993, large recall=0.993, FP=4, FN=4, middle p bins={'0.1_to_0.3': 4, '0.3_to_0.7': 0, '0.7_to_0.9': 0}
  macro F1 delta vs random baseline: -0.010
- source_ablation_drop_storage: rows=2905, macro F1=0.971, large precision=0.988, large recall=0.966, FP=7, FN=21, middle p bins={'0.1_to_0.3': 13, '0.3_to_0.7': 15, '0.7_to_0.9': 8}
  macro F1 delta vs random baseline: -0.011

## Interpretation

- These are stress diagnostics, not tuned headline metrics.
- Family holdouts identify hard families that still fail under scenario generalization.
- Source ablations should degrade performance when the dropped layer carries independent evidence.
- Middle probability bins indicate whether v1 has less all-or-nothing large-training probability mass than v0.
