# Synthetic Dataset Validation Report

Dataset: `data/synthetic_v0`
Errors: 0
Warnings: 0
Info: 6

## Manifest
- `dataset_id`: `synthetic_v0_seed_20260510`
- `days_per_site`: `30`
- `episode_count`: `210`
- `examples`: `{'0': 'examples/one_datapoint_label0.json', '1': 'examples/one_datapoint_label1.json', '2': 'examples/one_datapoint_label2.json', '3': 'examples/one_datapoint_label3.json', '4': 'examples/one_datapoint_label4.json'}`
- `feature_pipeline_version`: `features-v0.1.0`
- `feature_row_counts`: `{'window_features_15m.csv': 1663, 'window_features_1d.csv': 424, 'window_features_1h.csv': 1435, 'window_features_6h.csv': 858, 'window_features_all.csv': 4380}`
- `file_hashes`: `{'README.md': 'sha256:e42e61c92f66adf71940d09984d2ccffe39bfc053c11ee2a24a646e915e03737', 'examples/one_datapoint_label0.json': 'sha256:9eee2fbd2e2d6d7957f99efd844fcee65c288f57c496740307622e5b3b2e3bee', 'examples/one_datapoint_label1.json': 'sha256:a53e446ca441f3fc7386132bc6b3ffbdd5e494c2dcb5fcbe2188d8263040fe3c', 'examples/one_datapoint_label2.json': 'sha256:26d9f2788872a678460114fd6ca4830fb3fa2fa355bff97fc73b8a2aecf9516d', 'examples/one_datapoint_label3.json': 'sha256:cf7ddb8ce911649176fc519ad14fc55fe5cfd4a56c52fb701c7fd6a53a6f4a5c', 'examples/one_datapoint_label4.json': 'sha256:ab0078d6af7403c3f497a72ee844863d21c94e57ddaedd17b9c7ce94abee3ca0', 'features/window_features_15m.csv': 'sha256:6ad260f0cf74f9e4f3cbb2be62c6d1f21b79c661384a95ec3f0b3be22633be42', 'features/window_features_15m.parquet': 'sha256:51c237364035870d69a56ec11ff7ac986e381433fd62718ba790f484c741a4fd', 'features/window_features_1d.csv': 'sha256:48d51ee3cf350e627601b983b7246462f90e73f6db2f2c9302bc123531b3b7ec', 'features/window_features_1d.parquet': 'sha256:afae278f1b7662efade8148673c0b7dff13c647f48233815895dc9eccb5d511e', 'features/window_features_1h.csv': 'sha256:d3297b428c77724a50b674e80f90508f46794065a99d2a31ab10a701fd1590db', 'features/window_features_1h.parquet': 'sha256:6775b2d2fdc194346c9f930e3d7ecb1b356a0bd0c89decf1c6c0a6740643c54b', 'features/window_features_6h.csv': 'sha256:861c8456b1c7c0e33357842a16288a2f700595f80f8a2e75a72f48abd64c6ffc', 'features/window_features_6h.parquet': 'sha256:bc44c36a253b56729d31ac0b757b9eabe7f8c6b62e3340e42b4562b532f8e00a', 'features/window_features_all.csv': 'sha256:08cbf018975b343366cba8a4e5acc14f27a3e6672437c27d9ea46bd9d0fe0df6', 'features/window_features_all.parquet': 'sha256:df9d3c2080197d334d57282a6889a8e8ce76813b45f00ab1f70cc0dbc3942ba9', 'manifest.json': 'sha256:ef21444599f868825b9550ad70bf415444298312ddaa20b355b7adeac0faf10b', 'raw_normalized/event_records.jsonl': 'sha256:93236d7b861a454464fad5a6570206e2735242948a48c00cc0c93db4e68db889', 'raw_normalized/metric_samples.jsonl': 'sha256:b21f1aab9c60364faaf47deb8f4d922da0b48cf8eb7ed9a7207a9e4fcd1c17ac', 'raw_normalized/snapshot_records.jsonl': 'sha256:053151bf2d2351a2a01bf45fbe891a9660a3809a83d5362e94f704f9b930df2d', 'schemas/event_record.schema.json': 'sha256:5314c1970a6c229acbf176ea8f012820c80564445af704d0326d94f7dca874e4', 'schemas/metric_sample.schema.json': 'sha256:f34aa0dfd1654e9678fbf49adfe38b663b4b0adbde63b5a7f37198c37882926a', 'schemas/prediction_record.schema.json': 'sha256:dccfdd6fea4c336f0aa1049022205fd9ba74d37bec021051d6e8370078a79034', 'schemas/snapshot_record.schema.json': 'sha256:9e45294380c9e898a610ba49d327ff6d008c818a23580d52b1f778dcb610ba3d', 'schemas/window_feature_row.schema.json': 'sha256:5db3b740ea97a2b783fedca3f2d5fedbaf52848e734abde888c7a48ac72fc7d8', 'workbook_rules/composite_rules.json': 'sha256:d0b95e5f3c90969ce584a24bf27e1f5cbd2221eaa04f9d98e9d68df47dfb3fc3', 'workbook_rules/feature_engineering.json': 'sha256:594db087c3aa6f7e988f9a58c9894a0fb34651867401cbfadfc5b330db6da6a0', 'workbook_rules/ground_truth_ranges.json': 'sha256:0ca5963cd8347fbbd369a08a5717a66f9edb0a70cdc8bdb004394bf1094532d1', 'workbook_rules/label_definitions.json': 'sha256:4af94d211b2c20b613654d1a40f7720fa15cd8bdfdb46e7fe76a35c906796635', 'workbook_rules/observable_matrix.json': 'sha256:d6759919a13bbd767dbb40b2aa9263b06cc75904f73ab8c15cd1ebb3ea22ce4d', 'workbook_rules/windowing_guide.json': 'sha256:749d42d82cf442e9c3a38a73a88cee600647f40182abf986e6fc649f7e52c139'}`
- `generation_time`: `2026-05-10T21:32:30Z`
- `generator_version`: `synthetic-generator-v0.1.0`
- `hardware_normalization_version`: `h100e-normalization-v0.1.0`
- `label_distribution`: `{'0': 1632, '1': 1098, '2': 953, '3': 491, '4': 206}`
- `notes`: `['All data is synthetic and fictional.', 'Labels are generated from latent scenario truth plus workbook-inspired composite rules.', 'Missing telemetry is represented with coverage and missing-reason fields, not encoded as zero activity.']`
- `policy_threshold_version`: `policy-thresholds-v0.1.0`
- `raw_record_counts`: `{'event_records.jsonl': 670, 'metric_samples.jsonl': 42108, 'snapshot_records.jsonl': 183}`
- `scale`: `v0`
- `scenario_distribution`: `{'adversarial_fragmented_training': 260, 'cloud_reservation_used_for_training': 176, 'counter_suppressed_candidate_window': 129, 'hardware_burn_in': 66, 'hpc_mpi_simulation': 238, 'idle': 865, 'large_batch_inference': 273, 'large_etl_data_movement': 244, 'large_fine_tune': 203, 'maintenance_window': 91, 'nccl_benchmark': 96, 'normal_inference': 654, 'pretraining': 190, 'reserved_but_unused_capacity': 271, 'small_fine_tune': 208, 'storage_rebuild': 112, 'synthetic_data_generation': 176, 'underclocked_long_duration_training': 128}`
- `schema_version`: `synthetic-raw-v0.1.0`
- `seed`: `20260510`
- `site_count`: `3`
- `sites`: `[{'account_id_hash': 'acct_hmac_cad616ced762', 'baseline_it_mw': 1.08112, 'homogeneous_high_end_fraction': 0.83, 'largest_contiguous_domain_gpus': 3522, 'non_partitioned_fraction': 0.94, 'normalized_h100e_capacity': 4096.0, 'rack_power_design_mw': 4.43984, 'region_hash': 'region_hmac_a132ba32', 'scope_id_hash': 'fabricdom_hmac_e583d3d368', 'site_id': 'site_a', 'site_scope_id_hash': 'site_hmac_0a832da78c', 'site_type': 'self_managed_hpc', 'telemetry_stack': 'dcgm_slurm_ufm_bms', 'trust_tier': 'operator_signed'}, {'account_id_hash': 'acct_hmac_9618c7ff2bc9', 'baseline_it_mw': 0.63056, 'homogeneous_high_end_fraction': 0.865, 'largest_contiguous_domain_gpus': 1761, 'non_partitioned_fraction': 0.94, 'normalized_h100e_capacity': 2048.0, 'rack_power_design_mw': 2.30992, 'region_hash': 'region_hmac_f31fe97f', 'scope_id_hash': 'fabricdom_hmac_91e1474b6e', 'site_id': 'site_b', 'site_scope_id_hash': 'site_hmac_03dba13c6f', 'site_type': 'cloud_region', 'telemetry_stack': 'cloud_api_dcgm_billing', 'trust_tier': 'operator_signed'}, {'account_id_hash': 'acct_hmac_37028fd4feaf', 'baseline_it_mw': 0.26448, 'homogeneous_high_end_fraction': 0.9, 'largest_contiguous_domain_gpus': 276, 'non_partitioned_fraction': 0.66, 'normalized_h100e_capacity': 384.0, 'rack_power_design_mw': 0.57936, 'region_hash': 'region_hmac_d565fecc', 'scope_id_hash': 'fabricdom_hmac_8f408e3606', 'site_id': 'site_c', 'site_scope_id_hash': 'site_hmac_366c2af1d4', 'site_type': 'managed_ai_cloud', 'telemetry_stack': 'dcgm_slurm_ufm_bms', 'trust_tier': 'auditor_reconciled'}]`

## Raw Record Counts
- `raw_normalized/event_records.jsonl`: 670
- `raw_normalized/metric_samples.jsonl`: 42108
- `raw_normalized/snapshot_records.jsonl`: 183

## Findings
### INFO
- `feature_row_count`: Feature rows: 4380
- `label_distribution`: 0: 1632, 1: 1098, 2: 953, 3: 491, 4: 206
- `scenario_count`: Scenarios represented: 18
- `site_episode_count`: Sites: 3; episodes: 210
- `missing_reason_distribution`: observed: 59551, not_scheduled: 4251, privacy_redacted: 4248, routine_profiler_disabled: 3908, not_applicable: 1496, delayed_log_delivery: 604, collector_gap: 273, counter_disabled_by_cc_mode: 129
- `gpu_power_correlation`: GPU utilization and rack power correlation: 0.720

