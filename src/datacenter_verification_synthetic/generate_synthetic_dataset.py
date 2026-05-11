"""Generate synthetic datacenter training-run verification data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .build_window_features import build_features
    from .common import (
        FEATURE_PIPELINE_VERSION,
        GENERATOR_VERSION,
        HARD_GENERATION_SCALES,
        HARDWARE_NORMALIZATION_VERSION,
        OBSERVABLE_IDS,
        POLICY_CONCURRENCY_THRESHOLD_GPUS,
        POLICY_GPU_HOURS_THRESHOLD,
        POLICY_THRESHOLD_VERSION,
        SCALE_PRESETS,
        SCENARIO_CLASSES,
        SCHEMA_VERSION,
        SNAPSHOT_RECORD_FIELDS,
        EVENT_RECORD_FIELDS,
        METRIC_SAMPLE_FIELDS,
        TRAINING_SCENARIOS,
        V1_HARD_NEGATIVE_FAMILIES,
        V1_HARD_POSITIVE_FAMILIES,
        V1_SCENARIO_FAMILIES,
        clamp,
        directory_file_hashes,
        ensure_scale,
        json_dumps,
        raw_payload_hash,
        stable_hash,
        utc_iso,
        write_json,
        write_jsonl,
        write_schema_files,
    )
    from .export_workbook_rules import export_workbook_rules
except ImportError:  # pragma: no cover - direct script execution
    from build_window_features import build_features
    from common import (
        FEATURE_PIPELINE_VERSION,
        GENERATOR_VERSION,
        HARD_GENERATION_SCALES,
        HARDWARE_NORMALIZATION_VERSION,
        OBSERVABLE_IDS,
        POLICY_CONCURRENCY_THRESHOLD_GPUS,
        POLICY_GPU_HOURS_THRESHOLD,
        POLICY_THRESHOLD_VERSION,
        SCALE_PRESETS,
        SCENARIO_CLASSES,
        SCHEMA_VERSION,
        SNAPSHOT_RECORD_FIELDS,
        EVENT_RECORD_FIELDS,
        METRIC_SAMPLE_FIELDS,
        TRAINING_SCENARIOS,
        V1_HARD_NEGATIVE_FAMILIES,
        V1_HARD_POSITIVE_FAMILIES,
        V1_SCENARIO_FAMILIES,
        clamp,
        directory_file_hashes,
        ensure_scale,
        json_dumps,
        raw_payload_hash,
        stable_hash,
        utc_iso,
        write_json,
        write_jsonl,
        write_schema_files,
    )
    from export_workbook_rules import export_workbook_rules


SCENARIO_WEIGHTS = {
    "idle": 0.26,
    "normal_inference": 0.17,
    "large_batch_inference": 0.07,
    "synthetic_data_generation": 0.04,
    "small_fine_tune": 0.06,
    "large_fine_tune": 0.04,
    "pretraining": 0.025,
    "hpc_mpi_simulation": 0.06,
    "nccl_benchmark": 0.035,
    "hardware_burn_in": 0.04,
    "storage_rebuild": 0.035,
    "large_etl_data_movement": 0.035,
    "reserved_but_unused_capacity": 0.04,
    "cloud_reservation_used_for_training": 0.02,
    "maintenance_window": 0.04,
    "adversarial_fragmented_training": 0.025,
    "underclocked_long_duration_training": 0.02,
    "counter_suppressed_candidate_window": 0.015,
}

V1_SCENARIO_WEIGHTS = {
    "idle_or_low_activity": 1.15,
    "normal_inference": 0.18,
    "large_batch_inference": 0.045,
    "model_parallel_inference": 0.035,
    "embedding_generation": 0.04,
    "synthetic_data_generation_gpu_heavy": 0.035,
    "hpc_mpi_collective": 0.035,
    "nccl_extended_benchmark": 0.025,
    "hardware_burn_in_or_thermal_soak": 0.035,
    "storage_rebuild_or_replication": 0.045,
    "large_etl_or_data_movement": 0.055,
    "distributed_database_or_graph_analytics": 0.04,
    "reserved_but_unused_capacity": 0.08,
    "maintenance_with_collector_gaps": 0.045,
    "multi_tenant_fragmented_nontraining": 0.03,
    "counter_suppressed_candidate_window": 0.015,
    "capacity_or_integrity_only_warning": 0.04,
    "pretraining_standard": 0.014,
    "large_fine_tune_standard": 0.014,
    "cloud_training_redacted_runtime": 0.014,
    "training_without_semantic_logs": 0.014,
    "underclocked_energy_capped_training": 0.008,
    "elastic_preempted_training": 0.010,
    "fragmented_training_linked": 0.008,
    "sparse_or_moe_bursty_training": 0.009,
    "training_with_low_fabric_high_checkpoint": 0.014,
    "training_with_delayed_logs": 0.012,
    "multi_stage_training_pipeline": 0.014,
}

DATA_QUALITY_REGIMES = [
    "full_observability",
    "routine_privacy_redaction",
    "confidential_compute_counters_disabled",
    "collector_gap_short",
    "collector_gap_long",
    "delayed_log_delivery",
    "clock_drift_episode",
    "maintenance_observability_loss",
    "source_not_deployed_for_site",
    "partial_fabric_mapping",
]

PRIVACY_TIERS = ["standard_operator_access", "pseudonymized_account", "runtime_redacted", "semantic_logs_withheld"]
COLLECTOR_PROFILES = ["dcgm_ufm_bms_full", "cloud_api_redacted_runtime", "cc_mode_limited_counters", "partial_fabric_mapping"]
TOPOLOGY_CLASSES = ["single_fabric_domain", "multi_tenant_pod", "cloud_placement_group", "fragmented_accounts", "maintenance_partition"]


def _choice(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def make_sites(scale: str, seed: int) -> list[dict[str, Any]]:
    preset = ensure_scale(scale)
    site_count = int(preset["site_count"])
    capacities = [4096, 2048, 384, 8192, 1536, 6144, 768, 3072, 12288, 1024, 512, 16384]
    site_types = ["self_managed_hpc", "cloud_region", "managed_ai_cloud", "enterprise_on_prem"]
    sites: list[dict[str, Any]] = []
    for idx in range(site_count):
        capacity = capacities[idx % len(capacities)]
        if scale == "smoke":
            capacity = 2048
        if scale in HARD_GENERATION_SCALES and capacity < POLICY_CONCURRENCY_THRESHOLD_GPUS:
            capacity = int(POLICY_CONCURRENCY_THRESHOLD_GPUS * 1.5)
        site_id = f"site_{chr(ord('a') + idx)}"
        site_type = site_types[idx % len(site_types)]
        largest_domain = max(128, int(capacity * (0.86 if capacity >= 512 else 0.72)))
        baseline_it_mw = 0.18 + capacity * 0.00022
        design_it_mw = baseline_it_mw + capacity * 0.00082
        sites.append(
            {
                "site_id": site_id,
                "site_type": site_type,
                "region_hash": "region_hmac_" + stable_hash(seed, site_id, "region", length=8),
                "account_id_hash": "acct_hmac_" + stable_hash(seed, site_id, "account", length=12),
                "scope_id_hash": "fabricdom_hmac_" + stable_hash(seed, site_id, "domain", length=10),
                "site_scope_id_hash": "site_hmac_" + stable_hash(seed, site_id, "site", length=10),
                "normalized_h100e_capacity": float(capacity),
                "largest_contiguous_domain_gpus": largest_domain,
                "homogeneous_high_end_fraction": _round(0.83 + (idx % 4) * 0.035),
                "non_partitioned_fraction": 0.94 if capacity >= 512 else 0.66,
                "rack_power_design_mw": design_it_mw,
                "baseline_it_mw": baseline_it_mw,
                "trust_tier": "operator_signed" if idx % 3 != 2 else "auditor_reconciled",
                "telemetry_stack": "dcgm_slurm_ufm_bms" if idx % 2 == 0 else "cloud_api_dcgm_billing",
            }
        )
    return sites


def scenario_duration_hours(rng: np.random.Generator, scenario: str) -> float:
    ranges = {
        "idle": (6, 72),
        "normal_inference": (2, 96),
        "large_batch_inference": (1, 36),
        "synthetic_data_generation": (6, 96),
        "small_fine_tune": (2, 48),
        "large_fine_tune": (48, 240),
        "pretraining": (168, 480),
        "hpc_mpi_simulation": (2, 72),
        "nccl_benchmark": (0.25, 8),
        "hardware_burn_in": (6, 72),
        "storage_rebuild": (4, 48),
        "large_etl_data_movement": (4, 72),
        "reserved_but_unused_capacity": (24, 240),
        "cloud_reservation_used_for_training": (72, 336),
        "maintenance_window": (1, 12),
        "adversarial_fragmented_training": (48, 288),
        "underclocked_long_duration_training": (168, 480),
        "counter_suppressed_candidate_window": (24, 168),
        "idle_or_low_activity": (6, 96),
        "normal_inference": (4, 120),
        "large_batch_inference": (2, 48),
        "model_parallel_inference": (3, 72),
        "embedding_generation": (6, 96),
        "synthetic_data_generation_gpu_heavy": (8, 144),
        "hpc_mpi_collective": (4, 96),
        "nccl_extended_benchmark": (1, 24),
        "hardware_burn_in_or_thermal_soak": (8, 96),
        "storage_rebuild_or_replication": (6, 96),
        "large_etl_or_data_movement": (6, 120),
        "distributed_database_or_graph_analytics": (6, 96),
        "reserved_but_unused_capacity": (24, 240),
        "maintenance_with_collector_gaps": (2, 36),
        "multi_tenant_fragmented_nontraining": (12, 144),
        "capacity_or_integrity_only_warning": (8, 96),
        "pretraining_standard": (168, 720),
        "large_fine_tune_standard": (36, 240),
        "cloud_training_redacted_runtime": (72, 360),
        "training_without_semantic_logs": (48, 336),
        "underclocked_energy_capped_training": (168, 720),
        "elastic_preempted_training": (48, 360),
        "fragmented_training_linked": (72, 480),
        "sparse_or_moe_bursty_training": (48, 336),
        "training_with_low_fabric_high_checkpoint": (48, 240),
        "training_with_delayed_logs": (72, 360),
        "multi_stage_training_pipeline": (96, 480),
    }
    low, high = ranges[scenario]
    return _log_uniform(rng, low, high)


def scenario_sequence(rng: np.random.Generator, count: int) -> list[str]:
    scenarios = list(SCENARIO_CLASSES)
    if count <= len(scenarios):
        return scenarios[:count]
    remaining = count - len(scenarios)
    weights = np.array([SCENARIO_WEIGHTS[name] for name in scenarios], dtype=float)
    weights = weights / weights.sum()
    sampled = list(rng.choice(scenarios, size=remaining, replace=True, p=weights))
    all_scenarios = scenarios + sampled
    rng.shuffle(all_scenarios)
    return list(all_scenarios)


def v1_scenario_sequence(rng: np.random.Generator, count: int) -> list[str]:
    scenarios = list(V1_SCENARIO_FAMILIES)
    if count <= len(scenarios):
        return scenarios[:count]
    remaining = count - len(scenarios)
    weights = np.array([V1_SCENARIO_WEIGHTS[name] for name in scenarios], dtype=float)
    weights = weights / weights.sum()
    sampled = list(rng.choice(scenarios, size=remaining, replace=True, p=weights))
    all_scenarios = scenarios + sampled
    rng.shuffle(all_scenarios)
    return list(all_scenarios)


def choose_site(rng: np.random.Generator, sites: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    if scenario == "cloud_reservation_used_for_training":
        cloud_sites = [
            site
            for site in sites
            if "cloud" in site["site_type"] and site["normalized_h100e_capacity"] >= POLICY_CONCURRENCY_THRESHOLD_GPUS
        ]
        return _choice(rng, cloud_sites or sites)
    if scenario in TRAINING_SCENARIOS - {"small_fine_tune"}:
        capable = [site for site in sites if site["normalized_h100e_capacity"] >= POLICY_CONCURRENCY_THRESHOLD_GPUS]
        return _choice(rng, capable or sites)
    return _choice(rng, sites)


def make_base_template(site: dict[str, Any], scenario: str, duration_hours: float, rng: np.random.Generator) -> dict[str, Any]:
    capacity = float(site["normalized_h100e_capacity"])
    capacity_possible = capacity >= POLICY_CONCURRENCY_THRESHOLD_GPUS
    template: dict[str, Any] = {
        "scope_type": "topology_domain",
        "scope_id_hash": site["scope_id_hash"],
        "scenario_family": None,
        "scenario_variant": None,
        "evidence_recipe_id": None,
        "temporal_phase": None,
        "data_quality_regime": None,
        "privacy_tier": None,
        "counterfactual_group_id": None,
        "synthetic_counterfactual_role": None,
        "collector_profile": None,
        "topology_class": None,
        "synthetic_hard_case_tags": None,
        "capacity_possible": capacity_possible,
        "policy_compute_ratio": 0.0,
        "o1_normalized_h100e_capacity": capacity,
        "o1_largest_contiguous_domain_gpus": site["largest_contiguous_domain_gpus"],
        "o1_homogeneous_high_end_fraction": site["homogeneous_high_end_fraction"],
        "o1_non_partitioned_fraction": site["non_partitioned_fraction"],
        "o1_inventory_delta_rate": 0.0,
        "o17_external_capacity_conflict_score": 0.0,
        "o2_max_concurrent_normalized_gpus": 0.0,
        "o2_allocation_duration_hours": duration_hours,
        "o2_gpu_hours_policy_ratio": 0.0,
        "o2_concurrency_fraction_domain": 0.0,
        "o2_topology_contiguity_score": 0.0,
        "o2_declared_workload_class": "none",
        "o2_reservation_exclusive_flag": False,
        "o2_elastic_resize_count": 0,
        "o2_preemption_restart_count": 0,
        "o2_scheduler_queue_delay_hours": 0.0,
        "o2_account_linkage_confidence": 1.0,
        "o2_job_array_width": 1,
        "o2_reservation_reuse_count": 0,
        "o3_batch_provisioned_gpus": None,
        "o3_capacity_reservation_duration_hours": None,
        "o3_training_sku_fraction": None,
        "o3_billing_continuity_score": None,
        "o3_egress_tb": None,
        "o4_gpu_util_p50": 0.0,
        "o4_gpu_util_p95": 0.0,
        "o4_gpu_util_duty_gt_70": 0.0,
        "o4_sm_tensor_active_p95": 0.0,
        "o4_hbm_used_fraction_p50": 0.0,
        "o4_hbm_bandwidth_active_p95": 0.0,
        "o4_gpu_power_fraction_p95": 0.18,
        "o4_error_spike_score": 0.0,
        "o4_gpu_util_cv": 0.0,
        "o4_gpu_idle_gap_p95_minutes": 0.0,
        "o4_hbm_pressure_duration_fraction": 0.0,
        "o4_power_cap_active_fraction": 0.0,
        "o4_thermal_throttle_fraction": 0.0,
        "o5_kernel_training_motif_score": None,
        "o5_tensor_throughput_ratio": None,
        "o5_profiler_available": False,
        "o6_nvlink_util_p95": 0.0,
        "o6_nvlink_periodicity_score": 0.0,
        "o6_link_error_spike_score": 0.0,
        "o7_scaleout_port_util_p95": 0.0,
        "o7_synchronized_fabric_footprint": 0,
        "o7_collective_periodicity_score": 0.0,
        "o7_burst_duty_cycle": 0.0,
        "o7_rdma_congestion_score": 0.0,
        "o7_job_to_port_mapping_coverage": 0.96,
        "o7_flow_entropy_score": 0.25,
        "o7_cross_section_sync_score": 0.0,
        "o7_collective_jitter_score": 0.0,
        "o7_storage_traffic_fraction": 0.0,
        "o7_inference_fanout_score": 0.0,
        "o7_account_flow_linkage_confidence": 1.0,
        "o8_rack_power_fraction_p95": 0.2,
        "o8_facility_it_power_mw": site["rack_power_design_mw"],
        "o8_baseline_subtracted_energy_kwh": 0.0,
        "o8_power_continuity_days": duration_hours / 24.0,
        "o8_power_cv": 0.18,
        "o8_power_to_gpu_residual": 0.0,
        "o8_power_baseline_drift_score": 0.0,
        "o8_power_cap_or_curtailment_active": False,
        "o8_unattributed_power_fraction": 0.0,
        "o9_gpu_hbm_temp_score": 0.08,
        "o9_thermal_delta_t_score": 0.1,
        "o9_cooling_flow_duty": 0.12,
        "o9_cooling_maintenance_active": False,
        "o9_thermal_throttle_support_score": 0.0,
        "o10_world_size": 0,
        "o10_runtime_framework_class": "none",
        "o10_rank_stability_score": 0.0,
        "o10_same_image_gpu_count": 0,
        "o10_rendezvous_present": False,
        "o10_runtime_metadata_confidence": 1.0,
        "o10_declared_vs_observed_mismatch_score": 0.0,
        "o11_data_staging_tb": 0.0,
        "o11_checkpoint_write_tb_per_event": 0.0,
        "o11_checkpoint_periodicity_score": 0.0,
        "o11_read_write_training_pattern_score": 0.0,
        "o11_checkpoint_jitter_score": 0.0,
        "o11_artifact_write_pattern_score": 0.0,
        "o11_dataloader_read_pattern_score": 0.0,
        "o11_backup_or_replication_pattern_score": 0.0,
        "o11_storage_cotraffic_score": 0.0,
        "o12_signed_ml_logs_present": False,
        "o12_declared_parameter_count_b": None,
        "o12_training_tokens_b": None,
        "o12_step_count": None,
        "o12_loss_curve_present": False,
        "o12_optimizer_state_present": False,
        "o12_log_delivery_delay_hours": 0.0,
        "o12_log_completeness_fraction": 0.0,
        "o12_declaration_consistency_score": 0.0,
        "o13_attestation_valid_fraction": 0.98,
        "o13_confidential_compute_mode_fraction": 0.0,
        "o13_collector_measurement_valid": True,
        "o14_min_critical_coverage": 0.98,
        "o14_gap_fraction_critical": 0.01,
        "o14_clock_drift_max_ms": int(_uniform(rng, 3, 80)),
        "o14_counter_reset_count": 0,
        "o15_unapproved_physical_change_near_window": False,
        "o15_firmware_bmc_change_near_window": False,
        "o16_probe_throughput_ratio_min": None,
        "o16_probe_latency_inflation_max": None,
        "o16_vram_residency_conflict_score": None,
        "o17_external_capacity_assertion": "consistent_with_internal_capacity",
        "o17_energy_contract_alignment_score": 0.0,
        "o17_network_provider_utilization_score": 0.0,
        "o17_procurement_or_maintenance_explanation_score": 0.0,
        "scheduler_signature_status": "valid",
        "gpu_telemetry_trust_level": site["trust_tier"],
        "fabric_telemetry_trust_level": site["trust_tier"],
        "power_meter_trust_level": "bms_meter",
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "policy_threshold_version": POLICY_THRESHOLD_VERSION,
        "hardware_normalization_version": HARDWARE_NORMALIZATION_VERSION,
        "raw_input_manifest_hash": None,
        "capacity_evidence_only": False,
        "integrity_evidence_only": False,
        "physical_evidence_only": False,
        "synthetic_evidence_profile": "none",
    }
    for obs_id in OBSERVABLE_IDS:
        key = obs_id.lower()
        template[f"{key}_coverage_fraction"] = 1.0
        template[f"{key}_missing_reason"] = "observed"

    if "cloud" not in site["site_type"]:
        template["o3_coverage_fraction"] = 0.0
        template["o3_missing_reason"] = "not_applicable"
    template["o5_coverage_fraction"] = 0.0
    template["o5_missing_reason"] = "routine_profiler_disabled"
    template["o12_coverage_fraction"] = 0.0
    template["o12_missing_reason"] = "privacy_redacted"
    template["o16_coverage_fraction"] = 0.0
    template["o16_missing_reason"] = "not_scheduled"
    return template


def set_activity(
    template: dict[str, Any],
    site: dict[str, Any],
    *,
    gpus: float,
    duration_hours: float,
    util_p50: float,
    util_p95: float,
    duty: float,
    tensor: float,
    hbm: float,
    fabric: float,
    periodicity: float,
    contiguity: float,
    declared: str,
) -> None:
    capacity = max(1.0, float(site["normalized_h100e_capacity"]))
    gpus = min(gpus, capacity)
    gpu_hours_ratio = gpus * duration_hours / POLICY_GPU_HOURS_THRESHOLD
    template["policy_compute_ratio"] = _round(gpu_hours_ratio)
    template["o2_max_concurrent_normalized_gpus"] = _round(gpus, 2)
    template["o2_allocation_duration_hours"] = _round(duration_hours, 2)
    template["o2_gpu_hours_policy_ratio"] = _round(gpu_hours_ratio)
    template["o2_concurrency_fraction_domain"] = _round(gpus / capacity)
    template["o2_topology_contiguity_score"] = _round(contiguity)
    template["o2_declared_workload_class"] = declared
    template["o2_reservation_exclusive_flag"] = gpus >= 512 and duration_hours >= 24
    template["o2_account_linkage_confidence"] = max(template.get("o2_account_linkage_confidence", 0.0), _round(contiguity))
    template["o4_gpu_util_p50"] = _round(util_p50, 2)
    template["o4_gpu_util_p95"] = _round(util_p95, 2)
    template["o4_gpu_util_duty_gt_70"] = _round(duty)
    template["o4_sm_tensor_active_p95"] = _round(tensor, 2)
    template["o4_hbm_used_fraction_p50"] = _round(hbm)
    template["o4_hbm_bandwidth_active_p95"] = _round(clamp((tensor / 100.0) * 0.9 + hbm * 0.25))
    template["o4_gpu_power_fraction_p95"] = _round(clamp(0.17 + util_p95 / 115.0))
    template["o4_gpu_util_cv"] = _round(clamp(0.42 - duty * 0.28 + abs(util_p95 - util_p50) / 220.0))
    template["o4_gpu_idle_gap_p95_minutes"] = _round(max(0.0, (1.0 - duty) * 45.0), 2)
    template["o4_hbm_pressure_duration_fraction"] = _round(clamp(max(0.0, hbm - 0.25) * duty * 1.15))
    template["o6_nvlink_util_p95"] = _round(clamp(fabric * 0.85))
    template["o6_nvlink_periodicity_score"] = _round(clamp(periodicity * 0.92))
    template["o7_scaleout_port_util_p95"] = _round(clamp(fabric))
    template["o7_synchronized_fabric_footprint"] = int(max(0, min(gpus * 0.9, gpus * periodicity * 1.15)))
    template["o7_collective_periodicity_score"] = _round(clamp(periodicity))
    template["o7_burst_duty_cycle"] = _round(clamp(0.08 + periodicity * 0.35))
    template["o7_rdma_congestion_score"] = _round(clamp(fabric * periodicity * 0.75))
    template["o7_cross_section_sync_score"] = _round(clamp(periodicity * contiguity))
    template["o7_collective_jitter_score"] = _round(clamp(0.4 - periodicity * 0.28 + template["o4_gpu_util_cv"] * 0.3))
    template["o7_flow_entropy_score"] = _round(clamp(0.28 + fabric * 0.45 - periodicity * 0.2))
    template["o7_account_flow_linkage_confidence"] = max(template.get("o7_account_flow_linkage_confidence", 0.0), _round(contiguity))
    template["o8_rack_power_fraction_p95"] = _round(clamp(0.22 + (gpus / capacity) * (util_p95 / 100.0) * 0.78))
    if util_p95 >= 80:
        template["o8_rack_power_fraction_p95"] = max(template["o8_rack_power_fraction_p95"], 0.45)
    elif util_p95 >= 70:
        template["o8_rack_power_fraction_p95"] = max(template["o8_rack_power_fraction_p95"], 0.38)
    template["o8_power_cv"] = _round(max(0.02, 0.22 - duty * 0.16))
    expected_power = template["o4_gpu_power_fraction_p95"] * template["o2_concurrency_fraction_domain"]
    template["o8_power_to_gpu_residual"] = _round(template["o8_rack_power_fraction_p95"] - expected_power)
    template["o9_gpu_hbm_temp_score"] = _round(clamp(0.15 + template["o4_gpu_power_fraction_p95"] * 0.72))
    template["o9_thermal_delta_t_score"] = _round(clamp(0.12 + template["o8_rack_power_fraction_p95"] * 0.78))
    template["o9_cooling_flow_duty"] = _round(clamp(0.18 + template["o8_rack_power_fraction_p95"] * 0.7))
    template["o10_world_size"] = int(gpus) if gpus >= 2 else 1 if gpus > 0 else 0
    template["o10_same_image_gpu_count"] = int(gpus)
    template["o10_rank_stability_score"] = _round(clamp(0.25 + contiguity * 0.7))
    template["o10_rendezvous_present"] = gpus >= 16 and periodicity >= 0.35


def apply_scenario_features(
    site: dict[str, Any],
    scenario: str,
    duration_hours: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    template = make_base_template(site, scenario, duration_hours, rng)
    capacity = float(site["normalized_h100e_capacity"])

    def sample_gpus(low: float, high: float) -> float:
        return min(capacity, _log_uniform(rng, low, max(low, high)))

    if scenario == "idle":
        template["synthetic_evidence_profile"] = "none"
        template["o2_allocation_duration_hours"] = 0.0
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.12, 0.28))
        template["o8_power_cv"] = _round(_uniform(rng, 0.12, 0.24))
    elif scenario == "normal_inference":
        gpus = sample_gpus(4, min(256, max(8, capacity * 0.2)))
        set_activity(
            template,
            site,
            gpus=gpus,
            duration_hours=duration_hours,
            util_p50=_uniform(rng, 22, 56),
            util_p95=_uniform(rng, 45, 76),
            duty=_uniform(rng, 0.1, 0.38),
            tensor=_uniform(rng, 22, 58),
            hbm=_uniform(rng, 0.22, 0.58),
            fabric=_uniform(rng, 0.02, 0.18),
            periodicity=_uniform(rng, 0.0, 0.18),
            contiguity=_uniform(rng, 0.3, 0.72),
            declared="inference",
        )
        template["o10_runtime_framework_class"] = _choice(rng, ["vllm_inference", "tensorrt_llm_serving", "ray_batch_inference"])
    elif scenario == "large_batch_inference":
        gpus = sample_gpus(128, min(1024, max(128, capacity * 0.55)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=68, util_p95=88, duty=0.62, tensor=72, hbm=0.62, fabric=0.22, periodicity=0.15, contiguity=0.78, declared="inference")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "large_batch_inference"
        template["o11_data_staging_tb"] = _round(_uniform(rng, 10, 80), 2)
    elif scenario == "synthetic_data_generation":
        gpus = sample_gpus(128, min(1536, max(128, capacity * 0.6)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=72, util_p95=91, duty=0.72, tensor=80, hbm=0.7, fabric=0.35, periodicity=0.25, contiguity=0.75, declared="synthetic_data")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "synthetic_data_generation_pipeline"
        template["o11_data_staging_tb"] = _round(_uniform(rng, 80, 420), 2)
        template["o11_read_write_training_pattern_score"] = 0.28
    elif scenario == "small_fine_tune":
        gpus = sample_gpus(8, min(192, max(8, capacity * 0.25)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=68, util_p95=89, duty=0.7, tensor=78, hbm=0.68, fabric=0.32, periodicity=0.52, contiguity=0.82, declared="fine_tune")
        template["o10_runtime_framework_class"] = "pytorch_distributed_training"
        template["o11_checkpoint_periodicity_score"] = 0.48
        template["o11_checkpoint_write_tb_per_event"] = _round(_uniform(rng, 0.03, 0.22), 3)
        template["o12_coverage_fraction"] = 0.55
        template["o12_missing_reason"] = "delayed_log_delivery"
        template["o12_signed_ml_logs_present"] = bool(rng.random() < 0.35)
        template["o12_declared_parameter_count_b"] = _round(_uniform(rng, 1, 12), 2)
        template["o12_training_tokens_b"] = _round(_uniform(rng, 1, 25), 2)
        template["o12_step_count"] = int(_uniform(rng, 500, 6000))
        template["o12_loss_curve_present"] = True
        template["o12_optimizer_state_present"] = True
    elif scenario in {"large_fine_tune", "pretraining", "cloud_reservation_used_for_training"}:
        low = 512 if scenario != "pretraining" else 1536
        high = capacity if scenario == "pretraining" else min(capacity, 3072)
        gpus = sample_gpus(low, max(low, high))
        util = _uniform(rng, 78, 92)
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=util, util_p95=min(98, util + 9), duty=_uniform(rng, 0.78, 0.94), tensor=_uniform(rng, 82, 96), hbm=_uniform(rng, 0.72, 0.9), fabric=_uniform(rng, 0.68, 0.9), periodicity=_uniform(rng, 0.72, 0.94), contiguity=_uniform(rng, 0.82, 0.97), declared="train")
        template["synthetic_evidence_profile"] = "training_primary_semantic"
        template["o10_runtime_framework_class"] = "pytorch_distributed_training"
        template["o11_data_staging_tb"] = _round(_uniform(rng, 80, 800), 2)
        template["o11_checkpoint_write_tb_per_event"] = _round(_uniform(rng, 0.25, 2.4), 3)
        template["o11_checkpoint_periodicity_score"] = _round(_uniform(rng, 0.62, 0.92))
        template["o11_read_write_training_pattern_score"] = _round(_uniform(rng, 0.62, 0.94))
        if scenario == "pretraining" or rng.random() < 0.6:
            template["o12_coverage_fraction"] = _round(_uniform(rng, 0.65, 0.95))
            template["o12_missing_reason"] = "delayed_log_delivery"
            signed_probability = 0.62 if scenario in {"pretraining", "cloud_reservation_used_for_training"} else 0.2
            template["o12_signed_ml_logs_present"] = bool(rng.random() < signed_probability)
            template["o12_declared_parameter_count_b"] = _round(_uniform(rng, 35, 450), 2)
            template["o12_training_tokens_b"] = _round(_uniform(rng, 120, 4500), 2)
            template["o12_step_count"] = int(_uniform(rng, 12000, 650000))
            template["o12_loss_curve_present"] = True
            template["o12_optimizer_state_present"] = True
        if scenario == "cloud_reservation_used_for_training":
            template["o3_coverage_fraction"] = 1.0
            template["o3_missing_reason"] = "observed"
            template["o3_batch_provisioned_gpus"] = int(gpus)
            template["o3_capacity_reservation_duration_hours"] = _round(duration_hours, 2)
            template["o3_training_sku_fraction"] = _round(_uniform(rng, 0.82, 1.0))
            template["o3_billing_continuity_score"] = _round(_uniform(rng, 0.88, 0.99))
            template["o3_egress_tb"] = _round(_uniform(rng, 10, 120), 2)
    elif scenario == "hpc_mpi_simulation":
        gpus = sample_gpus(128, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=70, util_p95=92, duty=0.68, tensor=42, hbm=0.46, fabric=0.72, periodicity=0.58, contiguity=0.86, declared="hpc")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "mpi_hpc_simulation"
        template["o11_checkpoint_periodicity_score"] = 0.08
    elif scenario == "nccl_benchmark":
        gpus = sample_gpus(64, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=58, util_p95=96, duty=0.52, tensor=84, hbm=0.4, fabric=0.92, periodicity=0.86, contiguity=0.92, declared="benchmark")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "nccl_benchmark"
    elif scenario == "hardware_burn_in":
        gpus = sample_gpus(128, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=82, util_p95=98, duty=0.9, tensor=50, hbm=0.35, fabric=0.1, periodicity=0.05, contiguity=0.55, declared="burn_in")
        template["synthetic_evidence_profile"] = "physical_only" if gpus < 512 else "false_positive_primary"
        template["o10_runtime_framework_class"] = "hardware_burn_in"
        template["o4_error_spike_score"] = _round(_uniform(rng, 0.15, 0.48))
    elif scenario == "storage_rebuild":
        template["synthetic_evidence_profile"] = "physical_only"
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.42, 0.64))
        template["o11_data_staging_tb"] = _round(_uniform(rng, 300, 1800), 2)
        template["o11_read_write_training_pattern_score"] = _round(_uniform(rng, 0.05, 0.18))
        template["o10_runtime_framework_class"] = "storage_rebuild"
    elif scenario == "large_etl_data_movement":
        gpus = sample_gpus(0.0 + 1, min(128, max(1, capacity * 0.1)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=22, util_p95=48, duty=0.08, tensor=12, hbm=0.28, fabric=0.18, periodicity=0.05, contiguity=0.25, declared="data")
        template["synthetic_evidence_profile"] = "physical_only"
        template["o11_data_staging_tb"] = _round(_uniform(rng, 500, 4000), 2)
        template["o3_egress_tb"] = _round(_uniform(rng, 50, 600), 2)
        template["o10_runtime_framework_class"] = "etl_data_pipeline"
    elif scenario == "reserved_but_unused_capacity":
        gpus = sample_gpus(512, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=3, util_p95=12, duty=0.0, tensor=2, hbm=0.08, fabric=0.03, periodicity=0.0, contiguity=0.86, declared="reserved")
        template["synthetic_evidence_profile"] = "capacity_only"
        template["o2_reservation_exclusive_flag"] = True
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.18, 0.32))
    elif scenario == "maintenance_window":
        template["synthetic_evidence_profile"] = "integrity_only"
        template["o14_min_critical_coverage"] = _round(_uniform(rng, 0.48, 0.82))
        template["o14_gap_fraction_critical"] = _round(1.0 - template["o14_min_critical_coverage"])
        template["o14_counter_reset_count"] = int(_uniform(rng, 1, 5))
        template["o15_unapproved_physical_change_near_window"] = bool(rng.random() < 0.12)
        template["o15_firmware_bmc_change_near_window"] = True
        for obs in ["o4", "o7", "o14"]:
            template[f"{obs}_coverage_fraction"] = template["o14_min_critical_coverage"]
            template[f"{obs}_missing_reason"] = "collector_gap"
    elif scenario == "adversarial_fragmented_training":
        gpus = sample_gpus(192, min(capacity, 1024))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=76, util_p95=91, duty=0.78, tensor=86, hbm=0.74, fabric=0.48, periodicity=0.55, contiguity=0.42, declared="unknown")
        template["synthetic_evidence_profile"] = "training_fragmented"
        template["o17_external_capacity_conflict_score"] = _round(_uniform(rng, 0.2, 0.55))
        template["o10_runtime_framework_class"] = "pytorch_distributed_training"
        template["o11_checkpoint_periodicity_score"] = _round(_uniform(rng, 0.45, 0.7))
        template["o11_checkpoint_write_tb_per_event"] = _round(_uniform(rng, 0.1, 0.8), 3)
    elif scenario == "underclocked_long_duration_training":
        gpus = sample_gpus(512, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=58, util_p95=76, duty=0.62, tensor=72, hbm=0.7, fabric=0.55, periodicity=0.7, contiguity=0.86, declared="unknown")
        template["synthetic_evidence_profile"] = "training_primary_semantic"
        template["o10_runtime_framework_class"] = "pytorch_distributed_training"
        template["o8_power_to_gpu_residual"] = _round(_uniform(rng, -0.08, 0.03))
        template["o11_checkpoint_periodicity_score"] = _round(_uniform(rng, 0.5, 0.78))
        template["o11_checkpoint_write_tb_per_event"] = _round(_uniform(rng, 0.15, 1.2), 3)
    elif scenario == "counter_suppressed_candidate_window":
        gpus = sample_gpus(512, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=0, util_p95=0, duty=0.0, tensor=0, hbm=0.0, fabric=0.68, periodicity=0.72, contiguity=0.88, declared="unknown")
        template["synthetic_evidence_profile"] = "training_suppressed_counters"
        template["o4_gpu_util_p50"] = None
        template["o4_gpu_util_p95"] = None
        template["o4_gpu_util_duty_gt_70"] = None
        template["o4_sm_tensor_active_p95"] = None
        template["o4_hbm_used_fraction_p50"] = None
        template["o4_hbm_bandwidth_active_p95"] = None
        template["o4_gpu_power_fraction_p95"] = None
        template["o4_coverage_fraction"] = _round(_uniform(rng, 0.05, 0.22))
        template["o4_missing_reason"] = "counter_disabled_by_cc_mode"
        template["o13_confidential_compute_mode_fraction"] = _round(_uniform(rng, 0.75, 1.0))
        template["o14_min_critical_coverage"] = min(template["o14_min_critical_coverage"], template["o4_coverage_fraction"])
        template["o14_gap_fraction_critical"] = _round(1.0 - template["o14_min_critical_coverage"])
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.68, 0.88))
        template["o8_power_to_gpu_residual"] = _round(_uniform(rng, 0.22, 0.5))
        template["o10_runtime_framework_class"] = "unknown_cc_mode"
        template["o16_coverage_fraction"] = 1.0
        template["o16_missing_reason"] = "observed"
        template["o16_probe_throughput_ratio_min"] = _round(_uniform(rng, 0.45, 0.78))
        template["o16_probe_latency_inflation_max"] = _round(_uniform(rng, 1.2, 2.5))
        template["o16_vram_residency_conflict_score"] = _round(_uniform(rng, 0.3, 0.72))

    # Make profiler occasionally available in candidate and false-positive high-compute cases.
    if scenario in {
        "pretraining",
        "large_fine_tune",
        "cloud_reservation_used_for_training",
        "hpc_mpi_simulation",
        "nccl_benchmark",
    } and rng.random() < 0.45:
        template["o5_profiler_available"] = True
        template["o5_coverage_fraction"] = _round(_uniform(rng, 0.35, 0.8))
        template["o5_missing_reason"] = "privacy_redacted"
        if scenario in TRAINING_SCENARIOS:
            template["o5_kernel_training_motif_score"] = _round(_uniform(rng, 0.58, 0.9))
            template["o5_tensor_throughput_ratio"] = _round(_uniform(rng, 0.55, 0.9))
        else:
            template["o5_kernel_training_motif_score"] = _round(_uniform(rng, 0.22, 0.55))
            template["o5_tensor_throughput_ratio"] = _round(_uniform(rng, 0.2, 0.65))

    if template["o2_max_concurrent_normalized_gpus"] > 0:
        template["scope_type"] = "linked_job_group" if scenario in TRAINING_SCENARIOS else "topology_domain"
        template["scope_id_hash"] = ("jobgrp_hmac_" if template["scope_type"] == "linked_job_group" else "fabricdom_hmac_") + stable_hash(
            site["site_id"], scenario, duration_hours, template["o2_max_concurrent_normalized_gpus"], length=12
        )

    return template


def _v1_variant(rng: np.random.Generator, family: str) -> str:
    variants = {
        "pretraining_standard": ["full_stack_signed", "unsigned_full_stack", "evaluation_phase_mixed"],
        "large_fine_tune_standard": ["adapter_or_full_finetune", "checkpoint_heavy", "moderate_scale_signed"],
        "cloud_training_redacted_runtime": ["runtime_redacted", "hashed_container_only", "customer_privacy_tier"],
        "training_without_semantic_logs": ["no_ml_logs", "storage_and_fabric_only", "runtime_low_confidence"],
        "underclocked_energy_capped_training": ["energy_capped_long", "demand_response_overlap", "weak_early_window"],
        "elastic_preempted_training": ["elastic_resize", "preempted_restart", "gang_delay_then_train"],
        "fragmented_training_linked": ["weak_linkage", "strong_linkage", "cross_account_group"],
        "sparse_or_moe_bursty_training": ["moe_bursty", "sparse_expert_parallel", "idle_gaps_high_hbm"],
        "training_with_low_fabric_high_checkpoint": ["pipeline_parallel_low_scaleout", "checkpoint_dominant", "single_domain_training"],
        "training_with_delayed_logs": ["logs_delayed_6h", "logs_delayed_36h", "late_signed_declaration"],
        "multi_stage_training_pipeline": ["staging_then_train", "train_eval_export", "recovery_then_steady"],
        "idle_or_low_activity": ["fully_observed_idle", "capable_quiet", "small_background_jobs"],
        "normal_inference": ["steady_serving", "autoscaled_serving", "low_collective_serving"],
        "large_batch_inference": ["batch_decode", "offline_scoring", "large_object_reads"],
        "model_parallel_inference": ["high_local_fabric", "long_context_serving", "multi_node_inference"],
        "embedding_generation": ["high_util_low_collective", "batch_embedding", "vector_index_build"],
        "synthetic_data_generation_gpu_heavy": ["generation_pipeline", "sampling_heavy", "teacher_model_serving"],
        "hpc_mpi_collective": ["allreduce_simulation", "climate_or_cfd", "mpi_checkpoint_light"],
        "nccl_extended_benchmark": ["burnin_collectives", "fabric_validation", "short_benchmark"],
        "hardware_burn_in_or_thermal_soak": ["thermal_soak", "power_validation", "post_rma_burnin"],
        "storage_rebuild_or_replication": ["object_replication", "raid_rebuild", "backup_wave"],
        "large_etl_or_data_movement": ["gpu_light_etl", "object_store_migration", "shuffle_heavy"],
        "distributed_database_or_graph_analytics": ["graph_analytics", "distributed_join", "vector_database_build"],
        "reserved_but_unused_capacity": ["exclusive_unused", "reservation_reused_later", "capacity_block_idle"],
        "maintenance_with_collector_gaps": ["collector_gap_short", "collector_gap_long", "cooling_maintenance"],
        "multi_tenant_fragmented_nontraining": ["many_small_jobs", "shared_topology_overlap", "fragmented_batch_inference"],
        "counter_suppressed_candidate_window": ["cc_mode_counters_disabled", "gpu_counters_hidden", "probe_conflict"],
        "capacity_or_integrity_only_warning": ["capacity_external_conflict", "integrity_only_reset", "offledger_capacity_hint"],
    }
    return _choice(rng, variants.get(family, ["default"]))


def _set_v1_metadata(
    template: dict[str, Any],
    rng: np.random.Generator,
    *,
    family: str,
    variant: str | None = None,
    data_quality_regime: str | None = None,
    privacy_tier: str | None = None,
    counterfactual_group_id: str | None = None,
    counterfactual_role: str | None = None,
    hard_case_tags: list[str] | None = None,
) -> None:
    template["scenario_family"] = family
    template["scenario_variant"] = variant or _v1_variant(rng, family)
    template["evidence_recipe_id"] = "recipe_" + stable_hash(family, template["scenario_variant"], length=12)
    template["temporal_phase"] = "episode_summary"
    template["data_quality_regime"] = data_quality_regime or _choice(
        rng,
        [
            "full_observability",
            "full_observability",
            "routine_privacy_redaction",
            "delayed_log_delivery",
            "partial_fabric_mapping",
            "clock_drift_episode",
        ],
    )
    template["privacy_tier"] = privacy_tier or _choice(rng, PRIVACY_TIERS)
    template["counterfactual_group_id"] = counterfactual_group_id
    template["synthetic_counterfactual_role"] = counterfactual_role
    template["collector_profile"] = _choice(rng, COLLECTOR_PROFILES)
    template["topology_class"] = _choice(rng, TOPOLOGY_CLASSES)
    tags = list(hard_case_tags or [])
    if family in V1_HARD_POSITIVE_FAMILIES:
        tags.append("hard_positive")
    if family in V1_HARD_NEGATIVE_FAMILIES:
        tags.append("hard_negative")
    template["synthetic_hard_case_tags"] = ";".join(sorted(set(tags))) if tags else "none"


def _set_training_storage(template: dict[str, Any], rng: np.random.Generator, *, checkpoint: float, artifact: float, dataloader: float) -> None:
    template["o11_data_staging_tb"] = _round(_uniform(rng, 80, 1200), 2)
    template["o11_checkpoint_write_tb_per_event"] = _round(_uniform(rng, 0.15, 2.8), 3)
    template["o11_checkpoint_periodicity_score"] = _round(checkpoint)
    template["o11_read_write_training_pattern_score"] = _round(max(checkpoint, dataloader) * _uniform(rng, 0.85, 1.05))
    template["o11_checkpoint_jitter_score"] = _round(_uniform(rng, 0.04, 0.28))
    template["o11_artifact_write_pattern_score"] = _round(artifact)
    template["o11_dataloader_read_pattern_score"] = _round(dataloader)
    template["o11_storage_cotraffic_score"] = _round(_uniform(rng, 0.04, 0.24))
    template["o11_backup_or_replication_pattern_score"] = _round(_uniform(rng, 0.0, 0.18))


def _set_training_semantics(
    template: dict[str, Any],
    rng: np.random.Generator,
    *,
    signed: bool,
    redacted_runtime: bool = False,
    delayed_logs_hours: float = 0.0,
    completeness: float = 0.0,
) -> None:
    template["o10_runtime_framework_class"] = "pytorch_distributed_training"
    template["o10_runtime_metadata_confidence"] = _round(0.25 if redacted_runtime else _uniform(rng, 0.65, 0.95))
    if redacted_runtime:
        template["o10_missing_reason"] = "privacy_redacted"
        template["o10_coverage_fraction"] = _round(_uniform(rng, 0.25, 0.55))
    template["o12_coverage_fraction"] = _round(max(completeness, 0.15 if signed else 0.0))
    template["o12_missing_reason"] = (
        "observed"
        if signed and completeness >= 0.98
        else "delayed_log_delivery"
        if signed or delayed_logs_hours
        else "privacy_redacted"
    )
    template["o12_signed_ml_logs_present"] = signed
    template["o12_declared_parameter_count_b"] = _round(_uniform(rng, 35, 650), 2) if signed or completeness > 0 else None
    template["o12_training_tokens_b"] = _round(_uniform(rng, 80, 5200), 2) if signed or completeness > 0 else None
    template["o12_step_count"] = int(_uniform(rng, 8000, 750000)) if signed or completeness > 0 else None
    template["o12_loss_curve_present"] = signed or completeness >= 0.45
    template["o12_optimizer_state_present"] = signed or completeness >= 0.35
    template["o12_log_delivery_delay_hours"] = _round(delayed_logs_hours, 2)
    template["o12_log_completeness_fraction"] = _round(completeness if completeness else (0.82 if signed else 0.0))
    template["o12_declaration_consistency_score"] = _round(_uniform(rng, 0.75, 0.98) if signed else _uniform(rng, 0.0, 0.35))


def _apply_data_quality_regime(template: dict[str, Any], rng: np.random.Generator) -> None:
    regime = template.get("data_quality_regime")
    if regime == "full_observability":
        return
    if regime == "routine_privacy_redaction":
        for obs in ["o5", "o10", "o12"]:
            template[f"{obs}_coverage_fraction"] = min(float(template.get(f"{obs}_coverage_fraction") or 1.0), _round(_uniform(rng, 0.25, 0.75)))
            template[f"{obs}_missing_reason"] = "privacy_redacted"
        template["o10_runtime_metadata_confidence"] = min(float(template.get("o10_runtime_metadata_confidence") or 1.0), 0.55)
    elif regime == "confidential_compute_counters_disabled":
        for obs in ["o4", "o5"]:
            template[f"{obs}_coverage_fraction"] = _round(_uniform(rng, 0.08, 0.32))
            template[f"{obs}_missing_reason"] = "counter_disabled_by_cc_mode"
        template["o13_confidential_compute_mode_fraction"] = _round(_uniform(rng, 0.58, 1.0))
    elif regime in {"collector_gap_short", "collector_gap_long", "maintenance_observability_loss"}:
        low, high = (0.62, 0.84) if regime == "collector_gap_short" else (0.28, 0.62)
        if regime == "maintenance_observability_loss":
            low, high = (0.22, 0.55)
        coverage = _round(_uniform(rng, low, high))
        for obs in ["o4", "o7", "o8", "o14"]:
            template[f"{obs}_coverage_fraction"] = min(float(template.get(f"{obs}_coverage_fraction") or 1.0), coverage)
            template[f"{obs}_missing_reason"] = "collector_gap"
        template["o14_min_critical_coverage"] = min(float(template["o14_min_critical_coverage"]), coverage)
        template["o14_gap_fraction_critical"] = _round(max(float(template["o14_gap_fraction_critical"]), 1.0 - coverage))
        template["o14_counter_reset_count"] = max(int(template.get("o14_counter_reset_count") or 0), int(_uniform(rng, 1, 4)))
    elif regime == "delayed_log_delivery":
        template["o12_coverage_fraction"] = min(float(template.get("o12_coverage_fraction") or 1.0), _round(_uniform(rng, 0.2, 0.65)))
        template["o12_missing_reason"] = "delayed_log_delivery"
        template["o12_log_delivery_delay_hours"] = max(float(template.get("o12_log_delivery_delay_hours") or 0.0), _round(_uniform(rng, 4, 48), 2))
    elif regime == "clock_drift_episode":
        template["o14_clock_drift_max_ms"] = int(_uniform(rng, 700, 4500))
        template["o14_gap_fraction_critical"] = max(float(template["o14_gap_fraction_critical"]), _round(_uniform(rng, 0.04, 0.12)))
    elif regime == "source_not_deployed_for_site":
        for obs in ["o5", "o12", "o16"]:
            template[f"{obs}_coverage_fraction"] = 0.0
            template[f"{obs}_missing_reason"] = "source_not_deployed"
    elif regime == "partial_fabric_mapping":
        template["o7_job_to_port_mapping_coverage"] = _round(_uniform(rng, 0.35, 0.68))
        template["o7_coverage_fraction"] = min(float(template.get("o7_coverage_fraction") or 1.0), _round(_uniform(rng, 0.55, 0.82)))
        template["o7_missing_reason"] = "privacy_redacted"
    template["o14_min_critical_coverage"] = min(
        float(template["o14_min_critical_coverage"]),
        float(template.get("o2_coverage_fraction") or 1.0),
        float(template.get("o4_coverage_fraction") or 1.0),
        float(template.get("o7_coverage_fraction") or 1.0),
        float(template.get("o8_coverage_fraction") or 1.0),
    )


def apply_v1_scenario_features(
    site: dict[str, Any],
    family: str,
    duration_hours: float,
    rng: np.random.Generator,
    *,
    variant: str | None = None,
    counterfactual_group_id: str | None = None,
    counterfactual_role: str | None = None,
) -> dict[str, Any]:
    template = make_base_template(site, family, duration_hours, rng)
    variant = variant or _v1_variant(rng, family)
    capacity = float(site["normalized_h100e_capacity"])

    def sample_gpus(low: float, high: float) -> float:
        return min(capacity, _log_uniform(rng, max(1.0, low), max(low, high)))

    tags: list[str] = []
    data_quality = None
    privacy = None

    if family == "idle_or_low_activity":
        tags.extend(["negative_control"])
        template["o2_allocation_duration_hours"] = 0.0
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.12, 0.30))
        template["o8_power_cv"] = _round(_uniform(rng, 0.10, 0.24))
    elif family == "normal_inference":
        gpus = sample_gpus(4, min(192, max(8, capacity * 0.18)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 24, 54), util_p95=_uniform(rng, 48, 76), duty=_uniform(rng, 0.12, 0.35), tensor=_uniform(rng, 24, 58), hbm=_uniform(rng, 0.25, 0.58), fabric=_uniform(rng, 0.02, 0.16), periodicity=_uniform(rng, 0.01, 0.16), contiguity=_uniform(rng, 0.30, 0.70), declared="inference")
        template["o10_runtime_framework_class"] = _choice(rng, ["vllm_inference", "tensorrt_llm_serving", "ray_batch_inference"])
        template["o7_inference_fanout_score"] = _round(_uniform(rng, 0.45, 0.88))
    elif family == "large_batch_inference":
        gpus = sample_gpus(96, min(1536, max(128, capacity * 0.55)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 60, 78), util_p95=_uniform(rng, 78, 94), duty=_uniform(rng, 0.45, 0.70), tensor=_uniform(rng, 62, 82), hbm=_uniform(rng, 0.55, 0.78), fabric=_uniform(rng, 0.12, 0.32), periodicity=_uniform(rng, 0.05, 0.24), contiguity=_uniform(rng, 0.55, 0.84), declared="inference")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "large_batch_inference"
        template["o7_inference_fanout_score"] = _round(_uniform(rng, 0.68, 0.95))
        template["o11_data_staging_tb"] = _round(_uniform(rng, 40, 500), 2)
        tags.extend(["hard_high_load_negative", "inference_counterpart"])
    elif family == "model_parallel_inference":
        gpus = sample_gpus(64, min(1024, max(128, capacity * 0.45)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 55, 76), util_p95=_uniform(rng, 74, 92), duty=_uniform(rng, 0.34, 0.62), tensor=_uniform(rng, 58, 78), hbm=_uniform(rng, 0.65, 0.88), fabric=_uniform(rng, 0.44, 0.72), periodicity=_uniform(rng, 0.22, 0.48), contiguity=_uniform(rng, 0.72, 0.94), declared="inference")
        template["o10_runtime_framework_class"] = "model_parallel_inference"
        template["o7_inference_fanout_score"] = _round(_uniform(rng, 0.72, 0.96))
        template["o11_dataloader_read_pattern_score"] = _round(_uniform(rng, 0.05, 0.24))
        tags.extend(["hard_high_load_negative", "fabric_overlap"])
    elif family in {"embedding_generation", "synthetic_data_generation_gpu_heavy"}:
        lighter_variant = variant in {"vector_index_build", "generation_pipeline"}
        if lighter_variant:
            low_gpus, high_gpus = (8, min(48, max(16, capacity * 0.08))) if family.endswith("heavy") else (24, min(160, max(32, capacity * 0.16)))
            gpus = sample_gpus(low_gpus, high_gpus)
            set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 24, 44), util_p95=_uniform(rng, 38, 62), duty=_uniform(rng, 0.08, 0.28), tensor=_uniform(rng, 24, 52), hbm=_uniform(rng, 0.22, 0.50), fabric=_uniform(rng, 0.02, 0.14), periodicity=_uniform(rng, 0.0, 0.10), contiguity=_uniform(rng, 0.22, 0.52), declared="synthetic_data" if family.endswith("heavy") else "embedding")
        else:
            gpus = sample_gpus(128, min(1536, max(128, capacity * 0.60)))
            set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 62, 82), util_p95=_uniform(rng, 82, 96), duty=_uniform(rng, 0.52, 0.78), tensor=_uniform(rng, 72, 90), hbm=_uniform(rng, 0.62, 0.82), fabric=_uniform(rng, 0.18, 0.42), periodicity=_uniform(rng, 0.10, 0.30), contiguity=_uniform(rng, 0.55, 0.82), declared="synthetic_data" if family.endswith("heavy") else "embedding")
        template["o10_runtime_framework_class"] = "synthetic_data_generation_pipeline" if family.endswith("heavy") else "embedding_generation"
        template["o11_data_staging_tb"] = _round(_uniform(rng, 20, 240), 2) if lighter_variant else _round(_uniform(rng, 120, 1200), 2)
        template["o11_dataloader_read_pattern_score"] = _round(_uniform(rng, 0.05, 0.22)) if lighter_variant else _round(_uniform(rng, 0.22, 0.48))
        tags.extend(["ml_adjacent_negative"] if lighter_variant else ["hard_high_load_negative", "gpu_counterpart"])
    elif family == "hpc_mpi_collective":
        lighter_variant = variant == "mpi_checkpoint_light"
        if lighter_variant:
            gpus = sample_gpus(32, min(192, max(64, capacity * 0.18)))
            set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 30, 50), util_p95=_uniform(rng, 46, 66), duty=_uniform(rng, 0.14, 0.34), tensor=_uniform(rng, 18, 42), hbm=_uniform(rng, 0.20, 0.42), fabric=_uniform(rng, 0.12, 0.32), periodicity=_uniform(rng, 0.08, 0.32), contiguity=_uniform(rng, 0.48, 0.70), declared="hpc")
        else:
            gpus = sample_gpus(128, min(capacity, 3072))
            set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 64, 82), util_p95=_uniform(rng, 82, 96), duty=_uniform(rng, 0.55, 0.78), tensor=_uniform(rng, 35, 62), hbm=_uniform(rng, 0.35, 0.62), fabric=_uniform(rng, 0.62, 0.90), periodicity=_uniform(rng, 0.50, 0.82), contiguity=_uniform(rng, 0.78, 0.95), declared="hpc")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "mpi_hpc_simulation"
        template["o11_checkpoint_periodicity_score"] = _round(_uniform(rng, 0.02, 0.18))
        tags.extend(["hpc_counterpart"] if lighter_variant else ["hard_high_load_negative", "fabric_overlap"])
    elif family == "nccl_extended_benchmark":
        gpus = sample_gpus(64, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 52, 76), util_p95=_uniform(rng, 88, 98), duty=_uniform(rng, 0.35, 0.62), tensor=_uniform(rng, 74, 94), hbm=_uniform(rng, 0.32, 0.55), fabric=_uniform(rng, 0.78, 0.96), periodicity=_uniform(rng, 0.72, 0.94), contiguity=_uniform(rng, 0.82, 0.98), declared="benchmark")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "nccl_benchmark"
        tags.extend(["hard_high_load_negative", "fabric_overlap"])
    elif family == "hardware_burn_in_or_thermal_soak":
        gpus = sample_gpus(128, min(capacity, 2048))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 74, 88), util_p95=_uniform(rng, 90, 99), duty=_uniform(rng, 0.70, 0.94), tensor=_uniform(rng, 35, 58), hbm=_uniform(rng, 0.25, 0.45), fabric=_uniform(rng, 0.04, 0.18), periodicity=_uniform(rng, 0.0, 0.12), contiguity=_uniform(rng, 0.45, 0.70), declared="burn_in")
        template["synthetic_evidence_profile"] = "false_positive_primary"
        template["o10_runtime_framework_class"] = "hardware_burn_in"
        template["o4_error_spike_score"] = _round(_uniform(rng, 0.12, 0.50))
        template["o9_thermal_throttle_support_score"] = _round(_uniform(rng, 0.45, 0.82))
        tags.extend(["hard_high_load_negative", "power_counterpart"])
    elif family == "storage_rebuild_or_replication":
        template["synthetic_evidence_profile"] = "physical_only"
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.42, 0.68))
        template["o11_data_staging_tb"] = _round(_uniform(rng, 600, 6000), 2)
        template["o11_backup_or_replication_pattern_score"] = _round(_uniform(rng, 0.72, 0.98))
        template["o11_storage_cotraffic_score"] = _round(_uniform(rng, 0.58, 0.94))
        template["o7_storage_traffic_fraction"] = _round(_uniform(rng, 0.65, 0.96))
        template["o10_runtime_framework_class"] = "storage_rebuild"
        tags.extend(["hard_high_load_negative", "storage_counterpart"])
    elif family in {"large_etl_or_data_movement", "distributed_database_or_graph_analytics"}:
        gpus = sample_gpus(8, min(256, max(16, capacity * 0.12)))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 18, 42), util_p95=_uniform(rng, 42, 68), duty=_uniform(rng, 0.08, 0.32), tensor=_uniform(rng, 8, 35), hbm=_uniform(rng, 0.24, 0.52), fabric=_uniform(rng, 0.18, 0.48), periodicity=_uniform(rng, 0.03, 0.25), contiguity=_uniform(rng, 0.25, 0.60), declared="database" if family.endswith("analytics") else "data")
        template["synthetic_evidence_profile"] = "physical_only"
        template["o10_runtime_framework_class"] = "distributed_graph_analytics" if family.endswith("analytics") else "etl_data_pipeline"
        template["o11_data_staging_tb"] = _round(_uniform(rng, 500, 4500), 2)
        template["o7_storage_traffic_fraction"] = _round(_uniform(rng, 0.35, 0.72))
        tags.extend(["hard_high_load_negative"])
    elif family == "reserved_but_unused_capacity":
        gpus = sample_gpus(512, min(capacity, 3072))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 2, 12), util_p95=_uniform(rng, 8, 22), duty=0.0, tensor=_uniform(rng, 1, 8), hbm=_uniform(rng, 0.03, 0.16), fabric=_uniform(rng, 0.01, 0.08), periodicity=0.0, contiguity=_uniform(rng, 0.74, 0.95), declared="reserved")
        template["synthetic_evidence_profile"] = "capacity_only"
        template["o2_reservation_exclusive_flag"] = True
        template["o2_reservation_reuse_count"] = int(_uniform(rng, 0, 4))
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.18, 0.34))
        tags.extend(["counterfactual_candidate"])
    elif family == "maintenance_with_collector_gaps":
        template["synthetic_evidence_profile"] = "integrity_only"
        template["o9_cooling_maintenance_active"] = True
        template["o15_firmware_bmc_change_near_window"] = True
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.50, 0.68))
        template["o9_thermal_delta_t_score"] = _round(_uniform(rng, 0.48, 0.72))
        template["o9_cooling_flow_duty"] = _round(_uniform(rng, 0.55, 0.86))
        template["o8_power_baseline_drift_score"] = _round(_uniform(rng, 0.28, 0.68))
        template["o8_unattributed_power_fraction"] = _round(_uniform(rng, 0.18, 0.45))
        data_quality = _choice(rng, ["collector_gap_short", "collector_gap_long", "maintenance_observability_loss"])
        tags.extend(["missingness_edge", "boundary"])
    elif family == "multi_tenant_fragmented_nontraining":
        gpus = sample_gpus(192, min(capacity, 1536))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=_uniform(rng, 45, 68), util_p95=_uniform(rng, 70, 90), duty=_uniform(rng, 0.28, 0.58), tensor=_uniform(rng, 45, 78), hbm=_uniform(rng, 0.42, 0.70), fabric=_uniform(rng, 0.32, 0.64), periodicity=_uniform(rng, 0.20, 0.52), contiguity=_uniform(rng, 0.18, 0.48), declared="inference")
        template["o2_job_array_width"] = int(_uniform(rng, 12, 80))
        template["o2_account_linkage_confidence"] = _round(_uniform(rng, 0.10, 0.38))
        template["o7_account_flow_linkage_confidence"] = _round(_uniform(rng, 0.08, 0.35))
        template["o10_runtime_framework_class"] = "fragmented_batch_inference"
        tags.extend(["multi_tenant_overlap"])
    elif family == "counter_suppressed_candidate_window":
        integrity_only_variant = variant == "gpu_counters_hidden"
        gpus = sample_gpus(512 if not integrity_only_variant else 128, min(capacity, 2048 if not integrity_only_variant else 384))
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=0, util_p95=0, duty=0.0, tensor=0, hbm=0.0, fabric=_uniform(rng, 0.55, 0.82) if not integrity_only_variant else _uniform(rng, 0.08, 0.22), periodicity=_uniform(rng, 0.55, 0.78) if not integrity_only_variant else _uniform(rng, 0.02, 0.18), contiguity=_uniform(rng, 0.70, 0.92) if not integrity_only_variant else _uniform(rng, 0.36, 0.62), declared="unknown")
        template["synthetic_evidence_profile"] = "integrity_only" if integrity_only_variant else "training_suppressed_counters"
        template["o8_rack_power_fraction_p95"] = _round(_uniform(rng, 0.35, 0.52) if integrity_only_variant else _uniform(rng, 0.62, 0.86))
        template["o8_power_to_gpu_residual"] = _round(_uniform(rng, 0.04, 0.16) if integrity_only_variant else _uniform(rng, 0.22, 0.48))
        template["o16_coverage_fraction"] = 1.0
        template["o16_missing_reason"] = "observed"
        template["o16_probe_throughput_ratio_min"] = _round(_uniform(rng, 0.45, 0.78))
        template["o16_probe_latency_inflation_max"] = _round(_uniform(rng, 1.2, 2.5))
        template["o16_vram_residency_conflict_score"] = _round(_uniform(rng, 0.3, 0.72))
        data_quality = "confidential_compute_counters_disabled"
        tags.extend(["missingness_edge", "boundary"])
    elif family == "capacity_or_integrity_only_warning":
        if variant == "offledger_capacity_hint":
            template["o17_external_capacity_conflict_score"] = _round(_uniform(rng, 0.55, 0.88))
            template["o17_external_capacity_assertion"] = "additional_capacity_offledger_possible"
        template["synthetic_evidence_profile"] = "integrity_only" if variant != "offledger_capacity_hint" else "capacity_only"
        template["o14_counter_reset_count"] = int(_uniform(rng, 1, 5))
        template["o14_min_critical_coverage"] = _round(_uniform(rng, 0.55, 0.82))
        template["o14_gap_fraction_critical"] = _round(1.0 - template["o14_min_critical_coverage"])
        tags.extend(["integrity_warning", "boundary"])
    else:
        signed = variant in {"full_stack_signed", "late_signed_declaration", "moderate_scale_signed"}
        if family == "pretraining_standard":
            signed = variant == "full_stack_signed" or (variant == "evaluation_phase_mixed" and rng.random() < 0.45)
        elif family == "large_fine_tune_standard":
            signed = variant == "moderate_scale_signed" or rng.random() < 0.35
        elif family == "cloud_training_redacted_runtime":
            signed = True
        elif family == "training_with_delayed_logs":
            signed = rng.random() < 0.78
        elif family == "training_without_semantic_logs":
            signed = False
        redacted_runtime = family == "cloud_training_redacted_runtime" and variant != "hashed_container_only"
        if family == "pretraining_standard":
            gpus = sample_gpus(1536, capacity)
            util_low, util_high, fabric_low, fabric_high = 76, 94, 0.68, 0.92
            checkpoint, artifact, dataloader = _uniform(rng, 0.62, 0.92), _uniform(rng, 0.55, 0.90), _uniform(rng, 0.62, 0.88)
            if variant == "evaluation_phase_mixed":
                util_low, util_high, fabric_low, fabric_high = 58, 80, 0.36, 0.62
                checkpoint, artifact, dataloader = _uniform(rng, 0.30, 0.52), _uniform(rng, 0.28, 0.50), _uniform(rng, 0.34, 0.56)
        elif family == "large_fine_tune_standard":
            gpus = sample_gpus(512, min(capacity, 3072))
            util_low, util_high, fabric_low, fabric_high = 70, 90, 0.48, 0.78
            checkpoint, artifact, dataloader = _uniform(rng, 0.52, 0.82), _uniform(rng, 0.45, 0.78), _uniform(rng, 0.45, 0.72)
            if variant == "adapter_or_full_finetune":
                util_low, util_high, fabric_low, fabric_high = 54, 76, 0.26, 0.52
                checkpoint, artifact, dataloader = _uniform(rng, 0.30, 0.54), _uniform(rng, 0.28, 0.52), _uniform(rng, 0.30, 0.54)
        elif family == "training_without_semantic_logs":
            linked_variant = variant == "storage_and_fabric_only"
            gpus = sample_gpus(512 if not linked_variant else 384, min(capacity, 3072 if not linked_variant else 1536))
            util_low, util_high, fabric_low, fabric_high = (64, 90, 0.45, 0.82) if not linked_variant else (58, 84, 0.34, 0.68)
            checkpoint, artifact, dataloader = _uniform(rng, 0.50, 0.82), _uniform(rng, 0.42, 0.74), _uniform(rng, 0.48, 0.78)
            if linked_variant:
                template["o2_job_array_width"] = int(_uniform(rng, 8, 72))
                template["o2_account_linkage_confidence"] = _round(_uniform(rng, 0.52, 0.78))
                template["o7_account_flow_linkage_confidence"] = template["o2_account_linkage_confidence"]
                tags.append("linked_shards")
        elif family == "cloud_training_redacted_runtime":
            gpus = sample_gpus(512, min(capacity, 3072))
            util_low, util_high, fabric_low, fabric_high = 64, 88, 0.42, 0.78
            checkpoint, artifact, dataloader = _uniform(rng, 0.50, 0.82), _uniform(rng, 0.42, 0.74), _uniform(rng, 0.48, 0.78)
            if variant == "hashed_container_only":
                data_quality = "full_observability"
                privacy = "pseudonymized_account"
        elif family == "underclocked_energy_capped_training":
            weak_underclocked = variant in {"demand_response_overlap", "weak_early_window"}
            gpus = sample_gpus(512 if not weak_underclocked else 256, min(capacity, 3072 if not weak_underclocked else 1024))
            util_low, util_high, fabric_low, fabric_high = (50, 72, 0.45, 0.72) if not weak_underclocked else (42, 64, 0.18, 0.42)
            checkpoint, artifact, dataloader = (
                (_uniform(rng, 0.55, 0.82), _uniform(rng, 0.42, 0.72), _uniform(rng, 0.58, 0.82))
                if not weak_underclocked
                else (_uniform(rng, 0.20, 0.44), _uniform(rng, 0.16, 0.40), _uniform(rng, 0.22, 0.48))
            )
            template["o4_power_cap_active_fraction"] = _round(_uniform(rng, 0.45, 0.92))
            template["o8_power_cap_or_curtailment_active"] = True
            template["o8_power_to_gpu_residual"] = _round(_uniform(rng, -0.10, 0.03))
            signed = (not weak_underclocked) and rng.random() < 0.85
            tags.extend(["energy_capped_boundary"] if weak_underclocked else ["energy_capped"])
        elif family == "elastic_preempted_training":
            gpus = sample_gpus(512, min(capacity, 3072))
            util_low, util_high, fabric_low, fabric_high = 62, 88, 0.45, 0.82
            checkpoint, artifact, dataloader = _uniform(rng, 0.46, 0.78), _uniform(rng, 0.42, 0.76), _uniform(rng, 0.48, 0.78)
            template["o2_elastic_resize_count"] = int(_uniform(rng, 2, 8))
            template["o2_preemption_restart_count"] = int(_uniform(rng, 1, 5))
            template["o2_scheduler_queue_delay_hours"] = _round(_uniform(rng, 0.5, 18), 2)
            signed = True
            tags.extend(["elastic", "preempted"])
        elif family == "fragmented_training_linked":
            weak = variant == "weak_linkage"
            gpus = sample_gpus(384 if not weak else 128, min(capacity, 2048 if not weak else 768))
            util_low, util_high, fabric_low, fabric_high = (62, 88, 0.36, 0.72) if not weak else (52, 76, 0.18, 0.42)
            checkpoint, artifact, dataloader = (
                (_uniform(rng, 0.45, 0.76), _uniform(rng, 0.38, 0.68), _uniform(rng, 0.42, 0.72))
                if not weak
                else (_uniform(rng, 0.20, 0.44), _uniform(rng, 0.16, 0.40), _uniform(rng, 0.20, 0.46))
            )
            template["o2_job_array_width"] = int(_uniform(rng, 8, 96))
            template["o2_account_linkage_confidence"] = _round(_uniform(rng, 0.08, 0.32) if weak else _uniform(rng, 0.65, 0.92))
            template["o7_account_flow_linkage_confidence"] = template["o2_account_linkage_confidence"]
            signed = (not weak) and rng.random() < 0.80
            tags.extend(["weak_linkage_boundary"] if weak else ["linked_shards"])
        elif family == "sparse_or_moe_bursty_training":
            gpus = sample_gpus(512, min(capacity, 3072))
            util_low, util_high, fabric_low, fabric_high = 48, 86, 0.40, 0.82
            checkpoint, artifact, dataloader = _uniform(rng, 0.50, 0.80), _uniform(rng, 0.42, 0.70), _uniform(rng, 0.52, 0.84)
            template["o4_gpu_util_cv"] = _round(_uniform(rng, 0.32, 0.68))
            template["o4_gpu_idle_gap_p95_minutes"] = _round(_uniform(rng, 8, 42), 2)
            template["o7_collective_jitter_score"] = _round(_uniform(rng, 0.35, 0.72))
            signed = rng.random() < 0.75
            tags.extend(["bursty", "moe_like"])
        elif family == "training_with_low_fabric_high_checkpoint":
            gpus = sample_gpus(512, min(capacity, 2048))
            util_low, util_high, fabric_low, fabric_high = 68, 90, 0.18, 0.45
            checkpoint, artifact, dataloader = _uniform(rng, 0.72, 0.96), _uniform(rng, 0.65, 0.92), _uniform(rng, 0.62, 0.86)
            signed = rng.random() < 0.85
            tags.extend(["low_fabric", "checkpoint_heavy"])
        elif family == "training_with_delayed_logs":
            gpus = sample_gpus(512, min(capacity, 3072))
            util_low, util_high, fabric_low, fabric_high = 68, 91, 0.50, 0.82
            checkpoint, artifact, dataloader = _uniform(rng, 0.54, 0.86), _uniform(rng, 0.42, 0.74), _uniform(rng, 0.48, 0.78)
            data_quality = "delayed_log_delivery"
            signed = rng.random() < 0.85
            tags.extend(["log_latency"])
        elif family == "multi_stage_training_pipeline":
            gpus = sample_gpus(512, min(capacity, 4096))
            util_low, util_high, fabric_low, fabric_high = 58, 92, 0.36, 0.86
            checkpoint, artifact, dataloader = _uniform(rng, 0.54, 0.90), _uniform(rng, 0.55, 0.92), _uniform(rng, 0.55, 0.88)
            template["o2_elastic_resize_count"] = int(_uniform(rng, 1, 5))
            signed = rng.random() < 0.85
            tags.extend(["multi_stage"])
        else:
            gpus = sample_gpus(512, min(capacity, 3072))
            util_low, util_high, fabric_low, fabric_high = 64, 90, 0.45, 0.82
            checkpoint, artifact, dataloader = _uniform(rng, 0.45, 0.82), _uniform(rng, 0.40, 0.78), _uniform(rng, 0.48, 0.78)
            tags.append("boundary")

        util = _uniform(rng, util_low, util_high)
        set_activity(template, site, gpus=gpus, duration_hours=duration_hours, util_p50=max(35, util - _uniform(rng, 4, 16)), util_p95=min(98, util + _uniform(rng, 4, 9)), duty=_uniform(rng, 0.48, 0.92), tensor=_uniform(rng, 62, 94), hbm=_uniform(rng, 0.62, 0.92), fabric=_uniform(rng, fabric_low, fabric_high), periodicity=_uniform(rng, 0.48, 0.92), contiguity=_uniform(rng, 0.62, 0.96), declared="train" if not redacted_runtime else "unknown")
        if family == "fragmented_training_linked":
            weak = variant == "weak_linkage"
            template["o2_account_linkage_confidence"] = _round(_uniform(rng, 0.08, 0.32) if weak else _uniform(rng, 0.65, 0.92))
            template["o7_account_flow_linkage_confidence"] = template["o2_account_linkage_confidence"]
        template["synthetic_evidence_profile"] = "training_primary_semantic"
        _set_training_storage(template, rng, checkpoint=checkpoint, artifact=artifact, dataloader=dataloader)
        delayed = _uniform(rng, 6, 60) if family == "training_with_delayed_logs" else 0.0
        if signed:
            policy_signed_variant = variant in {"full_stack_signed", "late_signed_declaration", "moderate_scale_signed"}
            if policy_signed_variant and family in {"pretraining_standard", "large_fine_tune_standard", "training_with_delayed_logs"}:
                completeness = _uniform(rng, 0.65, 0.95)
            else:
                completeness = _uniform(rng, 0.28, 0.52)
        elif delayed:
            completeness = _uniform(rng, 0.35, 0.70)
        else:
            completeness = 0.0
        _set_training_semantics(template, rng, signed=signed, redacted_runtime=redacted_runtime, delayed_logs_hours=delayed, completeness=completeness)
        if (family == "pretraining_standard" and variant == "evaluation_phase_mixed") or (
            family == "large_fine_tune_standard" and variant == "adapter_or_full_finetune"
        ):
            for obs in ["o4", "o7", "o8", "o14"]:
                template[f"{obs}_coverage_fraction"] = min(float(template.get(f"{obs}_coverage_fraction") or 1.0), 0.68)
                template[f"{obs}_missing_reason"] = "collector_gap"
            template["o14_min_critical_coverage"] = min(float(template["o14_min_critical_coverage"]), 0.68)
            template["o14_gap_fraction_critical"] = max(float(template["o14_gap_fraction_critical"]), 0.32)
            tags.append("coverage_boundary")

    if family in {"pretraining_standard", "large_fine_tune_standard", "cloud_training_redacted_runtime", "training_with_delayed_logs"}:
        if "cloud" in site["site_type"] or rng.random() < 0.35:
            template["o3_coverage_fraction"] = 1.0
            template["o3_missing_reason"] = "observed"
            template["o3_batch_provisioned_gpus"] = int(template.get("o2_max_concurrent_normalized_gpus") or 0)
            template["o3_capacity_reservation_duration_hours"] = _round(duration_hours, 2)
            template["o3_training_sku_fraction"] = _round(_uniform(rng, 0.72, 1.0))
            template["o3_billing_continuity_score"] = _round(_uniform(rng, 0.78, 0.99))
            template["o3_egress_tb"] = _round(_uniform(rng, 5, 180), 2)

    if (family in V1_HARD_POSITIVE_FAMILIES or family in {"hpc_mpi_collective", "nccl_extended_benchmark", "hardware_burn_in_or_thermal_soak"}) and rng.random() < 0.42:
        template["o5_profiler_available"] = True
        template["o5_coverage_fraction"] = _round(_uniform(rng, 0.35, 0.78))
        template["o5_missing_reason"] = "privacy_redacted"
        if family in V1_HARD_POSITIVE_FAMILIES:
            template["o5_kernel_training_motif_score"] = _round(_uniform(rng, 0.54, 0.88))
            template["o5_tensor_throughput_ratio"] = _round(_uniform(rng, 0.50, 0.86))
        else:
            template["o5_kernel_training_motif_score"] = _round(_uniform(rng, 0.18, 0.54))
            template["o5_tensor_throughput_ratio"] = _round(_uniform(rng, 0.20, 0.62))

    if template["o2_max_concurrent_normalized_gpus"] > 0:
        template["scope_type"] = "linked_job_group" if family in V1_HARD_POSITIVE_FAMILIES or "fragmented" in family else "topology_domain"
        prefix = "jobgrp_hmac_" if template["scope_type"] == "linked_job_group" else "fabricdom_hmac_"
        template["scope_id_hash"] = prefix + stable_hash(site["site_id"], family, variant, duration_hours, template["o2_max_concurrent_normalized_gpus"], length=12)

    if data_quality is None:
        if family == "cloud_training_redacted_runtime":
            data_quality = "routine_privacy_redaction"
            privacy = "runtime_redacted"
        elif family == "training_without_semantic_logs":
            data_quality = (
                "routine_privacy_redaction"
                if variant in {"runtime_low_confidence", "storage_and_fabric_only"}
                else "full_observability"
            )
            privacy = "semantic_logs_withheld"
        elif rng.random() < 0.18:
            data_quality = _choice(rng, DATA_QUALITY_REGIMES[1:])
    _set_v1_metadata(
        template,
        rng,
        family=family,
        variant=variant,
        data_quality_regime=data_quality or "full_observability",
        privacy_tier=privacy,
        counterfactual_group_id=counterfactual_group_id,
        counterfactual_role=counterfactual_role,
        hard_case_tags=tags,
    )
    _apply_data_quality_regime(template, rng)
    return template


def make_episode(
    rng: np.random.Generator,
    dataset_id: str,
    seed: int,
    idx: int,
    site: dict[str, Any],
    scenario: str,
    scale_conf: dict[str, Any],
    base_time: datetime,
    *,
    hard_profile: bool = False,
    variant: str | None = None,
    counterfactual_group_id: str | None = None,
    counterfactual_role: str | None = None,
    fixed_start: datetime | None = None,
    fixed_duration_hours: float | None = None,
) -> dict[str, Any]:
    duration_hours = fixed_duration_hours if fixed_duration_hours is not None else scenario_duration_hours(rng, scenario)
    total_hours = float(scale_conf["days_per_site"]) * 24.0
    max_start_hour = max(0.0, total_hours - min(duration_hours, total_hours * 0.95))
    start = fixed_start or (base_time + timedelta(hours=_uniform(rng, 0, max_start_hour)))
    end = start + timedelta(hours=duration_hours)
    if hard_profile:
        template = apply_v1_scenario_features(
            site,
            scenario,
            duration_hours,
            rng,
            variant=variant,
            counterfactual_group_id=counterfactual_group_id,
            counterfactual_role=counterfactual_role,
        )
    else:
        template = apply_scenario_features(site, scenario, duration_hours, rng)
    episode_id = f"episode_{scenario}_{idx:05d}"
    return {
        "dataset_id": dataset_id,
        "seed": seed,
        "episode_id": episode_id,
        "site": site,
        "site_id": site["site_id"],
        "episode_start": utc_iso(start),
        "episode_end": utc_iso(end),
        "duration_hours": _round(duration_hours, 2),
        "latent_workload_class": scenario,
        "scenario_family": template.get("scenario_family"),
        "scenario_variant": template.get("scenario_variant"),
        "counterfactual_group_id": template.get("counterfactual_group_id"),
        "feature_template": template,
        "max_windows_per_episode_per_length": scale_conf["max_windows_per_episode_per_length"],
        "max_metric_samples_per_episode": scale_conf.get("max_metric_samples_per_episode", 32),
        "site_baseline_it_mw": site["baseline_it_mw"],
    }


COUNTERFACTUAL_ARCHETYPES = [
    [
        ("reserved_but_unused_capacity", "exclusive_unused", "cf_a_reserved_unused"),
        ("hardware_burn_in_or_thermal_soak", "thermal_soak", "cf_b_high_gpu_power_only"),
        ("training_without_semantic_logs", "no_ml_logs", "cf_c_training_no_logs"),
        ("pretraining_standard", "full_stack_signed", "cf_d_signed_training"),
    ],
    [
        ("hpc_mpi_collective", "allreduce_simulation", "cf_a_hpc_collective"),
        ("training_without_semantic_logs", "storage_and_fabric_only", "cf_b_training_same_fabric"),
        ("nccl_extended_benchmark", "fabric_validation", "cf_c_nccl_benchmark"),
        ("model_parallel_inference", "high_local_fabric", "cf_d_model_parallel_inference"),
    ],
    [
        ("capacity_or_integrity_only_warning", "capacity_external_conflict", "cf_a_demand_or_capacity_warning"),
        ("underclocked_energy_capped_training", "energy_capped_long", "cf_b_underclocked_training"),
        ("large_batch_inference", "large_object_reads", "cf_c_batch_inference"),
    ],
    [
        ("multi_tenant_fragmented_nontraining", "shared_topology_overlap", "cf_a_fragmented_nontraining"),
        ("fragmented_training_linked", "weak_linkage", "cf_b_weak_linkage_training"),
        ("fragmented_training_linked", "strong_linkage", "cf_c_linked_training"),
    ],
    [
        ("storage_rebuild_or_replication", "backup_wave", "cf_a_storage_replication"),
        ("training_with_low_fabric_high_checkpoint", "checkpoint_dominant", "cf_b_checkpoint_training"),
        ("large_etl_or_data_movement", "object_store_migration", "cf_c_etl_data_movement"),
    ],
    [
        ("cloud_training_redacted_runtime", "runtime_redacted", "cf_a_redacted_training"),
        ("training_with_delayed_logs", "late_signed_declaration", "cf_b_late_signed_logs"),
        ("hpc_mpi_collective", "climate_or_cfd", "cf_c_hpc_counterpart"),
    ],
]


def make_v1_counterfactual_episodes(
    rng: np.random.Generator,
    dataset_id: str,
    seed: int,
    sites: list[dict[str, Any]],
    scale_conf: dict[str, Any],
    base_time: datetime,
    group_count: int,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    capable_sites = [site for site in sites if site["normalized_h100e_capacity"] >= POLICY_CONCURRENCY_THRESHOLD_GPUS] or sites
    total_hours = float(scale_conf["days_per_site"]) * 24.0
    for group_index in range(group_count):
        archetype = COUNTERFACTUAL_ARCHETYPES[group_index % len(COUNTERFACTUAL_ARCHETYPES)]
        site = _choice(rng, capable_sites)
        duration_hours = _log_uniform(rng, 36, min(240, max(48, total_hours * 0.45)))
        max_start_hour = max(0.0, total_hours - duration_hours)
        start = base_time + timedelta(hours=_uniform(rng, 0, max_start_hour))
        group_id = "cfg_hmac_" + stable_hash(dataset_id, group_index, site["site_id"], length=14)
        for role_index, (family, variant, role) in enumerate(archetype):
            episode_index = len(episodes)
            # Keep the same site/time context but nudge starts slightly so window IDs remain unique.
            role_start = start + timedelta(minutes=role_index * 7)
            episodes.append(
                make_episode(
                    rng,
                    dataset_id,
                    seed,
                    episode_index,
                    site,
                    family,
                    scale_conf,
                    base_time,
                    hard_profile=True,
                    variant=variant,
                    counterfactual_group_id=group_id,
                    counterfactual_role=role,
                    fixed_start=role_start,
                    fixed_duration_hours=duration_hours,
                )
            )
    return episodes


def make_v1_episodes(
    rng: np.random.Generator,
    dataset_id: str,
    seed: int,
    sites: list[dict[str, Any]],
    scale_conf: dict[str, Any],
    base_time: datetime,
) -> list[dict[str, Any]]:
    target_count = int(scale_conf["episode_count"])
    group_count = min(int(scale_conf.get("counterfactual_group_count", 0)), max(0, target_count // 2))
    episodes = make_v1_counterfactual_episodes(rng, dataset_id, seed, sites, scale_conf, base_time, group_count)
    for site_index, site in enumerate(sites):
        if len(episodes) >= target_count:
            break
        family = "training_with_low_fabric_high_checkpoint" if site_index % 2 else "large_fine_tune_standard"
        variant = "checkpoint_dominant" if family == "training_with_low_fabric_high_checkpoint" else "moderate_scale_signed"
        fixed_start = base_time + timedelta(hours=12 + site_index * 9)
        episodes.append(
            make_episode(
                rng,
                dataset_id,
                seed,
                len(episodes),
                site,
                family,
                scale_conf,
                base_time,
                hard_profile=True,
                variant=variant,
                fixed_start=fixed_start,
                fixed_duration_hours=_uniform(rng, 72, 168),
            )
        )
    variant_anchors = [
        ("pretraining_standard", "evaluation_phase_mixed"),
        ("large_fine_tune_standard", "adapter_or_full_finetune"),
        ("hpc_mpi_collective", "mpi_checkpoint_light"),
        ("embedding_generation", "vector_index_build"),
        ("synthetic_data_generation_gpu_heavy", "generation_pipeline"),
        ("counter_suppressed_candidate_window", "gpu_counters_hidden"),
    ]
    capable_sites = [site for site in sites if site["normalized_h100e_capacity"] >= POLICY_CONCURRENCY_THRESHOLD_GPUS] or sites
    for anchor_index, (family, variant) in enumerate(variant_anchors):
        if len(episodes) >= target_count:
            break
        site = capable_sites[anchor_index % len(capable_sites)]
        episodes.append(
            make_episode(
                rng,
                dataset_id,
                seed,
                len(episodes),
                site,
                family,
                scale_conf,
                base_time,
                hard_profile=True,
                variant=variant,
                fixed_start=base_time + timedelta(hours=96 + anchor_index * 11),
                fixed_duration_hours=scenario_duration_hours(rng, family),
            )
        )
    remaining = max(0, target_count - len(episodes))
    scenarios = v1_scenario_sequence(rng, remaining)
    for scenario in scenarios:
        idx = len(episodes)
        site = choose_site(rng, sites, scenario)
        episodes.append(make_episode(rng, dataset_id, seed, idx, site, scenario, scale_conf, base_time, hard_profile=True))
    return episodes[:target_count]


def base_event(
    *,
    dataset_id: str,
    episode: dict[str, Any],
    observable_id: str,
    source_system: str,
    event_type: str,
    event_time: str,
    event_end_time: str | None,
    attributes: dict[str, Any],
    entity_type: str = "scope",
    entity_id_hash: str | None = None,
    trust_level: str = "operator_signed",
    signature_status: str = "valid",
) -> dict[str, Any]:
    template = episode["feature_template"]
    payload = {
        "dataset_id": dataset_id,
        "event_type": event_type,
        "episode_id": episode["episode_id"],
        "attributes": attributes,
    }
    return {
        "event_record_id": "evt_" + stable_hash(dataset_id, episode["episode_id"], observable_id, event_type, event_time, length=24),
        "site_id": episode["site_id"],
        "observable_id": observable_id,
        "source_system": source_system,
        "event_type": event_type,
        "scope_type": template.get("scope_type", "topology_domain"),
        "scope_id_hash": template.get("scope_id_hash") or episode["site"]["scope_id_hash"],
        "account_id_hash": episode["site"]["account_id_hash"],
        "job_id_hash": "job_hmac_" + stable_hash(episode["episode_id"], "job", length=12),
        "entity_type": entity_type,
        "entity_id_hash": entity_id_hash or template.get("scope_id_hash") or episode["site"]["scope_id_hash"],
        "event_time": event_time,
        "event_end_time": event_end_time,
        "ingest_time": utc_iso(datetime.now(timezone.utc)),
        "source_clock_offset_ms": template.get("o14_clock_drift_max_ms", 0),
        "attributes_json": json_dumps(attributes),
        "trust_level": trust_level,
        "signature_status": signature_status,
        "raw_payload_hash": raw_payload_hash(payload),
        "ingest_batch_id": "batch_" + stable_hash(dataset_id, episode["site_id"], "events", length=10),
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode["episode_id"],
        "latent_workload_class": episode["latent_workload_class"],
    }


def metric_record(
    *,
    dataset_id: str,
    episode: dict[str, Any],
    observable_id: str,
    source_system: str,
    metric_name: str,
    entity_type: str,
    event_time: datetime,
    sample_interval_ms: int,
    value_num: float | None,
    value_text: str | None,
    unit: str,
    coverage_fraction: float,
) -> dict[str, Any]:
    template = episode["feature_template"]
    collector_time = event_time + timedelta(milliseconds=int(template.get("o14_clock_drift_max_ms") or 0))
    entity_hash = template.get("scope_id_hash") or episode["site"]["scope_id_hash"]
    payload = {
        "dataset_id": dataset_id,
        "episode_id": episode["episode_id"],
        "observable_id": observable_id,
        "metric_name": metric_name,
        "event_time": utc_iso(event_time),
        "value_num": value_num,
        "value_text": value_text,
    }
    return {
        "metric_sample_id": "met_" + stable_hash(dataset_id, episode["episode_id"], observable_id, metric_name, utc_iso(event_time), length=24),
        "site_id": episode["site_id"],
        "observable_id": observable_id,
        "source_system": source_system,
        "metric_name": metric_name,
        "entity_type": entity_type,
        "entity_id_hash": entity_hash,
        "parent_entity_id_hash": episode["site"]["site_scope_id_hash"],
        "event_time": utc_iso(event_time),
        "ingest_time": utc_iso(event_time + timedelta(seconds=30)),
        "collector_time": utc_iso(collector_time),
        "source_clock_offset_ms": template.get("o14_clock_drift_max_ms", 0),
        "sample_interval_ms": sample_interval_ms,
        "value_num": value_num,
        "value_text": value_text,
        "unit": unit,
        "coverage_fraction": coverage_fraction,
        "trust_level": episode["site"]["trust_tier"],
        "signature_status": "valid",
        "raw_payload_hash": raw_payload_hash(payload),
        "ingest_batch_id": "batch_" + stable_hash(dataset_id, episode["site_id"], "metrics", length=10),
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode["episode_id"],
        "latent_workload_class": episode["latent_workload_class"],
    }


def snapshot_record(
    *,
    dataset_id: str,
    site: dict[str, Any],
    observable_id: str,
    snapshot_type: str,
    valid_from: datetime,
    valid_to: datetime,
    attributes: dict[str, Any],
    trust_level: str,
    signature_status: str = "valid",
) -> dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "site_id": site["site_id"],
        "observable_id": observable_id,
        "snapshot_type": snapshot_type,
        "valid_from": utc_iso(valid_from),
        "attributes": attributes,
    }
    return {
        "snapshot_id": "snap_" + stable_hash(dataset_id, site["site_id"], observable_id, snapshot_type, utc_iso(valid_from), length=24),
        "site_id": site["site_id"],
        "observable_id": observable_id,
        "snapshot_type": snapshot_type,
        "scope_type": "topology_domain",
        "scope_id_hash": site["scope_id_hash"],
        "valid_from": utc_iso(valid_from),
        "valid_to": utc_iso(valid_to),
        "observed_at": utc_iso(valid_from + timedelta(minutes=2)),
        "ingest_time": utc_iso(valid_from + timedelta(minutes=4)),
        "attributes_json": json_dumps(attributes),
        "trust_level": trust_level,
        "signature_status": signature_status,
        "raw_payload_hash": raw_payload_hash(payload),
        "ingest_batch_id": "batch_" + stable_hash(dataset_id, site["site_id"], "snapshots", length=10),
        "schema_version": SCHEMA_VERSION,
        "episode_id": None,
        "latent_workload_class": None,
    }


def records_for_episode(dataset_id: str, episode: dict[str, Any], sample_minutes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    template = episode["feature_template"]
    start = datetime.fromisoformat(episode["episode_start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(episode["episode_end"].replace("Z", "+00:00"))
    events: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    summary_attrs = {
        key: value for key, value in episode.items() if key not in {"site"}
    }
    summary_event = base_event(
        dataset_id=dataset_id,
        episode=episode,
        observable_id="O14",
        source_system="synthetic_generator",
        event_type="synthetic_episode_summary",
        event_time=episode["episode_start"],
        event_end_time=episode["episode_end"],
        attributes=summary_attrs,
        trust_level="synthetic",
    )
    summary_attrs["raw_event_hash"] = summary_event["raw_payload_hash"]
    summary_event["attributes_json"] = json_dumps(summary_attrs)
    events.append(summary_event)

    if template.get("o2_max_concurrent_normalized_gpus", 0) > 0:
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O2",
                source_system="slurm_orchestration",
                event_type="scheduler_allocation_interval",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "allocated_normalized_gpus": template["o2_max_concurrent_normalized_gpus"],
                    "duration_hours": template["o2_allocation_duration_hours"],
                    "declared_workload_class": template["o2_declared_workload_class"],
                    "exclusive_reservation": template["o2_reservation_exclusive_flag"],
                    "topology_contiguity_score": template["o2_topology_contiguity_score"],
                    "elastic_resize_count": template.get("o2_elastic_resize_count"),
                    "preemption_restart_count": template.get("o2_preemption_restart_count"),
                    "scheduler_queue_delay_hours": template.get("o2_scheduler_queue_delay_hours"),
                    "account_linkage_confidence": template.get("o2_account_linkage_confidence"),
                    "job_array_width": template.get("o2_job_array_width"),
                    "reservation_reuse_count": template.get("o2_reservation_reuse_count"),
                },
            )
        )
    if template.get("o3_coverage_fraction", 0) and template.get("o3_batch_provisioned_gpus"):
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O3",
                source_system="cloud_control_plane",
                event_type="capacity_reservation_or_batch_provisioning",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "batch_provisioned_gpus": template["o3_batch_provisioned_gpus"],
                    "training_sku_fraction": template["o3_training_sku_fraction"],
                    "billing_continuity_score": template["o3_billing_continuity_score"],
                    "egress_tb": template["o3_egress_tb"],
                },
            )
        )
    if template.get("o10_runtime_framework_class") not in {None, "none"}:
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O10",
                source_system="runtime_metadata",
                event_type="runtime_process_interval",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "framework_class": template["o10_runtime_framework_class"],
                    "world_size": template["o10_world_size"],
                    "rank_stability_score": template["o10_rank_stability_score"],
                    "rendezvous_present": template["o10_rendezvous_present"],
                    "runtime_metadata_confidence": template.get("o10_runtime_metadata_confidence"),
                    "declared_vs_observed_mismatch_score": template.get("o10_declared_vs_observed_mismatch_score"),
                },
            )
        )
    if template.get("o11_data_staging_tb", 0) or template.get("o11_checkpoint_periodicity_score", 0):
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O11",
                source_system="storage_logs",
                event_type="storage_activity_summary",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "data_staging_tb": template["o11_data_staging_tb"],
                    "checkpoint_write_tb_per_event": template["o11_checkpoint_write_tb_per_event"],
                    "checkpoint_periodicity_score": template["o11_checkpoint_periodicity_score"],
                    "read_write_training_pattern_score": template["o11_read_write_training_pattern_score"],
                    "checkpoint_jitter_score": template.get("o11_checkpoint_jitter_score"),
                    "artifact_write_pattern_score": template.get("o11_artifact_write_pattern_score"),
                    "dataloader_read_pattern_score": template.get("o11_dataloader_read_pattern_score"),
                    "backup_or_replication_pattern_score": template.get("o11_backup_or_replication_pattern_score"),
                    "storage_cotraffic_score": template.get("o11_storage_cotraffic_score"),
                },
            )
        )
    if template.get("o12_signed_ml_logs_present") or template.get("o12_log_delivery_delay_hours", 0):
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O12",
                source_system="signed_ml_declaration",
                event_type="signed_ml_training_log_summary" if template.get("o12_signed_ml_logs_present") else "delayed_or_incomplete_ml_log_status",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "declared_parameter_count_b": template["o12_declared_parameter_count_b"],
                    "training_tokens_b": template["o12_training_tokens_b"],
                    "step_count": template["o12_step_count"],
                    "loss_curve_present": template["o12_loss_curve_present"],
                    "optimizer_state_present": template["o12_optimizer_state_present"],
                    "log_delivery_delay_hours": template.get("o12_log_delivery_delay_hours"),
                    "log_completeness_fraction": template.get("o12_log_completeness_fraction"),
                    "declaration_consistency_score": template.get("o12_declaration_consistency_score"),
                },
            )
        )
    if template.get("o14_gap_fraction_critical", 0) > 0.05 or template.get("o14_counter_reset_count", 0):
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O14",
                source_system="monitoring_integrity",
                event_type="collector_gap_or_reset",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "min_critical_coverage": template["o14_min_critical_coverage"],
                    "gap_fraction_critical": template["o14_gap_fraction_critical"],
                    "counter_reset_count": template["o14_counter_reset_count"],
                    "clock_drift_max_ms": template["o14_clock_drift_max_ms"],
                },
            )
        )
    if template.get("o15_firmware_bmc_change_near_window") or template.get("o15_unapproved_physical_change_near_window"):
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O15",
                source_system="physical_security_cmms",
                event_type="maintenance_or_physical_change",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "unapproved_physical_change": template["o15_unapproved_physical_change_near_window"],
                    "firmware_bmc_change": template["o15_firmware_bmc_change_near_window"],
                },
            )
        )
    if template.get("o16_probe_throughput_ratio_min") is not None:
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O16",
                source_system="active_probe_controller",
                event_type="challenge_probe_result",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "throughput_ratio_min": template["o16_probe_throughput_ratio_min"],
                    "latency_inflation_max": template["o16_probe_latency_inflation_max"],
                    "vram_residency_conflict_score": template["o16_vram_residency_conflict_score"],
                },
            )
        )
    if template.get("o17_external_capacity_conflict_score", 0) > 0:
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O17",
                source_system="external_capacity_reconciliation",
                event_type="external_capacity_conflict",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "external_capacity_conflict_score": template["o17_external_capacity_conflict_score"],
                    "external_capacity_assertion": template.get("o17_external_capacity_assertion"),
                    "energy_contract_alignment_score": template.get("o17_energy_contract_alignment_score"),
                    "network_provider_utilization_score": template.get("o17_network_provider_utilization_score"),
                    "procurement_or_maintenance_explanation_score": template.get("o17_procurement_or_maintenance_explanation_score"),
                },
            )
        )

    sample_interval = int(sample_minutes * 60 * 1000)
    sample_cap = int(episode.get("max_metric_samples_per_episode") or 32)
    sample_count = max(1, min(sample_cap, int(math.ceil((end - start).total_seconds() / (sample_minutes * 60)))))
    if sample_count == 1:
        sample_times = [start + (end - start) / 2]
    else:
        step = (end - start) / sample_count
        sample_times = [start + step * idx for idx in range(sample_count)]

    metric_specs = [
        ("O4", "dcgm_exporter", "gpu_utilization_p95", "gpu_group", template.get("o4_gpu_util_p95"), "percent", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "sm_tensor_active_p95", "gpu_group", template.get("o4_sm_tensor_active_p95"), "percent", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "gpu_power_fraction_p95", "gpu_group", template.get("o4_gpu_power_fraction_p95"), "fraction", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "gpu_util_cv", "gpu_group", template.get("o4_gpu_util_cv"), "fraction", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "hbm_pressure_duration_fraction", "gpu_group", template.get("o4_hbm_pressure_duration_fraction"), "fraction", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "power_cap_active_fraction", "gpu_group", template.get("o4_power_cap_active_fraction"), "fraction", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "thermal_throttle_fraction", "gpu_group", template.get("o4_thermal_throttle_fraction"), "fraction", template.get("o4_coverage_fraction", 0)),
        ("O5", "dcgm_profiler", "kernel_training_motif_score", "gpu_group", template.get("o5_kernel_training_motif_score"), "score", template.get("o5_coverage_fraction", 0)),
        ("O6", "dcgm_fabric", "nvlink_util_p95", "topology_domain", template.get("o6_nvlink_util_p95"), "fraction", template.get("o6_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "scaleout_port_util_p95", "fabric_partition", template.get("o7_scaleout_port_util_p95"), "fraction", template.get("o7_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "collective_periodicity_score", "fabric_partition", template.get("o7_collective_periodicity_score"), "score", template.get("o7_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "cross_section_sync_score", "fabric_partition", template.get("o7_cross_section_sync_score"), "score", template.get("o7_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "flow_entropy_score", "fabric_partition", template.get("o7_flow_entropy_score"), "score", template.get("o7_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "storage_traffic_fraction", "fabric_partition", template.get("o7_storage_traffic_fraction"), "fraction", template.get("o7_coverage_fraction", 0)),
        ("O8", "bms_meter", "rack_power_fraction_p95", "power_zone", template.get("o8_rack_power_fraction_p95"), "fraction", template.get("o8_coverage_fraction", 0)),
        ("O8", "bms_meter", "power_baseline_drift_score", "power_zone", template.get("o8_power_baseline_drift_score"), "score", template.get("o8_coverage_fraction", 0)),
        ("O8", "bms_meter", "unattributed_power_fraction", "power_zone", template.get("o8_unattributed_power_fraction"), "fraction", template.get("o8_coverage_fraction", 0)),
        ("O9", "cooling_bms", "thermal_delta_t_score", "cooling_zone", template.get("o9_thermal_delta_t_score"), "score", template.get("o9_coverage_fraction", 0)),
        ("O9", "cooling_bms", "thermal_throttle_support_score", "cooling_zone", template.get("o9_thermal_throttle_support_score"), "score", template.get("o9_coverage_fraction", 0)),
        ("O13", "attestation_service", "attestation_valid_fraction", "topology_domain", template.get("o13_attestation_valid_fraction"), "fraction", template.get("o13_coverage_fraction", 0)),
        ("O14", "monitoring_integrity", "min_critical_coverage", "site", template.get("o14_min_critical_coverage"), "fraction", template.get("o14_coverage_fraction", 0)),
    ]
    for sample_time in sample_times:
        for obs_id, source, name, entity, base_value, unit, coverage in metric_specs:
            if base_value is None or coverage == 0:
                continue
            value = float(base_value)
            if unit in {"percent", "fraction", "score"}:
                noise_seed = int(stable_hash(episode["episode_id"], name, utc_iso(sample_time), length=8), 16)
                value = value * float(np.random.default_rng(noise_seed).normal(1.0, 0.025))
                if unit == "percent":
                    value = clamp(value, 0.0, 100.0)
                else:
                    value = clamp(value, 0.0, 1.0)
            metrics.append(
                metric_record(
                    dataset_id=dataset_id,
                    episode=episode,
                    observable_id=obs_id,
                    source_system=source,
                    metric_name=name,
                    entity_type=entity,
                    event_time=sample_time,
                    sample_interval_ms=sample_interval,
                    value_num=_round(value, 5),
                    value_text=None,
                    unit=unit,
                    coverage_fraction=coverage,
                )
            )
    return events, metrics


def make_snapshots(dataset_id: str, sites: list[dict[str, Any]], base_time: datetime, days_per_site: int) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for site in sites:
        for day in range(days_per_site):
            valid_from = base_time + timedelta(days=day)
            valid_to = valid_from + timedelta(days=1)
            snapshots.append(
                snapshot_record(
                    dataset_id=dataset_id,
                    site=site,
                    observable_id="O1",
                    snapshot_type="accelerator_inventory",
                    valid_from=valid_from,
                    valid_to=valid_to,
                    attributes={
                        "normalized_h100e_capacity": site["normalized_h100e_capacity"],
                        "largest_contiguous_domain_gpus": site["largest_contiguous_domain_gpus"],
                        "homogeneous_high_end_fraction": site["homogeneous_high_end_fraction"],
                        "non_partitioned_fraction": site["non_partitioned_fraction"],
                        "telemetry_stack": site["telemetry_stack"],
                    },
                    trust_level=site["trust_tier"],
                )
            )
            snapshots.append(
                snapshot_record(
                    dataset_id=dataset_id,
                    site=site,
                    observable_id="O13",
                    snapshot_type="attestation_state",
                    valid_from=valid_from,
                    valid_to=valid_to,
                    attributes={
                        "attestation_valid_fraction": 0.98,
                        "collector_measurement_valid": True,
                        "confidential_compute_mode_fraction": 0.0,
                    },
                    trust_level="device_or_collector_signed",
                )
            )
        snapshots.append(
            snapshot_record(
                dataset_id=dataset_id,
                site=site,
                observable_id="O7",
                snapshot_type="job_to_fabric_port_mapping",
                valid_from=base_time,
                valid_to=base_time + timedelta(days=days_per_site),
                attributes={
                    "mapping_coverage": 0.93,
                    "fabric_type": "infiniband_ndr" if "cloud" not in site["site_type"] else "efa_or_roce",
                    "largest_partition_ports": int(site["largest_contiguous_domain_gpus"] * 1.15),
                },
                trust_level=site["trust_tier"],
            )
        )
    return snapshots


def coerce_csv_value(value: str) -> Any:
    if value == "":
        return None
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def create_examples(dataset_dir: Path) -> dict[str, str]:
    features_path = dataset_dir / "features" / "window_features_all.csv"
    examples_dir = dataset_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples: dict[str, str] = {}
    with features_path.open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            label = str(raw_row["label_0_to_4"])
            if label in examples:
                continue
            row = {key: coerce_csv_value(value) for key, value in raw_row.items()}
            coverage = {}
            missing_reasons = {}
            for obs_id in OBSERVABLE_IDS:
                key = obs_id.lower()
                coverage_key = {
                    "O1": "o1_capacity",
                    "O2": "o2_scheduler",
                    "O3": "o3_cloud_billing",
                    "O4": "o4_gpu_telemetry",
                    "O5": "o5_profiler",
                    "O6": "o6_local_fabric",
                    "O7": "o7_scaleout_fabric",
                    "O8": "o8_power",
                    "O9": "o9_cooling",
                    "O10": "o10_runtime",
                    "O11": "o11_storage",
                    "O12": "o12_ml_logs",
                    "O13": "o13_attestation",
                    "O14": "o14_integrity",
                    "O15": "o15_physical",
                    "O16": "o16_probe",
                    "O17": "o17_external",
                }[obs_id]
                coverage[coverage_key] = row.get(f"{key}_coverage_fraction")
                reason = row.get(f"{key}_missing_reason")
                if reason and reason != "observed":
                    missing_reasons[coverage_key] = reason
            row["coverage"] = coverage
            row["missing_reasons"] = missing_reasons
            row["trust"] = {
                "scheduler_signature_status": row.get("scheduler_signature_status"),
                "gpu_telemetry_trust_level": row.get("gpu_telemetry_trust_level"),
                "fabric_telemetry_trust_level": row.get("fabric_telemetry_trust_level"),
                "power_meter_trust_level": row.get("power_meter_trust_level"),
                "feature_pipeline_version": row.get("feature_pipeline_version"),
                "policy_threshold_version": row.get("policy_threshold_version"),
                "hardware_normalization_version": row.get("hardware_normalization_version"),
            }
            output_path = examples_dir / f"one_datapoint_label{label}.json"
            write_json(output_path, row)
            examples[label] = str(output_path.relative_to(dataset_dir))
            if len(examples) == 5:
                break
    return examples


def read_label_distribution(dataset_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with (dataset_dir / "features" / "window_features_all.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[str(row["label_0_to_4"])] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def read_scenario_distribution(dataset_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with (dataset_dir / "features" / "window_features_all.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts[row["latent_workload_class"]] += 1
    return dict(sorted(counts.items()))


def read_scenario_family_distribution(dataset_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with (dataset_dir / "features" / "window_features_all.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            family = row.get("scenario_family") or row.get("latent_workload_class") or ""
            if family:
                counts[family] += 1
    return dict(sorted(counts.items()))


def read_scenario_variant_count(dataset_dir: Path) -> int:
    variants: set[str] = set()
    with (dataset_dir / "features" / "window_features_all.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            variant = row.get("scenario_variant")
            if variant:
                variants.add(variant)
    return len(variants)


def read_counterfactual_group_count(dataset_dir: Path) -> int:
    groups: set[str] = set()
    with (dataset_dir / "features" / "window_features_all.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group = row.get("counterfactual_group_id")
            if group:
                groups.add(group)
    return len(groups)


def write_data_readme(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    title_version = "v1 hard" if manifest["scale"] in HARD_GENERATION_SCALES else manifest["scale"]
    hard_text = ""
    if manifest["scale"] in HARD_GENERATION_SCALES:
        hard_text = f"""
## What changed from v0

The v0 dataset proved that the pipeline worked, but the evidence patterns were too cleanly separable. V1 deliberately adds overlapping distributions, hard positives without semantic logs, and high-load negatives that share GPU, fabric, power, or storage signatures with real training.

V1 separates latent scenario metadata from labels:

- `scenario_family` and `scenario_variant` describe the synthetic episode family.
- `evidence_recipe_id`, `temporal_phase`, `data_quality_regime`, `privacy_tier`, `collector_profile`, `topology_class`, and counterfactual metadata are audit fields.
- Those audit fields are leakage columns and are excluded by the modeling package.

## Hard positives and hard negatives

Hard positive families include underclocked energy-capped training, elastic/preempted training, fragmented linked training, sparse or MoE-like bursty training, training with low fabric but high checkpoint cadence, redacted runtime, missing semantic logs, delayed logs, and multi-stage pipelines.

Hard negative and ambiguous families include model-parallel inference, HPC MPI collectives, extended NCCL benchmarks, hardware burn-in, storage rebuild/replication, large ETL/data movement, distributed graph/database analytics, maintenance gaps, fragmented non-training activity, and capacity/integrity-only warnings.

## Counterfactual groups

Counterfactual groups contain related episodes sharing a site, time span, and capacity context while changing one or two evidence layers. For example, a group may contain reserved-but-unused capacity, high GPU/power burn-in, training without semantic logs, and the same pattern with signed ML logs. The default generation created `{manifest.get('counterfactual_group_count', 0)}` counterfactual groups.

## Missingness and trust

V1 uses named data-quality regimes rather than random nulls: full observability, privacy redaction, confidential-compute counters disabled, short and long collector gaps, delayed log delivery, clock drift, maintenance observability loss, source-not-deployed, and partial fabric mapping. Missing telemetry is represented with coverage fractions and missing reasons; it is not encoded as inactivity.

## Labels

Labels are derived by composite evidence rules. Capacity alone is capped at label 1, integrity-only evidence is not training proof, physical-only evidence is capped at label 2, label 3 requires coherent primary plus supportive or semantic evidence, and label 4 requires authenticated policy-scale semantic evidence or full-stack corroboration under high integrity.

## Known limitations

The raw records remain group-level synthetic telemetry rather than per-GPU or per-port traces. The generator is designed for modeling and validation research, not operational claims. Counterfactuals are synthetic by construction, so real deployment would still require controlled drills and calibration against real telemetry.
"""
    text = f"""# Synthetic Datacenter Verification Dataset {title_version}

This directory is generated synthetic study data. It contains fictional raw-like datacenter telemetry, windowed feature rows, workbook-derived rule exports, schemas, examples, and validation artifacts.

Dataset ID: `{manifest['dataset_id']}`  
Scale: `{manifest['scale']}`  
Seed: `{manifest['seed']}`  
Generator: `{GENERATOR_VERSION}`  

The model training unit is one row in `features/window_features_all.csv`, not an individual raw metric sample or event record.
{hard_text}
## Regenerate

```bash
python src/datacenter_verification_synthetic/generate_synthetic_dataset.py \\
  --output {dataset_dir.as_posix()} \\
  --scale {manifest['scale']} \\
  --seed {manifest['seed']}
```

## Validate

```bash
python src/datacenter_verification_synthetic/validate_synthetic_dataset.py \\
  --dataset {dataset_dir.as_posix()}
```

```bash
python src/datacenter_verification_synthetic/validate_synthetic_hardness.py \\
  --dataset {dataset_dir.as_posix()}
```

```bash
python -m src.datacenter_verification_validators \\
  --dataset {dataset_dir.as_posix()}
```
"""
    (dataset_dir / "README.md").write_text(text, encoding="utf-8")


def generate_dataset(output: Path, scale: str, seed: int, workbook: Path, overwrite: bool = True) -> dict[str, Any]:
    if scale not in SCALE_PRESETS:
        ensure_scale(scale)
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ["schemas", "workbook_rules", "raw_normalized", "features", "examples", "validation"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    scale_conf = ensure_scale(scale)
    dataset_id = f"synthetic_{scale}_seed_{seed}"
    generation_time = utc_iso(datetime.now(timezone.utc))
    base_time = datetime(2026, 5, 10, tzinfo=timezone.utc)
    sites = make_sites(scale, seed)
    hard_profile = scale in HARD_GENERATION_SCALES

    if hard_profile:
        episodes = make_v1_episodes(rng, dataset_id, seed, sites, scale_conf, base_time)
    else:
        scenarios = scenario_sequence(rng, int(scale_conf["episode_count"]))
        episodes = []
        for idx, scenario in enumerate(scenarios):
            site = choose_site(rng, sites, scenario)
            episodes.append(make_episode(rng, dataset_id, seed, idx, site, scenario, scale_conf, base_time))

    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for episode in episodes:
        events, metrics = records_for_episode(dataset_id, episode, int(scale_conf["metric_sample_minutes"]))
        event_rows.extend(events)
        metric_rows.extend(metrics)
    snapshot_rows = make_snapshots(dataset_id, sites, base_time, int(scale_conf["days_per_site"]))

    # Ensure all raw rows use the declared canonical fields.
    metric_rows = [{field: row.get(field) for field in METRIC_SAMPLE_FIELDS} for row in metric_rows]
    event_rows = [{field: row.get(field) for field in EVENT_RECORD_FIELDS} for row in event_rows]
    snapshot_rows = [{field: row.get(field) for field in SNAPSHOT_RECORD_FIELDS} for row in snapshot_rows]

    write_schema_files(output / "schemas")
    export_workbook_rules(workbook, output / "workbook_rules")
    raw_dir = output / "raw_normalized"
    raw_counts = {
        "metric_samples.jsonl": write_jsonl(raw_dir / "metric_samples.jsonl", metric_rows),
        "event_records.jsonl": write_jsonl(raw_dir / "event_records.jsonl", event_rows),
        "snapshot_records.jsonl": write_jsonl(raw_dir / "snapshot_records.jsonl", snapshot_rows),
    }
    feature_counts = build_features(raw_dir, output / "features", seed=seed)
    examples = create_examples(output)

    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "scale": scale,
        "seed": seed,
        "generation_time": generation_time,
        "generator_version": GENERATOR_VERSION,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "policy_threshold_version": POLICY_THRESHOLD_VERSION,
        "hardware_normalization_version": HARDWARE_NORMALIZATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "site_count": len(sites),
        "days_per_site": scale_conf["days_per_site"],
        "episode_count": len(episodes),
        "sites": sites,
        "raw_record_counts": raw_counts,
        "feature_row_counts": feature_counts,
        "label_distribution": read_label_distribution(output),
        "scenario_distribution": read_scenario_distribution(output),
        "scenario_family_distribution": read_scenario_family_distribution(output),
        "scenario_variant_count": read_scenario_variant_count(output),
        "counterfactual_group_count": read_counterfactual_group_count(output),
        "examples": examples,
        "file_hashes": directory_file_hashes(output),
        "notes": [
            "All data is synthetic and fictional.",
            "Labels are generated from latent scenario truth plus workbook-inspired composite rules.",
            "Missing telemetry is represented with coverage and missing-reason fields, not encoded as zero activity.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    write_data_readme(output, manifest)
    # Refresh manifest hashes after README/manifest writes.
    manifest["file_hashes"] = directory_file_hashes(output)
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_v0"))
    parser.add_argument("--scale", choices=sorted(SCALE_PRESETS), default="v0")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("xx_private/docs/ai_training_run_ground_truth_ranges.xlsx"),
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output exists instead of replacing it.")
    args = parser.parse_args()
    if args.output.exists() and args.no_overwrite:
        raise FileExistsError(f"output already exists: {args.output}")
    manifest = generate_dataset(args.output, args.scale, args.seed, args.workbook, overwrite=not args.no_overwrite)
    print(f"dataset_id: {manifest['dataset_id']}")
    print(f"raw_records: {sum(manifest['raw_record_counts'].values())}")
    print(f"feature_rows: {manifest['feature_row_counts']['window_features_all.csv']}")
    print(f"label_distribution: {manifest['label_distribution']}")


if __name__ == "__main__":
    main()
