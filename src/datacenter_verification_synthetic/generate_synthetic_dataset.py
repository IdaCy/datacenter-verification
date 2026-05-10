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
        "o8_rack_power_fraction_p95": 0.2,
        "o8_facility_it_power_mw": site["rack_power_design_mw"],
        "o8_baseline_subtracted_energy_kwh": 0.0,
        "o8_power_continuity_days": duration_hours / 24.0,
        "o8_power_cv": 0.18,
        "o8_power_to_gpu_residual": 0.0,
        "o9_gpu_hbm_temp_score": 0.08,
        "o9_thermal_delta_t_score": 0.1,
        "o9_cooling_flow_duty": 0.12,
        "o10_world_size": 0,
        "o10_runtime_framework_class": "none",
        "o10_rank_stability_score": 0.0,
        "o10_same_image_gpu_count": 0,
        "o10_rendezvous_present": False,
        "o11_data_staging_tb": 0.0,
        "o11_checkpoint_write_tb_per_event": 0.0,
        "o11_checkpoint_periodicity_score": 0.0,
        "o11_read_write_training_pattern_score": 0.0,
        "o12_signed_ml_logs_present": False,
        "o12_declared_parameter_count_b": None,
        "o12_training_tokens_b": None,
        "o12_step_count": None,
        "o12_loss_curve_present": False,
        "o12_optimizer_state_present": False,
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
    template["o4_gpu_util_p50"] = _round(util_p50, 2)
    template["o4_gpu_util_p95"] = _round(util_p95, 2)
    template["o4_gpu_util_duty_gt_70"] = _round(duty)
    template["o4_sm_tensor_active_p95"] = _round(tensor, 2)
    template["o4_hbm_used_fraction_p50"] = _round(hbm)
    template["o4_hbm_bandwidth_active_p95"] = _round(clamp((tensor / 100.0) * 0.9 + hbm * 0.25))
    template["o4_gpu_power_fraction_p95"] = _round(clamp(0.17 + util_p95 / 115.0))
    template["o6_nvlink_util_p95"] = _round(clamp(fabric * 0.85))
    template["o6_nvlink_periodicity_score"] = _round(clamp(periodicity * 0.92))
    template["o7_scaleout_port_util_p95"] = _round(clamp(fabric))
    template["o7_synchronized_fabric_footprint"] = int(max(0, min(gpus * 0.9, gpus * periodicity * 1.15)))
    template["o7_collective_periodicity_score"] = _round(clamp(periodicity))
    template["o7_burst_duty_cycle"] = _round(clamp(0.08 + periodicity * 0.35))
    template["o7_rdma_congestion_score"] = _round(clamp(fabric * periodicity * 0.75))
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


def make_episode(
    rng: np.random.Generator,
    dataset_id: str,
    seed: int,
    idx: int,
    site: dict[str, Any],
    scenario: str,
    scale_conf: dict[str, Any],
    base_time: datetime,
) -> dict[str, Any]:
    duration_hours = scenario_duration_hours(rng, scenario)
    total_hours = float(scale_conf["days_per_site"]) * 24.0
    max_start_hour = max(0.0, total_hours - min(duration_hours, total_hours * 0.95))
    start = base_time + timedelta(hours=_uniform(rng, 0, max_start_hour))
    end = start + timedelta(hours=duration_hours)
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
        "feature_template": template,
        "max_windows_per_episode_per_length": scale_conf["max_windows_per_episode_per_length"],
        "site_baseline_it_mw": site["baseline_it_mw"],
    }


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
                },
            )
        )
    if template.get("o12_signed_ml_logs_present"):
        events.append(
            base_event(
                dataset_id=dataset_id,
                episode=episode,
                observable_id="O12",
                source_system="signed_ml_declaration",
                event_type="signed_ml_training_log_summary",
                event_time=episode["episode_start"],
                event_end_time=episode["episode_end"],
                attributes={
                    "declared_parameter_count_b": template["o12_declared_parameter_count_b"],
                    "training_tokens_b": template["o12_training_tokens_b"],
                    "step_count": template["o12_step_count"],
                    "loss_curve_present": template["o12_loss_curve_present"],
                    "optimizer_state_present": template["o12_optimizer_state_present"],
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
                attributes={"external_capacity_conflict_score": template["o17_external_capacity_conflict_score"]},
            )
        )

    sample_interval = int(sample_minutes * 60 * 1000)
    sample_count = max(1, min(32, int(math.ceil((end - start).total_seconds() / (sample_minutes * 60)))))
    if sample_count == 1:
        sample_times = [start + (end - start) / 2]
    else:
        step = (end - start) / sample_count
        sample_times = [start + step * idx for idx in range(sample_count)]

    metric_specs = [
        ("O4", "dcgm_exporter", "gpu_utilization_p95", "gpu_group", template.get("o4_gpu_util_p95"), "percent", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "sm_tensor_active_p95", "gpu_group", template.get("o4_sm_tensor_active_p95"), "percent", template.get("o4_coverage_fraction", 0)),
        ("O4", "dcgm_exporter", "gpu_power_fraction_p95", "gpu_group", template.get("o4_gpu_power_fraction_p95"), "fraction", template.get("o4_coverage_fraction", 0)),
        ("O5", "dcgm_profiler", "kernel_training_motif_score", "gpu_group", template.get("o5_kernel_training_motif_score"), "score", template.get("o5_coverage_fraction", 0)),
        ("O6", "dcgm_fabric", "nvlink_util_p95", "topology_domain", template.get("o6_nvlink_util_p95"), "fraction", template.get("o6_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "scaleout_port_util_p95", "fabric_partition", template.get("o7_scaleout_port_util_p95"), "fraction", template.get("o7_coverage_fraction", 0)),
        ("O7", "ufm_telemetry", "collective_periodicity_score", "fabric_partition", template.get("o7_collective_periodicity_score"), "score", template.get("o7_coverage_fraction", 0)),
        ("O8", "bms_meter", "rack_power_fraction_p95", "power_zone", template.get("o8_rack_power_fraction_p95"), "fraction", template.get("o8_coverage_fraction", 0)),
        ("O9", "cooling_bms", "thermal_delta_t_score", "cooling_zone", template.get("o9_thermal_delta_t_score"), "score", template.get("o9_coverage_fraction", 0)),
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


def write_data_readme(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    text = f"""# Synthetic Datacenter Verification Dataset v0

This directory is generated synthetic study data. It contains fictional raw-like
datacenter telemetry, windowed feature rows, workbook-derived rule exports,
schemas, examples, and validation artifacts.

Dataset ID: `{manifest['dataset_id']}`

Scale: `{manifest['scale']}`

Seed: `{manifest['seed']}`

Generator: `{GENERATOR_VERSION}`

Regenerate:

```bash
python src/datacenter_verification_synthetic/generate_synthetic_dataset.py \\
  --output {dataset_dir.as_posix()} \\
  --scale {manifest['scale']} \\
  --seed {manifest['seed']}
```

Validate:

```bash
python src/datacenter_verification_synthetic/validate_synthetic_dataset.py \\
  --dataset {dataset_dir.as_posix()}
```

The model training unit is one row in `features/window_features_all.csv`, not an
individual raw metric sample or event record.
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
    scenarios = scenario_sequence(rng, int(scale_conf["episode_count"]))

    episodes: list[dict[str, Any]] = []
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
