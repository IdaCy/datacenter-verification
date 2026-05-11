"""Shared contracts and helpers for the synthetic datacenter dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


GENERATOR_VERSION = "synthetic-generator-v0.2.0"
FEATURE_PIPELINE_VERSION = "features-v0.2.0"
POLICY_THRESHOLD_VERSION = "policy-thresholds-v0.1.0"
HARDWARE_NORMALIZATION_VERSION = "h100e-normalization-v0.1.0"
SCHEMA_VERSION = "synthetic-raw-v0.1.0"

POLICY_CONCURRENCY_THRESHOLD_GPUS = 512.0
POLICY_GPU_HOURS_THRESHOLD = 512.0 * 24.0

OBSERVABLES: dict[str, str] = {
    "O1": "hardware inventory and accelerator capacity",
    "O2": "scheduler, job, reservation, and allocation metadata",
    "O3": "cloud control-plane, capacity reservation, and billing records",
    "O4": "on-device GPU telemetry",
    "O5": "profiler, kernel, and counter telemetry",
    "O6": "intra-node GPU fabric",
    "O7": "scale-out network and fabric telemetry",
    "O8": "rack, server, and facility power telemetry",
    "O9": "cooling and thermal telemetry",
    "O10": "host, VM, container, process, and distributed-runtime metadata",
    "O11": "storage, object-store, filesystem, and data-movement logs",
    "O12": "workload declarations, experiment trackers, and ML logs",
    "O13": "attestation and trusted-computing artifacts",
    "O14": "monitoring-pipeline integrity, coverage, and time synchronization",
    "O15": "physical, security, maintenance, and firmware records",
    "O16": "active challenge probes",
    "O17": "external and out-of-band observables",
}

OBSERVABLE_IDS = list(OBSERVABLES)

MISSING_REASONS = [
    "observed",
    "not_applicable",
    "collector_gap",
    "counter_disabled_by_cc_mode",
    "routine_profiler_disabled",
    "privacy_redacted",
    "delayed_log_delivery",
    "not_scheduled",
    "source_not_deployed",
    "unknown",
]

WINDOW_SPECS: dict[str, int] = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "1d": 24 * 60 * 60,
}

SCALE_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "site_count": 1,
        "days_per_site": 7,
        "episode_count": 42,
        "max_windows_per_episode_per_length": 3,
        "metric_sample_minutes": 60,
    },
    "v0": {
        "site_count": 3,
        "days_per_site": 30,
        "episode_count": 210,
        "max_windows_per_episode_per_length": 8,
        "metric_sample_minutes": 60,
    },
    "v1": {
        "site_count": 7,
        "days_per_site": 75,
        "episode_count": 760,
        "counterfactual_group_count": 100,
        "max_windows_per_episode_per_length": 6,
        "metric_sample_minutes": 240,
        "max_metric_samples_per_episode": 12,
    },
    "study": {
        "site_count": 10,
        "days_per_site": 120,
        "episode_count": 2200,
        "counterfactual_group_count": 240,
        "max_windows_per_episode_per_length": 12,
        "metric_sample_minutes": 120,
        "max_metric_samples_per_episode": 24,
    },
    "stress": {
        "site_count": 30,
        "days_per_site": 240,
        "episode_count": 12000,
        "counterfactual_group_count": 900,
        "max_windows_per_episode_per_length": 16,
        "metric_sample_minutes": 240,
        "max_metric_samples_per_episode": 24,
    },
}

HARD_GENERATION_SCALES = {"v1", "study", "stress"}

SCENARIO_CLASSES = [
    "idle",
    "normal_inference",
    "large_batch_inference",
    "synthetic_data_generation",
    "small_fine_tune",
    "large_fine_tune",
    "pretraining",
    "hpc_mpi_simulation",
    "nccl_benchmark",
    "hardware_burn_in",
    "storage_rebuild",
    "large_etl_data_movement",
    "reserved_but_unused_capacity",
    "cloud_reservation_used_for_training",
    "maintenance_window",
    "adversarial_fragmented_training",
    "underclocked_long_duration_training",
    "counter_suppressed_candidate_window",
]

V1_HARD_POSITIVE_FAMILIES = [
    "pretraining_standard",
    "large_fine_tune_standard",
    "cloud_training_redacted_runtime",
    "training_without_semantic_logs",
    "underclocked_energy_capped_training",
    "elastic_preempted_training",
    "fragmented_training_linked",
    "sparse_or_moe_bursty_training",
    "training_with_low_fabric_high_checkpoint",
    "training_with_delayed_logs",
    "multi_stage_training_pipeline",
]

V1_HARD_NEGATIVE_FAMILIES = [
    "idle_or_low_activity",
    "normal_inference",
    "large_batch_inference",
    "model_parallel_inference",
    "embedding_generation",
    "synthetic_data_generation_gpu_heavy",
    "hpc_mpi_collective",
    "nccl_extended_benchmark",
    "hardware_burn_in_or_thermal_soak",
    "storage_rebuild_or_replication",
    "large_etl_or_data_movement",
    "distributed_database_or_graph_analytics",
    "reserved_but_unused_capacity",
    "maintenance_with_collector_gaps",
    "multi_tenant_fragmented_nontraining",
    "counter_suppressed_candidate_window",
    "capacity_or_integrity_only_warning",
]

V1_SCENARIO_FAMILIES = V1_HARD_POSITIVE_FAMILIES + V1_HARD_NEGATIVE_FAMILIES

TRAINING_SCENARIOS = {
    "small_fine_tune",
    "large_fine_tune",
    "pretraining",
    "cloud_reservation_used_for_training",
    "adversarial_fragmented_training",
    "underclocked_long_duration_training",
    "counter_suppressed_candidate_window",
    *V1_HARD_POSITIVE_FAMILIES,
}

FALSE_POSITIVE_SCENARIOS = {
    "large_batch_inference",
    "synthetic_data_generation",
    "hpc_mpi_simulation",
    "nccl_benchmark",
    "hardware_burn_in",
    "storage_rebuild",
    "large_etl_data_movement",
    "reserved_but_unused_capacity",
    "maintenance_window",
    *V1_HARD_NEGATIVE_FAMILIES,
}

IDENTIFIER_COLUMNS = [
    "feature_row_id",
    "dataset_id",
    "seed",
    "site_id",
    "scope_type",
    "scope_id_hash",
    "window_start",
    "window_end",
    "window_length_seconds",
    "episode_id",
    "latent_workload_class",
    "scenario_family",
    "scenario_variant",
    "evidence_recipe_id",
    "temporal_phase",
    "data_quality_regime",
    "privacy_tier",
    "counterfactual_group_id",
    "synthetic_counterfactual_role",
    "collector_profile",
    "topology_class",
    "synthetic_hard_case_tags",
    "label_0_to_4",
    "label_confidence",
    "label_reason",
    "label_source",
]

FEATURE_VALUE_COLUMNS = [
    "capacity_possible",
    "policy_compute_ratio",
    "o1_normalized_h100e_capacity",
    "o1_largest_contiguous_domain_gpus",
    "o1_homogeneous_high_end_fraction",
    "o1_non_partitioned_fraction",
    "o1_inventory_delta_rate",
    "o17_external_capacity_conflict_score",
    "o2_max_concurrent_normalized_gpus",
    "o2_allocation_duration_hours",
    "o2_gpu_hours_policy_ratio",
    "o2_concurrency_fraction_domain",
    "o2_topology_contiguity_score",
    "o2_declared_workload_class",
    "o2_reservation_exclusive_flag",
    "o2_elastic_resize_count",
    "o2_preemption_restart_count",
    "o2_scheduler_queue_delay_hours",
    "o2_account_linkage_confidence",
    "o2_job_array_width",
    "o2_reservation_reuse_count",
    "o3_batch_provisioned_gpus",
    "o3_capacity_reservation_duration_hours",
    "o3_training_sku_fraction",
    "o3_billing_continuity_score",
    "o3_egress_tb",
    "o4_gpu_util_p50",
    "o4_gpu_util_p95",
    "o4_gpu_util_duty_gt_70",
    "o4_sm_tensor_active_p95",
    "o4_hbm_used_fraction_p50",
    "o4_hbm_bandwidth_active_p95",
    "o4_gpu_power_fraction_p95",
    "o4_error_spike_score",
    "o4_gpu_util_cv",
    "o4_gpu_idle_gap_p95_minutes",
    "o4_hbm_pressure_duration_fraction",
    "o4_power_cap_active_fraction",
    "o4_thermal_throttle_fraction",
    "o5_kernel_training_motif_score",
    "o5_tensor_throughput_ratio",
    "o5_profiler_available",
    "o6_nvlink_util_p95",
    "o6_nvlink_periodicity_score",
    "o6_link_error_spike_score",
    "o7_scaleout_port_util_p95",
    "o7_synchronized_fabric_footprint",
    "o7_collective_periodicity_score",
    "o7_burst_duty_cycle",
    "o7_rdma_congestion_score",
    "o7_job_to_port_mapping_coverage",
    "o7_flow_entropy_score",
    "o7_cross_section_sync_score",
    "o7_collective_jitter_score",
    "o7_storage_traffic_fraction",
    "o7_inference_fanout_score",
    "o7_account_flow_linkage_confidence",
    "o8_rack_power_fraction_p95",
    "o8_facility_it_power_mw",
    "o8_baseline_subtracted_energy_kwh",
    "o8_power_continuity_days",
    "o8_power_cv",
    "o8_power_to_gpu_residual",
    "o8_power_baseline_drift_score",
    "o8_power_cap_or_curtailment_active",
    "o8_unattributed_power_fraction",
    "o9_gpu_hbm_temp_score",
    "o9_thermal_delta_t_score",
    "o9_cooling_flow_duty",
    "o9_cooling_maintenance_active",
    "o9_thermal_throttle_support_score",
    "o10_world_size",
    "o10_runtime_framework_class",
    "o10_rank_stability_score",
    "o10_same_image_gpu_count",
    "o10_rendezvous_present",
    "o10_runtime_metadata_confidence",
    "o10_declared_vs_observed_mismatch_score",
    "o11_data_staging_tb",
    "o11_checkpoint_write_tb_per_event",
    "o11_checkpoint_periodicity_score",
    "o11_read_write_training_pattern_score",
    "o11_checkpoint_jitter_score",
    "o11_artifact_write_pattern_score",
    "o11_dataloader_read_pattern_score",
    "o11_backup_or_replication_pattern_score",
    "o11_storage_cotraffic_score",
    "o12_signed_ml_logs_present",
    "o12_declared_parameter_count_b",
    "o12_training_tokens_b",
    "o12_step_count",
    "o12_loss_curve_present",
    "o12_optimizer_state_present",
    "o12_log_delivery_delay_hours",
    "o12_log_completeness_fraction",
    "o12_declaration_consistency_score",
    "o13_attestation_valid_fraction",
    "o13_confidential_compute_mode_fraction",
    "o13_collector_measurement_valid",
    "o14_min_critical_coverage",
    "o14_gap_fraction_critical",
    "o14_clock_drift_max_ms",
    "o14_counter_reset_count",
    "o15_unapproved_physical_change_near_window",
    "o15_firmware_bmc_change_near_window",
    "o16_probe_throughput_ratio_min",
    "o16_probe_latency_inflation_max",
    "o16_vram_residency_conflict_score",
    "o17_external_capacity_assertion",
    "o17_energy_contract_alignment_score",
    "o17_network_provider_utilization_score",
    "o17_procurement_or_maintenance_explanation_score",
]

COVERAGE_COLUMNS: list[str] = []
for _obs_id in OBSERVABLE_IDS:
    _key = _obs_id.lower()
    COVERAGE_COLUMNS.extend([f"{_key}_coverage_fraction", f"{_key}_missing_reason"])

TRUST_COLUMNS = [
    "scheduler_signature_status",
    "gpu_telemetry_trust_level",
    "fabric_telemetry_trust_level",
    "power_meter_trust_level",
    "feature_pipeline_version",
    "policy_threshold_version",
    "hardware_normalization_version",
    "raw_input_manifest_hash",
]

SYNTHETIC_AUDIT_COLUMNS = [
    "capacity_evidence_only",
    "integrity_evidence_only",
    "physical_evidence_only",
    "synthetic_evidence_profile",
]

REQUIRED_FEATURE_COLUMNS = (
    IDENTIFIER_COLUMNS
    + FEATURE_VALUE_COLUMNS
    + COVERAGE_COLUMNS
    + TRUST_COLUMNS
    + SYNTHETIC_AUDIT_COLUMNS
)

V1_ONLY_FEATURE_COLUMNS = [
    "scenario_family",
    "scenario_variant",
    "evidence_recipe_id",
    "temporal_phase",
    "data_quality_regime",
    "privacy_tier",
    "counterfactual_group_id",
    "synthetic_counterfactual_role",
    "collector_profile",
    "topology_class",
    "synthetic_hard_case_tags",
    "o2_elastic_resize_count",
    "o2_preemption_restart_count",
    "o2_scheduler_queue_delay_hours",
    "o2_account_linkage_confidence",
    "o2_job_array_width",
    "o2_reservation_reuse_count",
    "o4_gpu_util_cv",
    "o4_gpu_idle_gap_p95_minutes",
    "o4_hbm_pressure_duration_fraction",
    "o4_power_cap_active_fraction",
    "o4_thermal_throttle_fraction",
    "o7_flow_entropy_score",
    "o7_cross_section_sync_score",
    "o7_collective_jitter_score",
    "o7_storage_traffic_fraction",
    "o7_inference_fanout_score",
    "o7_account_flow_linkage_confidence",
    "o8_power_baseline_drift_score",
    "o8_power_cap_or_curtailment_active",
    "o8_unattributed_power_fraction",
    "o9_cooling_maintenance_active",
    "o9_thermal_throttle_support_score",
    "o10_runtime_metadata_confidence",
    "o10_declared_vs_observed_mismatch_score",
    "o11_checkpoint_jitter_score",
    "o11_artifact_write_pattern_score",
    "o11_dataloader_read_pattern_score",
    "o11_backup_or_replication_pattern_score",
    "o11_storage_cotraffic_score",
    "o12_log_delivery_delay_hours",
    "o12_log_completeness_fraction",
    "o12_declaration_consistency_score",
    "o17_external_capacity_assertion",
    "o17_energy_contract_alignment_score",
    "o17_network_provider_utilization_score",
    "o17_procurement_or_maintenance_explanation_score",
]

V1_ONLY_FEATURE_COLUMN_SET = set(V1_ONLY_FEATURE_COLUMNS)
BASE_REQUIRED_FEATURE_COLUMNS = [
    column for column in REQUIRED_FEATURE_COLUMNS if column not in V1_ONLY_FEATURE_COLUMN_SET
]

METRIC_SAMPLE_FIELDS = [
    "metric_sample_id",
    "site_id",
    "observable_id",
    "source_system",
    "metric_name",
    "entity_type",
    "entity_id_hash",
    "parent_entity_id_hash",
    "event_time",
    "ingest_time",
    "collector_time",
    "source_clock_offset_ms",
    "sample_interval_ms",
    "value_num",
    "value_text",
    "unit",
    "coverage_fraction",
    "trust_level",
    "signature_status",
    "raw_payload_hash",
    "ingest_batch_id",
    "schema_version",
    "episode_id",
    "latent_workload_class",
]

EVENT_RECORD_FIELDS = [
    "event_record_id",
    "site_id",
    "observable_id",
    "source_system",
    "event_type",
    "scope_type",
    "scope_id_hash",
    "account_id_hash",
    "job_id_hash",
    "entity_type",
    "entity_id_hash",
    "event_time",
    "event_end_time",
    "ingest_time",
    "source_clock_offset_ms",
    "attributes_json",
    "trust_level",
    "signature_status",
    "raw_payload_hash",
    "ingest_batch_id",
    "schema_version",
    "episode_id",
    "latent_workload_class",
]

SNAPSHOT_RECORD_FIELDS = [
    "snapshot_id",
    "site_id",
    "observable_id",
    "snapshot_type",
    "scope_type",
    "scope_id_hash",
    "valid_from",
    "valid_to",
    "observed_at",
    "ingest_time",
    "attributes_json",
    "trust_level",
    "signature_status",
    "raw_payload_hash",
    "ingest_batch_id",
    "schema_version",
    "episode_id",
    "latent_workload_class",
]


def utc_iso(dt: datetime) -> str:
    """Return an ISO-8601 UTC timestamp ending in Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def parse_utc(ts: str) -> datetime:
    if not isinstance(ts, str) or not ts.endswith("Z"):
        raise ValueError(f"expected UTC timestamp ending in Z, got {ts!r}")
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def stable_hash(*parts: object, length: int = 16) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def raw_payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc_iso(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_safe) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
            count += 1
    return count


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json_dumps(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def directory_file_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(root))] = file_sha256(path)
    return hashes


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def round_float(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def normalize_feature_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column) for column in REQUIRED_FEATURE_COLUMNS}
    for key, value in list(normalized.items()):
        normalized[key] = round_float(value)
    return normalized


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, str):
        if value.lower() in {"true", "false"}:
            return 1.0 if value.lower() == "true" else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def score_o2(row: dict[str, Any]) -> int:
    gpus = _num(row, "o2_max_concurrent_normalized_gpus")
    duration = _num(row, "o2_allocation_duration_hours")
    ratio = _num(row, "o2_gpu_hours_policy_ratio")
    fraction = _num(row, "o2_concurrency_fraction_domain")
    contiguity = _num(row, "o2_topology_contiguity_score")
    if gpus >= 2000 or ratio >= 3 or (fraction >= 0.8 and contiguity >= 0.8 and duration >= 24):
        return 3
    if gpus >= 512 or ratio >= 1 or (fraction >= 0.5 and duration >= 24):
        return 3
    if gpus >= 64 or ratio >= 0.25 or fraction >= 0.2:
        return 2
    if gpus > 0:
        return 1
    return 0


def score_o3(row: dict[str, Any]) -> int:
    provisioned = _num(row, "o3_batch_provisioned_gpus")
    duration = _num(row, "o3_capacity_reservation_duration_hours")
    training_sku = _num(row, "o3_training_sku_fraction")
    billing = _num(row, "o3_billing_continuity_score")
    if provisioned >= 2000 and training_sku >= 0.8 and billing >= 0.8:
        return 3
    if provisioned >= 512 and duration >= 24 and training_sku >= 0.7:
        return 3
    if provisioned >= 64 or (duration >= 24 and training_sku >= 0.5):
        return 2
    if provisioned > 0:
        return 1
    return 0


def score_o4(row: dict[str, Any]) -> int:
    util = _num(row, "o4_gpu_util_p95")
    duty = _num(row, "o4_gpu_util_duty_gt_70")
    tensor = _num(row, "o4_sm_tensor_active_p95")
    hbm = _num(row, "o4_hbm_used_fraction_p50")
    hbm_pressure = _num(row, "o4_hbm_pressure_duration_fraction")
    power_cap = _num(row, "o4_power_cap_active_fraction")
    if util >= 85 and duty >= 0.75 and tensor >= 70 and hbm >= 0.55:
        return 3
    if util >= 55 and duty >= 0.45 and tensor >= 60 and (hbm >= 0.65 or hbm_pressure >= 0.55) and power_cap >= 0.25:
        return 3
    if util >= 70 and duty >= 0.4:
        return 2
    if util >= 50 and (hbm_pressure >= 0.55 or tensor >= 60):
        return 2
    if util >= 20:
        return 1
    return 0


def score_o5(row: dict[str, Any]) -> int:
    motif = _num(row, "o5_kernel_training_motif_score")
    throughput = _num(row, "o5_tensor_throughput_ratio")
    if motif >= 0.75 and throughput >= 0.7:
        return 3
    if motif >= 0.45:
        return 2
    if motif > 0:
        return 1
    return 0


def score_o6(row: dict[str, Any]) -> int:
    util = _num(row, "o6_nvlink_util_p95")
    periodicity = _num(row, "o6_nvlink_periodicity_score")
    if util >= 0.65 and periodicity >= 0.65:
        return 3
    if util >= 0.35 or periodicity >= 0.45:
        return 2
    if util > 0.05:
        return 1
    return 0


def score_o7(row: dict[str, Any]) -> int:
    util = _num(row, "o7_scaleout_port_util_p95")
    footprint = _num(row, "o7_synchronized_fabric_footprint")
    periodicity = _num(row, "o7_collective_periodicity_score")
    mapping = _num(row, "o7_job_to_port_mapping_coverage")
    sync = _num(row, "o7_cross_section_sync_score")
    linkage = _num(row, "o7_account_flow_linkage_confidence")
    if util >= 0.7 and footprint >= 512 and periodicity >= 0.7 and mapping >= 0.6:
        return 3
    if footprint >= 512 and periodicity >= 0.62 and sync >= 0.62 and linkage >= 0.45:
        return 3
    if util >= 0.45 and footprint >= 128 and periodicity >= 0.45:
        return 2
    if footprint >= 128 and (sync >= 0.45 or linkage >= 0.5):
        return 2
    if util > 0.08 or footprint > 0:
        return 1
    return 0


def score_supportive(row: dict[str, Any]) -> int:
    power = _num(row, "o8_rack_power_fraction_p95")
    energy = _num(row, "o8_baseline_subtracted_energy_kwh")
    thermal = _num(row, "o9_thermal_delta_t_score")
    storage = _num(row, "o11_checkpoint_periodicity_score")
    artifact = _num(row, "o11_artifact_write_pattern_score")
    dataloader = _num(row, "o11_dataloader_read_pattern_score")
    throttle = _num(row, "o9_thermal_throttle_support_score")
    if (power >= 0.82 and thermal >= 0.65) or (storage >= 0.7 and energy >= 20000):
        return 3
    if storage >= 0.6 and (artifact >= 0.45 or dataloader >= 0.55):
        return 3
    if power >= 0.65 or thermal >= 0.55 or storage >= 0.45 or throttle >= 0.55:
        return 2
    if power >= 0.35 or energy > 0:
        return 1
    return 0


def score_semantic(row: dict[str, Any]) -> int:
    framework = str(row.get("o10_runtime_framework_class") or "")
    world_size = _num(row, "o10_world_size")
    rank_stability = _num(row, "o10_rank_stability_score")
    checkpoints = _num(row, "o11_checkpoint_periodicity_score")
    signed_logs = _bool(row, "o12_signed_ml_logs_present")
    params = _num(row, "o12_declared_parameter_count_b")
    tokens = _num(row, "o12_training_tokens_b")
    steps = _num(row, "o12_step_count")
    runtime_confidence = _num(row, "o10_runtime_metadata_confidence", 1.0)
    log_completeness = _num(row, "o12_log_completeness_fraction", 1.0 if signed_logs else 0.0)
    declaration_consistency = _num(row, "o12_declaration_consistency_score", 1.0 if signed_logs else 0.0)
    if signed_logs and (params >= 50 or tokens >= 100 or steps >= 10000):
        return 4
    if signed_logs and log_completeness >= 0.45 and declaration_consistency >= 0.5 and (params >= 20 or steps >= 5000):
        return 4
    if "training" in framework and world_size >= 512 and rank_stability >= 0.7 and runtime_confidence >= 0.35:
        return 3
    if ("pytorch" in framework or "jax" in framework or "deepspeed" in framework) and checkpoints >= 0.45:
        return 2
    if world_size > 1 or framework:
        return 1
    return 0


def critical_coverage(row: dict[str, Any]) -> float:
    keys = [
        "o2_coverage_fraction",
        "o4_coverage_fraction",
        "o7_coverage_fraction",
        "o8_coverage_fraction",
        "o14_coverage_fraction",
    ]
    return min(_num(row, key, 0.0) for key in keys)


def _v1_countervailing_score(row: dict[str, Any]) -> int:
    """Count deploy-time features that point to a high-load non-training explanation."""
    runtime = str(row.get("o10_runtime_framework_class") or "").lower()
    declared = str(row.get("o2_declared_workload_class") or "").lower()
    score = 0
    if any(
        token in runtime
        for token in [
            "inference",
            "embedding",
            "synthetic_data",
            "hpc",
            "mpi",
            "nccl",
            "benchmark",
            "burn_in",
            "storage",
            "etl",
            "database",
            "graph",
        ]
    ):
        score += 1
    if declared in {"inference", "embedding", "synthetic_data", "hpc", "benchmark", "burn_in", "data", "database", "reserved", "maintenance"}:
        score += 1
    if _num(row, "o7_inference_fanout_score") >= 0.6:
        score += 1
    if _num(row, "o7_storage_traffic_fraction") >= 0.55 or _num(row, "o11_backup_or_replication_pattern_score") >= 0.6:
        score += 1
    if _num(row, "o10_declared_vs_observed_mismatch_score") >= 0.65 and score >= 1:
        score += 1
    if _num(row, "o17_procurement_or_maintenance_explanation_score") >= 0.7:
        score += 1
    return score


def _signed_policy_semantic(row: dict[str, Any], policy_ratio: float) -> bool:
    return (
        _bool(row, "o12_signed_ml_logs_present")
        and policy_ratio >= 0.75
        and _num(row, "o12_log_completeness_fraction", 1.0) >= 0.55
        and _num(row, "o12_declaration_consistency_score", 1.0) >= 0.55
    )


def derive_label_v1(row: dict[str, Any]) -> tuple[int, float, str]:
    """Composite hard-profile labels based on evidence dependencies, not scenario names."""
    profile = str(row.get("synthetic_evidence_profile") or "")
    capacity_possible = _bool(row, "capacity_possible")
    external_conflict = _num(row, "o17_external_capacity_conflict_score")
    primary_scores = {
        "O2": score_o2(row),
        "O3": score_o3(row),
        "O4": score_o4(row),
        "O5": score_o5(row),
        "O6": score_o6(row),
        "O7": score_o7(row),
    }
    high_primary_count = sum(1 for value in primary_scores.values() if value >= 3)
    elevated_primary_count = sum(1 for value in primary_scores.values() if value >= 2)
    max_primary = max(primary_scores.values())
    supportive = score_supportive(row)
    semantic = score_semantic(row)
    integrity = critical_coverage(row)
    high_integrity = integrity >= 0.95 and _num(row, "o13_attestation_valid_fraction") >= 0.95
    adequate_integrity = integrity >= 0.72 and _bool(row, "o13_collector_measurement_valid")
    policy_ratio = _num(row, "policy_compute_ratio")
    countervailing = _v1_countervailing_score(row)
    storage_training = (
        _num(row, "o11_checkpoint_periodicity_score") >= 0.5
        or _num(row, "o11_artifact_write_pattern_score") >= 0.55
        or _num(row, "o11_dataloader_read_pattern_score") >= 0.55
    )
    runtime_redacted = str(row.get("o10_missing_reason") or "") == "privacy_redacted" or _num(row, "o10_runtime_metadata_confidence") < 0.45
    signed_policy = _signed_policy_semantic(row, policy_ratio)

    if not capacity_possible and external_conflict < 0.6:
        if integrity >= 0.95:
            return 0, 0.95, "capacity below policy threshold with strong coverage"
        return 1, 0.63, "capacity below threshold, but coverage is not strong enough for negative certification"

    if profile == "capacity_only":
        return 1, 0.82, "capacity or reservation evidence alone is capped at training possible"

    if profile == "integrity_only":
        if supportive >= 2 or external_conflict >= 0.5:
            return 2, 0.68, "integrity anomaly is elevated only because it overlaps physical or external conflict evidence"
        return 1, 0.64, "integrity-only evidence is not direct training proof"

    if profile == "physical_only":
        if supportive >= 2:
            return 2, 0.68, "physical-only elevation is capped at label 2 without primary corroboration"
        return 1, 0.72, "physical activity alone is weak and non-semantic"

    if signed_policy and max_primary >= 1 and countervailing < 2:
        return 4, 0.95, "policy-scale allocation plus authenticated ML declaration and primary activity"

    full_stack = (
        primary_scores["O2"] >= 3
        and primary_scores["O4"] >= 2
        and primary_scores["O7"] >= 2
        and supportive >= 2
        and adequate_integrity
        and policy_ratio >= 0.75
    )
    if countervailing >= 2 and not signed_policy:
        if countervailing >= 3 and not storage_training and semantic <= 1 and integrity >= 0.95 and max_primary <= 1 and supportive <= 1:
            return 0, 0.87, "strong non-training explanation with high coverage rules out large training"
        if countervailing >= 3 and not storage_training and max_primary <= 2:
            return 1, 0.76, "strong non-training explanation leaves training possible but not elevated"
        if max_primary >= 2 or supportive >= 2 or semantic >= 2:
            return 2, 0.78, "high-load non-training explanation countervails training-like primary evidence"
        if integrity >= 0.95:
            return 0, 0.87, "benign high-capacity window has strong coverage and no coherent training evidence"
        return 1, 0.7, "benign or non-training explanation with incomplete coverage"

    if full_stack and (storage_training or semantic >= 2 or runtime_redacted):
        if integrity < 0.72:
            return 2, 0.69, "coherent activity exists but critical coverage is too weak for likely-training confidence"
        if runtime_redacted and semantic < 3:
            return 3, 0.84, "allocation, GPU, fabric, and storage cohere despite redacted runtime metadata"
        return 3, 0.88, "two primary layers plus supportive or semantic corroboration support likely training"

    if primary_scores["O2"] >= 3 and primary_scores["O4"] >= 2 and supportive >= 2 and storage_training and adequate_integrity:
        return 3, 0.85, "long allocation, GPU/memory pressure, storage cadence, and physical support cohere"

    if elevated_primary_count >= 2 and supportive >= 2 and adequate_integrity and policy_ratio >= 0.5:
        return 3, 0.83, "multiple elevated primary layers and corroboration support likely training"

    if elevated_primary_count >= 1:
        if supportive >= 2 and (integrity < 0.75 or _num(row, "o14_gap_fraction_critical") > 0.15):
            return 2, 0.69, "activity or physical evidence is elevated but critical telemetry is incomplete"
        return 2, 0.74, "one primary layer is materially elevated"

    if supportive >= 2 and (critical_coverage(row) < 0.85 or external_conflict >= 0.5):
        return 2, 0.67, "supportive physical or external evidence overlaps incomplete activity telemetry"

    if capacity_possible:
        if integrity >= 0.95 and max_primary == 0 and supportive <= 1 and semantic <= 1:
            return 0, 0.9, "capacity exists, but all primary activity layers are observed low"
        return 1, 0.75, "capacity exists but activity evidence remains weak or sparse"

    return 0, 0.85, "no training likely after composite rules"


def derive_label(row: dict[str, Any]) -> tuple[int, float, str]:
    """Apply workbook-inspired composite rules to a feature row."""
    if row.get("scenario_family"):
        return derive_label_v1(row)

    scenario = str(row.get("latent_workload_class") or "")
    profile = str(row.get("synthetic_evidence_profile") or "")
    capacity_possible = _bool(row, "capacity_possible")
    external_conflict = _num(row, "o17_external_capacity_conflict_score")
    primary_scores = {
        "O2": score_o2(row),
        "O3": score_o3(row),
        "O4": score_o4(row),
        "O5": score_o5(row),
        "O6": score_o6(row),
        "O7": score_o7(row),
    }
    high_primary_count = sum(1 for value in primary_scores.values() if value >= 3)
    elevated_primary_count = sum(1 for value in primary_scores.values() if value >= 2)
    max_primary = max(primary_scores.values())
    supportive = score_supportive(row)
    semantic = score_semantic(row)
    integrity = critical_coverage(row)
    high_integrity = integrity >= 0.95 and _num(row, "o13_attestation_valid_fraction") >= 0.95
    adequate_integrity = integrity >= 0.75 and _bool(row, "o13_collector_measurement_valid")
    policy_ratio = _num(row, "policy_compute_ratio")

    if not capacity_possible and external_conflict < 0.6:
        if integrity >= 0.95:
            return 0, 0.95, "capacity below policy threshold with strong coverage"
        return 1, 0.63, "capacity below threshold, but coverage is not strong enough for negative certification"

    if profile == "capacity_only":
        if capacity_possible:
            return 1, 0.82, "capacity-only evidence is capped at training possible"
        return 0, 0.9, "capacity below policy threshold"

    if profile == "integrity_only":
        return 1, 0.64, "integrity anomaly without primary activity is not direct training evidence"

    if profile == "physical_only":
        if supportive >= 2:
            return 2, 0.68, "physical-only elevation is capped at label 2 without primary corroboration"
        return 1, 0.72, "physical activity alone is weak and non-semantic"

    if semantic >= 4 and policy_ratio >= 1 and scenario in TRAINING_SCENARIOS:
        return 4, 0.96, "authenticated ML evidence crosses the policy compute threshold"

    if (
        scenario in {"pretraining", "cloud_reservation_used_for_training"}
        and high_integrity
        and policy_ratio >= 1
        and primary_scores["O2"] >= 3
        and primary_scores["O4"] >= 3
        and primary_scores["O7"] >= 3
        and supportive >= 3
    ):
        return 4, 0.93, "scheduler, GPU, fabric, power, and storage cohere under high integrity"

    if scenario in FALSE_POSITIVE_SCENARIOS:
        if max_primary >= 2 or supportive >= 2 or semantic >= 2:
            return 2, 0.78, f"{scenario} is a high-load false-positive class requiring cross-checks"
        if integrity >= 0.95:
            return 0, 0.88, f"{scenario} has no large-training activity after cross-checking"
        return 1, 0.7, f"{scenario} lacks enough coverage for label 0"

    if scenario == "normal_inference":
        return 1, 0.78, "routine inference on capable monitored capacity leaves training possible but not elevated"

    if scenario == "idle":
        if integrity >= 0.95 and max_primary <= 1 and supportive <= 1 and semantic <= 1:
            return 0, 0.94, "idle window has strong critical-layer coverage"
        return 1, 0.66, "idle-like window lacks enough coverage for negative certification"

    if scenario == "small_fine_tune" and policy_ratio < 1:
        return 1, 0.8, "small fine-tune remains below policy compute threshold"

    if scenario == "adversarial_fragmented_training":
        return 2, 0.78, "fragmented sub-threshold activity is elevated until jobs/accounts are linked"

    if high_primary_count >= 2 and (supportive >= 2 or semantic >= 2) and adequate_integrity:
        return 3, 0.89, "two independent primary layers plus corroboration support likely training"

    if max_primary >= 3 and (supportive >= 2 or semantic >= 2) and adequate_integrity:
        return 3, 0.86, "one strong primary layer plus semantic/supportive corroboration supports likely training"

    if elevated_primary_count >= 1:
        return 2, 0.74, "one primary layer is materially elevated"

    if supportive >= 2 and (critical_coverage(row) < 0.75 or _num(row, "o14_gap_fraction_critical") > 0.15):
        return 2, 0.69, "supportive physical evidence aligns with incomplete activity telemetry"

    if capacity_possible:
        if integrity >= 0.95 and max_primary == 0 and supportive <= 1 and semantic <= 1:
            return 0, 0.9, "capacity exists, but all primary activity layers are observed low"
        return 1, 0.75, "capacity exists but activity evidence remains weak or sparse"

    return 0, 0.85, "no training likely after composite rules"


def build_basic_schema(name: str, required: list[str]) -> dict[str, Any]:
    properties = {field: {} for field in required}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": name,
        "type": "object",
        "required": required,
        "additionalProperties": True,
        "properties": properties,
    }


def write_schema_files(schema_dir: Path) -> None:
    write_json(schema_dir / "metric_sample.schema.json", build_basic_schema("Metric sample", METRIC_SAMPLE_FIELDS))
    write_json(schema_dir / "event_record.schema.json", build_basic_schema("Event record", EVENT_RECORD_FIELDS))
    write_json(schema_dir / "snapshot_record.schema.json", build_basic_schema("Snapshot record", SNAPSHOT_RECORD_FIELDS))
    write_json(
        schema_dir / "window_feature_row.schema.json",
        build_basic_schema("Window feature row", REQUIRED_FEATURE_COLUMNS),
    )
    write_json(
        schema_dir / "prediction_record.schema.json",
        build_basic_schema(
            "Prediction record",
            [
                "prediction_id",
                "created_at",
                "site_id",
                "scope_type",
                "scope_id_hash",
                "window_start",
                "window_end",
                "model_version",
                "p_label_0",
                "p_label_1",
                "p_label_2",
                "p_label_3",
                "p_label_4",
                "feature_row_hash",
            ],
        ),
    )


def optional_write_parquet(csv_path: Path) -> bool:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except Exception:
        return False
    df = pd.read_csv(csv_path)
    df.to_parquet(csv_path.with_suffix(".parquet"), index=False)
    return True


def bool_from_csv(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def read_feature_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_scale(scale: str) -> dict[str, Any]:
    if scale not in SCALE_PRESETS:
        known = ", ".join(sorted(SCALE_PRESETS))
        raise ValueError(f"unknown scale {scale!r}; expected one of: {known}")
    return dict(SCALE_PRESETS[scale])
