#!/usr/bin/env python3
"""Validate synthetic datacenter training-run verification datasets.

The checks in this file are intentionally stronger than plain schema checks.
They test whether the generated feature rows encode the intended evidence
logic: capacity is only a gate, missing data is not zero, training labels need
multi-layer coherent signals, and false-positive workload families exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABEL_TARGETS = {
    0: (0.35, 0.55),
    1: (0.20, 0.35),
    2: (0.10, 0.25),
    3: (0.05, 0.15),
    4: (0.01, 0.08),
}

SCALE_MINIMA = {
    "smoke": {"sites": 1, "episodes": 30},
    "v0": {"sites": 3, "episodes": 180},
    "study": {"sites": 8, "episodes": 1500},
    "stress": {"sites": 20, "episodes": 10000},
}

REQUIRED_SCENARIOS = {
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
}

HARD_FALSE_POSITIVE_SCENARIOS = {
    "large_batch_inference",
    "synthetic_data_generation",
    "hpc_mpi_simulation",
    "nccl_benchmark",
    "hardware_burn_in",
    "storage_rebuild",
    "large_etl_data_movement",
    "reserved_but_unused_capacity",
    "maintenance_window",
}

TRAINING_SCENARIOS = {
    "large_fine_tune",
    "pretraining",
    "cloud_reservation_used_for_training",
    "adversarial_fragmented_training",
    "underclocked_long_duration_training",
}

OBSERVABLE_IDS = [f"o{i}" for i in range(1, 18)]
MISSING_VALUES = {"", "null", "none", "nan", "na", "n/a", "nil"}

REQUIRED_RAW_FILES = {
    "raw_normalized/metric_samples.jsonl": {
        "metric_sample_id",
        "site_id",
        "observable_id",
        "source_system",
        "metric_name",
        "entity_type",
        "entity_id_hash",
        "event_time",
        "ingest_time",
        "value_num",
        "coverage_fraction",
        "trust_level",
        "signature_status",
        "episode_id",
        "latent_workload_class",
    },
    "raw_normalized/event_records.jsonl": {
        "event_record_id",
        "site_id",
        "observable_id",
        "source_system",
        "event_type",
        "scope_type",
        "scope_id_hash",
        "event_time",
        "ingest_time",
        "attributes_json",
        "trust_level",
        "signature_status",
        "episode_id",
        "latent_workload_class",
    },
    "raw_normalized/snapshot_records.jsonl": {
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
        "episode_id",
        "latent_workload_class",
    },
}

REQUIRED_FEATURE_COLUMNS = {
    "feature_row_id",
    "site_id",
    "scope_type",
    "scope_id_hash",
    "window_start",
    "window_end",
    "window_length_seconds",
    "episode_id",
    "latent_workload_class",
    "label_0_to_4",
    "label_confidence",
    "label_reason",
    "label_source",
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
    "o8_rack_power_fraction_p95",
    "o8_facility_it_power_mw",
    "o8_baseline_subtracted_energy_kwh",
    "o8_power_continuity_days",
    "o8_power_cv",
    "o8_power_to_gpu_residual",
    "o9_gpu_hbm_temp_score",
    "o9_thermal_delta_t_score",
    "o9_cooling_flow_duty",
    "o10_world_size",
    "o10_runtime_framework_class",
    "o10_rank_stability_score",
    "o10_same_image_gpu_count",
    "o10_rendezvous_present",
    "o11_data_staging_tb",
    "o11_checkpoint_write_tb_per_event",
    "o11_checkpoint_periodicity_score",
    "o11_read_write_training_pattern_score",
    "o12_signed_ml_logs_present",
    "o12_declared_parameter_count_b",
    "o12_training_tokens_b",
    "o12_step_count",
    "o12_loss_curve_present",
    "o12_optimizer_state_present",
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
}

for observable_id in OBSERVABLE_IDS:
    REQUIRED_FEATURE_COLUMNS.add(f"{observable_id}_coverage_fraction")
    REQUIRED_FEATURE_COLUMNS.add(f"{observable_id}_missing_reason")


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    count: int = 1
    examples: list[str] = field(default_factory=list)


@dataclass
class ValidationState:
    dataset: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    feature_rows: list[dict[str, str]] = field(default_factory=list)
    feature_columns: set[str] = field(default_factory=set)
    raw_counts: dict[str, int] = field(default_factory=dict)
    raw_observables: set[str] = field(default_factory=set)
    findings: list[Finding] = field(default_factory=list)

    def error(self, code: str, message: str, examples: list[str] | None = None, count: int = 1) -> None:
        self.findings.append(Finding("ERROR", code, message, count, examples or []))

    def warn(self, code: str, message: str, examples: list[str] | None = None, count: int = 1) -> None:
        self.findings.append(Finding("WARNING", code, message, count, examples or []))

    def info(self, code: str, message: str, examples: list[str] | None = None, count: int = 1) -> None:
        self.findings.append(Finding("INFO", code, message, count, examples or []))


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_VALUES


def as_float(row: dict[str, str], key: str, default: float | None = None) -> float | None:
    value = row.get(key)
    if is_missing(value):
        return default
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def num(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = as_float(row, key, None)
    return default if value is None else value


def as_int(row: dict[str, str], key: str, default: int | None = None) -> int | None:
    value = as_float(row, key)
    if value is None:
        return default
    return int(value)


def as_bool(row: dict[str, str], key: str, default: bool | None = None) -> bool | None:
    value = row.get(key)
    if is_missing(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return default


def parse_utc_z(value: str) -> datetime | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    if not text.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(text[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError:
        return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def label(row: dict[str, str]) -> int | None:
    return as_int(row, "label_0_to_4")


def capacity_possible(row: dict[str, str]) -> bool:
    return bool(as_bool(row, "capacity_possible", False))


def offledger_warning(row: dict[str, str]) -> bool:
    return (as_float(row, "o17_external_capacity_conflict_score", 0.0) or 0.0) >= 0.5


def allocation_signal(row: dict[str, str]) -> bool:
    return any(
        [
            (as_float(row, "o2_max_concurrent_normalized_gpus", 0.0) or 0.0) >= 512,
            (as_float(row, "o2_gpu_hours_policy_ratio", 0.0) or 0.0) >= 0.75,
            (as_float(row, "o2_concurrency_fraction_domain", 0.0) or 0.0) >= 0.5,
            (as_float(row, "o3_batch_provisioned_gpus", 0.0) or 0.0) >= 512,
        ]
    )


def gpu_signal(row: dict[str, str]) -> bool:
    return any(
        [
            (as_float(row, "o4_gpu_util_p95", 0.0) or 0.0) >= 70,
            (as_float(row, "o4_gpu_util_duty_gt_70", 0.0) or 0.0) >= 0.5,
            (as_float(row, "o4_sm_tensor_active_p95", 0.0) or 0.0) >= 70,
            (as_float(row, "o5_tensor_throughput_ratio", 0.0) or 0.0) >= 0.6,
        ]
    )


def fabric_signal(row: dict[str, str]) -> bool:
    return any(
        [
            (as_float(row, "o7_synchronized_fabric_footprint", 0.0) or 0.0) >= 512,
            (as_float(row, "o7_collective_periodicity_score", 0.0) or 0.0) >= 0.65,
            (as_float(row, "o7_scaleout_port_util_p95", 0.0) or 0.0) >= 0.60,
            (as_float(row, "o6_nvlink_periodicity_score", 0.0) or 0.0) >= 0.65,
        ]
    )


def power_signal(row: dict[str, str]) -> bool:
    return any(
        [
            (as_float(row, "o8_rack_power_fraction_p95", 0.0) or 0.0) >= 0.70,
            (
                (as_float(row, "o8_rack_power_fraction_p95", 0.0) or 0.0) >= 0.50
                and (as_float(row, "o8_power_continuity_days", 0.0) or 0.0) >= 1.0
            ),
            (as_float(row, "o8_baseline_subtracted_energy_kwh", 0.0) or 0.0) >= 10000,
        ]
    )


def semantic_signal(row: dict[str, str]) -> bool:
    framework = str(row.get("o10_runtime_framework_class", "")).lower()
    return any(
        [
            (as_float(row, "o10_world_size", 0.0) or 0.0) >= 512,
            any(token in framework for token in ["training", "deepspeed", "megatron", "fsdp", "pytorch_distributed"]),
            bool(as_bool(row, "o12_signed_ml_logs_present", False)),
            bool(as_bool(row, "o12_loss_curve_present", False)),
            bool(as_bool(row, "o12_optimizer_state_present", False)),
            (as_float(row, "o12_step_count", 0.0) or 0.0) >= 1000,
        ]
    )


def storage_signal(row: dict[str, str]) -> bool:
    return any(
        [
            (as_float(row, "o11_checkpoint_periodicity_score", 0.0) or 0.0) >= 0.5,
            (as_float(row, "o11_checkpoint_write_tb_per_event", 0.0) or 0.0) >= 0.1,
            (as_float(row, "o11_read_write_training_pattern_score", 0.0) or 0.0) >= 0.5,
            (as_float(row, "o11_data_staging_tb", 0.0) or 0.0) >= 10,
        ]
    )


def integrity_anomaly(row: dict[str, str]) -> bool:
    return any(
        [
            num(row, "o14_min_critical_coverage", 1.0) < 0.80,
            (as_float(row, "o14_gap_fraction_critical", 0.0) or 0.0) > 0.05,
            num(row, "o13_attestation_valid_fraction", 1.0) < 0.90,
            (as_float(row, "o13_confidential_compute_mode_fraction", 0.0) or 0.0) > 0.50,
            bool(as_bool(row, "o15_unapproved_physical_change_near_window", False)),
            (as_float(row, "o14_counter_reset_count", 0.0) or 0.0) > 0,
        ]
    )


def primary_count(row: dict[str, str]) -> int:
    return sum([allocation_signal(row), gpu_signal(row), fabric_signal(row)])


def coherence_count(row: dict[str, str]) -> int:
    return sum(
        [
            allocation_signal(row),
            gpu_signal(row),
            fabric_signal(row),
            power_signal(row),
            semantic_signal(row),
            storage_signal(row),
        ]
    )


def row_id(row: dict[str, str]) -> str:
    return row.get("feature_row_id") or f"{row.get('site_id')}:{row.get('window_start')}:{row.get('scope_id_hash')}"


class DatasetValidator:
    def __init__(self, dataset: Path, strict: bool = False, max_raw_records: int | None = None) -> None:
        self.state = ValidationState(dataset=dataset)
        self.strict = strict
        self.max_raw_records = max_raw_records

    def run(self) -> ValidationState:
        self.check_required_paths()
        self.load_manifest()
        self.validate_raw_files()
        self.load_features()
        if self.state.feature_rows:
            self.validate_feature_schema()
            self.validate_feature_values()
            self.validate_distributions_and_occurrences()
            self.validate_label_logic()
            self.validate_missingness_logic()
            self.validate_cross_feature_dependencies()
            self.validate_scenario_patterns()
        self.validate_examples()
        self.write_report()
        return self.state

    def check_required_paths(self) -> None:
        dataset = self.state.dataset
        if not dataset.exists():
            self.state.error("dataset_missing", f"Dataset directory does not exist: {dataset}")
            return

        required = [
            "README.md",
            "manifest.json",
            "schemas/metric_sample.schema.json",
            "schemas/event_record.schema.json",
            "schemas/snapshot_record.schema.json",
            "schemas/window_feature_row.schema.json",
            "raw_normalized/metric_samples.jsonl",
            "raw_normalized/event_records.jsonl",
            "raw_normalized/snapshot_records.jsonl",
            "features/window_features_all.csv",
            "examples/one_datapoint_label0.json",
            "examples/one_datapoint_label1.json",
            "examples/one_datapoint_label2.json",
            "examples/one_datapoint_label3.json",
            "examples/one_datapoint_label4.json",
        ]
        for relpath in required:
            path = dataset / relpath
            if not path.exists():
                self.state.error("required_file_missing", f"Missing required file: {relpath}")

        for schema in (dataset / "schemas").glob("*.schema.json") if (dataset / "schemas").exists() else []:
            try:
                json.loads(schema.read_text())
            except json.JSONDecodeError as exc:
                self.state.error("schema_json_invalid", f"Schema is not valid JSON: {schema}", [str(exc)])

    def load_manifest(self) -> None:
        path = self.state.dataset / "manifest.json"
        if not path.exists():
            return
        try:
            self.state.manifest = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            self.state.error("manifest_json_invalid", "manifest.json is not valid JSON", [str(exc)])
            return

        for key in [
            "dataset_id",
            "scale",
            "seed",
            "generator_version",
            "feature_pipeline_version",
            "policy_threshold_version",
            "hardware_normalization_version",
        ]:
            if key not in self.state.manifest:
                self.state.warn("manifest_key_missing", f"manifest.json is missing `{key}`")

    def validate_raw_files(self) -> None:
        for relpath, required_fields in REQUIRED_RAW_FILES.items():
            path = self.state.dataset / relpath
            if not path.exists():
                continue

            count = 0
            bad_json = 0
            missing_field_examples: list[str] = []
            timestamp_examples: list[str] = []
            range_examples: list[str] = []

            with path.open() as handle:
                for line_number, line in enumerate(handle, start=1):
                    if self.max_raw_records is not None and count >= self.max_raw_records:
                        break
                    if not line.strip():
                        continue
                    count += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        bad_json += 1
                        continue

                    missing = sorted(field for field in required_fields if field not in obj)
                    if missing and len(missing_field_examples) < 5:
                        missing_field_examples.append(f"{relpath}:{line_number} missing {missing}")

                    observable = str(obj.get("observable_id", "")).lower()
                    if observable.startswith("o"):
                        self.state.raw_observables.add(observable)

                    for time_key in ["event_time", "event_end_time", "valid_from", "valid_to", "observed_at", "ingest_time"]:
                        if time_key in obj and obj.get(time_key) is not None and parse_utc_z(str(obj.get(time_key))) is None:
                            if len(timestamp_examples) < 5:
                                timestamp_examples.append(f"{relpath}:{line_number} bad {time_key}={obj.get(time_key)}")

                    unit = obj.get("unit")
                    value = obj.get("value_num")
                    if value is not None:
                        try:
                            value_float = float(value)
                        except (TypeError, ValueError):
                            value_float = None
                        if value_float is not None:
                            if unit == "percent" and not (0.0 <= value_float <= 100.0) and len(range_examples) < 10:
                                range_examples.append(
                                    f"{relpath}:{line_number} {obj.get('metric_name')}={value_float} percent"
                                )
                            if unit in {"fraction", "score"} and not (0.0 <= value_float <= 1.0) and len(range_examples) < 10:
                                range_examples.append(
                                    f"{relpath}:{line_number} {obj.get('metric_name')}={value_float} {unit}"
                                )

            self.state.raw_counts[relpath] = count
            if count == 0:
                self.state.error("raw_file_empty", f"Raw file has no records: {relpath}")
            if bad_json:
                self.state.error("raw_jsonl_invalid", f"{relpath} contains invalid JSONL rows", count=bad_json)
            if missing_field_examples:
                self.state.error("raw_required_fields_missing", f"{relpath} rows are missing required fields", missing_field_examples)
            if timestamp_examples:
                self.state.error("raw_timestamp_invalid", f"{relpath} has invalid UTC `Z` timestamps", timestamp_examples)
            if range_examples:
                self.state.error(
                    "raw_metric_value_out_of_range",
                    f"{relpath} has bounded metric values outside their declared unit range",
                    range_examples,
                )

        missing_observables = {f"o{i}" for i in range(1, 18)} - self.state.raw_observables
        if missing_observables:
            self.state.warn(
                "raw_observable_coverage_incomplete",
                "Raw normalized records do not include every observable family O1-O17",
                sorted(missing_observables),
            )

    def load_features(self) -> None:
        path = self.state.dataset / "features/window_features_all.csv"
        if not path.exists():
            return
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.state.feature_columns = set(reader.fieldnames or [])
                self.state.feature_rows = list(reader)
        except csv.Error as exc:
            self.state.error("feature_csv_invalid", "features/window_features_all.csv is not valid CSV", [str(exc)])
            return

        if not self.state.feature_rows:
            self.state.error("feature_rows_empty", "features/window_features_all.csv has no rows")

    def validate_feature_schema(self) -> None:
        missing = sorted(REQUIRED_FEATURE_COLUMNS - self.state.feature_columns)
        if missing:
            self.state.error("feature_columns_missing", "Feature table is missing required columns", missing[:80], len(missing))

        for window_file in ["window_features_15m.csv", "window_features_1h.csv", "window_features_6h.csv", "window_features_1d.csv"]:
            if not (self.state.dataset / "features" / window_file).exists():
                self.state.warn("window_file_missing", f"Expected window feature file is missing: features/{window_file}")

    def validate_feature_values(self) -> None:
        bad_labels: list[str] = []
        bad_times: list[str] = []
        duplicate_ids: list[str] = []
        seen_ids: set[str] = set()

        for row in self.state.feature_rows:
            rid = row_id(row)
            if rid in seen_ids and len(duplicate_ids) < 5:
                duplicate_ids.append(rid)
            seen_ids.add(rid)

            lab = label(row)
            if lab not in {0, 1, 2, 3, 4} and len(bad_labels) < 5:
                bad_labels.append(f"{rid}: label={row.get('label_0_to_4')}")

            start = parse_utc_z(row.get("window_start", ""))
            end = parse_utc_z(row.get("window_end", ""))
            if start is None or end is None or start >= end:
                if len(bad_times) < 5:
                    bad_times.append(f"{rid}: {row.get('window_start')} -> {row.get('window_end')}")

        if bad_labels:
            self.state.error("feature_label_invalid", "Feature rows contain labels outside 0-4", bad_labels)
        if bad_times:
            self.state.error("feature_window_time_invalid", "Feature rows have invalid UTC windows", bad_times)
        if duplicate_ids:
            self.state.error("feature_row_id_duplicate", "Feature row IDs must be unique", duplicate_ids)

    def validate_distributions_and_occurrences(self) -> None:
        rows = self.state.feature_rows
        labels = Counter(label(row) for row in rows)
        scenarios = Counter(row.get("latent_workload_class", "") for row in rows)
        sites = {row.get("site_id") for row in rows if row.get("site_id")}
        episodes = {row.get("episode_id") for row in rows if row.get("episode_id")}

        self.state.info("feature_row_count", f"Feature rows: {len(rows)}")
        self.state.info("label_distribution", ", ".join(f"{k}: {labels.get(k, 0)}" for k in range(5)))
        self.state.info("scenario_count", f"Scenarios represented: {len(scenarios)}")
        self.state.info("site_episode_count", f"Sites: {len(sites)}; episodes: {len(episodes)}")

        scale = str(self.state.manifest.get("scale", "unknown")).lower()
        minima = SCALE_MINIMA.get(scale)
        if minima:
            if len(sites) < minima["sites"]:
                self.state.error("scale_site_count_low", f"Scale `{scale}` requires at least {minima['sites']} sites; found {len(sites)}")
            if len(episodes) < minima["episodes"]:
                self.state.error(
                    "scale_episode_count_low",
                    f"Scale `{scale}` requires at least {minima['episodes']} episodes; found {len(episodes)}",
                )

        missing_labels = [str(lab) for lab in range(5) if labels.get(lab, 0) == 0]
        if missing_labels:
            severity = self.state.error if len(rows) >= 1000 else self.state.warn
            severity("labels_missing", "Dataset does not contain examples of every label 0-4", missing_labels)

        for lab, (low, high) in LABEL_TARGETS.items():
            ratio = labels.get(lab, 0) / max(len(rows), 1)
            if ratio < low or ratio > high:
                self.state.warn(
                    "label_distribution_outside_target",
                    f"Label {lab} ratio {ratio:.3f} is outside target {low:.2f}-{high:.2f}",
                )

        missing_scenarios = sorted(REQUIRED_SCENARIOS - set(scenarios))
        if missing_scenarios:
            self.state.warn("scenario_classes_missing", "Not every required latent scenario class is represented", missing_scenarios)

        represented_false_positives = HARD_FALSE_POSITIVE_SCENARIOS & set(scenarios)
        if len(represented_false_positives) < 5:
            self.state.error(
                "hard_false_positives_insufficient",
                "Dataset needs several non-training high-load false-positive scenario families",
                sorted(represented_false_positives),
            )

    def validate_label_logic(self) -> None:
        errors: dict[str, list[str]] = defaultdict(list)

        for row in self.state.feature_rows:
            lab = label(row)
            if lab is None:
                continue
            rid = row_id(row)
            cap = capacity_possible(row)
            offledger = offledger_warning(row)
            alloc = allocation_signal(row)
            gpu = gpu_signal(row)
            fabric = fabric_signal(row)
            power = power_signal(row)
            semantic = semantic_signal(row)
            storage = storage_signal(row)
            integ = integrity_anomaly(row)
            coherent = coherence_count(row)
            primary = primary_count(row)
            coverage = num(row, "o14_min_critical_coverage", 1.0)

            if not cap and lab > 1 and not offledger:
                errors["capacity_gate_violation"].append(f"{rid}: label {lab} with capacity_possible=false")

            if lab == 0 and cap and coverage < 0.80:
                errors["label0_low_coverage"].append(f"{rid}: label 0 with coverage={coverage:.2f}")

            if lab == 0 and cap and (alloc or gpu or fabric or power or semantic or storage):
                errors["label0_activity_present"].append(f"{rid}: label 0 despite activity/supportive signal")

            if cap and not (alloc or gpu or fabric or power or semantic or storage or integ) and lab > 1:
                errors["capacity_only_above_label1"].append(f"{rid}: label {lab} from capacity-only evidence")

            if power and not (alloc or gpu or fabric or semantic or storage) and lab > 2:
                errors["physical_only_above_label2"].append(f"{rid}: label {lab} from physical-only evidence")

            if integ and not (alloc or gpu or fabric or power or semantic or storage) and lab > 2:
                errors["integrity_only_positive"].append(f"{rid}: label {lab} from integrity-only anomaly")

            if lab == 2 and not (alloc or gpu or fabric or power or semantic or storage or integ):
                errors["label2_without_elevated_signal"].append(f"{rid}: label 2 has no elevated signal")

            if lab == 3:
                if not cap:
                    errors["label3_without_capacity"].append(f"{rid}: label 3 without capacity_possible")
                if primary == 0:
                    errors["label3_without_primary"].append(f"{rid}: label 3 without primary activity signal")
                if coherent < 2:
                    errors["label3_low_coherence"].append(f"{rid}: label 3 coherence_count={coherent}")

            if lab == 4:
                signed_semantic = bool(as_bool(row, "o12_signed_ml_logs_present", False))
                policy_crossed = (as_float(row, "policy_compute_ratio", 0.0) or 0.0) >= 1.0
                has_o12_definite = signed_semantic and policy_crossed
                has_full_stack_warning = coherent >= 4 and alloc and gpu and fabric and coverage >= 0.85
                if not (has_o12_definite or has_full_stack_warning):
                    errors["label4_without_definite_or_full_stack"].append(
                        f"{rid}: label 4 lacks signed O12 threshold evidence or full-stack coherence"
                    )

        for code, examples in errors.items():
            self.state.error(code, f"{len(examples)} feature rows violate `{code}`", examples[:10], len(examples))

    def validate_missingness_logic(self) -> None:
        bad_missing_reasons: list[str] = []
        zero_encoded_missing: list[str] = []
        missing_reason_counts: Counter[str] = Counter()
        null_cell_count = 0

        for row in self.state.feature_rows:
            rid = row_id(row)
            for observable_id in OBSERVABLE_IDS:
                coverage_col = f"{observable_id}_coverage_fraction"
                reason_col = f"{observable_id}_missing_reason"
                coverage = as_float(row, coverage_col, None)
                reason = str(row.get(reason_col, "")).strip()
                if reason:
                    missing_reason_counts[reason] += 1
                if coverage is not None and coverage < 0.999 and reason.lower() in {"", "observed"}:
                    if len(bad_missing_reasons) < 20:
                        bad_missing_reasons.append(f"{rid}: {observable_id} coverage={coverage} reason={reason!r}")

                if coverage == 0:
                    prefix = f"{observable_id}_"
                    for key, value in row.items():
                        if not key.startswith(prefix):
                            continue
                        if key.endswith("_coverage_fraction") or key.endswith("_missing_reason"):
                            continue
                        if str(value).strip() == "0" and len(zero_encoded_missing) < 20:
                            zero_encoded_missing.append(f"{rid}: {key}=0 with {observable_id} coverage=0")
                            break

            null_cell_count += sum(1 for value in row.values() if is_missing(value))

        if bad_missing_reasons:
            self.state.error(
                "missing_reason_absent_for_partial_coverage",
                "Rows with partial/missing coverage need explicit missing reasons",
                bad_missing_reasons,
            )
        if zero_encoded_missing:
            self.state.warn(
                "possible_zero_encoded_missing",
                "Some observable values are zero when observable coverage is zero; verify these are observed zeros, not missing data",
                zero_encoded_missing,
            )

        non_observed_reasons = {reason for reason in missing_reason_counts if reason.lower() not in {"observed", ""}}
        if not non_observed_reasons:
            self.state.error("missingness_regimes_absent", "Dataset has no explicit non-observed missingness regimes")
        else:
            self.state.info("missing_reason_distribution", ", ".join(f"{k}: {v}" for k, v in missing_reason_counts.most_common(12)))

        if null_cell_count == 0:
            self.state.warn("null_values_absent", "Feature table contains no null-like cells; this is unrealistic for routine monitoring")

    def validate_cross_feature_dependencies(self) -> None:
        rows = self.state.feature_rows
        gpu_power_pairs: list[tuple[float, float]] = []
        high_gpu_low_power: list[str] = []
        high_power_hidden_activity: list[str] = []
        label34_without_fabric: list[str] = []
        label34_without_storage_or_semantic: list[str] = []

        for row in rows:
            gpu_util = as_float(row, "o4_gpu_util_p95", None)
            rack_power = as_float(row, "o8_rack_power_fraction_p95", None)
            if gpu_util is not None and rack_power is not None:
                gpu_power_pairs.append((gpu_util, rack_power))

            if (gpu_util or 0.0) >= 80 and (rack_power or 0.0) < 0.45 and len(high_gpu_low_power) < 10:
                high_gpu_low_power.append(f"{row_id(row)}: gpu_p95={gpu_util}, rack_power={rack_power}")

            if (
                (rack_power or 0.0) >= 0.75
                and not gpu_signal(row)
                and not fabric_signal(row)
                and not integrity_anomaly(row)
                and len(high_power_hidden_activity) < 10
            ):
                high_power_hidden_activity.append(f"{row_id(row)}: high power without GPU/fabric/integrity explanation")

            lab = label(row)
            if lab in {3, 4}:
                if not (fabric_signal(row) or semantic_signal(row)):
                    label34_without_fabric.append(f"{row_id(row)}: label {lab}")
                if not (storage_signal(row) or semantic_signal(row)):
                    label34_without_storage_or_semantic.append(f"{row_id(row)}: label {lab}")

        if len(gpu_power_pairs) >= 20:
            xs = [pair[0] for pair in gpu_power_pairs]
            ys = [pair[1] for pair in gpu_power_pairs]
            corr = pearson(xs, ys)
            if corr is None:
                self.state.warn("gpu_power_correlation_unavailable", "Could not compute GPU/power correlation")
            elif corr < 0.35:
                self.state.warn(
                    "gpu_power_correlation_low",
                    f"GPU utilization and rack power correlation is low ({corr:.3f}); synthetic physics may be weak",
                )
            else:
                self.state.info("gpu_power_correlation", f"GPU utilization and rack power correlation: {corr:.3f}")

        if high_gpu_low_power:
            self.state.warn("high_gpu_low_power", "High GPU utilization rows have unexpectedly low rack power", high_gpu_low_power)
        if high_power_hidden_activity:
            self.state.warn(
                "high_power_without_explanation",
                "High power should usually align with GPU/fabric activity or an integrity/missingness anomaly",
                high_power_hidden_activity,
            )
        if label34_without_fabric:
            self.state.error(
                "training_labels_without_fabric_or_semantic",
                "Label 3/4 rows should include fabric synchronization or semantic/runtime evidence",
                label34_without_fabric[:10],
                len(label34_without_fabric),
            )

        label34_count = sum(1 for row in rows if label(row) in {3, 4})
        if label34_count:
            missing_ratio = len(label34_without_storage_or_semantic) / label34_count
            if missing_ratio > 0.50:
                self.state.warn(
                    "training_labels_storage_semantic_weak",
                    f"{missing_ratio:.1%} of label 3/4 rows lack storage/checkpoint or semantic evidence",
                    label34_without_storage_or_semantic[:10],
                )

    def validate_scenario_patterns(self) -> None:
        by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_episode: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.state.feature_rows:
            by_scenario[row.get("latent_workload_class", "")].append(row)
            by_episode[row.get("episode_id", "")].append(row)

        def scenario_rows(name: str) -> list[dict[str, str]]:
            return by_scenario.get(name, [])

        pretraining = scenario_rows("pretraining")
        if pretraining:
            strong = sum(1 for row in pretraining if label(row) in {3, 4} and allocation_signal(row) and gpu_signal(row) and fabric_signal(row))
            if strong / len(pretraining) < 0.25:
                self.state.warn(
                    "pretraining_pattern_weak",
                    "Pretraining rows should often show label 3/4 with allocation+GPU+fabric coherence",
                )

        counter_suppressed = scenario_rows("counter_suppressed_candidate_window")
        if counter_suppressed:
            explained = sum(
                1
                for row in counter_suppressed
                if (
                    power_signal(row)
                    and (
                        num(row, "o4_coverage_fraction", 1.0) < 0.8
                        or (as_float(row, "o13_confidential_compute_mode_fraction", 0.0) or 0.0) > 0.5
                        or (as_float(row, "o14_gap_fraction_critical", 0.0) or 0.0) > 0.05
                    )
                )
            )
            if explained / len(counter_suppressed) < 0.50:
                self.state.warn(
                    "counter_suppressed_pattern_weak",
                    "Counter-suppressed scenarios should combine high power/activity with missing GPU counters, CC mode, or gaps",
                )

        reserved_unused = scenario_rows("reserved_but_unused_capacity")
        if reserved_unused:
            likely_bad = [
                row_id(row)
                for row in reserved_unused
                if label(row) in {3, 4} or ((as_float(row, "o4_gpu_util_p95", 0.0) or 0.0) > 50 and label(row) > 2)
            ]
            if likely_bad:
                self.state.warn(
                    "reserved_unused_too_training_like",
                    "Reserved-but-unused capacity should not usually look like active training",
                    likely_bad[:10],
                    len(likely_bad),
                )

        false_positive_rows = [row for scenario in HARD_FALSE_POSITIVE_SCENARIOS for row in by_scenario.get(scenario, [])]
        high_load_false_positives = [
            row
            for row in false_positive_rows
            if (gpu_signal(row) or power_signal(row) or fabric_signal(row) or storage_signal(row)) and label(row) in {1, 2, 3}
        ]
        if false_positive_rows and len(high_load_false_positives) / len(false_positive_rows) < 0.20:
            self.state.warn(
                "hard_false_positive_patterns_weak",
                "Hard false-positive scenarios should include substantial high-load non-training examples",
            )

        label34_episodes = {
            episode_id
            for episode_id, rows in by_episode.items()
            if episode_id and any(label(row) in {3, 4} for row in rows)
        }
        short_training_episodes = [
            episode_id for episode_id in label34_episodes if len(by_episode.get(episode_id, [])) < 2
        ]
        if short_training_episodes:
            self.state.warn(
                "training_episode_too_short",
                "Likely/highest-warning training episodes should usually span multiple feature windows",
                short_training_episodes[:10],
                len(short_training_episodes),
            )

    def validate_examples(self) -> None:
        for lab in range(5):
            path = self.state.dataset / "examples" / f"one_datapoint_label{lab}.json"
            if not path.exists():
                continue
            try:
                obj = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                self.state.error("example_json_invalid", f"Example datapoint label {lab} is invalid JSON", [str(exc)])
                continue
            if obj.get("label_0_to_4") != lab:
                self.state.error(
                    "example_label_mismatch",
                    f"Example datapoint one_datapoint_label{lab}.json does not have label_0_to_4={lab}",
                )
            missing = sorted(REQUIRED_FEATURE_COLUMNS - set(obj.keys()))
            # Example JSON may nest coverage/trust details, so only require top-level core columns.
            core_missing = [
                col
                for col in missing
                if not col.endswith("_coverage_fraction") and not col.endswith("_missing_reason")
            ]
            if core_missing:
                self.state.warn(
                    "example_core_fields_missing",
                    f"Example datapoint label {lab} is missing some core feature fields",
                    core_missing[:20],
                )

    def write_report(self) -> None:
        validation_dir = self.state.dataset / "validation"
        try:
            validation_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        errors = [finding for finding in self.state.findings if finding.severity == "ERROR"]
        warnings = [finding for finding in self.state.findings if finding.severity == "WARNING"]
        infos = [finding for finding in self.state.findings if finding.severity == "INFO"]

        lines: list[str] = []
        lines.append("# Synthetic Dataset Validation Report")
        lines.append("")
        lines.append(f"Dataset: `{self.state.dataset}`")
        lines.append(f"Errors: {len(errors)}")
        lines.append(f"Warnings: {len(warnings)}")
        lines.append(f"Info: {len(infos)}")
        lines.append("")

        if self.state.manifest:
            lines.append("## Manifest")
            for key in sorted(self.state.manifest):
                lines.append(f"- `{key}`: `{self.state.manifest[key]}`")
            lines.append("")

        lines.append("## Raw Record Counts")
        if self.state.raw_counts:
            for path, count in sorted(self.state.raw_counts.items()):
                lines.append(f"- `{path}`: {count}")
        else:
            lines.append("- No raw records loaded.")
        lines.append("")

        lines.append("## Findings")
        for severity in ["ERROR", "WARNING", "INFO"]:
            group = [finding for finding in self.state.findings if finding.severity == severity]
            if not group:
                continue
            lines.append(f"### {severity}")
            for finding in group:
                suffix = f" ({finding.count})" if finding.count != 1 else ""
                lines.append(f"- `{finding.code}`{suffix}: {finding.message}")
                for example in finding.examples[:10]:
                    lines.append(f"  - {example}")
            lines.append("")

        report_path = validation_dir / "validation_report.md"
        report_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate synthetic datacenter verification datasets.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory, e.g. data/synthetic_v0")
    parser.add_argument("--strict", action="store_true", help="Return non-zero for warnings as well as errors")
    parser.add_argument(
        "--max-raw-records",
        type=int,
        default=None,
        help="Optional cap for raw JSONL rows inspected per raw file. Feature rows are always fully loaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = DatasetValidator(args.dataset, strict=args.strict, max_raw_records=args.max_raw_records)
    state = validator.run()
    errors = [finding for finding in state.findings if finding.severity == "ERROR"]
    warnings = [finding for finding in state.findings if finding.severity == "WARNING"]

    print(f"Validated dataset: {args.dataset}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    if args.dataset.exists():
        print(f"Report: {args.dataset / 'validation' / 'validation_report.md'}")

    if errors:
        return 1
    if args.strict and warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
