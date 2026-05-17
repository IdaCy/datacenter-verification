#!/usr/bin/env python3
"""generate synthetic observable sites and website demo data"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datacenter_verification.observable_algorithm import (
    POLICY_THRESHOLD_OPERATIONS,
    THRESHOLDS,
    WEBSITE_CONTROL_KEYS,
    evaluate_site,
)

SYNTHETIC_DIR = ROOT / "synthetic"
SITES_JSON = SYNTHETIC_DIR / "sites.json"
BASE_START = datetime(2026, 4, 1, tzinfo=timezone.utc)
DEFAULT_PEAK_RATE = 2.0e15
SCENARIO_COUNTER = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SITES_JSON)
    parser.add_argument("--website-dir", type=Path, default=None)
    args = parser.parse_args()

    sites = build_sites()
    results = []
    for site in sites:
        result = evaluate_site(site)
        assert_expected(site, result)
        site["expected_algorithm_outputs"] = expected_from_result(result)
        results.append(result)

    dataset = {
        "metadata": {
            "dataset_id": "synthetic_observable_sites_2026_05_16",
            "version": "0.1",
            "created": "2026-05-16",
            "source_of_truth": [
                "observables/observables.yaml",
                "observables/rules/README.md",
                "observables/rules/source_ledger.yaml",
                "observables/rules/feature_rule_index.yaml",
                "observables/rules/derived_signals.yaml",
                "observables/rules/capability_rules.yaml",
                "observables/rules/training_core_rules.yaml",
                "observables/rules/training_support_rules.yaml",
                "observables/rules/discrepancy_rules.yaml",
                "observables/rules/aggregation_rules.yaml",
            ],
            "synthetic_notice": "All site names, SKUs, telemetry, and workloads are synthetic. Numeric telemetry values are scenario fixtures, not vendor defaults.",
            "policy_threshold_operations": POLICY_THRESHOLD_OPERATIONS,
            "thresholds_from_observable_rules": THRESHOLDS,
            "scenario_count": len(sites),
        },
        "sites": sites,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.website_dir:
        write_website_data(args.website_dir, dataset, results)

    print(json.dumps({"sites": len(sites), "output": str(args.output)}, indent=2))


def build_sites() -> list[dict[str, Any]]:
    defs = [
        scenario(
            "site_a_clean_training",
            "A_clean_threshold_training",
            "Clean threshold-scale training-like run",
            "high_training_like_warning",
            days=30,
            count=8192,
            activity=0.88,
            achieved=1.15e25,
            fabric=0.86,
            participants=6144,
            checkpoint=0.82,
            non_serving=0.88,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=[
                "large_compute_candidate",
                "distributed_training_like_candidate",
                "checkpoint_training_like_candidate",
                "sparse_large_compute_training_like_candidate",
            ],
            expected_c=[],
        ),
        scenario(
            "site_b_covered_negative",
            "B_covered_negative",
            "Clean covered negative",
            "no_training_like_candidate_detected_in_covered_live_segment",
            days=3,
            count=1024,
            peak=1.6e14,
            activity=0.05,
            achieved=0.0,
            fabric=0.05,
            checkpoint=0.0,
            capacity_coverage=0.82,
            allocation=False,
            expected_a="capacity_limited_but_not_ruled_out",
            expected_b=[],
            expected_c=["negative_screen_coverage_sufficient"],
        ),
        scenario(
            "site_c_capacity_ruleout",
            "C_capacity_ruled_out",
            "Capacity ruled out",
            "capacity_ruled_out_for_scope",
            days=7,
            count=256,
            peak=1.0e14,
            activity=0.0,
            achieved=0.0,
            fabric=0.0,
            checkpoint=0.0,
            capacity_coverage=0.97,
            allocation=False,
            expected_a="capacity_ruled_out_for_scope",
            expected_b=[],
            expected_c=[],
            short_circuit=True,
        ),
        scenario(
            "site_d_missing_negative_screen",
            "D_missingness_blocks_negative_screen",
            "Missingness blocks negative screen",
            "inconclusive_due_to_missingness",
            days=30,
            count=4096,
            activity=0.04,
            achieved=0.0,
            fabric=0.0,
            checkpoint=0.0,
            capacity_coverage=0.65,
            activity_coverage=0.42,
            fabric_coverage=0.38,
            storage_coverage=0.44,
            identity_shape_coverage=0.40,
            scope_mapping_coverage=0.93,
            clock_alignment=0.91,
            allocation=False,
            expected_a="capacity_possible_for_scope",
            expected_b=[],
            expected_c=["negative_screen_blocked_by_missingness", "inconclusive_due_to_missingness"],
        ),
        scenario(
            "site_e_storage_explains_checkpoint",
            "E_storage_operation_explains_checkpoint",
            "Storage operation explains checkpoint-like candidate",
            "candidate_explained_or_demoted",
            days=30,
            count=4096,
            activity=0.76,
            achieved=2.5e24,
            fabric=0.15,
            checkpoint=0.84,
            checkpoint_bursts=5,
            storage_operation_overlap=0.92,
            bytes_explained=0.88,
            non_serving=0.80,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=["checkpoint_training_like_candidate"],
            expected_c=["candidate_explained_by_storage_operation"],
        ),
        scenario(
            "site_f_serving_counterevidence",
            "F_serving_inference_counterevidence",
            "Serving or inference-like counterevidence",
            "candidate_explained_or_demoted",
            days=30,
            count=4096,
            activity=0.82,
            achieved=1.08e25,
            fabric=0.20,
            checkpoint=0.10,
            serving=0.92,
            non_serving=0.08,
            serving_overlap=0.87,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=["large_compute_candidate"],
            expected_c=["candidate_explained_by_serving"],
        ),
        scenario(
            "site_g_hpc_benchmark_alternative",
            "G_hpc_mpi_benchmark_alternative",
            "HPC/MPI or benchmark alternative",
            "candidate_explained_or_demoted",
            days=30,
            count=4096,
            activity=0.86,
            achieved=4.2e24,
            fabric=0.89,
            participants=3072,
            checkpoint=0.05,
            benchmark_regularity=0.94,
            benchmark_duration=5400,
            hpc_mpi=0.82,
            hpc_overlap=0.86,
            non_serving=0.82,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=["distributed_training_like_candidate"],
            expected_c=["candidate_benchmark_like", "candidate_hpc_mpi_alternative"],
        ),
        scenario(
            "site_h_capacity_claim_conflict",
            "H_capacity_claim_conflict",
            "Capacity claim conflict",
            "integrity_review_required",
            days=30,
            count=2048,
            peak=2.0e15,
            activity=0.78,
            achieved=1.35e25,
            fabric=0.30,
            checkpoint=0.0,
            non_serving=0.78,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=["large_compute_candidate"],
            expected_c=["capacity_claim_conflict", "integrity_review_required"],
        ),
        scenario(
            "site_i_activity_attribution_conflict",
            "I_activity_attribution_conflict",
            "Activity attribution conflict",
            "integrity_review_required",
            days=30,
            count=4096,
            activity=0.88,
            achieved=4.0e24,
            fabric=0.74,
            participants=2048,
            checkpoint=0.0,
            attribution_overlap=0.0,
            attribution_coverage=0.94,
            allocation=False,
            non_serving=0.82,
            expected_a="capacity_possible_for_scope",
            expected_b=["distributed_training_like_candidate"],
            expected_c=["activity_attribution_conflict", "integrity_review_required"],
        ),
        scenario(
            "site_j_physical_topology_health_conflict",
            "J_physical_timeline_topology_health_conflict",
            "Physical, topology, and health conflict",
            "integrity_review_required",
            days=30,
            count=4096,
            activity=0.82,
            achieved=3.0e24,
            fabric=0.72,
            participants=3072,
            checkpoint=0.64,
            physical_timeline_conflict=True,
            health_throttle_conflict=True,
            topology_route_conflict=True,
            power_activity_conflict=True,
            non_serving=0.80,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=["distributed_training_like_candidate", "checkpoint_training_like_candidate"],
            expected_c=["physical_timeline_conflict", "health_throttle_conflict", "topology_route_conflict", "power_activity_conflict", "integrity_review_required"],
        ),
        scenario(
            "site_k_large_compute_alone",
            "K_large_compute_alone",
            "Large compute alone",
            "weak_training_like_candidate",
            days=30,
            count=4096,
            activity=0.18,
            achieved=1.12e25,
            fabric=0.05,
            checkpoint=0.0,
            non_serving=0.78,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=["large_compute_candidate"],
            expected_c=[],
        ),
        scenario(
            "site_l_activity_alone",
            "L_activity_alone",
            "Activity alone",
            "no_training_like_candidate_detected_in_covered_live_segment",
            days=30,
            count=4096,
            activity=0.91,
            achieved=1.8e24,
            fabric=0.05,
            checkpoint=0.0,
            non_serving=0.82,
            allocation=True,
            expected_a="capacity_possible_for_scope",
            expected_b=[],
            expected_c=["negative_screen_coverage_sufficient"],
        ),
        scenario(
            "site_m_fabric_alone",
            "M_fabric_alone",
            "Fabric alone",
            "no_training_like_candidate_detected_in_covered_live_segment",
            days=30,
            count=4096,
            activity=0.08,
            achieved=0.0,
            fabric=0.88,
            participants=2048,
            checkpoint=0.0,
            allocation=False,
            expected_a="capacity_possible_for_scope",
            expected_b=[],
            expected_c=["negative_screen_coverage_sufficient"],
        ),
        scenario(
            "site_n_storage_writes_alone",
            "N_storage_writes_alone",
            "Storage writes alone",
            "no_training_like_candidate_detected_in_covered_live_segment",
            days=30,
            count=4096,
            activity=0.07,
            achieved=0.0,
            fabric=0.02,
            checkpoint=0.88,
            checkpoint_bursts=5,
            allocation=False,
            expected_a="capacity_possible_for_scope",
            expected_b=[],
            expected_c=["negative_screen_coverage_sufficient"],
        ),
    ]
    return defs


def scenario(
    site_id: str,
    key: str,
    name: str,
    final_route: str,
    *,
    days: int,
    count: int,
    peak: float = DEFAULT_PEAK_RATE,
    activity: float,
    achieved: float,
    fabric: float,
    checkpoint: float,
    participants: int = 0,
    checkpoint_bursts: int = 0,
    serving: float | None = None,
    serving_overlap: float = 0.0,
    non_serving: float | None = None,
    storage_operation_overlap: float = 0.0,
    bytes_explained: float = 0.0,
    benchmark_regularity: float = 0.0,
    benchmark_duration: float | None = None,
    hpc_mpi: float = 0.0,
    hpc_overlap: float = 0.0,
    capacity_coverage: float = 0.96,
    activity_coverage: float = 0.94,
    achieved_coverage: float = 0.94,
    fabric_coverage: float = 0.92,
    storage_coverage: float = 0.92,
    serving_coverage: float = 0.90,
    storage_operation_coverage: float = 0.90,
    benchmark_hpc_coverage: float = 0.90,
    attribution_coverage: float = 0.92,
    attribution_overlap: float = 1.0,
    scope_mapping_coverage: float = 0.94,
    identity_shape_coverage: float | None = None,
    clock_alignment: float = 0.93,
    hidden_capacity: bool = False,
    capacity_adjustment: float = 1.0,
    allocation: bool = True,
    physical_timeline_conflict: bool = False,
    health_throttle_conflict: bool = False,
    topology_route_conflict: bool = False,
    power_activity_conflict: bool = False,
    expected_a: str,
    expected_b: list[str],
    expected_c: list[str],
    short_circuit: bool = False,
) -> dict[str, Any]:
    global SCENARIO_COUNTER
    index = SCENARIO_COUNTER
    SCENARIO_COUNTER += 1
    start = BASE_START + timedelta(days=index * 2)
    end = start + timedelta(days=days)
    audit_window = {"start": iso(start), "end": iso(end)}
    if serving is None:
        serving = max(0.0, 1.0 - (non_serving if non_serving is not None else 0.85))
    if non_serving is None:
        non_serving = max(0.0, 1.0 - serving)
    if checkpoint_bursts == 0 and checkpoint > 0:
        checkpoint_bursts = 3
    if benchmark_duration is None:
        benchmark_duration = days * 86400
    if identity_shape_coverage is None:
        identity_shape_coverage = min(fabric_coverage, storage_coverage)

    coverage = {
        "capacity": capacity_coverage,
        "activity": activity_coverage,
        "achieved_ops": achieved_coverage,
        "fabric": fabric_coverage,
        "storage": storage_coverage,
        "serving": serving_coverage,
        "storage_operations": storage_operation_coverage,
        "benchmark_hpc": benchmark_hpc_coverage,
        "attribution": attribution_coverage,
        "scope_mapping": scope_mapping_coverage,
        "identity_shape": identity_shape_coverage,
        "clock_alignment": clock_alignment,
    }
    signals = {
        "capacity_adjustment_factor": capacity_adjustment,
        "hidden_or_unmonitored_capacity_possible": hidden_capacity,
        "activity_score": activity,
        "activity_duration_seconds": days * 86400,
        "achieved_operations": achieved,
        "achieved_operations_unit_normalized": True,
        "collective_cadence_score": fabric,
        "activity_fabric_overlap_fraction": 0.82 if activity >= 0.50 and fabric >= 0.50 else 0.0,
        "participant_count": participants,
        "checkpoint_periodicity_score": checkpoint,
        "checkpoint_burst_count": checkpoint_bursts,
        "checkpoint_activity_adjacency_fraction": 0.78 if activity >= 0.50 and checkpoint >= 0.50 else 0.0,
        "non_serving_score": non_serving,
        "serving_counterevidence_score": serving,
        "serving_activity_overlap_fraction": serving_overlap,
        "storage_operation_overlap_fraction": storage_operation_overlap,
        "bytes_explained_fraction": bytes_explained,
        "benchmark_regularity_score": benchmark_regularity,
        "benchmark_duration_seconds": benchmark_duration,
        "hpc_mpi_score": hpc_mpi,
        "hpc_overlap_fraction": hpc_overlap,
        "attribution_overlap_fraction": attribution_overlap,
        "physical_timeline_conflict": physical_timeline_conflict,
        "health_throttle_conflict": health_throttle_conflict,
        "topology_route_conflict": topology_route_conflict,
        "power_activity_conflict": power_activity_conflict,
    }
    raw_features = raw_feature_values(
        audit_window,
        count=count,
        peak=peak,
        activity=activity,
        achieved=achieved,
        fabric=fabric,
        participants=participants,
        checkpoint=checkpoint,
        checkpoint_bursts=checkpoint_bursts,
        serving=serving,
        storage_operation_overlap=storage_operation_overlap,
        bytes_explained=bytes_explained,
        benchmark_regularity=benchmark_regularity,
        allocation=allocation,
        physical_timeline_conflict=physical_timeline_conflict,
        health_throttle_conflict=health_throttle_conflict,
        topology_route_conflict=topology_route_conflict,
        power_activity_conflict=power_activity_conflict,
    )

    return {
        "site_id": site_id,
        "scenario_key": key,
        "scenario_name": name,
        "scope": f"{site_id}/accelerator_pool",
        "audit_window": audit_window,
        "operator_context": {
            "operator_type": "synthetic_monitored_operator",
            "telemetry_stack": "inventory_scheduler_activity_fabric_storage_power_network",
            "trust_tier": "synthetic_operator_signed",
        },
        "coverage": coverage,
        "normalized_signals": signals,
        "raw_features": raw_features,
        "expected": {
            "A_capacity_gate_label": expected_a,
            "B_training_candidate_detection_labels": expected_b,
            "C_discrepancy_and_explanation_review_labels": expected_c,
            "final_route": final_route,
            "capacity_short_circuit": short_circuit,
        },
        "caveats": caveats_for(key),
    }


def raw_feature_values(
    audit_window: dict[str, str],
    *,
    count: int,
    peak: float,
    activity: float,
    achieved: float,
    fabric: float,
    participants: int,
    checkpoint: float,
    checkpoint_bursts: int,
    serving: float,
    storage_operation_overlap: float,
    bytes_explained: float,
    benchmark_regularity: float,
    allocation: bool,
    physical_timeline_conflict: bool,
    health_throttle_conflict: bool,
    topology_route_conflict: bool,
    power_activity_conflict: bool,
) -> dict[str, list[dict[str, Any]]]:
    start = parse(audit_window["start"])
    end = parse(audit_window["end"])
    duration_seconds = (end - start).total_seconds()
    mid = start + (end - start) / 2
    raw: dict[str, list[dict[str, Any]]] = {
        "accelerator_count_by_family_sku": [
            {
                "valid_from": audit_window["start"],
                "valid_to": audit_window["end"],
                "accelerator_family": "SYN",
                "accelerator_sku": "SYN-ACCEL",
                "memory_class": "synthetic_high_bandwidth",
                "form_factor": "synthetic_module",
                "count": count,
            }
        ],
        "advertised_peak_rate_by_precision": [
            {
                "valid_from": audit_window["start"],
                "valid_to": audit_window["end"],
                "precision_or_mode": "synthetic_tensor_ops",
                "peak_rate": peak,
            }
        ],
        "scaleout_fabric_domain_graph": [
            {
                "valid_from": audit_window["start"],
                "valid_to": audit_window["end"],
                "scaleout_fabric_type": "synthetic_low_latency_fabric",
                "node_count": max(1, count // 8),
                "link_count": max(1, count * 4),
                "link_bandwidth_gbps": 800,
                "switch_count": max(1, count // 64),
            }
        ],
        "electrical_service_status_intervals": [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "service_status": "energized",
                "service_capacity_mw": round(count * 0.0009, 4),
                "service_capacity_mva": round(count * 0.001, 4),
                "service_voltage_kv": 34.5,
                "service_class": "synthetic_datacenter_service",
            }
        ],
    }
    if allocation:
        allocated = max(1, min(count, participants or int(count * max(activity, 0.1))))
        raw["allocated_accelerator_count_by_sku"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "accelerator_sku": "SYN-ACCEL",
                "accelerator_profile": "full",
                "partition_scope": "accelerator_pool",
                "count": allocated,
            }
        ]
        raw["compute_running_intervals"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "compute_resource_state": "running",
                "accelerator_count": allocated,
                "accelerator_shape_or_sku": "SYN-ACCEL",
            }
        ]
        raw["accelerator_compute_billing_usage_intervals"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "usage_quantity": allocated * duration_seconds,
                "usage_unit": "accelerator_seconds",
                "accelerator_shape_or_sku": "SYN-ACCEL",
                "billing_usage_type": "synthetic_running_accelerator_usage",
            }
        ]
    if activity > 0:
        raw["accelerator_busy_or_utilization_fraction"] = [{"sample_time": iso(mid), "value": activity}]
        raw["tensor_matrix_mxu_neuron_or_engine_active_fraction"] = [
            {"sample_time": iso(mid), "value": min(1.0, activity + 0.03), "engine_scope": "all_accelerators"}
        ]
    if achieved > 0:
        raw["generic_achieved_operation_rate"] = [
            {
                "sample_time": iso(mid),
                "operation_rate": achieved / max(duration_seconds, 1.0),
                "operation_unit": "synthetic_normalized_operations",
                "counter_scope": "accelerator_pool",
            }
        ]
    if fabric > 0:
        raw["scaleout_port_tx_rx_bytes_packets"] = [
            {
                "sample_time": iso(mid),
                "tx_bytes": int(fabric * 10**16),
                "rx_bytes": int(fabric * 10**16),
                "tx_packets": int(fabric * 10**9),
                "rx_packets": int(fabric * 10**9),
            }
        ]
        raw["fabric_port_device_sample_counters"] = [
            {"sample_time": iso(mid), "counter_name": "collective_cadence_score", "counter_value": fabric, "counter_unit": "score_0_to_1", "monitored_scope_category": "accelerator_pool"},
            {"sample_time": iso(mid), "counter_name": "participant_count", "counter_value": participants, "counter_unit": "accelerators", "monitored_scope_category": "accelerator_pool"},
            {"sample_time": iso(mid), "counter_name": "regularity_score", "counter_value": benchmark_regularity, "counter_unit": "score_0_to_1", "monitored_scope_category": "accelerator_pool"},
        ]
    if checkpoint > 0:
        raw["storage_write_operation_bytes"] = []
        raw["object_storage_operation_counts"] = []
        for idx in range(max(1, checkpoint_bursts)):
            burst_start = start + timedelta(seconds=(idx + 1) * duration_seconds / (checkpoint_bursts + 2))
            burst_end = burst_start + timedelta(minutes=45)
            raw["storage_write_operation_bytes"].append(
                {
                    "start_time": iso(burst_start),
                    "end_time": iso(burst_end),
                    "write_operation_count": int(1000 + checkpoint * 10000),
                    "write_bytes": int(checkpoint * 10**15),
                }
            )
            raw["object_storage_operation_counts"].append(
                {
                    "start_time": iso(burst_start),
                    "end_time": iso(burst_end),
                    "operation_type": "synthetic_checkpoint_state_write",
                    "operation_count": int(1000 + checkpoint * 10000),
                    "object_count": int(128 + checkpoint * 4096),
                    "bytes": int(checkpoint * 10**15),
                    "object_count_type": "distinct_objects",
                }
            )
    if serving > 0:
        raw["load_balancer_gateway_flow_activity"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "connection_count": int(serving * 10_000_000),
                "bytes": int(serving * 10**15),
            }
        ]
        raw["north_south_external_egress"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "bytes": int(serving * 10**15),
                "flow_count": int(serving * 1_000_000),
                "direction": "egress",
            }
        ]
    if storage_operation_overlap > 0 or bytes_explained > 0:
        raw["storage_operation_intervals"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "operation_type": "backup",
                "bytes_moved": int(bytes_explained * 10**16),
            }
        ]
    if physical_timeline_conflict:
        raw["electrical_service_status_intervals"] = [
            {
                "start_time": audit_window["start"],
                "end_time": audit_window["end"],
                "service_status": "not_energized",
                "service_capacity_mw": 0,
                "service_capacity_mva": 0,
                "service_voltage_kv": 34.5,
                "service_class": "synthetic_datacenter_service",
            }
        ]
        raw["asset_receiving_installation_events"] = [
            {
                "event_time": iso(end + timedelta(days=2)),
                "event_type": "installed",
                "asset_category": "accelerator",
                "asset_quantity": count,
            }
        ]
    if health_throttle_conflict:
        raw["accelerator_health_error_state"] = [
            {
                "sample_time": iso(mid),
                "ecc_error_count": 0,
                "retired_page_count": 0,
                "xid_or_equivalent_error_count": 12,
                "reset_count": 4,
                "link_error_count": 50,
                "throttle_event_count": 200,
                "degraded_state": True,
            }
        ]
        raw["accelerator_throttle_state"] = [{"sample_time": iso(mid), "state": "throttled"}]
    if topology_route_conflict:
        raw["topology_change_events"] = [
            {
                "event_time": iso(mid),
                "topology_change_type": "route_change",
                "affected_asset_category": "fabric_port",
                "affected_link_or_port_count": max(1, count // 4),
            }
        ]
        raw["network_gateway_nat_route_state"] = [
            {
                "event_time": iso(mid),
                "network_control_type": "route",
                "state_event_type": "updated",
                "state": "visibility_updated",
            }
        ]
    if power_activity_conflict:
        raw["rack_pdu_it_power"] = [
            {
                "sample_time": iso(mid),
                "power_watts": count * 900,
                "energy_joules": count * 900 * duration_seconds,
            }
        ]
    return raw


def expected_from_result(result: dict[str, Any]) -> dict[str, Any]:
    stages = result["stage_outputs"]
    return {
        "minimal_sparse_labels": [item["label"] for item in result["minimal_sparse_outputs"]],
        "derived_signal_checks": {
            "capacity_upper_bound_operations": stages["final_claim_routing"]["reader_numbers"]["capacity_upper_bound_operations"],
            "achieved_operations": stages["final_claim_routing"]["reader_numbers"]["achieved_operations"],
            "activity_score": stages["final_claim_routing"]["reader_numbers"]["activity_score"],
            "fabric_cadence_score": stages["final_claim_routing"]["reader_numbers"]["fabric_cadence_score"],
            "checkpoint_score": stages["final_claim_routing"]["reader_numbers"]["checkpoint_score"],
        },
        "aggregation_outputs": {
            "A_capacity_gate": stages["A_capacity_gate"]["labels"],
            "B_training_candidate_detection": stages["B_training_candidate_detection"]["labels"],
            "C_discrepancy_and_explanation_review": stages["C_discrepancy_and_explanation_review"]["labels"],
            "final_claim_routing": [stages["final_claim_routing"]["label"]],
        },
        "final_route": result["final_route"],
    }


def assert_expected(site: dict[str, Any], result: dict[str, Any]) -> None:
    expected = site["expected"]
    stages = result["stage_outputs"]
    actual_a = stages["A_capacity_gate"]["label"]
    actual_b = stages["B_training_candidate_detection"]["labels"]
    actual_c = stages["C_discrepancy_and_explanation_review"]["labels"]
    actual_final = result["final_route"]
    actual_short = stages["A_capacity_gate"]["short_circuited"]

    errors = []
    if actual_a != expected["A_capacity_gate_label"]:
        errors.append(f"A expected {expected['A_capacity_gate_label']} got {actual_a}")
    for label in expected["B_training_candidate_detection_labels"]:
        if label not in actual_b:
            errors.append(f"B missing {label}; got {actual_b}")
    if len(expected["B_training_candidate_detection_labels"]) == 0 and actual_b:
        errors.append(f"B expected no labels; got {actual_b}")
    for label in expected["C_discrepancy_and_explanation_review_labels"]:
        if label not in actual_c:
            errors.append(f"C missing {label}; got {actual_c}")
    if actual_final != expected["final_route"]:
        errors.append(f"final expected {expected['final_route']} got {actual_final}")
    if bool(actual_short) != bool(expected["capacity_short_circuit"]):
        errors.append(f"short circuit expected {expected['capacity_short_circuit']} got {actual_short}")
    if errors:
        joined = "; ".join(errors)
        raise AssertionError(f"{site['site_id']}: {joined}")


def write_website_data(website_dir: Path, dataset: dict[str, Any], results: list[dict[str, Any]]) -> None:
    data_dir = website_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for site, result in zip(dataset["sites"], results):
        rows.append(
            {
                "site_id": site["site_id"],
                "scenario_key": site["scenario_key"],
                "scenario_name": site["scenario_name"],
                "scope": site["scope"],
                "audit_window": site["audit_window"],
                "duration_seconds": result["duration_seconds"],
                "raw_feature_count": len(site["raw_features"]),
                "coverage": site["coverage"],
                "normalized_signals": site["normalized_signals"],
                "demo_state": result["demo_state"],
                "expected": site["expected"],
                "result": public_result(result),
                "raw_features": site["raw_features"],
                "caveats": site["caveats"],
            }
        )
    demo = {
        "metadata": {
            "dataset_id": dataset["metadata"]["dataset_id"],
            "generated_from": "datacenter-verification/synthetic/sites.json",
            "policy_threshold_operations": POLICY_THRESHOLD_OPERATIONS,
            "thresholds": THRESHOLDS,
            "control_keys": WEBSITE_CONTROL_KEYS,
            "site_count": len(rows),
            "synthetic_notice": dataset["metadata"]["synthetic_notice"],
        },
        "sites": rows,
    }
    json_text = json.dumps(demo, separators=(",", ":"), sort_keys=True)
    (data_dir / "observable-demo-data.json").write_text(json_text + "\n", encoding="utf-8")
    (data_dir / "observable-demo-data.js").write_text(
        "window.DCVObservableDemoData = " + json_text + ";\n",
        encoding="utf-8",
    )


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    stages = result["stage_outputs"]
    return {
        "final_route": result["final_route"],
        "stage_outputs": stages,
        "positive_evidence_paths": result["positive_evidence_paths"],
        "suppressors_or_explanations": result["suppressors_or_explanations"],
        "discrepancy_findings": result["discrepancy_findings"],
        "missing_channels": result["missing_channels"],
        "caveats": result["caveats"],
        "reader_numbers": result["reader_numbers"],
        "minimal_sparse_outputs": result["minimal_sparse_outputs"],
    }


def caveats_for(key: str) -> list[str]:
    caveats = {
        "A_clean_threshold_training": ["synthetic high-warning fixture, not a calibrated probability"],
        "B_covered_negative": ["capacity is live but limited, so the negative claim depends on primary and identity-shape coverage"],
        "C_capacity_ruled_out": ["rule-out applies only to the monitored scope/window, hidden capacity remains outside the claim"],
        "D_missingness_blocks_negative_screen": ["missing primary activity and identity-shape coverage blocks the no-candidate conclusion"],
        "E_storage_operation_explains_checkpoint": ["checkpoint-like writes are explained by an overlapping backup operation"],
        "F_serving_inference_counterevidence": ["large compute and activity are demoted by serving-like traffic shape"],
        "G_hpc_mpi_benchmark_alternative": ["fabric-heavy collective behavior is benchmark/HPC-like"],
        "H_capacity_claim_conflict": ["achieved-operation claim exceeds cap-adjusted capacity and requires integrity review"],
        "I_activity_attribution_conflict": ["activity attribution channel is covered but lacks allocation/running/billing overlap"],
        "J_physical_timeline_topology_health_conflict": ["physical and topology contradictions block ordinary candidate routing"],
        "K_large_compute_alone": ["large compute alone is not workload identity"],
        "L_activity_alone": ["activity alone does not classify training"],
        "M_fabric_alone": ["fabric alone does not classify training"],
        "N_storage_writes_alone": ["storage writes alone do not classify training"],
    }
    return caveats.get(key, [])


def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
