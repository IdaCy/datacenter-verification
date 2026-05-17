"""staged observable algorithm for synthetic datacenter scenarios"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from math import inf
from typing import Any

POLICY_THRESHOLD_OPERATIONS = 1.0e25

THRESHOLDS: dict[str, float] = {
    "policy_threshold_operations": POLICY_THRESHOLD_OPERATIONS,
    "min_capacity_ruleout_coverage_fraction": 0.90,
    "min_capacity_clock_alignment_confidence": 0.80,
    "min_capacity_possible_coverage_fraction": 0.60,
    "min_large_compute_coverage_fraction": 0.75,
    "min_primary_B_channel_coverage_fraction": 0.75,
    "min_activity_screen_score": 0.50,
    "min_activity_screen_duration_seconds": 600.0,
    "min_activity_for_fabric_pair": 0.55,
    "min_fabric_cadence_pair_score": 0.60,
    "min_activity_fabric_overlap_fraction": 0.50,
    "min_activity_fabric_duration_seconds": 1800.0,
    "min_activity_for_checkpoint_pair": 0.50,
    "min_checkpoint_pair_score": 0.55,
    "min_checkpoint_activity_adjacency_fraction": 0.50,
    "min_checkpoint_pair_burst_count": 2.0,
    "min_non_serving_pair_score": 0.60,
    "min_serving_channel_coverage_fraction": 0.80,
    "min_serving_suppression_score": 0.70,
    "min_serving_activity_overlap_fraction": 0.50,
    "min_operation_activity_overlap_fraction": 0.80,
    "min_bytes_explained_fraction": 0.70,
    "min_benchmark_like_cadence_score": 0.75,
    "min_benchmark_regularity_score": 0.90,
    "max_benchmark_duration_seconds": 7200.0,
    "min_sparse_hpc_fabric_cadence_score": 0.60,
    "min_sparse_hpc_activity_score": 0.50,
    "min_sparse_hpc_overlap_fraction": 0.50,
    "min_negative_screen_primary_coverage_fraction": 0.75,
    "min_negative_screen_identity_coverage_fraction": 0.75,
    "min_negative_screen_scope_mapping_coverage_fraction": 0.75,
    "min_negative_screen_clock_alignment_confidence": 0.80,
    "min_unattributed_activity_score": 0.70,
    "min_unattributed_activity_duration_seconds": 600.0,
    "max_attribution_overlap_fraction": 0.05,
    "min_attribution_channel_coverage_fraction": 0.80,
    "benign_attribution_explanation_overlap_fraction": 0.80,
    "max_sparse_achieved_to_capacity_ratio": 1.10,
    "min_sparse_capacity_conflict_coverage_fraction": 0.75,
    "unit_or_hidden_capacity_suppression_score": 0.70,
}

WEBSITE_CONTROL_KEYS = [
    "accelerator_count",
    "peak_rate_ops_per_second",
    "duration_hours",
    "capacity_coverage",
    "activity_score",
    "activity_coverage",
    "achieved_operations",
    "achieved_ops_coverage",
    "fabric_cadence_score",
    "fabric_coverage",
    "participant_count",
    "checkpoint_score",
    "checkpoint_burst_count",
    "storage_coverage",
    "serving_counterevidence_score",
    "serving_coverage",
    "serving_activity_overlap_fraction",
    "storage_operation_explained_fraction",
    "storage_operation_overlap_fraction",
    "storage_operation_coverage",
    "benchmark_regularity_score",
    "benchmark_duration_seconds",
    "hpc_mpi_score",
    "hpc_overlap_fraction",
    "benchmark_hpc_coverage",
    "attribution_coverage",
    "attribution_overlap_fraction",
    "scope_mapping_coverage",
    "clock_alignment_confidence",
    "hidden_or_unmonitored_capacity_possible",
    "physical_timeline_conflict",
    "health_throttle_conflict",
    "topology_route_conflict",
    "power_activity_conflict",
]


def evaluate_sites(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [evaluate_site(site) for site in sites]


def evaluate_site(site: dict[str, Any]) -> dict[str, Any]:
    scenario = deepcopy(site)
    audit_window = scenario["audit_window"]
    duration_seconds = _window_seconds(audit_window)
    raw_features = scenario.get("raw_features", {})
    coverage = scenario.get("coverage", {})
    signals = scenario.get("normalized_signals", {})

    derived = _derive_signals(scenario, duration_seconds)
    sparse_outputs = _sparse_outputs(scenario, derived)
    stage_a = _capacity_gate(scenario, derived)

    if stage_a["short_circuited"]:
        stage_b = {
            "stage": "B_training_candidate_detection",
            "mode": "skipped_due_to_capacity_ruleout",
            "labels": [],
            "positive_evidence_paths": [],
            "warning_height": "none",
            "notes": ["B was not run because A emitted capacity_ruled_out_for_scope."],
        }
        stage_c = {
            "stage": "C_discrepancy_and_explanation_review",
            "mode": "skipped_due_to_capacity_ruleout",
            "labels": [],
            "suppressors": [],
            "explanations": [],
            "discrepancies": [],
            "missing_channels": [],
            "notes": ["General C was not run after the capacity rule-out short circuit."],
        }
    else:
        stage_b = _candidate_detection(scenario, derived, stage_a)
        stage_c = _targeted_c_review(scenario, derived, stage_a, stage_b)

    final = _final_route(scenario, derived, stage_a, stage_b, stage_c)
    demo_state = _demo_state(scenario, derived)

    return {
        "algorithm_version": "observable_staged_v0.1",
        "site_id": scenario["site_id"],
        "scenario_key": scenario["scenario_key"],
        "scenario_name": scenario["scenario_name"],
        "scope": scenario["scope"],
        "audit_window": audit_window,
        "duration_seconds": duration_seconds,
        "raw_feature_ids": sorted(raw_features),
        "minimal_sparse_outputs": sparse_outputs,
        "derived_signals": derived,
        "stage_outputs": {
            "A_capacity_gate": stage_a,
            "B_training_candidate_detection": stage_b,
            "C_discrepancy_and_explanation_review": stage_c,
            "final_claim_routing": final,
        },
        "final_route": final["label"],
        "positive_evidence_paths": stage_b["positive_evidence_paths"],
        "suppressors_or_explanations": stage_c["suppressors"] + stage_c["explanations"],
        "discrepancy_findings": stage_c["discrepancies"],
        "missing_channels": stage_c["missing_channels"],
        "caveats": final["caveats"],
        "reader_numbers": final["reader_numbers"],
        "coverage": coverage,
        "normalized_signals": signals,
        "demo_state": demo_state,
    }


def _derive_signals(site: dict[str, Any], duration_seconds: float) -> dict[str, Any]:
    raw = site.get("raw_features", {})
    coverage = site.get("coverage", {})
    signals = site.get("normalized_signals", {})

    count = _sum(raw.get("accelerator_count_by_family_sku", []), "count")
    peak_rate = _max(raw.get("advertised_peak_rate_by_precision", []), "peak_rate")
    capacity_factor = _number(signals.get("capacity_adjustment_factor"), 1.0)
    capacity_upper = count * peak_rate * duration_seconds * capacity_factor

    achieved_rate = _max(raw.get("generic_achieved_operation_rate", []), "operation_rate")
    achieved_operations = _number(signals.get("achieved_operations"), achieved_rate * duration_seconds)
    achieved_unit_normalized = bool(signals.get("achieved_operations_unit_normalized", True))

    activity_score = max(
        _max(raw.get("accelerator_busy_or_utilization_fraction", []), "value"),
        _max(raw.get("tensor_matrix_mxu_neuron_or_engine_active_fraction", []), "value"),
        _number(signals.get("activity_score"), 0.0),
    )
    fabric_score = _number(signals.get("collective_cadence_score"), _counter(raw, "collective_cadence_score"))
    participant_count = _number(signals.get("participant_count"), _counter(raw, "participant_count"))
    checkpoint_score = _number(signals.get("checkpoint_periodicity_score"), 0.0)
    checkpoint_bursts = _number(
        signals.get("checkpoint_burst_count"),
        float(len(raw.get("storage_write_operation_bytes", []))),
    )
    serving_score = _number(signals.get("serving_counterevidence_score"), 0.0)
    non_serving_score = _number(signals.get("non_serving_score"), max(0.0, 1.0 - serving_score))
    benchmark_regularity = _number(signals.get("benchmark_regularity_score"), _counter(raw, "regularity_score"))
    benchmark_duration = _number(signals.get("benchmark_duration_seconds"), duration_seconds)
    hpc_score = _number(signals.get("hpc_mpi_score"), 0.0)

    capacity_coverage = _coverage(coverage, "capacity")
    primary_min = min(
        _coverage(coverage, "activity"),
        _coverage(coverage, "achieved_ops"),
    )
    identity_coverage = _coverage(
        coverage,
        "identity_shape",
        min(_coverage(coverage, "fabric"), _coverage(coverage, "storage")),
    )
    suppressor_coverage = min(
        _coverage(coverage, "serving"),
        _coverage(coverage, "storage_operations"),
        _coverage(coverage, "benchmark_hpc"),
    )
    scope_mapping = _coverage(coverage, "scope_mapping")
    clock_alignment = _coverage(coverage, "clock_alignment")

    capacity_ratio = achieved_operations / capacity_upper if capacity_upper > 0 else (inf if achieved_operations > 0 else 0.0)

    return {
        "capacity_segment_window": {
            "start_time": site["audit_window"]["start"],
            "end_time": site["audit_window"]["end"],
            "duration_seconds": duration_seconds,
            "capacity_coverage_fraction": capacity_coverage,
        },
        "capacity_upper_bound_flop": {
            "unit_normalized": bool(signals.get("capacity_unit_normalized", True)),
            "capacity_upper_bound_operations": capacity_upper,
            "coverage_fraction": capacity_coverage,
            "hidden_or_unmonitored_capacity_possible": bool(
                signals.get("hidden_or_unmonitored_capacity_possible", False)
            ),
            "missing_required_peak_rate": peak_rate <= 0,
            "count": count,
            "peak_rate_ops_per_second": peak_rate,
            "capacity_adjustment_factor": capacity_factor,
        },
        "achieved_operation_integral": {
            "operation_count": achieved_operations,
            "coverage_fraction": _coverage(coverage, "achieved_ops"),
            "unit_normalized": achieved_unit_normalized,
            "operation_count_to_capacity_upper_bound_ratio": capacity_ratio,
        },
        "accelerator_activity_score": {
            "activity_score": activity_score,
            "duration_seconds": _number(signals.get("activity_duration_seconds"), duration_seconds),
            "coverage_fraction": _coverage(coverage, "activity"),
        },
        "collective_cadence_score": {
            "cadence_score": fabric_score,
            "coverage_fraction": _coverage(coverage, "fabric"),
            "participant_count": participant_count,
            "activity_fabric_overlap_fraction": _number(signals.get("activity_fabric_overlap_fraction"), 0.0),
        },
        "checkpoint_periodicity_score": {
            "checkpoint_score": checkpoint_score,
            "burst_count": checkpoint_bursts,
            "coverage_fraction": _coverage(coverage, "storage"),
            "checkpoint_activity_adjacency_fraction": _number(
                signals.get("checkpoint_activity_adjacency_fraction"),
                0.0,
            ),
        },
        "non_serving_score": {
            "non_serving_score": non_serving_score,
            "serving_counterevidence_score": serving_score,
            "coverage_fraction": _coverage(coverage, "serving"),
            "serving_activity_overlap_fraction": _number(signals.get("serving_activity_overlap_fraction"), 0.0),
        },
        "storage_operation_explanation": {
            "operation_overlap_fraction": _number(signals.get("storage_operation_overlap_fraction"), 0.0),
            "bytes_explained_fraction": _number(signals.get("bytes_explained_fraction"), 0.0),
            "coverage_fraction": _coverage(coverage, "storage_operations"),
        },
        "benchmark_hpc_alternative": {
            "benchmark_regularity_score": benchmark_regularity,
            "benchmark_duration_seconds": benchmark_duration,
            "hpc_mpi_score": hpc_score,
            "hpc_overlap_fraction": _number(signals.get("hpc_overlap_fraction"), 0.0),
            "coverage_fraction": _coverage(coverage, "benchmark_hpc"),
        },
        "negative_screen_required_coverage_state": {
            "primary_activity_coverage_fraction": _coverage(coverage, "activity"),
            "achieved_ops_coverage_fraction": _coverage(coverage, "achieved_ops"),
            "identity_shape_coverage_fraction": identity_coverage,
            "suppressor_channel_coverage_fraction": suppressor_coverage,
            "scope_mapping_coverage_fraction": scope_mapping,
            "clock_alignment_confidence": clock_alignment,
            "decision_blocking_missingness": bool(signals.get("decision_blocking_missingness", False)),
        },
        "telemetry_coverage_fraction": {
            "primary_channel_min": primary_min,
            "identity_shape_coverage_fraction": identity_coverage,
            "suppressor_channel_coverage_fraction": suppressor_coverage,
            "scope_mapping_coverage_fraction": scope_mapping,
            "clock_alignment_confidence": clock_alignment,
            "missing_primary_channels": _missing_channels_for_threshold(site, threshold=0.75),
        },
        "clock_aligned_overlap_window": {
            "clock_alignment_confidence": clock_alignment,
        },
    }


def _sparse_outputs(site: dict[str, Any], derived: dict[str, Any]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    cap = derived["capacity_upper_bound_flop"]
    activity = derived["accelerator_activity_score"]
    achieved = derived["achieved_operation_integral"]
    fabric = derived["collective_cadence_score"]
    checkpoint = derived["checkpoint_periodicity_score"]
    non_serving = derived["non_serving_score"]
    storage_op = derived["storage_operation_explanation"]
    alternative = derived["benchmark_hpc_alternative"]

    if cap["count"] > 0:
        outputs.append(_output("accelerator_count_capacity_screen", "accelerator_count_threshold_capacity_screen", "capacity_count_or_shape"))
    if cap["peak_rate_ops_per_second"] > 0:
        outputs.append(_output("peak_rate_capacity_screen", "peak_rate_capacity_screen", "capacity_peak_rate"))
    if cap["capacity_upper_bound_operations"] >= POLICY_THRESHOLD_OPERATIONS:
        outputs.append(_output("capacity_threshold_possible", "threshold_scale_capacity_possible", "capacity_policy_scale"))
    elif cap["coverage_fraction"] >= THRESHOLDS["min_capacity_possible_coverage_fraction"]:
        outputs.append(_output("capacity_threshold_ruled_out", "threshold_scale_capacity_not_supported_sparse", "capacity_policy_scale"))

    if (
        activity["activity_score"] >= THRESHOLDS["min_activity_screen_score"]
        and activity["duration_seconds"] >= THRESHOLDS["min_activity_screen_duration_seconds"]
    ):
        outputs.append(_output("activity_only_screen", "accelerator_activity_screen", "accelerator_activity_run_existence"))
    if (
        achieved["operation_count"] >= POLICY_THRESHOLD_OPERATIONS
        and achieved["coverage_fraction"] >= THRESHOLDS["min_large_compute_coverage_fraction"]
        and achieved["unit_normalized"]
    ):
        outputs.append(_output("large_compute_screen", "large_compute_candidate", "large_compute_scale"))
    if _activity_fabric_pair(derived):
        outputs.append(_output("activity_fabric_cadence_candidate", "collective_like_training_candidate", "fabric_parameter_update_like"))
    if _activity_checkpoint_pair(derived):
        outputs.append(_output("activity_checkpoint_candidate", "checkpoint_like_training_candidate", "storage_checkpoint_state_like"))
    if (
        activity["activity_score"] >= THRESHOLDS["min_activity_for_fabric_pair"]
        and non_serving["non_serving_score"] >= THRESHOLDS["min_non_serving_pair_score"]
        and non_serving["coverage_fraction"] >= THRESHOLDS["min_serving_channel_coverage_fraction"]
    ):
        outputs.append(_output("activity_nonserving_candidate", "non_serving_accelerator_compute_candidate", "non_serving_workload_shape"))
    if non_serving["serving_counterevidence_score"] >= 0.65 and non_serving["coverage_fraction"] >= 0.60:
        outputs.append(_output("serving_like_network_counterevidence_screen", "serving_like_counterevidence_screen", "serving_counterevidence"))
    if (
        storage_op["operation_overlap_fraction"] >= THRESHOLDS["min_operation_activity_overlap_fraction"]
        and storage_op["bytes_explained_fraction"] >= THRESHOLDS["min_bytes_explained_fraction"]
    ):
        outputs.append(_output("storage_operation_explanation_screen", "storage_operation_explanation_screen", "storage_operation_explanation"))
    if (
        alternative["benchmark_regularity_score"] >= 0.85
        or alternative["hpc_mpi_score"] >= THRESHOLDS["min_sparse_hpc_fabric_cadence_score"]
    ):
        outputs.append(_output("benchmark_hpc_alternative_screen", "benchmark_or_hpc_alternative_screen", "benchmark_hpc_alternative"))
    return outputs


def _capacity_gate(site: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    cap = derived["capacity_upper_bound_flop"]
    coverage = cap["coverage_fraction"]
    clock = derived["clock_aligned_overlap_window"]["clock_alignment_confidence"]
    missing_required = (
        not cap["unit_normalized"]
        or cap["missing_required_peak_rate"]
        or cap["count"] <= 0
        or coverage < THRESHOLDS["min_capacity_possible_coverage_fraction"]
        or clock < THRESHOLDS["min_capacity_clock_alignment_confidence"]
    )

    if (
        cap["capacity_upper_bound_operations"] < POLICY_THRESHOLD_OPERATIONS
        and coverage >= THRESHOLDS["min_capacity_ruleout_coverage_fraction"]
        and clock >= THRESHOLDS["min_capacity_clock_alignment_confidence"]
        and not cap["hidden_or_unmonitored_capacity_possible"]
        and not cap["missing_required_peak_rate"]
        and cap["unit_normalized"]
    ):
        label = "capacity_ruled_out_for_scope"
        confidence = "conservative_bound"
        short_circuited = True
    elif missing_required:
        label = "capacity_unknown_due_to_missing_inputs"
        confidence = "weak"
        short_circuited = False
    elif cap["capacity_upper_bound_operations"] >= POLICY_THRESHOLD_OPERATIONS:
        label = "capacity_possible_for_scope"
        confidence = "screen"
        short_circuited = False
    else:
        label = "capacity_limited_but_not_ruled_out"
        confidence = "weak"
        short_circuited = False

    return {
        "stage": "A_capacity_gate",
        "label": label,
        "labels": [label],
        "confidence": confidence,
        "short_circuited": short_circuited,
        "capacity_upper_bound_operations": cap["capacity_upper_bound_operations"],
        "coverage_fraction": coverage,
        "clock_alignment_confidence": clock,
        "hidden_or_unmonitored_capacity_possible": cap["hidden_or_unmonitored_capacity_possible"],
        "missing_inputs": _capacity_missing_inputs(cap, coverage, clock),
        "notes": _capacity_notes(label),
    }


def _candidate_detection(site: dict[str, Any], derived: dict[str, Any], stage_a: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    evidence_paths: list[str] = []
    notes: list[str] = []

    achieved = derived["achieved_operation_integral"]
    telemetry = derived["telemetry_coverage_fraction"]
    primary_coverage = telemetry["primary_channel_min"]
    suppressor_coverage = telemetry["suppressor_channel_coverage_fraction"]

    if (
        achieved["operation_count"] >= POLICY_THRESHOLD_OPERATIONS
        and achieved["coverage_fraction"] >= THRESHOLDS["min_large_compute_coverage_fraction"]
        and achieved["unit_normalized"]
    ):
        labels.append("large_compute_candidate")
        evidence_paths.append("large_compute_scale")
        notes.append("Achieved-operation integral crosses T_sys; workload identity remains unresolved without independent identity-shape evidence.")

    if _activity_fabric_pair(derived) and primary_coverage >= THRESHOLDS["min_primary_B_channel_coverage_fraction"]:
        labels.append("distributed_training_like_candidate")
        evidence_paths.extend(["accelerator_activity_run_existence", "fabric_parameter_update_like"])

    if _activity_checkpoint_pair(derived) and primary_coverage >= THRESHOLDS["min_primary_B_channel_coverage_fraction"]:
        labels.append("checkpoint_training_like_candidate")
        evidence_paths.extend(["accelerator_activity_run_existence", "storage_checkpoint_state_like"])

    if (
        "large_compute_candidate" in labels
        and ("distributed_training_like_candidate" in labels or "checkpoint_training_like_candidate" in labels)
    ):
        labels.append("sparse_large_compute_training_like_candidate")

    identity_count = _identity_category_count(derived, labels)
    if not labels:
        warning_height = "none"
    elif suppressor_coverage < THRESHOLDS["min_primary_B_channel_coverage_fraction"]:
        warning_height = "weak_training_like_candidate"
        notes.append("Suppressor coverage is incomplete, so B cannot promote medium or high warning.")
    elif identity_count >= 2 and primary_coverage >= 0.85:
        warning_height = "high_training_like_warning"
    elif "distributed_training_like_candidate" in labels or "checkpoint_training_like_candidate" in labels:
        warning_height = "medium_training_like_warning"
    else:
        warning_height = "weak_training_like_candidate"

    return {
        "stage": "B_training_candidate_detection",
        "mode": "candidate_detection",
        "capacity_input_label": stage_a["label"],
        "labels": _unique(labels),
        "positive_evidence_paths": _unique(evidence_paths),
        "identity_category_count": identity_count,
        "warning_height": warning_height,
        "primary_channel_coverage_fraction": primary_coverage,
        "suppressor_channel_coverage_fraction": suppressor_coverage,
        "notes": notes,
    }


def _targeted_c_review(
    site: dict[str, Any],
    derived: dict[str, Any],
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
) -> dict[str, Any]:
    labels: list[str] = []
    suppressors: list[str] = []
    explanations: list[str] = []
    discrepancies: list[str] = []
    missing_channels: list[str] = []
    notes: list[str] = []
    candidate_labels = set(stage_b["labels"])
    has_candidate = bool(candidate_labels)

    if not has_candidate:
        neg = derived["negative_screen_required_coverage_state"]
        missing_channels = _negative_screen_missing_channels(neg)
        if missing_channels or neg["decision_blocking_missingness"]:
            labels.extend(["negative_screen_blocked_by_missingness", "inconclusive_due_to_missingness"])
            notes.append("C1 ran only the coverage, scope, and clock checks needed to adjudicate the negative screen.")
        else:
            labels.append("negative_screen_coverage_sufficient")
            notes.append("C1 verified primary, identity-shape, scope-mapping, and clock coverage for the no-candidate segment.")
        return {
            "stage": "C_discrepancy_and_explanation_review",
            "mode": "C1_negative_screen_integrity",
            "labels": labels,
            "suppressors": suppressors,
            "explanations": explanations,
            "discrepancies": discrepancies,
            "missing_channels": missing_channels,
            "notes": notes,
        }

    non_serving = derived["non_serving_score"]
    storage_op = derived["storage_operation_explanation"]
    alternative = derived["benchmark_hpc_alternative"]
    activity = derived["accelerator_activity_score"]
    achieved = derived["achieved_operation_integral"]
    cap = derived["capacity_upper_bound_flop"]
    telemetry = derived["telemetry_coverage_fraction"]
    signals = site.get("normalized_signals", {})
    coverage = site.get("coverage", {})

    if (
        non_serving["serving_counterevidence_score"] >= THRESHOLDS["min_serving_suppression_score"]
        and non_serving["serving_activity_overlap_fraction"] >= THRESHOLDS["min_serving_activity_overlap_fraction"]
        and non_serving["coverage_fraction"] >= 0.60
    ):
        labels.append("candidate_explained_by_serving")
        suppressors.append("serving_counterevidence")
        explanations.append("serving-like network shape overlaps the activity window.")

    if (
        ("checkpoint_training_like_candidate" in candidate_labels or "distributed_training_like_candidate" in candidate_labels)
        and storage_op["operation_overlap_fraction"] >= THRESHOLDS["min_operation_activity_overlap_fraction"]
        and storage_op["bytes_explained_fraction"] >= THRESHOLDS["min_bytes_explained_fraction"]
    ):
        labels.append("candidate_explained_by_storage_operation")
        suppressors.append("storage_operation_explanation")
        explanations.append("Explicit storage operation explains most overlapping checkpoint/fabric bytes.")

    if (
        alternative["benchmark_regularity_score"] >= THRESHOLDS["min_benchmark_regularity_score"]
        and alternative["benchmark_duration_seconds"] <= THRESHOLDS["max_benchmark_duration_seconds"]
        and alternative["coverage_fraction"] >= 0.60
    ):
        labels.append("candidate_benchmark_like")
        suppressors.append("benchmark_hpc_alternative")
        explanations.append("Regular short collective cadence is benchmark-like.")

    if (
        alternative["hpc_mpi_score"] >= THRESHOLDS["min_sparse_hpc_fabric_cadence_score"]
        and activity["activity_score"] >= THRESHOLDS["min_sparse_hpc_activity_score"]
        and alternative["hpc_overlap_fraction"] >= THRESHOLDS["min_sparse_hpc_overlap_fraction"]
    ):
        labels.append("candidate_hpc_mpi_alternative")
        suppressors.append("benchmark_hpc_alternative")
        explanations.append("Fabric-heavy activity aligns with an HPC/MPI alternative.")

    if _capacity_claim_conflict(derived, signals):
        labels.append("capacity_claim_conflict")
        discrepancies.append("capacity_claim_conflict")
    if _activity_attribution_conflict(derived, coverage, signals):
        labels.append("activity_attribution_conflict")
        discrepancies.append("activity_attribution_conflict")

    for key, label in [
        ("physical_timeline_conflict", "physical_timeline_conflict"),
        ("health_throttle_conflict", "health_throttle_conflict"),
        ("topology_route_conflict", "topology_route_conflict"),
        ("power_activity_conflict", "power_activity_conflict"),
    ]:
        if signals.get(key):
            labels.append(label)
            discrepancies.append(label)

    for channel in _candidate_missing_channels(derived, candidate_labels):
        if channel not in missing_channels:
            missing_channels.append(channel)
    if missing_channels:
        labels.append("inconclusive_due_to_missingness")

    if discrepancies:
        labels.append("integrity_review_required")
    elif missing_channels:
        labels.append("candidate_requires_manual_review")
    elif not _checked_suppressors(derived):
        labels.append("candidate_demoted_by_unresolved_suppressor")
        suppressors.append("missing_suppressor_coverage")

    if "large_compute_candidate" in candidate_labels and _identity_category_count(derived, list(candidate_labels)) == 0:
        notes.append("Large compute candidate has no independent fabric/checkpoint/non-serving identity support.")

    return {
        "stage": "C_discrepancy_and_explanation_review",
        "mode": "C2_candidate_conflict_adjudication",
        "labels": _unique(labels),
        "suppressors": _unique(suppressors),
        "explanations": _unique(explanations),
        "discrepancies": _unique(discrepancies),
        "missing_channels": _unique(missing_channels),
        "notes": notes,
        "achieved_to_capacity_ratio": achieved["operation_count_to_capacity_upper_bound_ratio"],
        "capacity_upper_bound_operations": cap["capacity_upper_bound_operations"],
        "suppressor_channel_coverage_fraction": telemetry["suppressor_channel_coverage_fraction"],
    }


def _final_route(
    site: dict[str, Any],
    derived: dict[str, Any],
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
    stage_c: dict[str, Any],
) -> dict[str, Any]:
    c_labels = set(stage_c["labels"])
    b_labels = set(stage_b["labels"])
    caveats: list[str] = []

    if stage_a["label"] == "capacity_ruled_out_for_scope":
        label = "capacity_ruled_out_for_scope"
    elif "integrity_review_required" in c_labels:
        label = "integrity_review_required"
    elif "inconclusive_due_to_missingness" in c_labels and not b_labels:
        label = "inconclusive_due_to_missingness"
    elif c_labels.intersection(
        {
            "candidate_explained_by_serving",
            "candidate_explained_by_storage_operation",
            "candidate_benchmark_like",
            "candidate_hpc_mpi_alternative",
            "candidate_demoted_by_unresolved_suppressor",
        }
    ):
        label = "candidate_explained_or_demoted"
    elif "inconclusive_due_to_missingness" in c_labels:
        label = "inconclusive_due_to_missingness"
    elif stage_b["warning_height"] == "high_training_like_warning":
        label = "high_training_like_warning"
    elif stage_b["warning_height"] == "medium_training_like_warning":
        label = "medium_training_like_warning"
    elif b_labels:
        label = "weak_training_like_candidate"
    elif "negative_screen_coverage_sufficient" in c_labels:
        label = "no_training_like_candidate_detected_in_covered_live_segment"
    else:
        label = "inconclusive_due_to_missingness"

    if "large_compute_candidate" in b_labels and not (
        {"distributed_training_like_candidate", "checkpoint_training_like_candidate"} & b_labels
    ):
        caveats.append("large_compute_training_identity_unresolved")
    if stage_c["missing_channels"]:
        caveats.append("decision_blocking_missing_channels: " + ", ".join(stage_c["missing_channels"]))
    if stage_c["suppressors"]:
        caveats.append("candidate suppressors/explanations checked: " + ", ".join(stage_c["suppressors"]))
    if stage_a["missing_inputs"]:
        caveats.append("capacity missing inputs: " + ", ".join(stage_a["missing_inputs"]))

    reader_numbers = {
        "capacity_upper_bound_operations": derived["capacity_upper_bound_flop"]["capacity_upper_bound_operations"],
        "achieved_operations": derived["achieved_operation_integral"]["operation_count"],
        "achieved_to_capacity_ratio": derived["achieved_operation_integral"]["operation_count_to_capacity_upper_bound_ratio"],
        "activity_score": derived["accelerator_activity_score"]["activity_score"],
        "fabric_cadence_score": derived["collective_cadence_score"]["cadence_score"],
        "checkpoint_score": derived["checkpoint_periodicity_score"]["checkpoint_score"],
        "serving_counterevidence_score": derived["non_serving_score"]["serving_counterevidence_score"],
        "capacity_coverage": derived["capacity_upper_bound_flop"]["coverage_fraction"],
        "primary_channel_coverage": derived["telemetry_coverage_fraction"]["primary_channel_min"],
        "identity_shape_coverage": derived["telemetry_coverage_fraction"]["identity_shape_coverage_fraction"],
        "scope_mapping_coverage": derived["telemetry_coverage_fraction"]["scope_mapping_coverage_fraction"],
        "clock_alignment_confidence": derived["clock_aligned_overlap_window"]["clock_alignment_confidence"],
    }

    return {
        "stage": "final_claim_routing",
        "label": label,
        "warning_height": label if label.endswith("_warning") or label == "weak_training_like_candidate" else "none",
        "caveats": caveats,
        "reader_numbers": reader_numbers,
        "statement": _final_statement(label),
    }


def _demo_state(site: dict[str, Any], derived: dict[str, Any]) -> dict[str, Any]:
    cap = derived["capacity_upper_bound_flop"]
    achieved = derived["achieved_operation_integral"]
    activity = derived["accelerator_activity_score"]
    fabric = derived["collective_cadence_score"]
    checkpoint = derived["checkpoint_periodicity_score"]
    non_serving = derived["non_serving_score"]
    storage_op = derived["storage_operation_explanation"]
    alternative = derived["benchmark_hpc_alternative"]
    negative = derived["negative_screen_required_coverage_state"]
    signals = site.get("normalized_signals", {})
    coverage = site.get("coverage", {})

    return {
        "accelerator_count": cap["count"],
        "peak_rate_ops_per_second": cap["peak_rate_ops_per_second"],
        "duration_hours": derived["capacity_segment_window"]["duration_seconds"] / 3600.0,
        "capacity_coverage": cap["coverage_fraction"],
        "activity_score": activity["activity_score"],
        "activity_coverage": activity["coverage_fraction"],
        "achieved_operations": achieved["operation_count"],
        "achieved_ops_coverage": achieved["coverage_fraction"],
        "fabric_cadence_score": fabric["cadence_score"],
        "fabric_coverage": fabric["coverage_fraction"],
        "participant_count": fabric["participant_count"],
        "checkpoint_score": checkpoint["checkpoint_score"],
        "checkpoint_burst_count": checkpoint["burst_count"],
        "storage_coverage": checkpoint["coverage_fraction"],
        "serving_counterevidence_score": non_serving["serving_counterevidence_score"],
        "serving_coverage": non_serving["coverage_fraction"],
        "serving_activity_overlap_fraction": non_serving["serving_activity_overlap_fraction"],
        "storage_operation_explained_fraction": storage_op["bytes_explained_fraction"],
        "storage_operation_overlap_fraction": storage_op["operation_overlap_fraction"],
        "storage_operation_coverage": storage_op["coverage_fraction"],
        "benchmark_regularity_score": alternative["benchmark_regularity_score"],
        "benchmark_duration_seconds": alternative["benchmark_duration_seconds"],
        "hpc_mpi_score": alternative["hpc_mpi_score"],
        "hpc_overlap_fraction": alternative["hpc_overlap_fraction"],
        "benchmark_hpc_coverage": alternative["coverage_fraction"],
        "attribution_coverage": _coverage(coverage, "attribution"),
        "attribution_overlap_fraction": _number(signals.get("attribution_overlap_fraction"), 1.0),
        "scope_mapping_coverage": negative["scope_mapping_coverage_fraction"],
        "clock_alignment_confidence": negative["clock_alignment_confidence"],
        "hidden_or_unmonitored_capacity_possible": cap["hidden_or_unmonitored_capacity_possible"],
        "physical_timeline_conflict": bool(signals.get("physical_timeline_conflict", False)),
        "health_throttle_conflict": bool(signals.get("health_throttle_conflict", False)),
        "topology_route_conflict": bool(signals.get("topology_route_conflict", False)),
        "power_activity_conflict": bool(signals.get("power_activity_conflict", False)),
    }


def _activity_fabric_pair(derived: dict[str, Any]) -> bool:
    activity = derived["accelerator_activity_score"]
    fabric = derived["collective_cadence_score"]
    return (
        activity["activity_score"] >= THRESHOLDS["min_activity_for_fabric_pair"]
        and fabric["cadence_score"] >= THRESHOLDS["min_fabric_cadence_pair_score"]
        and fabric["activity_fabric_overlap_fraction"] >= THRESHOLDS["min_activity_fabric_overlap_fraction"]
        and activity["duration_seconds"] >= THRESHOLDS["min_activity_fabric_duration_seconds"]
    )


def _activity_checkpoint_pair(derived: dict[str, Any]) -> bool:
    activity = derived["accelerator_activity_score"]
    checkpoint = derived["checkpoint_periodicity_score"]
    return (
        activity["activity_score"] >= THRESHOLDS["min_activity_for_checkpoint_pair"]
        and checkpoint["checkpoint_score"] >= THRESHOLDS["min_checkpoint_pair_score"]
        and checkpoint["checkpoint_activity_adjacency_fraction"] >= THRESHOLDS["min_checkpoint_activity_adjacency_fraction"]
        and checkpoint["burst_count"] >= THRESHOLDS["min_checkpoint_pair_burst_count"]
    )


def _identity_category_count(derived: dict[str, Any], labels: list[str]) -> int:
    categories = set()
    if "distributed_training_like_candidate" in labels:
        categories.add("fabric_parameter_update_like")
    if "checkpoint_training_like_candidate" in labels:
        categories.add("storage_checkpoint_state_like")
    non_serving = derived["non_serving_score"]
    if (
        non_serving["non_serving_score"] >= THRESHOLDS["min_non_serving_pair_score"]
        and non_serving["coverage_fraction"] >= THRESHOLDS["min_serving_channel_coverage_fraction"]
    ):
        categories.add("non_serving_workload_shape")
    return len(categories)


def _checked_suppressors(derived: dict[str, Any]) -> bool:
    return derived["telemetry_coverage_fraction"]["suppressor_channel_coverage_fraction"] >= THRESHOLDS[
        "min_primary_B_channel_coverage_fraction"
    ]


def _capacity_claim_conflict(derived: dict[str, Any], signals: dict[str, Any]) -> bool:
    achieved = derived["achieved_operation_integral"]
    cap = derived["capacity_upper_bound_flop"]
    suppression = _number(signals.get("unit_mismatch_or_hidden_capacity_explanation_score"), 0.0)
    return (
        achieved["unit_normalized"]
        and achieved["operation_count_to_capacity_upper_bound_ratio"] > THRESHOLDS["max_sparse_achieved_to_capacity_ratio"]
        and cap["coverage_fraction"] >= THRESHOLDS["min_sparse_capacity_conflict_coverage_fraction"]
        and suppression < THRESHOLDS["unit_or_hidden_capacity_suppression_score"]
    )


def _activity_attribution_conflict(
    derived: dict[str, Any],
    coverage: dict[str, Any],
    signals: dict[str, Any],
) -> bool:
    activity = derived["accelerator_activity_score"]
    return (
        activity["activity_score"] >= THRESHOLDS["min_unattributed_activity_score"]
        and activity["duration_seconds"] >= THRESHOLDS["min_unattributed_activity_duration_seconds"]
        and _number(signals.get("attribution_overlap_fraction"), 1.0) <= THRESHOLDS["max_attribution_overlap_fraction"]
        and _coverage(coverage, "attribution") >= THRESHOLDS["min_attribution_channel_coverage_fraction"]
        and _number(signals.get("benign_attribution_explanation_overlap_fraction"), 0.0)
        < THRESHOLDS["benign_attribution_explanation_overlap_fraction"]
    )


def _candidate_missing_channels(derived: dict[str, Any], labels: set[str]) -> list[str]:
    missing = []
    telemetry = derived["telemetry_coverage_fraction"]
    if telemetry["primary_channel_min"] < THRESHOLDS["min_primary_B_channel_coverage_fraction"]:
        missing.append("primary_activity_or_achieved_ops")
    if (
        {"distributed_training_like_candidate", "checkpoint_training_like_candidate", "sparse_large_compute_training_like_candidate"}
        & labels
        and telemetry["identity_shape_coverage_fraction"] < THRESHOLDS["min_primary_B_channel_coverage_fraction"]
    ):
        missing.append("identity_shape")
    if telemetry["scope_mapping_coverage_fraction"] < THRESHOLDS["min_negative_screen_scope_mapping_coverage_fraction"]:
        missing.append("scope_mapping")
    if telemetry["clock_alignment_confidence"] < THRESHOLDS["min_negative_screen_clock_alignment_confidence"]:
        missing.append("clock_alignment")
    if telemetry["suppressor_channel_coverage_fraction"] < THRESHOLDS["min_primary_B_channel_coverage_fraction"]:
        missing.append("suppressor_channels")
    return missing


def _negative_screen_missing_channels(negative: dict[str, Any]) -> list[str]:
    missing = []
    if negative["primary_activity_coverage_fraction"] < THRESHOLDS["min_negative_screen_primary_coverage_fraction"]:
        missing.append("primary_activity")
    if negative["identity_shape_coverage_fraction"] < THRESHOLDS["min_negative_screen_identity_coverage_fraction"]:
        missing.append("identity_shape")
    if negative["scope_mapping_coverage_fraction"] < THRESHOLDS["min_negative_screen_scope_mapping_coverage_fraction"]:
        missing.append("scope_mapping")
    if negative["clock_alignment_confidence"] < THRESHOLDS["min_negative_screen_clock_alignment_confidence"]:
        missing.append("clock_alignment")
    return missing


def _missing_channels_for_threshold(site: dict[str, Any], threshold: float) -> list[str]:
    coverage = site.get("coverage", {})
    return sorted(key for key, value in coverage.items() if isinstance(value, (int, float)) and value < threshold)


def _capacity_missing_inputs(cap: dict[str, Any], coverage: float, clock: float) -> list[str]:
    missing = []
    if cap["count"] <= 0:
        missing.append("accelerator_count_by_family_sku")
    if cap["missing_required_peak_rate"]:
        missing.append("advertised_peak_rate_by_precision")
    if coverage < THRESHOLDS["min_capacity_possible_coverage_fraction"]:
        missing.append("capacity_coverage")
    if clock < THRESHOLDS["min_capacity_clock_alignment_confidence"]:
        missing.append("clock_alignment")
    if cap["hidden_or_unmonitored_capacity_possible"]:
        missing.append("hidden_or_unmonitored_capacity_possible")
    if not cap["unit_normalized"]:
        missing.append("capacity_unit_normalization")
    return missing


def _capacity_notes(label: str) -> list[str]:
    if label == "capacity_ruled_out_for_scope":
        return ["Conservative capacity upper bound is below T_sys with high coverage; B and C are skipped."]
    if label == "capacity_possible_for_scope":
        return ["Capacity remains live; A does not imply workload identity."]
    if label == "capacity_limited_but_not_ruled_out":
        return ["Capacity is below T_sys under current inputs, but coverage is not strong enough for a rule-out."]
    return ["Capacity inputs are missing or insufficiently aligned; B/C remain live with missingness caveats."]


def _final_statement(label: str) -> str:
    statements = {
        "capacity_ruled_out_for_scope": "Threshold-scale compute is conservatively ruled out for this monitored scope/window.",
        "no_training_like_candidate_detected_in_covered_live_segment": "No training-like candidate was detected and C1 coverage is sufficient to trust the negative screen.",
        "weak_training_like_candidate": "A sparse candidate remains, but evidence is too incomplete or underdetermined for medium/high warning.",
        "medium_training_like_warning": "Aligned activity and one training-like identity pathway survive targeted C review.",
        "high_training_like_warning": "Multiple independent aligned identity/support pathways survive targeted C review.",
        "candidate_explained_or_demoted": "Candidate evidence is explained or demoted by serving, storage-operation, benchmark, or HPC counterevidence.",
        "integrity_review_required": "Targeted C found an unresolved discrepancy that blocks ordinary warning routing.",
        "inconclusive_due_to_missingness": "Decision-blocking telemetry, scope, or clock coverage prevents a trusted positive or negative route.",
    }
    return statements.get(label, label)


def _output(rule_id: str, label: str, category: str) -> dict[str, str]:
    return {"rule_id": rule_id, "label": label, "category": category}


def _sum(records: list[dict[str, Any]], key: str) -> float:
    return sum(_number(record.get(key), 0.0) for record in records)


def _max(records: list[dict[str, Any]], key: str) -> float:
    values = [_number(record.get(key), 0.0) for record in records]
    return max(values) if values else 0.0


def _counter(raw_features: dict[str, Any], counter_name: str) -> float:
    values = []
    for record in raw_features.get("fabric_port_device_sample_counters", []):
        if record.get("counter_name") == counter_name:
            values.append(_number(record.get("counter_value"), 0.0))
    return max(values) if values else 0.0


def _coverage(coverage: dict[str, Any], key: str, default: float = 1.0) -> float:
    return _clamp(_number(coverage.get(key), default))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _window_seconds(window: dict[str, str]) -> float:
    start = _parse_time(window["start"])
    end = _parse_time(window["end"])
    return (end - start).total_seconds()


def _parse_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
