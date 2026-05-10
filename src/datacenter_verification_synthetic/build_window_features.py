"""Build window feature rows from normalized synthetic raw records."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

try:
    from .common import (
        FEATURE_PIPELINE_VERSION,
        HARDWARE_NORMALIZATION_VERSION,
        POLICY_THRESHOLD_VERSION,
        REQUIRED_FEATURE_COLUMNS,
        WINDOW_SPECS,
        derive_label,
        normalize_feature_row,
        optional_write_parquet,
        parse_utc,
        raw_payload_hash,
        read_jsonl,
        stable_hash,
        utc_iso,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        FEATURE_PIPELINE_VERSION,
        HARDWARE_NORMALIZATION_VERSION,
        POLICY_THRESHOLD_VERSION,
        REQUIRED_FEATURE_COLUMNS,
        WINDOW_SPECS,
        derive_label,
        normalize_feature_row,
        optional_write_parquet,
        parse_utc,
        raw_payload_hash,
        read_jsonl,
        stable_hash,
        utc_iso,
        write_csv,
    )


def load_episode_summaries(raw_dir: Path) -> list[dict[str, Any]]:
    events_path = raw_dir / "event_records.jsonl"
    summaries: list[dict[str, Any]] = []
    for event in read_jsonl(events_path):
        if event.get("event_type") != "synthetic_episode_summary":
            continue
        attrs = json.loads(event["attributes_json"])
        attrs["episode_id"] = event["episode_id"]
        attrs["site_id"] = event["site_id"]
        attrs["event_time"] = event["event_time"]
        attrs["event_end_time"] = event["event_end_time"]
        summaries.append(attrs)
    if not summaries:
        raise ValueError(f"no synthetic_episode_summary events found in {events_path}")
    return summaries


def _window_starts(summary: dict[str, Any], length_seconds: int) -> list[Any]:
    start = parse_utc(summary["episode_start"])
    end = parse_utc(summary["episode_end"])
    duration_seconds = max(1, int((end - start).total_seconds()))
    max_windows = int(summary.get("max_windows_per_episode_per_length") or 8)
    if duration_seconds <= length_seconds:
        return [start]

    available = max(1, duration_seconds - length_seconds)
    full_count = available // length_seconds + 1
    if full_count <= max_windows:
        return [start + timedelta(seconds=length_seconds * idx) for idx in range(full_count)]

    starts = []
    for idx in range(max_windows):
        offset = round(idx * available / max(1, max_windows - 1))
        starts.append(start + timedelta(seconds=offset))
    return starts


def build_row(summary: dict[str, Any], window_name: str, window_start: Any, length_seconds: int) -> dict[str, Any]:
    template = dict(summary["feature_template"])
    window_end = window_start + timedelta(seconds=length_seconds)
    window_hours = length_seconds / 3600.0
    row_id = "feat_" + stable_hash(
        summary["dataset_id"],
        summary["episode_id"],
        window_name,
        utc_iso(window_start),
        template.get("scope_type"),
        template.get("scope_id_hash"),
        length=24,
    )

    rack_power_fraction = float(template.get("o8_rack_power_fraction_p95") or 0.0)
    facility_mw = float(template.get("o8_facility_it_power_mw") or 0.0)
    baseline_mw = float(summary.get("site_baseline_it_mw") or max(0.1, facility_mw * 0.22))
    baseline_subtracted = max(0.0, (facility_mw * rack_power_fraction - baseline_mw) * 1000.0 * window_hours)
    duration_hours = float(template.get("o2_allocation_duration_hours") or summary.get("duration_hours") or 0.0)

    row = {
        **template,
        "feature_row_id": row_id,
        "dataset_id": summary["dataset_id"],
        "seed": summary["seed"],
        "site_id": summary["site_id"],
        "window_start": utc_iso(window_start),
        "window_end": utc_iso(window_end),
        "window_length_seconds": length_seconds,
        "episode_id": summary["episode_id"],
        "latent_workload_class": summary["latent_workload_class"],
        "o8_baseline_subtracted_energy_kwh": baseline_subtracted,
        "o8_power_continuity_days": min(duration_hours / 24.0, length_seconds / 86400.0 if duration_hours == 0 else duration_hours / 24.0),
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "policy_threshold_version": POLICY_THRESHOLD_VERSION,
        "hardware_normalization_version": HARDWARE_NORMALIZATION_VERSION,
    }
    row["raw_input_manifest_hash"] = raw_payload_hash(
        {
            "episode_id": summary["episode_id"],
            "window_name": window_name,
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "raw_event_hash": summary.get("raw_event_hash"),
        }
    )
    label, confidence, reason = derive_label(row)
    row["label_0_to_4"] = label
    row["label_confidence"] = confidence
    row["label_reason"] = reason
    row["label_source"] = "synthetic_latent_truth_plus_composite_rules"

    profile = str(row.get("synthetic_evidence_profile") or "")
    row["capacity_evidence_only"] = profile == "capacity_only"
    row["integrity_evidence_only"] = profile == "integrity_only"
    row["physical_evidence_only"] = profile == "physical_only"
    return normalize_feature_row(row)


def build_features(raw_dir: Path, output_dir: Path, seed: int | None = None) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = load_episode_summaries(raw_dir)
    if seed is not None:
        for summary in summaries:
            summary.setdefault("seed", seed)

    rows_by_window: dict[str, list[dict[str, Any]]] = {name: [] for name in WINDOW_SPECS}
    for summary in summaries:
        for window_name, length_seconds in WINDOW_SPECS.items():
            for window_start in _window_starts(summary, length_seconds):
                rows_by_window[window_name].append(build_row(summary, window_name, window_start, length_seconds))

    counts: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []
    for window_name, rows in rows_by_window.items():
        rows.sort(key=lambda row: (row["window_start"], row["site_id"], row["feature_row_id"]))
        path = output_dir / f"window_features_{window_name}.csv"
        counts[path.name] = write_csv(path, rows, REQUIRED_FEATURE_COLUMNS)
        optional_write_parquet(path)
        all_rows.extend(rows)

    all_rows.sort(key=lambda row: (row["window_start"], row["window_length_seconds"], row["site_id"], row["feature_row_id"]))
    all_path = output_dir / "window_features_all.csv"
    counts[all_path.name] = write_csv(all_path, all_rows, REQUIRED_FEATURE_COLUMNS)
    optional_write_parquet(all_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing raw_normalized JSONL files.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for feature CSV files.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    counts = build_features(args.input, args.output, seed=args.seed)
    for file_name, count in counts.items():
        print(f"{file_name}: {count} rows")


if __name__ == "__main__":
    main()

