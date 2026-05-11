"""Validate the synthetic datacenter verification dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (
        BASE_REQUIRED_FEATURE_COLUMNS,
        HARD_GENERATION_SCALES,
        OBSERVABLE_IDS,
        REQUIRED_FEATURE_COLUMNS,
        V1_ONLY_FEATURE_COLUMNS,
        bool_from_csv,
        parse_utc,
        read_feature_csv,
        read_jsonl,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from common import (
        BASE_REQUIRED_FEATURE_COLUMNS,
        HARD_GENERATION_SCALES,
        OBSERVABLE_IDS,
        REQUIRED_FEATURE_COLUMNS,
        V1_ONLY_FEATURE_COLUMNS,
        bool_from_csv,
        parse_utc,
        read_feature_csv,
        read_jsonl,
        write_csv,
    )


REQUIRED_FILES = [
    "README.md",
    "manifest.json",
    "schemas/metric_sample.schema.json",
    "schemas/event_record.schema.json",
    "schemas/snapshot_record.schema.json",
    "schemas/window_feature_row.schema.json",
    "schemas/prediction_record.schema.json",
    "workbook_rules/ground_truth_ranges.json",
    "workbook_rules/composite_rules.json",
    "workbook_rules/observable_matrix.json",
    "workbook_rules/label_definitions.json",
    "workbook_rules/windowing_guide.json",
    "workbook_rules/feature_engineering.json",
    "raw_normalized/metric_samples.jsonl",
    "raw_normalized/event_records.jsonl",
    "raw_normalized/snapshot_records.jsonl",
    "features/window_features_15m.csv",
    "features/window_features_1h.csv",
    "features/window_features_6h.csv",
    "features/window_features_1d.csv",
    "features/window_features_all.csv",
    "examples/one_datapoint_label0.json",
    "examples/one_datapoint_label1.json",
    "examples/one_datapoint_label2.json",
    "examples/one_datapoint_label3.json",
    "examples/one_datapoint_label4.json",
]

CRITICAL_COVERAGE_COLUMNS = [
    "o2_coverage_fraction",
    "o4_coverage_fraction",
    "o7_coverage_fraction",
    "o8_coverage_fraction",
    "o14_coverage_fraction",
]


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _write_counter_csv(path: Path, counter: Counter[Any], fieldnames: tuple[str, str]) -> None:
    rows = [{fieldnames[0]: key, fieldnames[1]: value} for key, value in sorted(counter.items(), key=lambda item: str(item[0]))]
    write_csv(path, rows, list(fieldnames))


def validate_dataset(dataset_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    validation_dir = dataset_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            errors.append(f"manifest does not parse: manifest.json: {exc}")

    for rel_path in REQUIRED_FILES:
        path = dataset_dir / rel_path
        if not path.exists():
            errors.append(f"missing required file: {rel_path}")

    for schema_path in sorted((dataset_dir / "schemas").glob("*.schema.json")):
        try:
            json.loads(schema_path.read_text())
        except Exception as exc:
            errors.append(f"schema does not parse: {schema_path.relative_to(dataset_dir)}: {exc}")

    raw_counts: dict[str, int] = {}
    for rel_path in [
        "raw_normalized/metric_samples.jsonl",
        "raw_normalized/event_records.jsonl",
        "raw_normalized/snapshot_records.jsonl",
    ]:
        path = dataset_dir / rel_path
        if path.exists():
            try:
                rows = read_jsonl(path)
                raw_counts[Path(rel_path).name] = len(rows)
                if rel_path.endswith("metric_samples.jsonl"):
                    for idx, row in enumerate(rows, start=1):
                        unit = row.get("unit")
                        value = row.get("value_num")
                        if value is None:
                            continue
                        try:
                            value_float = float(value)
                        except (TypeError, ValueError):
                            errors.append(f"{rel_path}:{idx}: value_num is not numeric: {value}")
                            continue
                        if unit == "percent" and not (0.0 <= value_float <= 100.0):
                            errors.append(f"{rel_path}:{idx}: percent value outside 0-100: {row.get('metric_name')}={value}")
                        if unit in {"fraction", "score"} and not (0.0 <= value_float <= 1.0):
                            errors.append(f"{rel_path}:{idx}: {unit} value outside 0-1: {row.get('metric_name')}={value}")
            except Exception as exc:
                errors.append(f"raw JSONL does not parse: {rel_path}: {exc}")

    feature_files = [
        "features/window_features_15m.csv",
        "features/window_features_1h.csv",
        "features/window_features_6h.csv",
        "features/window_features_1d.csv",
        "features/window_features_all.csv",
    ]
    feature_counts: dict[str, int] = {}
    all_rows: list[dict[str, Any]] = []
    for rel_path in feature_files:
        path = dataset_dir / rel_path
        if path.exists():
            try:
                rows = read_feature_csv(path)
                feature_counts[Path(rel_path).name] = len(rows)
                if rel_path.endswith("window_features_all.csv"):
                    all_rows = rows
            except Exception as exc:
                errors.append(f"feature CSV does not parse: {rel_path}: {exc}")

    if all_rows:
        header = set(all_rows[0].keys())
        scale = str(manifest.get("scale") or "").lower()
        hard_profile = scale in HARD_GENERATION_SCALES or (not scale and bool(header & set(V1_ONLY_FEATURE_COLUMNS)))
        required_feature_columns = REQUIRED_FEATURE_COLUMNS if hard_profile else BASE_REQUIRED_FEATURE_COLUMNS
        for column in required_feature_columns:
            if column not in header:
                errors.append(f"missing required feature column: {column}")
        for obs_id in OBSERVABLE_IDS:
            key = obs_id.lower()
            for suffix in ["coverage_fraction", "missing_reason"]:
                column = f"{key}_{suffix}"
                if column not in header:
                    errors.append(f"missing observable coverage/missingness column: {column}")

        label_counter: Counter[int] = Counter()
        scenario_counter: Counter[str] = Counter()
        scenario_family_counter: Counter[str] = Counter()
        counterfactual_group_counter: Counter[str] = Counter()
        missing_counter: Counter[str] = Counter()
        window_lengths: Counter[str] = Counter()
        for idx, row in enumerate(all_rows, start=2):
            try:
                label = _int(row, "label_0_to_4")
            except Exception:
                errors.append(f"features/window_features_all.csv:{idx}: label is not an integer")
                continue
            if label not in {0, 1, 2, 3, 4}:
                errors.append(f"features/window_features_all.csv:{idx}: label outside 0-4: {label}")
            label_counter[label] += 1
            scenario_counter[row.get("latent_workload_class", "")] += 1
            if row.get("scenario_family"):
                scenario_family_counter[row.get("scenario_family", "")] += 1
            if row.get("counterfactual_group_id"):
                counterfactual_group_counter[row.get("counterfactual_group_id", "")] += 1
            window_lengths[row.get("window_length_seconds", "")] += 1

            for obs_id in OBSERVABLE_IDS:
                reason = row.get(f"{obs_id.lower()}_missing_reason", "")
                if reason and reason != "observed":
                    missing_counter[f"{obs_id}:{reason}"] += 1

            if label == 0:
                weak = [column for column in CRITICAL_COVERAGE_COLUMNS if _float(row, column) < 0.95]
                if weak:
                    errors.append(
                        f"features/window_features_all.csv:{idx}: label 0 has weak critical coverage in {','.join(weak)}"
                    )
            if bool_from_csv(row.get("capacity_evidence_only")) or row.get("synthetic_evidence_profile") == "capacity_only":
                if label > 1:
                    errors.append(f"features/window_features_all.csv:{idx}: capacity-only row has label {label}")
            if bool_from_csv(row.get("integrity_evidence_only")) or row.get("synthetic_evidence_profile") == "integrity_only":
                if label > 2:
                    errors.append(f"features/window_features_all.csv:{idx}: integrity-only row has label {label}")
            if bool_from_csv(row.get("physical_evidence_only")) or row.get("synthetic_evidence_profile") == "physical_only":
                if label > 2:
                    errors.append(f"features/window_features_all.csv:{idx}: physical-only row has label {label}")

            try:
                start = parse_utc(row["window_start"])
                end = parse_utc(row["window_end"])
                if not start < end:
                    errors.append(f"features/window_features_all.csv:{idx}: window_start is not before window_end")
            except Exception as exc:
                errors.append(f"features/window_features_all.csv:{idx}: invalid UTC window timestamps: {exc}")

        for label in range(5):
            if not (dataset_dir / "examples" / f"one_datapoint_label{label}.json").exists():
                errors.append(f"missing example datapoint for label {label}")

        _write_counter_csv(validation_dir / "label_distribution.csv", Counter({str(k): v for k, v in label_counter.items()}), ("label_0_to_4", "row_count"))
        _write_counter_csv(validation_dir / "scenario_distribution.csv", scenario_counter, ("latent_workload_class", "row_count"))
        if scenario_family_counter:
            _write_counter_csv(validation_dir / "scenario_family_distribution.csv", scenario_family_counter, ("scenario_family", "row_count"))
        if counterfactual_group_counter:
            _write_counter_csv(validation_dir / "counterfactual_group_distribution.csv", counterfactual_group_counter, ("counterfactual_group_id", "row_count"))
        _write_counter_csv(validation_dir / "missingness_distribution.csv", missing_counter, ("observable_missing_reason", "row_count"))
        feature_dictionary_rows = []
        for column in required_feature_columns:
            if column.startswith("o") and "_" in column:
                family = column.split("_", 1)[0].upper()
            elif column in {"feature_row_id", "dataset_id", "seed", "site_id", "scope_type", "scope_id_hash", "window_start", "window_end"}:
                family = "identifier"
            elif "coverage" in column or "missing" in column:
                family = "coverage_missingness"
            elif column in {"label_0_to_4", "label_confidence", "label_reason", "label_source"}:
                family = "label"
            else:
                family = "metadata_or_trust"
            feature_dictionary_rows.append({"feature_name": column, "feature_family": family})
        write_csv(validation_dir / "feature_dictionary.csv", feature_dictionary_rows, ["feature_name", "feature_family"])

        summary = {
            "raw_counts": raw_counts,
            "feature_counts": feature_counts,
            "label_distribution": dict(sorted(label_counter.items())),
            "scenario_distribution": dict(sorted(scenario_counter.items())),
            "scenario_family_distribution": dict(sorted(scenario_family_counter.items())),
            "counterfactual_group_distribution": dict(sorted(counterfactual_group_counter.items())),
            "missingness_distribution": dict(sorted(missing_counter.items())),
            "window_lengths": dict(sorted(window_lengths.items())),
        }
    else:
        summary = {"raw_counts": raw_counts, "feature_counts": feature_counts}
        errors.append("no rows loaded from features/window_features_all.csv")

    report_lines = [
        "# Synthetic Dataset Validation Report",
        "",
        f"Dataset: `{dataset_dir}`",
        "",
        "## Status",
        "",
        "PASS" if not errors else "FAIL",
        "",
        "## Counts",
        "",
        f"- Raw records: {sum(raw_counts.values())} ({raw_counts})",
        f"- Feature rows: {feature_counts.get('window_features_all.csv', 0)} ({feature_counts})",
        "",
        "## Label Distribution",
        "",
    ]
    for label, count in summary.get("label_distribution", {}).items():
        report_lines.append(f"- {label}: {count}")
    report_lines.extend(["", "## Scenario Distribution", ""])
    for scenario, count in summary.get("scenario_distribution", {}).items():
        report_lines.append(f"- {scenario}: {count}")
    report_lines.extend(["", "## Missingness Distribution", ""])
    for reason, count in summary.get("missingness_distribution", {}).items():
        report_lines.append(f"- {reason}: {count}")
    report_lines.extend(["", "## Checks", ""])
    if errors:
        for error in errors:
            report_lines.append(f"- ERROR: {error}")
    else:
        report_lines.append("- All required files exist.")
        report_lines.append("- Schemas, JSONL, and feature CSV files parse.")
        report_lines.append("- Labels are restricted to 0-4.")
        report_lines.append("- Required feature, coverage, and missingness columns are present.")
        report_lines.append("- Label caps for capacity-only, integrity-only, and physical-only rows hold.")
        report_lines.append("- Label 0 rows have strong critical-layer coverage.")
        report_lines.append("- Example datapoints exist for labels 0-4.")
        report_lines.append("- Feature windows have ordered UTC timestamps ending in Z.")
    (validation_dir / "validation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return not errors, errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    ok, errors, summary = validate_dataset(args.dataset)
    print("PASS" if ok else "FAIL")
    print(f"raw_records: {sum(summary.get('raw_counts', {}).values())}")
    print(f"feature_rows: {summary.get('feature_counts', {}).get('window_features_all.csv', 0)}")
    if summary.get("label_distribution"):
        print(f"label_distribution: {summary['label_distribution']}")
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
