#!/usr/bin/env python3
"""validate synthetic observable sites, expected outputs, and demo data"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datacenter_verification.observable_algorithm import evaluate_site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "synthetic" / "sites.json")
    parser.add_argument("--website-dir", type=Path)
    parser.add_argument("--check-website-js", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    feature_schemas = load_feature_schemas(ROOT / "observables" / "observables.yaml")
    source_refs = load_source_refs(ROOT / "observables" / "rules" / "source_ledger.yaml")

    errors: list[str] = []
    results: dict[str, dict[str, Any]] = {}
    for site in data["sites"]:
        errors.extend(validate_raw_features(site, feature_schemas))
        errors.extend(validate_source_refs(site, source_refs))
        result = evaluate_site(site)
        results[site["site_id"]] = result
        errors.extend(validate_expected(site, result))
        errors.extend(validate_stage_invariants(site, result))

    if args.website_dir:
        errors.extend(validate_website_data(args.website_dir, results))
        if args.check_website_js:
            errors.extend(validate_website_js(args.website_dir))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "sites": len(data["sites"]),
                "raw_feature_ids_validated": len(feature_schemas),
                "website_checked": bool(args.website_dir),
                "website_js_checked": bool(args.check_website_js),
            },
            indent=2,
        )
    )


def load_feature_schemas(path: Path) -> dict[str, set[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    schemas: dict[str, set[str]] = {}
    for observable in payload["observables"]:
        for feature in observable.get("features", []):
            fields = {
                field["name"]
                for field in feature.get("value_schema", {}).get("fields", [])
                if isinstance(field, dict) and "name" in field
            }
            schemas[feature["id"]] = fields
    return schemas


def load_source_refs(path: Path) -> set[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    refs = set()
    for item in payload.get("sources", []):
        if "id" in item:
            refs.add(item["id"])
    return refs


def validate_raw_features(site: dict[str, Any], schemas: dict[str, set[str]]) -> list[str]:
    errors = []
    for feature_id, records in site.get("raw_features", {}).items():
        if feature_id not in schemas:
            errors.append(f"{site['site_id']}: raw feature id {feature_id!r} is not in observables/observables.yaml")
            continue
        if not isinstance(records, list):
            errors.append(f"{site['site_id']}: raw feature {feature_id!r} must be a list")
            continue
        allowed = schemas[feature_id]
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"{site['site_id']}: raw feature {feature_id}[{index}] must be an object")
                continue
            unknown = sorted(set(record) - allowed)
            if unknown:
                errors.append(f"{site['site_id']}: raw feature {feature_id}[{index}] has unknown fields {unknown}")
    return errors


def validate_source_refs(site: dict[str, Any], source_refs: set[str]) -> list[str]:
    errors = []
    for ref in site.get("source_refs", []):
        if ref not in source_refs:
            errors.append(f"{site['site_id']}: source ref {ref!r} missing from source ledger")
    return errors


def validate_expected(site: dict[str, Any], result: dict[str, Any]) -> list[str]:
    expected = site["expected"]
    stages = result["stage_outputs"]
    errors = []
    if stages["A_capacity_gate"]["label"] != expected["A_capacity_gate_label"]:
        errors.append(
            f"{site['site_id']}: A expected {expected['A_capacity_gate_label']} got {stages['A_capacity_gate']['label']}"
        )
    for label in expected["B_training_candidate_detection_labels"]:
        if label not in stages["B_training_candidate_detection"]["labels"]:
            errors.append(f"{site['site_id']}: B missing expected label {label}")
    if not expected["B_training_candidate_detection_labels"] and stages["B_training_candidate_detection"]["labels"]:
        errors.append(f"{site['site_id']}: B expected no labels got {stages['B_training_candidate_detection']['labels']}")
    for label in expected["C_discrepancy_and_explanation_review_labels"]:
        if label not in stages["C_discrepancy_and_explanation_review"]["labels"]:
            errors.append(f"{site['site_id']}: C missing expected label {label}")
    if result["final_route"] != expected["final_route"]:
        errors.append(f"{site['site_id']}: final expected {expected['final_route']} got {result['final_route']}")
    if stages["A_capacity_gate"]["short_circuited"] != expected["capacity_short_circuit"]:
        errors.append(f"{site['site_id']}: capacity short-circuit mismatch")
    return errors


def validate_stage_invariants(site: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors = []
    stages = result["stage_outputs"]
    a = stages["A_capacity_gate"]
    b = stages["B_training_candidate_detection"]
    c = stages["C_discrepancy_and_explanation_review"]
    final = result["final_route"]

    if a["label"] == "capacity_ruled_out_for_scope":
        if b["mode"] != "skipped_due_to_capacity_ruleout" or c["mode"] != "skipped_due_to_capacity_ruleout":
            errors.append(f"{site['site_id']}: capacity rule-out did not short-circuit B/C")
    if not b["labels"] and a["label"] != "capacity_ruled_out_for_scope":
        if c["mode"] != "C1_negative_screen_integrity":
            errors.append(f"{site['site_id']}: no-candidate live segment did not use C1")
    if b["labels"]:
        if c["mode"] != "C2_candidate_conflict_adjudication":
            errors.append(f"{site['site_id']}: candidate segment did not use C2")
    if final == "no_training_like_candidate_detected_in_covered_live_segment":
        if "negative_screen_coverage_sufficient" not in c["labels"]:
            errors.append(f"{site['site_id']}: no-training final route lacks sufficient negative-screen coverage")
    if "inconclusive_due_to_missingness" in c["labels"] and final == "no_training_like_candidate_detected_in_covered_live_segment":
        errors.append(f"{site['site_id']}: missingness produced a false negative route")
    if final in {"medium_training_like_warning", "high_training_like_warning"}:
        if not ({"distributed_training_like_candidate", "checkpoint_training_like_candidate"} & set(b["labels"])):
            errors.append(f"{site['site_id']}: warning height lacks identity-shape candidate")
    return errors


def validate_website_data(website_dir: Path, results: dict[str, dict[str, Any]]) -> list[str]:
    path = website_dir / "data" / "observable-demo-data.json"
    if not path.exists():
        return [f"website data missing: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    rows = payload.get("sites", [])
    if len(rows) != len(results):
        errors.append(f"website row count {len(rows)} does not match synthetic result count {len(results)}")
    for row in rows:
        site_id = row.get("site_id")
        result = results.get(site_id)
        if not result:
            errors.append(f"website row {site_id!r} is not in synthetic sites")
            continue
        if row.get("result", {}).get("final_route") != result["final_route"]:
            errors.append(f"website row {site_id}: final route does not match algorithm")
    return errors


def validate_website_js(website_dir: Path) -> list[str]:
    node = shutil.which("node")
    if not node:
        return ["node is not available for website JS validation"]
    script = f"""
global.window = {{}};
require({json.dumps(str((website_dir / 'data' / 'observable-demo-data.js').resolve()))});
require({json.dumps(str((website_dir / 'scoring.js').resolve()))});
const data = window.DCVObservableDemoData;
const scoring = window.DCVObservableScoring;
const errors = [];
for (const row of data.sites) {{
  const result = scoring.scoreState(row.demo_state);
  if (result.final_route !== row.result.final_route) {{
    errors.push(`${{row.site_id}} expected ${{row.result.final_route}} got ${{result.final_route}}`);
  }}
}}
if (errors.length) {{
  console.error(errors.join('\\n'));
  process.exit(1);
}}
console.log(JSON.stringify({{rows: data.sites.length}}));
"""
    proc = subprocess.run([node, "-e", script], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return [f"website JS scoring mismatch: {proc.stderr.strip() or proc.stdout.strip()}"]
    return []


if __name__ == "__main__":
    main()
