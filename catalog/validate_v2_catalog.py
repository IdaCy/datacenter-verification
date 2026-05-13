"""Validate the clean v2 feature catalogue YAML files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml


CATALOG_DIR = Path(__file__).resolve().parent


FILES = {
    "index": CATALOG_DIR / "catalog_index.v2.yaml",
    "features": CATALOG_DIR / "observable_feature_catalog.v2.yaml",
    "effects": CATALOG_DIR / "training_probability_effects.v2.yaml",
    "dependencies": CATALOG_DIR / "feature_dependencies.v2.yaml",
    "windows": CATALOG_DIR / "conditional_plausibility_windows.v2.yaml",
    "discrepancies": CATALOG_DIR / "discrepancy_evasion_rules.v2.yaml",
}

FEATURE_CARD_FIELDS = {
    "id",
    "name",
    "kind",
    "unit",
    "range",
    "source_refs",
    "raw_data_objects",
    "normalization",
    "aggregation_windows",
    "missingness_behavior",
    "privacy_sensitivity",
    "trust_requirements",
    "training_probability_effect_ref",
    "dependencies_in",
    "dependencies_out",
    "known_false_positives",
}

RANGE_FIELDS = {
    "type",
    "min",
    "max",
    "max_type",
    "values",
    "notes",
}

BANNED_LEGACY_REFS = {
    "feature_importance.csv",
    "predictions_all.csv",
    "predictions_test.csv",
    "model.joblib",
    "empirical_synthetic_v1_range",
    "importance_mean",
    "permuted_macro_f1",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def source_refs_from(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        if "source_refs" in value and isinstance(value["source_refs"], list):
            refs.extend(str(ref) for ref in value["source_refs"])
        for child in value.values():
            refs.extend(source_refs_from(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(source_refs_from(child))
    return refs


def main() -> int:
    errors: list[str] = []
    loaded = {name: load_yaml(path) for name, path in FILES.items()}

    features_catalog = loaded["features"]
    observables = features_catalog.get("observables", {})
    sources = features_catalog.get("sources", {})
    source_ids = set(sources)

    feature_ids: set[str] = set()
    effect_refs: set[str] = set()
    for oid, observable in observables.items():
        for feature in observable.get("features", []):
            feature_id = feature.get("id")
            if not feature_id:
                errors.append(f"{oid} has feature without id")
                continue
            if feature_id in feature_ids:
                errors.append(f"duplicate feature id {feature_id}")
            feature_ids.add(feature_id)
            missing = sorted(FEATURE_CARD_FIELDS - set(feature))
            if missing:
                errors.append(f"{feature_id} missing feature card fields: {missing}")
            feature_range = feature.get("range")
            if isinstance(feature_range, dict):
                extra_range_fields = sorted(set(feature_range) - RANGE_FIELDS)
                if extra_range_fields:
                    errors.append(f"{feature_id} range has unexpected fields: {extra_range_fields}")
            effect_ref = feature.get("training_probability_effect_ref")
            if isinstance(effect_ref, str):
                effect_refs.add(effect_ref)

    if len(observables) != 17:
        errors.append(f"expected 17 observables, found {len(observables)}")
    if len(feature_ids) != 70:
        errors.append(f"expected 70 feature ids, found {len(feature_ids)}")

    effects = loaded["effects"].get("feature_effects", {})
    effect_ids = set(effects)
    if effect_refs != effect_ids:
        errors.append(
            "feature effect refs do not match effect ids: "
            f"missing_effects={sorted(effect_refs - effect_ids)[:10]}, "
            f"orphan_effects={sorted(effect_ids - effect_refs)[:10]}"
        )
    for effect_id, effect in effects.items():
        feature_id = effect.get("feature_id")
        if feature_id not in feature_ids:
            errors.append(f"{effect_id} references unknown feature_id {feature_id}")
        for field in ("direction", "strength", "shape", "label_cap", "confidence", "source_refs"):
            if field not in effect:
                errors.append(f"{effect_id} missing {field}")

    dependencies = loaded["dependencies"]
    derived_concepts = set(dependencies.get("derived_concepts", []))
    dependency_ids: set[str] = set()
    for edge in dependencies.get("edges", []):
        edge_id = edge.get("id")
        if not edge_id:
            errors.append("dependency edge without id")
            continue
        if edge_id in dependency_ids:
            errors.append(f"duplicate dependency id {edge_id}")
        dependency_ids.add(edge_id)
        for parent in edge.get("parents", []):
            if parent not in feature_ids and parent not in derived_concepts:
                errors.append(f"{edge_id} unknown parent {parent}")
        for child in edge.get("children", []):
            if child not in feature_ids and child not in derived_concepts:
                errors.append(f"{edge_id} unknown child {child}")

    for feature in (f for obs in observables.values() for f in obs.get("features", [])):
        for edge_id in feature.get("dependencies_in", []) + feature.get("dependencies_out", []):
            if edge_id not in dependency_ids:
                errors.append(f"{feature['id']} references unknown dependency {edge_id}")

    for name, document in loaded.items():
        if name == "index":
            continue
        for ref in source_refs_from(document):
            if ref not in source_ids:
                errors.append(f"{name} references unknown source {ref}")

    for source_id, source in sources.items():
        if "tier" not in source:
            errors.append(f"source {source_id} missing tier")

    for name in ("features", "effects", "dependencies", "windows", "discrepancies"):
        text = FILES[name].read_text(encoding="utf-8")
        for banned in BANNED_LEGACY_REFS:
            if banned in text:
                errors.append(f"{name} contains banned legacy reference {banned}")

    summary = {
        "valid": not errors,
        "observable_count": len(observables),
        "feature_count": len(feature_ids),
        "effect_count": len(effect_ids),
        "dependency_count": len(dependency_ids),
        "plausibility_window_count": len(loaded["windows"].get("rules", [])),
        "discrepancy_rule_count": len(loaded["discrepancies"].get("rules", [])),
        "source_count": len(source_ids),
        "errors": errors,
    }
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
