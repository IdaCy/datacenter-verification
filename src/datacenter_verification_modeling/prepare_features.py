"""Prepare leakage-safe feature metadata for the synthetic v0 modeling run."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .common import DEFAULT_SEED, determine_feature_columns, ensure_dir, load_feature_table, make_episode_split, write_json
except ImportError:  # pragma: no cover - direct script execution
    from common import DEFAULT_SEED, determine_feature_columns, ensure_dir, load_feature_table, make_episode_split, write_json


def prepare_features(features_path: Path, output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, object]:
    df = load_feature_table(features_path)
    split_df, split_manifest = make_episode_split(df, seed=seed)
    feature_columns, excluded_columns = determine_feature_columns(split_df)

    ensure_dir(output_dir)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "feature_columns.json", feature_columns)
    write_json(output_dir / "excluded_columns.json", excluded_columns)

    return {
        "rows": int(len(split_df)),
        "episodes": int(split_df["episode_id"].nunique()),
        "feature_count": int(len(feature_columns)),
        "split_summary": split_manifest["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    summary = prepare_features(args.features, args.output, seed=args.seed)
    print(f"rows: {summary['rows']}")
    print(f"episodes: {summary['episodes']}")
    print(f"model_features: {summary['feature_count']}")
    for split, values in summary["split_summary"].items():
        print(f"{split}: rows={values['rows']} episodes={values['episodes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

