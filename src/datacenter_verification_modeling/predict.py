"""Generate calibrated and governance post-processed predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .evaluate_model import predict_for_model_run
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_model import predict_for_model_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-run", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _, predictions = predict_for_model_run(args.model_run, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"wrote_predictions: {args.output}")
    print(f"rows: {len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

