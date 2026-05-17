#!/usr/bin/env python3
"""run the staged observable algorithm over synthetic sites"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from datacenter_verification.observable_algorithm import evaluate_site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "synthetic" / "sites.json")
    parser.add_argument("--site", help="site_id or scenario_key to run")
    parser.add_argument("--json", action="store_true", help="emit full JSON results")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    sites = data["sites"]
    if args.site:
        sites = [
            site
            for site in sites
            if site["site_id"] == args.site or site["scenario_key"] == args.site
        ]
        if not sites:
            raise SystemExit(f"No site or scenario matched {args.site!r}")

    results = [evaluate_site(site) for site in sites]
    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    for result in results:
        stages = result["stage_outputs"]
        print(
            f"{result['site_id']} | {result['scenario_key']} | "
            f"A={stages['A_capacity_gate']['label']} | "
            f"B={','.join(stages['B_training_candidate_detection']['labels']) or 'none'} | "
            f"C={','.join(stages['C_discrepancy_and_explanation_review']['labels']) or 'none'} | "
            f"final={result['final_route']}"
        )


if __name__ == "__main__":
    main()
