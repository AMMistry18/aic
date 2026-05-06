#!/usr/bin/env python3
"""Summarize SC full-insertion reliability from scoring.yaml files."""

import argparse
import json
from pathlib import Path

import yaml


def iter_trial_blocks(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.startswith("trial_") and isinstance(v, dict):
                yield k, v
            yield from iter_trial_blocks(v)
    elif isinstance(node, list):
        for item in node:
            yield from iter_trial_blocks(item)


def flatten_items(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k).lower(), v
            yield from flatten_items(v)
    elif isinstance(node, list):
        for v in node:
            yield from flatten_items(v)


def extract_success(trial_block: dict):
    text_hits = []
    score_hits = []
    for key, value in flatten_items(trial_block):
        if "insertion" not in key and "cable" not in key and "tier" not in key:
            continue
        if isinstance(value, bool):
            text_hits.append((key, bool(value)))
        elif isinstance(value, (int, float)):
            score_hits.append((key, float(value)))
        elif isinstance(value, str):
            s = value.lower()
            if "success" in s or "inserted" in s:
                text_hits.append((key, True))
            if "fail" in s or "wrong-port" in s or "wrong port" in s:
                text_hits.append((key, False))

    for key, val in text_hits:
        if "full" in key or "success" in key or "correct" in key:
            return bool(val), {"source": key, "value": val}

    for key, val in score_hits:
        if "tier_3" in key or "tier3" in key or "cable" in key:
            if val >= 55.0:
                return True, {"source": key, "value": val}
            if val < 0.0:
                return False, {"source": key, "value": val}
    return None, None


def load_trial_results(scoring_file: Path):
    with open(scoring_file, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    rows = []
    for trial_name, block in iter_trial_blocks(doc):
        success, meta = extract_success(block)
        rows.append(
            {
                "trial": trial_name,
                "full_insertion_success": success,
                "evidence": meta,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        required=True,
        help="Directory containing one or more scoring.yaml files",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "outputs" / "sc_pose_pipeline" / "policy_eval_summary.json"),
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    scoring_files = sorted(root.rglob("scoring.yaml"))
    all_trials = []
    per_run = []
    for sf in scoring_files:
        trials = load_trial_results(sf)
        known = [t for t in trials if t["full_insertion_success"] is not None]
        run_success = sum(1 for t in known if t["full_insertion_success"])
        run_total = len(known)
        per_run.append(
            {
                "scoring_file": str(sf),
                "known_trials": run_total,
                "full_insertion_successes": run_success,
                "success_rate": (run_success / run_total) if run_total > 0 else None,
                "trials": known,
            }
        )
        all_trials.extend(known)

    total_success = sum(1 for t in all_trials if t["full_insertion_success"])
    total_trials = len(all_trials)
    summary = {
        "results_root": str(root),
        "scoring_files_found": len(scoring_files),
        "known_trials": total_trials,
        "full_insertion_successes": total_success,
        "success_rate": (total_success / total_trials) if total_trials > 0 else None,
        "runs": per_run,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
