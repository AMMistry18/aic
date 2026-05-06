#!/usr/bin/env python3
"""Summarize practical assisted-RL final insertion logs.

The deployment logs split useful evidence across processes:

* aic_engine logs contain trial scores and simulator/setup errors.
* policy/model logs contain SC seating lines with assisted_rl=applied/skipped.

This script intentionally does not launch evaluation. It only reads existing
logs so we can tell whether a new run actually exercised assisted RL.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SCORE_RE = re.compile(r"Finished scoring trial, total score is:\s*([-+]?\d+(?:\.\d+)?)")
ASSIST_RE = re.compile(r"\bassisted_rl=(applied|skipped)\b")
REASON_RE = re.compile(r"\breason=([^\s]+)")
ERROR_RE = re.compile(
    r"(EVALUATION ERROR:.*|Simulator setup failed.*|TareFt service timed out.*|"
    r"Failed to spawn .*|process has died .*|Could not contact service .*|"
    r"Simulation interface encountered an error.*)"
)


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def iter_log_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("*.log"))
        else:
            continue
        for path in candidates:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def parse_float_field(line: str, name: str, scale: float = 1.0) -> float | None:
    match = re.search(rf"\b{re.escape(name)}=([-+]?\d+(?:\.\d+)?)(?:mm|N)?\b", line)
    if match is None:
        return None
    return float(match.group(1)) * scale


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def parse_logs(paths: Iterable[Path], success_score: float, max_file_mb: float | None) -> dict:
    scores: list[dict] = []
    assist_events: list[dict] = []
    setup_errors: list[dict] = []
    skipped_files: list[dict] = []

    for path in paths:
        try:
            file_size = path.stat().st_size
            if max_file_mb is not None and file_size > max_file_mb * 1024 * 1024:
                skipped_files.append(
                    {
                        "file": str(path),
                        "size_mb": file_size / (1024 * 1024),
                        "reason": "over_max_file_mb",
                    }
                )
                continue
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for lineno, raw_line in enumerate(f, 1):
                    line = clean_line(raw_line)
                    score_match = SCORE_RE.search(line)
                    if score_match is not None:
                        score = float(score_match.group(1))
                        scores.append(
                            {
                                "file": str(path),
                                "line": lineno,
                                "score": score,
                                "success": score >= success_score,
                            }
                        )

                    assist_match = ASSIST_RE.search(line)
                    if assist_match is not None:
                        status = assist_match.group(1)
                        reason = "ok" if status == "applied" else "unknown"
                        reason_match = REASON_RE.search(line)
                        if reason_match is not None:
                            reason = reason_match.group(1)
                        assist_events.append(
                            {
                                "file": str(path),
                                "line": lineno,
                                "status": status,
                                "reason": reason,
                                "tip_xy_err_mm": parse_float_field(line, "tip_xy_err"),
                                "depth_mm": parse_float_field(line, "depth"),
                                "fts_delta_n": parse_float_field(line, "fts_delta"),
                                "handoff_xy_mm": parse_float_field(line, "handoff_xy"),
                                "handoff_depth_mm": parse_float_field(line, "handoff_depth"),
                                "axis": parse_float_field(line, "axis"),
                                "twist": parse_float_field(line, "twist"),
                            }
                        )

                    error_match = ERROR_RE.search(line)
                    if error_match is not None:
                        setup_errors.append(
                            {
                                "file": str(path),
                                "line": lineno,
                                "message": error_match.group(1),
                            }
                        )
        except OSError as exc:
            setup_errors.append({"file": str(path), "line": None, "message": f"read_failed: {exc}"})

    applied = [event for event in assist_events if event["status"] == "applied"]
    skipped = [event for event in assist_events if event["status"] == "skipped"]
    known_scores = len(scores)
    successes = sum(1 for row in scores if row["success"])
    skip_reasons = Counter(event["reason"] for event in skipped)

    metric_summary = {}
    for key in (
        "tip_xy_err_mm",
        "depth_mm",
        "fts_delta_n",
        "handoff_xy_mm",
        "handoff_depth_mm",
        "axis",
        "twist",
    ):
        metric_summary[key] = summarize_values(
            [event[key] for event in assist_events if event.get(key) is not None]
        )

    return {
        "score_success_threshold": success_score,
        "score_trials": known_scores,
        "score_successes": successes,
        "score_success_rate": (successes / known_scores) if known_scores else None,
        "scores": scores,
        "assist_events": len(assist_events),
        "assist_applied": len(applied),
        "assist_skipped": len(skipped),
        "assist_apply_rate": (len(applied) / len(assist_events)) if assist_events else None,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "metrics": metric_summary,
        "setup_error_count": len(setup_errors),
        "setup_errors": setup_errors[-20:],
        "skipped_file_count": len(skipped_files),
        "skipped_files": skipped_files[-20:],
    }


def print_human(summary: dict) -> None:
    rate = summary["score_success_rate"]
    rate_text = "n/a" if rate is None else f"{rate * 100.0:.1f}%"
    apply_rate = summary["assist_apply_rate"]
    apply_text = "n/a" if apply_rate is None else f"{apply_rate * 100.0:.1f}%"
    print("Assisted-RL log summary")
    print(f"  score threshold: >= {summary['score_success_threshold']:.1f}")
    print(
        "  scored trials: "
        f"{summary['score_trials']} | successes: {summary['score_successes']} | rate: {rate_text}"
    )
    print(
        "  assist events: "
        f"{summary['assist_events']} | applied: {summary['assist_applied']} | "
        f"skipped: {summary['assist_skipped']} | apply rate: {apply_text}"
    )
    if summary["skip_reasons"]:
        print("  skip reasons:")
        for reason, count in summary["skip_reasons"].items():
            print(f"    {reason}: {count}")
    else:
        print("  skip reasons: none")
    print(f"  setup/error lines: {summary['setup_error_count']}")
    print(f"  skipped oversized files: {summary['skipped_file_count']}")

    for key in ("handoff_xy_mm", "handoff_depth_mm", "axis", "twist", "fts_delta_n"):
        stats = summary["metrics"][key]
        if stats["count"]:
            print(
                f"  {key}: n={stats['count']} median={stats['median']:.3f} "
                f"mean={stats['mean']:.3f} range=[{stats['min']:.3f}, {stats['max']:.3f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("/home/ubuntu/.ros/log"), Path("/home/ubuntu/aic_results")],
        help="Log files or directories to scan. Defaults to .ros/log and aic_results.",
    )
    parser.add_argument(
        "--success-score",
        type=float,
        default=50.0,
        help="Score treated as a practical high-band success. Default: 50.0",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=100.0,
        help="Skip individual log files larger than this many MB. Use 0 for no limit. Default: 100.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--out", type=Path, help="Optional path for JSON summary output.")
    args = parser.parse_args()

    log_files = list(iter_log_files(args.paths))
    max_file_mb = None if args.max_file_mb <= 0 else args.max_file_mb
    summary = parse_logs(log_files, args.success_score, max_file_mb)
    summary["files_scanned"] = len(log_files)
    summary["paths"] = [str(path) for path in args.paths]

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_human(summary)
        if args.out is not None:
            print(f"  saved json: {args.out}")


if __name__ == "__main__":
    main()
