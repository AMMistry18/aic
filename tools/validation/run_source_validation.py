#!/usr/bin/env python3
"""Run canonical source-test suites and write provenance-rich evidence reports.

This tool intentionally validates source tests only. It does not start the
simulator, control hardware, or measure end-to-end cable insertion success.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


TOOL_VERSION = "1"
LIMITATION = (
    "This report records source-test results only. It is not evidence of "
    "end-to-end simulated or physical cable-insertion success."
)


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    pythonpath_entry: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class SuiteResult:
    name: str
    command: list[str]
    environment: dict[str, str]
    status: str
    exit_code: int | None
    duration_seconds: float
    counts: dict[str, int]
    output: str


CANONICAL_SUITES = (
    SuiteConfig(
        name="participant_model_and_frozen_validation",
        pythonpath_entry="aic_model",
        paths=("aic_model/test", "testing/sfp_v50_validation/tests"),
    ),
    SuiteConfig(
        name="flowstate_perception",
        pythonpath_entry="flowstate/aic_perception",
        paths=("flowstate/aic_perception/test",),
    ),
)

_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<label>passed|failed|skipped|xfailed|xpassed|error|errors|"
    r"deselected|warnings?)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(r"=+.*?in\s+[\d.]+s\s*=+|\bno tests ran\b", re.IGNORECASE)


def parse_pytest_counts(output: str) -> dict[str, int]:
    """Return pytest result counts from its final summary line when available."""
    summary_lines = [line for line in output.splitlines() if _SUMMARY_RE.search(line)]
    line = summary_lines[-1] if summary_lines else output
    counts: dict[str, int] = {}
    for match in _COUNT_RE.finditer(line):
        label = match.group("label").lower().rstrip("s")
        if label == "error":
            label = "errors"
        counts[label] = counts.get(label, 0) + int(match.group("count"))
    return dict(sorted(counts.items()))


def build_suite_environment(pythonpath_entry: str, inherited: dict[str, str] | None = None) -> dict[str, str]:
    """Build the documented test environment while preserving an existing PYTHONPATH."""
    environment = dict(os.environ if inherited is None else inherited)
    previous = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        pythonpath_entry if not previous else f"{pythonpath_entry}{os.pathsep}{previous}"
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def run_suite(repo_root: Path, python: Path, suite: SuiteConfig) -> SuiteResult:
    command = [str(python), "-m", "pytest", "-q", *suite.paths]
    environment = build_suite_environment(suite.pythonpath_entry)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        exit_code: int | None = completed.returncode
        output = completed.stdout
    except OSError as exc:
        exit_code = None
        output = f"Could not execute pytest: {exc}\n"
    duration_seconds = round(time.monotonic() - started, 3)
    counts = parse_pytest_counts(output)
    status = (
        "passed"
        if exit_code == 0
        and counts.get("passed", 0) > 0
        and counts.get("failed", 0) == 0
        and counts.get("errors", 0) == 0
        else "failed"
    )
    return SuiteResult(
        name=suite.name,
        command=command,
        environment={
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"],
            "PYTHONPATH": environment["PYTHONPATH"],
        },
        status=status,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        counts=counts,
        output=output,
    )


def _git_output(repo_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def source_metadata(repo_root: Path, python: Path) -> dict[str, Any]:
    dirty = _git_output(repo_root, "status", "--porcelain=v1")
    return {
        "git": {
            "branch": _git_output(repo_root, "branch", "--show-current"),
            "dirty": bool(dirty),
            "dirty_paths": dirty.splitlines() if dirty else [],
            "revision": _git_output(repo_root, "rev-parse", "HEAD"),
        },
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_executable": str(python),
            "python_version": sys.version.replace("\n", " "),
        },
    }


def build_evidence(repo_root: Path, python: Path, suite_results: Iterable[SuiteResult]) -> dict[str, Any]:
    results = [asdict(result) for result in suite_results]
    return {
        "schema_version": TOOL_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "limitation": LIMITATION,
        "metadata": source_metadata(repo_root, python),
        "overall_status": "passed" if all(result["status"] == "passed" for result in results) else "failed",
        "repo_root": str(repo_root.resolve()),
        "suites": results,
    }


def markdown_report(evidence: dict[str, Any]) -> str:
    """Render a concise, stable report from an evidence document."""
    git = evidence["metadata"]["git"]
    platform_metadata = evidence["metadata"]["platform"]
    lines = [
        "# Source Validation Evidence",
        "",
        f"Overall status: **{evidence['overall_status'].upper()}**",
        "",
        "## Scope",
        "",
        evidence["limitation"],
        "",
        "## Provenance",
        "",
        f"- Generated (UTC): `{evidence['generated_at_utc']}`",
        f"- Git revision: `{git['revision'] or 'unavailable'}`",
        f"- Git branch: `{git['branch'] or 'unavailable'}`",
        f"- Git working tree dirty: `{str(git['dirty']).lower()}`",
        f"- Python: `{platform_metadata['python_version']}`",
        f"- Platform: `{platform_metadata['platform']}`",
        "",
        "## Canonical Source-Test Suites",
        "",
        "| Suite | Status | Counts | Duration |",
        "| --- | --- | --- | ---: |",
    ]
    for suite in evidence["suites"]:
        counts = ", ".join(f"{value} {key}" for key, value in suite["counts"].items()) or "not parsed"
        lines.append(
            f"| `{suite['name']}` | {suite['status']} | {counts} | {suite['duration_seconds']:.3f}s |"
        )
    lines.extend(["", "The JSON companion report retains each invoked command and complete pytest output.", ""])
    return "\n".join(lines)


def write_evidence(evidence: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "source-validation.json"
    markdown_path = output_dir / "source-validation.md"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(evidence), encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing the canonical test paths.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Python executable to use; defaults to .pixi/envs/default/bin/python under repo root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/source_validation"),
        help="Directory for source-validation.json and source-validation.md.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    python = (args.python or repo_root / ".pixi/envs/default/bin/python").resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    suite_results = [run_suite(repo_root, python, suite) for suite in CANONICAL_SUITES]
    evidence = build_evidence(repo_root, python, suite_results)
    json_path, markdown_path = write_evidence(evidence, output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    print(f"Source validation {evidence['overall_status']}")
    return 0 if evidence["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
