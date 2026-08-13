from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from tools.validation.run_source_validation import (
    LIMITATION,
    SuiteConfig,
    SuiteResult,
    build_evidence,
    build_suite_environment,
    markdown_report,
    parse_pytest_counts,
    run_suite,
    write_evidence,
)


def test_parse_pytest_counts_uses_final_summary() -> None:
    output = """====================== 1 failed, 2 passed in 0.01s ======================\n====================== 302 passed in 18.75s ======================\n"""
    assert parse_pytest_counts(output) == {"passed": 302}


def test_parse_pytest_counts_normalizes_plural_errors() -> None:
    output = "================ 2 errors, 1 failed, 3 skipped in 0.15s ================\n"
    assert parse_pytest_counts(output) == {"errors": 2, "failed": 1, "skipped": 3}


def test_build_suite_environment_preserves_existing_pythonpath() -> None:
    environment = build_suite_environment("aic_model", {"PYTHONPATH": "existing"})
    assert environment["PYTHONPATH"] == f"aic_model{os.pathsep}existing"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


def test_run_suite_fails_when_pytest_fails(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=1, stdout="1 failed, 301 passed in 1.00s\n")

    monkeypatch.setattr("tools.validation.run_source_validation.subprocess.run", fake_run)
    suite = SuiteConfig("synthetic", "aic_model", ("aic_model/test",))
    result = run_suite(tmp_path, Path("/pixi/python"), suite)

    assert result.status == "failed"
    assert result.counts == {"failed": 1, "passed": 301}
    assert captured["command"] == ["/pixi/python", "-m", "pytest", "-q", "aic_model/test"]
    assert captured["environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert captured["environment"]["PYTHONPATH"].split(os.pathsep)[0] == "aic_model"


def test_run_suite_fails_when_pytest_output_has_no_test_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "tools.validation.run_source_validation.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="no tests ran in 0.01s\n"),
    )
    result = run_suite(tmp_path, Path("/pixi/python"), SuiteConfig("synthetic", "aic_model", ("test",)))
    assert result.status == "failed"


def test_markdown_and_json_evidence_are_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.validation.run_source_validation.source_metadata",
        lambda _root, _python: {
            "git": {"branch": "main", "dirty": False, "dirty_paths": [], "revision": "abc123"},
            "platform": {"machine": "arm64", "platform": "test", "python_executable": "/python", "python_version": "3.x"},
        },
    )
    suite = SuiteResult(
        name="synthetic",
        command=["python", "-m", "pytest"],
        environment={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": "aic_model"},
        status="passed",
        exit_code=0,
        duration_seconds=1.25,
        counts={"passed": 302},
        output="302 passed in 1.25s\n",
    )
    evidence = build_evidence(tmp_path, Path("/python"), [suite])
    evidence["generated_at_utc"] = "2026-08-13T00:00:00+00:00"
    json_path, markdown_path = write_evidence(evidence, tmp_path / "evidence")

    assert json.loads(json_path.read_text())["overall_status"] == "passed"
    report = markdown_path.read_text()
    assert "302 passed" in report
    assert LIMITATION in report
    assert markdown_report(evidence) == report
