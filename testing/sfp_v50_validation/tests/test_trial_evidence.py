from __future__ import annotations

import pytest

from trial_evidence import TrialEvidenceRecorder, canonical_insertion_event


def test_event_truth_uses_scorer_namespace_and_monotonic_time() -> None:
    recorder = TrialEvidenceRecorder("trial", "nic_card_mount_2/sfp_port_1")
    recorder.start(1_000_000_000)
    recorder.observe_insertion_event("/nic_card_mount_2/sfp_port_1/touched", 45_000_000_000)
    result = recorder.finish("succeeded", 46_000_000_000)
    assert canonical_insertion_event("//module/port/extra") == "module/port"
    assert result["clock"] == "time.monotonic_ns"
    assert result["force_clock"] == "wrench_header_stamp"
    assert result["correct_event_elapsed_s"] == 44.0
    assert result["local_gate_pass"]


def test_wrong_port_event_fails_even_if_correct_event_arrives() -> None:
    recorder = TrialEvidenceRecorder("trial", "module/port_0")
    recorder.start(0)
    recorder.observe_insertion_event("module/port_1", 1_000_000_000)
    recorder.observe_insertion_event("module/port_0", 2_000_000_000)
    result = recorder.finish("succeeded", 3_000_000_000)
    assert result["correct_insertion_event"]
    assert result["wrong_port_event"]
    assert not result["local_gate_pass"]


@pytest.mark.parametrize("elapsed,passes", [(45.0, True), (45.000001, False)])
def test_wall_deadline_is_inclusive(elapsed: float, passes: bool) -> None:
    recorder = TrialEvidenceRecorder("trial", "module/port")
    recorder.start(1_000_000_000)
    recorder.observe_insertion_event("module/port", int((elapsed + 1.0) * 1e9))
    result = recorder.finish("succeeded", int((elapsed + 2.0) * 1e9))
    assert result["correct_event_within_limit"] is passes


def test_force_penalty_matches_scorer_and_offlimit_is_fatal() -> None:
    recorder = TrialEvidenceRecorder("trial", "module/port")
    recorder.start(0)
    recorder.observe_force(0.0, 0.0, 0.0, 100_000_000)
    recorder.observe_force(21.0, 0.0, 0.0, 700_000_000)
    recorder.observe_force(21.0, 0.0, 0.0, 1_300_000_001)
    recorder.observe_offlimit(1_400_000_000)
    recorder.observe_insertion_event("module/port", 2_000_000_000)
    result = recorder.finish("succeeded", 3_000_000_000)
    assert result["force_above_threshold_s"] == pytest.approx(1.200000001)
    assert result["force_penalty"]
    assert result["offlimit_count"] == 1
    assert not result["local_gate_pass"]


def test_prestart_messages_are_not_trial_evidence() -> None:
    recorder = TrialEvidenceRecorder("trial", "module/port")
    recorder.observe_insertion_event("wrong/port", 1)
    recorder.observe_force(100.0, 0.0, 0.0, 1)
    recorder.observe_offlimit(1)
    recorder.start(2)
    recorder.observe_insertion_event("module/port", 3)
    result = recorder.finish("succeeded", 4)
    assert result["prestart_event_count"] == 1
    assert not result["wrong_port_event"]
    assert result["offlimit_count"] == 0
    assert result["force_sample_count"] == 0
