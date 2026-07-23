"""Authoritative per-trial evidence captured on a monotonic wall clock."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable


Clock = Callable[[], int]


def canonical_insertion_event(value: str) -> str | None:
    """Match the scorer's first-two-nonempty-token namespace semantics."""

    tokens = [token for token in value.strip().split("/") if token]
    if len(tokens) < 2:
        return None
    return "/".join(tokens[:2])


@dataclass
class TrialEvidenceRecorder:
    trial_id: str
    expected_insertion_event: str
    max_event_wall_seconds: float = 45.0
    force_threshold_n: float = 20.0
    force_penalty_duration_seconds: float = 1.0
    clock: Clock = time.monotonic_ns
    frozen_gate_sha256: str | None = None
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    runtime_image_ids: dict[str, str] = field(default_factory=dict)
    config_sha256: str | None = None

    start_ns: int | None = None
    terminal_ns: int | None = None
    terminal_status: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    prestart_event_count: int = 0
    offlimit_count: int = 0
    force_sample_count: int = 0
    max_force_n: float = 0.0
    force_above_threshold_ns: int = 0
    _previous_force_ns: int | None = None

    def __post_init__(self) -> None:
        canonical = canonical_insertion_event(self.expected_insertion_event)
        if canonical is None:
            raise ValueError(
                f"Invalid expected insertion event {self.expected_insertion_event!r}"
            )
        self.expected_insertion_event = canonical
        if self.max_event_wall_seconds <= 0.0:
            raise ValueError("max_event_wall_seconds must be positive")

    def start(self, timestamp_ns: int | None = None) -> None:
        if self.start_ns is not None:
            raise RuntimeError("Trial evidence recorder already started")
        self.start_ns = self.clock() if timestamp_ns is None else timestamp_ns

    def observe_insertion_event(
        self, value: str, timestamp_ns: int | None = None
    ) -> None:
        now_ns = self.clock() if timestamp_ns is None else timestamp_ns
        if self.start_ns is None:
            self.prestart_event_count += 1
            return
        canonical = canonical_insertion_event(value)
        self.events.append(
            {
                "raw": value,
                "canonical": canonical,
                "elapsed_s": (now_ns - self.start_ns) / 1e9,
            }
        )

    def observe_offlimit(self, timestamp_ns: int | None = None) -> None:
        # Timestamp is accepted so callers use the same receipt-time API as the
        # other evidence sources.  A single authoritative contact is a penalty.
        _ = self.clock() if timestamp_ns is None else timestamp_ns
        if self.start_ns is not None:
            self.offlimit_count += 1

    def observe_force(
        self,
        force_x: float,
        force_y: float,
        force_z: float,
        timestamp_ns: int | None = None,
    ) -> None:
        now_ns = self.clock() if timestamp_ns is None else timestamp_ns
        if self.start_ns is None:
            return
        magnitude = math.sqrt(force_x * force_x + force_y * force_y + force_z * force_z)
        self.force_sample_count += 1
        self.max_force_n = max(self.max_force_n, magnitude)
        # Mirror ScoringTier2: when the current sample is above threshold, add
        # dt from the immediately preceding wrench sample.
        if magnitude > self.force_threshold_n and self._previous_force_ns is not None:
            self.force_above_threshold_ns += now_ns - self._previous_force_ns
        self._previous_force_ns = now_ns

    def finish(
        self, terminal_status: str, timestamp_ns: int | None = None
    ) -> dict[str, Any]:
        if self.start_ns is None:
            raise RuntimeError("Cannot finish evidence before the trial starts")
        if self.terminal_ns is not None:
            raise RuntimeError("Trial evidence recorder already finished")
        self.terminal_ns = self.clock() if timestamp_ns is None else timestamp_ns
        self.terminal_status = terminal_status
        return self.to_result()

    def to_result(self) -> dict[str, Any]:
        if self.start_ns is None:
            raise RuntimeError("Trial evidence has not started")
        correct_events = [
            event
            for event in self.events
            if event["canonical"] == self.expected_insertion_event
        ]
        wrong_events = [
            event
            for event in self.events
            if event["canonical"] is not None
            and event["canonical"] != self.expected_insertion_event
        ]
        first_correct_elapsed = (
            min(event["elapsed_s"] for event in correct_events)
            if correct_events
            else None
        )
        force_above_s = self.force_above_threshold_ns / 1e9
        force_penalty = force_above_s > self.force_penalty_duration_seconds
        terminal_elapsed = (
            (self.terminal_ns - self.start_ns) / 1e9
            if self.terminal_ns is not None
            else None
        )
        correct_in_time = (
            first_correct_elapsed is not None
            and first_correct_elapsed <= self.max_event_wall_seconds
        )
        gate_pass = (
            correct_in_time
            and not wrong_events
            and self.offlimit_count == 0
            and not force_penalty
        )
        return {
            "schema_version": 1,
            "trial_id": self.trial_id,
            "clock": "time.monotonic_ns",
            "expected_insertion_event": self.expected_insertion_event,
            "observed_insertion_events": self.events,
            "correct_insertion_event": bool(correct_events),
            "correct_event_elapsed_s": first_correct_elapsed,
            "correct_event_within_limit": correct_in_time,
            "max_event_wall_seconds": self.max_event_wall_seconds,
            "wrong_port_event": bool(wrong_events),
            "wrong_port_events": wrong_events,
            "prestart_event_count": self.prestart_event_count,
            "offlimit_count": self.offlimit_count,
            "force_sample_count": self.force_sample_count,
            "force_clock": "wrench_header_stamp",
            "force_tare": "aic_controller/controller_state.fts_tare_offset",
            "max_force_n": self.max_force_n,
            "force_above_threshold_s": force_above_s,
            "force_threshold_n": self.force_threshold_n,
            "force_penalty_duration_seconds": self.force_penalty_duration_seconds,
            "force_penalty": force_penalty,
            "action_terminal_status": self.terminal_status,
            "terminal_wall_elapsed_s": terminal_elapsed,
            "local_gate_pass": gate_pass,
            "frozen_gate_sha256": self.frozen_gate_sha256,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "runtime_image_ids": dict(sorted(self.runtime_image_ids.items())),
            "config_sha256": self.config_sha256,
        }
