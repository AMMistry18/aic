"""Adversarial synthetic traces test the verifier, not robotic performance."""
from dataclasses import replace
import pytest

from experiments.verification import SeatingProfile, verify_seating


@pytest.fixture
def profile():
    return SeatingProfile("synthetic-test-geometry", "unit-test-only",
                          .015, .0005, .0004, 2., .0001, .5, .06)


def trace(**changes):
    return [dict(sim_time_s=i / 20, depth_m=.015, lateral_error_m=.0001,
                 angular_error_deg=.5, max_penetration_m=0.,
                 contact_consistent=True, fixture_active=False, **changes)
            for i in range(15)]


def altered(key, value):
    return [{**r, key: value} for r in trace()]


def test_stable_contact_passes(profile):
    assert verify_seating(trace(), profile)["verified_success"]


@pytest.mark.parametrize("key,value", [
    ("depth_m", 0.), ("depth_m", .008), ("depth_m", .020),
    ("lateral_error_m", .002), ("angular_error_deg", 10.),
    ("max_penetration_m", .001), ("contact_consistent", False),
    ("fixture_active", True), ("depth_m", float("nan")),
    ("lateral_error_m", -.001), ("fixture_active", "false"),
])
def test_invalid_seats_never_pass(profile, key, value):
    assert not verify_seating(altered(key, value), profile)["verified_success"]


def test_latch_cannot_supply_dwell(profile):
    rows = trace()
    rows[4]["fixture_active"] = True
    assert not verify_seating(rows, profile)["verified_success"]


def test_terminal_unseat_fails(profile):
    rows = trace()
    rows[-1]["depth_m"] = 0.
    assert not verify_seating(rows, profile)["verified_success"]


def test_gaps_and_time_reset_fail(profile):
    rows = trace()
    rows[7:9] = []
    assert not verify_seating(rows, profile)["verified_success"]
    assert not verify_seating(trace() + trace(), profile)["verified_success"]


def test_missing_contact_fails_closed(profile):
    rows = trace()
    del rows[-1]["contact_consistent"]
    assert not verify_seating(rows, profile)["verified_success"]


def test_empty_or_short_trace_fails(profile):
    assert not verify_seating([], profile)["verified_success"]
    assert not verify_seating(trace()[:3], profile)["verified_success"]


def test_profile_cannot_include_zero_depth(profile):
    with pytest.raises(ValueError):
        replace(profile, depth_tolerance_m=.02)
