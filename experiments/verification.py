"""Evaluator-only seating gate. Never feed these fields to the robot policy.

This checks an instrumented trace, not raw policy logs. A calibrated profile and
contact/fixture recorder are required before using it for publication outcomes.
Depth is positive into the port from its physical mouth; angular error is the
full plug/port frame error (including twist), not merely axis tilt.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SeatingProfile:
    asset_revision: str
    geometry_evidence: str
    seated_depth_m: float
    depth_tolerance_m: float
    lateral_tolerance_m: float
    angular_tolerance_deg: float
    max_penetration_m: float
    dwell_s: float
    max_sample_gap_s: float

    def __post_init__(self):
        if not self.asset_revision or not self.geometry_evidence:
            raise ValueError("Profile needs asset revision and geometry evidence")
        for key, val in self.__dict__.items():
            if isinstance(val, (int, float)) and (not math.isfinite(val) or val <= 0):
                raise ValueError(f"Invalid positive profile limit: {key}")
        if self.depth_tolerance_m >= self.seated_depth_m:
            raise ValueError("Depth tolerance cannot include an uninserted plug")


def verify_seating(samples, profile: SeatingProfile):
    """Require a continuous terminal dwell; fixture-assisted dwell is unverified.

    Schema requires explicit collision/contact observability. Missing fields or
    invalid timestamps never fall back to an event-based success decision.
    This remains a necessary-condition verifier, not a proof of electrical mating.
    """
    def result(ok, reason):
        return {"verified_success": ok, "reason": reason,
                "asset_revision": profile.asset_revision}
    if not samples:
        return result(False, "missing_trace")
    numeric = ("sim_time_s", "depth_m", "lateral_error_m", "angular_error_deg",
               "max_penetration_m")
    boolean = ("contact_consistent", "fixture_active")
    previous = None
    valid_since = None
    reason = "insufficient_dwell"
    for row in samples:
        if any(k not in row or type(row[k]) not in (int, float)
               or not math.isfinite(row[k]) for k in numeric):
            return result(False, "missing_or_nonfinite_geometry")
        if any(type(row.get(k)) is not bool for k in boolean):
            return result(False, "missing_contact_or_fixture_observation")
        t = row["sim_time_s"]
        if previous is not None and t <= previous:
            return result(False, "nonmonotonic_sim_time")
        if any(row[k] < 0 for k in ("lateral_error_m", "angular_error_deg", "max_penetration_m")):
            return result(False, "invalid_unsigned_measurement")
        if previous is not None and t - previous > profile.max_sample_gap_s:
            valid_since = None
        previous = t
        if row["fixture_active"]:
            # Once the simulator latches, later apparent stability is confounded.
            return result(False, "simulator_fixture_active")
        if abs(row["depth_m"] - profile.seated_depth_m) > profile.depth_tolerance_m:
            reason = "depth_outside_seat"
        elif row["lateral_error_m"] > profile.lateral_tolerance_m:
            reason = "lateral_misalignment"
        elif row["angular_error_deg"] > profile.angular_tolerance_deg:
            reason = "angular_misalignment"
        elif row["max_penetration_m"] > profile.max_penetration_m:
            reason = "excess_interpenetration"
        elif not row["contact_consistent"]:
            reason = "contact_inconsistent"
        else:
            if valid_since is None:
                valid_since = t
            reason = "insufficient_dwell"
            continue
        valid_since = None
    if valid_since is not None and samples[-1]["sim_time_s"] - valid_since >= profile.dwell_s - 1e-9:
        return result(True, "stable_geometric_seat_with_contact")
    return result(False, reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path, help="Evaluator-only JSONL")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profile = SeatingProfile(**json.loads(args.profile.read_text()))
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    args.out.write_text(json.dumps(verify_seating(rows, profile), indent=2) + "\n")


if __name__ == "__main__":
    main()
