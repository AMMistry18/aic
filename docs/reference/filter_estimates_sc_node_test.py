"""Does the SC filter survive a rotated / tilted board?

Builds synthetic IVM detections for the five SC ports at a known board pose,
then runs both the original fixed-world-axis filter and the board-frame fix.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(
    0,
    r"C:/Users/anshu/AppData/Local/Temp/claude/c--Users-anshu-College-aic/"
    r"343bcc0d-7ca5-451e-93b3-195b12814928/scratchpad",
)
import filter_estimates_sc_fixed as fixed  # noqa: E402


# --- minimal stand-ins for the Flowstate protos -----------------------------
@dataclass
class _V:
    x: float
    y: float
    z: float


@dataclass
class _Q:
    x: float
    y: float
    z: float
    w: float


@dataclass
class _Pose:
    position: _V
    orientation: _Q


@dataclass
class _Est:
    root_t_target: _Pose
    score: float


def quat_from_matrix(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return _Q(x, y, z, w)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


# Board-frame port entrances: 3 on Y +0.0295, 2 on Y +0.0705, Z 0.0301.
# Index order along +X, matching the node's PORT_LABELS_BY_RAIL assumption.
PORTS_BOARD = np.array([
    [-0.120, 0.0295, 0.0301],   # sc_port_0
    [-0.075, 0.0295, 0.0301],   # sc_port_1
    [-0.030, 0.0295, 0.0301],   # sc_port_2
    [-0.100, 0.0705, 0.0301],   # sc_port_3
    [-0.045, 0.0705, 0.0301],   # sc_port_4
])


def make_detections(yaw_deg, tilt_deg=0.0, noise_m=0.0, seed=0):
    rng = np.random.default_rng(seed)
    R = Rx(math.radians(tilt_deg)) @ Rz(math.radians(yaw_deg))
    origin = np.array([-0.3445, 0.2602, 0.0])
    ests = []
    for p in PORTS_BOARD:
        xyz = R @ p + origin + rng.normal(0, noise_m, 3)
        ests.append(_Est(_Pose(_V(*xyz), quat_from_matrix(R)), 0.9))
    return ests, R


# --- the ORIGINAL node's geometry, extracted verbatim -----------------------
ALONG_ORIG = np.array([1.0, 0.0, 0.0])
BETWEEN_ORIG = np.array([0.0, 1.0, 0.0])


def original_layout_ok(ests):
    """Reproduces the original's gates: fixed world axes, 3/2 split by +Y."""
    pos = [np.array([e.root_t_target.position.x, e.root_t_target.position.y,
                     e.root_t_target.position.z]) for e in ests]
    between = np.array([p @ BETWEEN_ORIG for p in pos])
    order = np.argsort(between)
    r0, r1 = between[order[:3]], between[order[3:]]
    spread0, spread1 = float(np.ptp(r0)), float(np.ptp(r1))
    sep = float(r1.mean() - r0.mean())
    if spread0 > 0.012 or spread1 > 0.012:
        return False, f"within-rail spread {spread0*1000:.0f}/{spread1*1000:.0f}mm > 12mm"
    if abs(sep - 0.041) > 0.015:
        return False, f"rail separation {sep*1000:.0f}mm outside 41+/-15mm"
    return True, f"ok (sep {sep*1000:.0f}mm, spreads {spread0*1000:.0f}/{spread1*1000:.0f}mm)"


def fixed_labels_ok(ests, R):
    """Run the fix and check every port gets its correct index."""
    try:
        _best, labeled, _layout, along, _between, _cand, _fo = fixed.select_target(
            ests, "sc_port_0"
        )
    except RuntimeError as exc:
        return False, str(exc)[:70]
    # labeled is [(label, estimate)]; recover which synthetic port each is.
    got = {}
    for label, est in labeled:
        p = np.array([est.root_t_target.position.x, est.root_t_target.position.y,
                      est.root_t_target.position.z])
        idx = int(np.argmin([np.linalg.norm(p - (R @ q + np.array([-0.3445, 0.2602, 0.0])))
                             for q in PORTS_BOARD]))
        got[label] = idx
    correct = all(got.get(i) == i for i in range(5))
    return correct, f"label->port {dict(sorted(got.items()))}"


if __name__ == "__main__":
    print(f"{'board yaw':>10} {'tilt':>5} | {'ORIGINAL':<46} | FIXED")
    print("-" * 110)
    for tilt in (0.0, 8.0):
        for yaw in (0.0, 5.0, 10.0, 20.0, 45.0, 90.0, 140.0, 180.0, 250.0, 315.0):
            ests, R = make_detections(yaw, tilt, noise_m=0.0015, seed=1)
            ok_o, why_o = original_layout_ok(ests)
            ok_f, why_f = fixed_labels_ok(ests, R)
            print(
                f"{yaw:9.0f} {tilt:5.0f} | {'PASS' if ok_o else 'FAIL':4} {why_o:<41} | "
                f"{'PASS' if ok_f else 'FAIL':4} {why_f}"
            )
