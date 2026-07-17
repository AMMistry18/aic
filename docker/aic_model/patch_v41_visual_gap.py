"""Strictly add wedge-only visual recovery to the baked v41 RLInsert source."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


V41_SHA256 = "dbf9f0bb004f3d4ab541e3002178e4ce300de00c1ef1567ac500334e7e6f16ff"

IMPORT_OLD = "from .seat_actor_adapter import SeatActorHistory\n"
IMPORT_NEW = IMPORT_OLD + "from .visual_gap_recovery import VisualGapRecoveryMixin\n"
CLASS_OLD = "class RLInsert(Policy):\n"
CLASS_NEW = "class RLInsert(VisualGapRecoveryMixin, Policy):\n"
SWEEP_OLD = """                    if self._sweep_into_port(
                            get_observation, move_robot, Rp=Rp,
                            reference_port_pos=port_pos, R_seat=R_seat,
                            insert_axis=insert_axis, start_depth=depth_est):
                        deepest = -np.inf
                        stall_steps = 0
                        stall_depth = float("nan")
                        stall_lat_min, stall_lat_max = np.inf, -np.inf
                        continue
"""
SWEEP_NEW = """                    # v42 changes only this recovery decision: visual centering
                    # gets first refusal at the old low-force adaptive-sweep hook.
                    # Disabled, unavailable, or failed visual recovery falls through
                    # to the byte-identical v41 sweep below.
                    if self._visual_gap_wedge_enabled():
                        try:
                            send_feedback("wedge detected -- visual gap recovery")
                            visual_hole = self._run_visual_gap_wedge_recovery(
                                get_observation, move_robot,
                                raw_port_pos=raw_port_pos, Rp=Rp, R_seat=R_seat,
                                local_port_kps=LOCAL_SFP_PORT_KPS,
                                stiffness=GUIDED_STIFFNESS,
                                damping=GUIDED_DAMPING, step_dt=STEP_DT)
                            if visual_hole is not None:
                                visual_offset = Rp.T @ (visual_hole - port_pos)
                                port_pos = port_pos + (
                                    Rp[:, 0] * visual_offset[0]
                                    + Rp[:, 1] * visual_offset[1])
                                log.info("[visual-gap] replacing adaptive sweep with "
                                         "visual opening target; delta_mm="
                                         f"{np.round(visual_offset[:2]*1000.0, 2).tolist()}")
                                deepest = -np.inf
                                stall_steps = 0
                                stall_depth = float("nan")
                                stall_lat_min, stall_lat_max = np.inf, -np.inf
                                continue
                            log.warn("[visual-gap] recovery unavailable -- running "
                                     "the original adaptive sweep")
                        except Exception:
                            log.warn("[visual-gap] recovery exception -- running the "
                                     "original adaptive sweep:\\n" + traceback.format_exc())
                    if self._sweep_into_port(
                            get_observation, move_robot, Rp=Rp,
                            reference_port_pos=port_pos, R_seat=R_seat,
                            insert_axis=insert_axis, start_depth=depth_est):
                        deepest = -np.inf
                        stall_steps = 0
                        stall_depth = float("nan")
                        stall_lat_min, stall_lat_max = np.inf, -np.inf
                        continue
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source_bytes = path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != V41_SHA256:
        raise RuntimeError(
            f"refusing to patch non-v41 RLInsert at {path}: {digest} != {V41_SHA256}"
        )
    source = source_bytes.decode("utf-8")
    source = replace_once(source, IMPORT_OLD, IMPORT_NEW, "import")
    source = replace_once(source, CLASS_OLD, CLASS_NEW, "class")
    source = replace_once(source, SWEEP_OLD, SWEEP_NEW, "sweep")
    path.write_text(source, encoding="utf-8")
    print(f"patched {path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: patch_v41_visual_gap.py RLInsert.py [RLInsert.py ...]")
    for argument in sys.argv[1:]:
        patch(Path(argument))
