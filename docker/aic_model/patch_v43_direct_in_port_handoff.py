#!/usr/bin/env python3
"""Strictly patch v43 so confirmed in-port rescue hands directly to seat-RL."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


EXPECTED_V43_SHA256 = (
    "c27a764b9e64837ddee1748272abadd229cfa888ff05847324bfe819df77db57"
)

OLD = '''                    if in_port:
                        if handoff_mode == "rescue":
                            log.info(f"[script] IN-PORT RESCUE candidate at "
                                     f"{depth_est*1000:.1f}mm -- adaptive search gets "
                                     "first refusal before the shallow actor")
                        else:
                            log.info("[script] IN-PORT but outside nominal/rescue actor gates")
                    else:
                        log.info(f"[script] NOT in port (calibrated lateral "
                                 f"{lateral_est*1000:.2f}mm, depth {depth_est*1000:.1f}mm, "
                                 f"rot {np.degrees(rotation_est):.2f}deg) -- sweeping to enter.")
'''

NEW = '''                    if handoff_mode == "rescue":
                        log.info(f"[script] IN-PORT RESCUE (calibrated lateral "
                                 f"{lateral_est*1000:.2f}mm, depth {depth_est*1000:.1f}mm, "
                                 f"rot {np.degrees(rotation_est):.2f}deg, force {f_mag:.2f}N) "
                                 "-> handing directly to seat RL; no visual/sweep recovery.")
                        log.info("[seat_rl] RESCUE handoff gate passed")
                        seat_status = self._run_seat_rl(
                            get_observation, move_robot, send_feedback,
                            port_pos=port_pos, port_quat=port_quat, Rp=Rp,
                            rotation_reference=R_seat)
                        if seat_status == SEAT_RL_SEATED:
                            return True
                        if seat_status == SEAT_RL_HARD_FAILURE:
                            return False
                        log.warn("[seat_rl] recoverable rescue actor exit -- "
                                 "returning control to scripted squaring/descent")
                        deepest = -np.inf
                        stall_steps = 0
                        stall_depth = float("nan")
                        stall_lat_min, stall_lat_max = np.inf, -np.inf
                        continue
                    if in_port:
                        log.info("[script] IN-PORT but outside nominal/rescue actor gates -- "
                                 "holding; no visual/sweep recovery while inserted")
                        return False
                    log.info(f"[script] NOT in port (calibrated lateral "
                             f"{lateral_est*1000:.2f}mm, depth {depth_est*1000:.1f}mm, "
                             f"rot {np.degrees(rotation_est):.2f}deg) -- sweeping to enter.")
'''


def patch(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_V43_SHA256:
        raise SystemExit(
            f"refusing to patch {path}: expected exact v43 SHA256 "
            f"{EXPECTED_V43_SHA256}, got {digest}"
        )
    text = data.decode("utf-8")
    if text.count(OLD) != 1:
        raise SystemExit(
            f"refusing to patch {path}: expected one in-port rescue block, "
            f"found {text.count(OLD)}"
        )
    text = text.replace(OLD, NEW)
    if "adaptive search gets " in text and "first refusal before the shallow actor" in text:
        raise SystemExit(f"stale rescue-first-search text remains in {path}")
    path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} RLInsert.py [RLInsert.py ...]")
    for argument in sys.argv[1:]:
        path = Path(argument)
        print(f"patched {path}: {patch(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
