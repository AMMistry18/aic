"""Scripted PRIVILEGED teacher for the SFP insertion scene (Phase 1 / Fix A).

FULL last-inch controller: pure ``action_mode="cartesian_residual"`` with the
scripted base motion OFF (``base_script_enabled=False``). The teacher drives the
WHOLE last-inch motion via the Cartesian residual -- the same action space the
distilled student will use -- just scripted and privileged.

It does NOT touch the impedance controller (cart_kp_pos / cart_kd_* /
cart_max_wrench are the real-robot controller, left exactly as-is). It only shapes
the residual within the residual envelope (cart_pos_limit_m / cart_rot_limit_rad),
sized by the eval (0.20 m axial) so the residual can reach the seated goal and
lead far past it (the UNCHANGED impedance can then ramp seating force to the cap).

Key dynamics: the plug hangs off the tool on a compliant CABLE (free joint), so
the tip is NOT a rigid TCP offset -- it swings like a PENDULUM when the arm moves,
and a rigid TCP feed-forward mis-places it. So the control law is entirely on the
*measured* plug tip (which includes the cable swing):
  * LATERAL: PD on the measured tip offset from the port axis. The D term damps
    the cable pendulum -- a pure P resonates it (drives the tip to >20 mm).
  * ANGULAR: gentle P nulling the plug axis + keyed roll toward goal (reset roll
    <3 deg is already inside the 8.6 deg success tol; a stiff gain winds up and
    swings the arm under the soft controller, so it is kept low).
  * AXIAL: FORCE-REGULATED advance. The residual (envelope 0.20 m) grows to build
    seating force up to force_cap, then holds -- this works the plug past the
    chamfer wedge and seats it, while capping force so the big residual never rams
    (overinserts). The forward push is GATED down when the tip is off-axis (never
    ram the port wall) and BACKED OFF when port-wall penetration or plug tilt rise
    toward the bad_collision thresholds. A small yaw "screw" breaks static
    friction while stuck. Success (depth_norm>=0.99) ends the episode ~0.5 mm
    before the bottom, so no overinsert builds.
"""
from __future__ import annotations

import numpy as np


class ScriptedTeacher:
    def __init__(
        self,
        *,
        action_dim: int = 6,
        kp_lat: float = 0.25,          # lateral P (on measured plug-tip offset)
        kd_lat: float = 2.5,           # lateral D: damps the plug-on-cable PENDULUM
                                       # (a pure P on the pendulum tip resonates it)
        kp_rot: float = 0.3,           # angular P (tilt + keyed roll toward goal)
        axis_soft_rad: float = 4.4e-2, # plug-axis tilt (~2.5 deg) that triggers a backoff
        # ---- axial force regulation ----
        force_cap: float = 5.0,        # target/cap axial contact force (N); < 10 N wrench cap
        kf: float = 0.4,               # force-error -> axial residual velocity gain
        lat_full_m: float = 1.0e-3,    # full axial push when tip within this of the axis
        lat_gate_m: float = 5.0e-3,    # push tapers to a trickle by this lateral offset
        pen_soft_m: float = 6.0e-4,    # port-wall penetration excess that triggers a backoff
                                       # (< bad_collision threshold 1.5 mm)
        v_seat: float = 0.6,           # max forward axial residual velocity (action units)
        v_retract: float = 0.6,        # max backoff axial residual velocity
        depth_eps: float = 1.5e-3,     # normalized-depth gain counted as progress
        # ---- yaw "screw" friction aid while stuck ----
        seat_yaw_screw: float = 0.5,
        seat_screw_w: float = 0.35,
        seat_screw_stall: int = 8,
        success_depth: float = 0.99,
    ) -> None:
        self.action_dim = int(action_dim)
        self.kp_lat = float(kp_lat)
        self.kd_lat = float(kd_lat)
        self.kp_rot = float(kp_rot)
        self.axis_soft_rad = float(axis_soft_rad)
        self.force_cap = float(force_cap)
        self.kf = float(kf)
        self.lat_full_m = float(lat_full_m)
        self.lat_gate_m = float(lat_gate_m)
        self.pen_soft_m = float(pen_soft_m)
        self.v_seat = float(v_seat)
        self.v_retract = float(v_retract)
        self.depth_eps = float(depth_eps)
        self.seat_yaw_screw = float(seat_yaw_screw)
        self.seat_screw_w = float(seat_screw_w)
        self.seat_screw_stall = int(seat_screw_stall)
        self.success_depth = float(success_depth)
        self.reset()

    def reset(self) -> None:
        self._t = 0
        self._best_depth = -np.inf
        self._stall = 0
        self._prev_lat = np.zeros(2)

    # ------------------------------------------------------------------ #
    def act(self, env, obs=None) -> np.ndarray:
        cfg = env.cfg
        trans_scale = float(cfg.cart_trans_scale_m)
        rot_scale = float(cfg.cart_rot_scale_rad)
        insert_axis = np.asarray(env._insert_axis, dtype=np.float64)
        R = np.asarray(env._cart_frame_R, dtype=np.float64)   # [lat_x, lat_y, insert_axis]

        # ---- GT (measured plug tip; includes cable swing) ----
        tip = env.data.xpos[env._plug_tip_id].copy()
        _axial, lat_vec = env._tip_port_errors(tip)
        lat_vec = np.asarray(lat_vec, dtype=np.float64)      # (2,) m offset from port axis
        lat_err = float(np.linalg.norm(lat_vec))
        roll_err = float(env._plug_roll_error())
        axis_err = float(env._plug_axis_error())
        ins = float(env._insertion_depth_m())                # tip depth past entrance (m)
        depth = float(env._depth_norm())

        # ---- progress / stall ----
        if depth > self._best_depth + self.depth_eps:
            self._best_depth = depth
            self._stall = 0
        else:
            self._stall += 1

        # ---- lateral PD (P on measured tip offset + D damps the cable pendulum) ----
        lat_vel = lat_vec - self._prev_lat
        self._prev_lat = lat_vec.copy()
        a_xy = -(self.kp_lat * lat_vec + self.kd_lat * lat_vel) / trans_scale

        # ---- angular P (gentle: plug axis + keyed roll toward goal) ----
        plug_axis = np.asarray(env._plug_axis(), dtype=np.float64)
        rv_tilt = np.asarray(
            env._quat_to_rotvec(env._quat_between(plug_axis, insert_axis)),
            dtype=np.float64)
        roll_axis = np.asarray(env._plug_roll_axis(), dtype=np.float64)
        target_roll = np.asarray(env._target_roll_axis(), dtype=np.float64)
        theta_roll = float(np.arctan2(
            float(np.dot(np.cross(roll_axis, target_roll), insert_axis)),
            float(np.dot(roll_axis, target_roll))))
        rv_world = rv_tilt + insert_axis * theta_roll
        a_rot = self.kp_rot * (R.T @ rv_world) / rot_scale

        # ---- axial: FORCE-REGULATED advance ----
        # Advance the axial residual to build seating force up to force_cap, then
        # hold there (no more residual growth). This gives the sustained force
        # that works the plug past the chamfer wedge and seats it, WITHOUT the
        # 200 mm-residual ram that overinserts/collides. The residual (envelope
        # 0.20 m) is free to grow while the tip is stuck, but the regulator pulls
        # it back the moment contact force exceeds the cap.
        fz = abs(float(getattr(env, "_f_axial", 0.0)))   # axial contact force (prev step)
        pen = float(env._contact_summary().get("plug_port_penetration_excess_m", 0.0))
        if depth < self.success_depth and (pen > self.pen_soft_m
                                           or axis_err > self.axis_soft_rad):
            # about to gouge the port wall (off-axis jam) OR the plug is pivoting
            # (tilt) -> back off axially to unload the wall and let the lateral +
            # angular loops re-centre before re-pushing (keeps pen/tilt below the
            # bad_collision thresholds).
            a_z = -self.v_retract
        else:
            a_z = self.kf * (self.force_cap - fz)
            a_z = float(np.clip(a_z, -self.v_retract, self.v_seat))
            # gate forward push on lateral alignment: never ram the plug into the
            # port while swung off-axis. Full push when centred, taper when off.
            if a_z > 0.0:
                gate = float(np.clip(
                    (self.lat_gate_m - lat_err) / max(self.lat_gate_m - self.lat_full_m, 1e-6),
                    0.15, 1.0))
                a_z *= gate
        # yaw "screw" to break static friction while stuck (entrance wedge or seat)
        if self._stall >= self.seat_screw_stall and self.action_dim == 6:
            a_rot[2] += self.seat_yaw_screw * np.sin(self._t * self.seat_screw_w)

        # ------------------------------------------------------------------ #
        action = np.zeros(self.action_dim, dtype=np.float32)
        action[0], action[1], action[2] = a_xy[0], a_xy[1], a_z
        action[3], action[4] = a_rot[0], a_rot[1]
        if self.action_dim == 6:
            action[5] = a_rot[2]
        self._t += 1
        return np.clip(action, -1.0, 1.0)
