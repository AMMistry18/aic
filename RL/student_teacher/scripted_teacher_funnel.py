"""Scripted PRIVILEGED teacher -- FUNNEL entry variant (Phase 1 / Fix A, option a).

Same deployable-student setup as v1 (RL/teacher/scripted_teacher.py): pure
``action_mode="cartesian_residual"`` with ``base_script_enabled=False``; the
teacher drives the whole last-inch motion via the residual. Impedance controller
(cart_kp_pos=100, cart_kd_*, cart_max_wrench=10 N) is UNTOUCHED.

Difference vs v1 -- ENTRY philosophy only (v1's force-regulated SEAT is kept):
The plug is underactuated (swings on the cable); v1's active lateral nulling still
excites the pendulum near the chamfer -> off-axis entry -> wall penetration/tilt
collisions (v1's dominant failure). This variant instead lets the chamfer geometry
+ the soft impedance passively centre the plug (remote-centre-compliance style):
  1. SETTLE  -- while outside the port, damp the tip's lateral VELOCITY (strong D,
     modest centring P) and, near the entrance lip, ease the axial creep so the
     pendulum goes quiescent BEFORE committing to enter (don't snap it).
  2. CENTRE  -- the settle P gently places the (now quiescent) tip over the mouth.
  3. FUNNEL-IN -- across the chamfer, drop the lateral POSITION gain to near zero
     (keep damping) so we stop fighting the chamfer; let the funnel guide the plug
     in while the axial force-regulator pushes gently.
  4. SEAT    -- v1's force-regulated advance, unchanged; plus v1's penetration/tilt
     force-backoff so a bad engage bails out (retract, re-settle, re-approach).
"""
from __future__ import annotations

import numpy as np


class ScriptedTeacher:
    def __init__(
        self,
        *,
        action_dim: int = 6,
        # ---- lateral: settle (far) vs funnel (at chamfer) ----
        kp_center: float = 0.30,       # centring P while outside (settle the tip over the mouth)
        kp_funnel: float = 0.15,       # low (not zero) lateral P across the chamfer (let it guide)
        kd_lat: float = 3.0,           # strong D everywhere: damp the cable pendulum
        funnel_zone_m: float = 5.0e-3, # within this of the entrance -> funnel (low-gain) region
        # ---- SETTLE gating (go quiescent before committing to enter) ----
        settle_band_m: float = 6.0e-3, # just outside the mouth: ease creep until settled
        settle_vel_tol: float = 4.0e-4,# lateral speed (m/step) counted as "quiescent"
        center_tol_m: float = 1.8e-3,  # tip centred enough for the chamfer to capture
        settle_creep: float = 0.04,    # slow axial creep while settling (no hard stop)
        # ---- angular P (gentle: reset roll<3deg << 8.6deg tol) ----
        kp_rot: float = 0.3,
        axis_soft_rad: float = 4.4e-2, # plug-axis tilt (~2.5 deg) -> axial backoff
        # ---- axial force regulation (v1 SEAT, unchanged) ----
        force_cap: float = 5.0,
        kf: float = 0.4,
        lat_full_m: float = 1.0e-3,
        lat_gate_m: float = 5.0e-3,
        pen_soft_m: float = 6.0e-4,    # port-wall penetration excess -> backoff (<1.5mm gate)
        v_seat: float = 0.6,
        v_retract: float = 0.6,
        depth_eps: float = 1.5e-3,
        # ---- yaw screw friction aid while stuck ----
        seat_yaw_screw: float = 0.5,
        seat_screw_w: float = 0.35,
        seat_screw_stall: int = 8,
        success_depth: float = 0.99,
    ) -> None:
        self.action_dim = int(action_dim)
        self.kp_center = float(kp_center)
        self.kp_funnel = float(kp_funnel)
        self.kd_lat = float(kd_lat)
        self.funnel_zone_m = float(funnel_zone_m)
        self.settle_band_m = float(settle_band_m)
        self.settle_vel_tol = float(settle_vel_tol)
        self.center_tol_m = float(center_tol_m)
        self.settle_creep = float(settle_creep)
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
        R = np.asarray(env._cart_frame_R, dtype=np.float64)

        # ---- GT (measured plug tip; includes cable swing) ----
        tip = env.data.xpos[env._plug_tip_id].copy()
        _axial, lat_vec = env._tip_port_errors(tip)
        lat_vec = np.asarray(lat_vec, dtype=np.float64)
        lat_err = float(np.linalg.norm(lat_vec))
        axis_err = float(env._plug_axis_error())
        ins = float(env._insertion_depth_m())
        depth = float(env._depth_norm())

        lat_vel = lat_vec - self._prev_lat
        self._prev_lat = lat_vec.copy()
        lat_speed = float(np.linalg.norm(lat_vel))

        if depth > self._best_depth + self.depth_eps:
            self._best_depth = depth
            self._stall = 0
        else:
            self._stall += 1

        # ===== lateral: SETTLE (far) or FUNNEL (at chamfer) =====
        # Across the chamfer, drop the position gain so we stop fighting the funnel;
        # keep strong velocity damping everywhere so the pendulum never resonates.
        in_funnel = ins > -self.funnel_zone_m
        kpl = self.kp_funnel if in_funnel else self.kp_center
        a_xy = -(kpl * lat_vec + self.kd_lat * lat_vel) / trans_scale

        # ===== angular P (gentle) =====
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

        # ===== axial: force-regulated (v1 SEAT) + SETTLE gate + backoffs =====
        fz = abs(float(getattr(env, "_f_axial", 0.0)))
        pen = float(env._contact_summary().get("plug_port_penetration_excess_m", 0.0))
        if depth < self.success_depth and (pen > self.pen_soft_m
                                           or axis_err > self.axis_soft_rad):
            a_z = -self.v_retract               # bail out of a bad engage
        else:
            a_z = self.kf * (self.force_cap - fz)
            a_z = float(np.clip(a_z, -self.v_retract, self.v_seat))
            # SETTLE: at the entrance lip but not yet quiescent+centred -> ease to a
            # slow creep (let the damping settle the pendulum) instead of committing.
            at_lip = (-self.settle_band_m < ins < 0.0)
            ready = (lat_speed < self.settle_vel_tol and lat_err < self.center_tol_m)
            if at_lip and not ready:
                a_z = min(a_z, self.settle_creep)
            # gate forward push on lateral alignment (never ram the wall off-axis)
            if a_z > 0.0:
                gate = float(np.clip(
                    (self.lat_gate_m - lat_err) / max(self.lat_gate_m - self.lat_full_m, 1e-6),
                    0.15, 1.0))
                a_z *= gate
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
