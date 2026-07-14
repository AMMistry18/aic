"""Force-reactive seat environment starting from a validated lateral wedge."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from RL.student_teacher.parity.evaluate_guided_controller import guided_action
from RL.student_teacher.student_env_a import (
    DEPLOY_POS_SCALE,
    make_student_env_a,
)
from RL.student_teacher.teacher_contract import build_teacher_obs21


HISTORY = 8
ACTOR_FRAME_DIM = 34
PRIVILEGED_DIM = 32

# Reverse-curriculum retraction is measured FROM FULLY SEATED.  The hardest
# stage retracts 40.5 mm from the 45.8 mm seated pose, placing the target near
# the 5 mm end of the Phase-1 variable-depth wedge band.
SEAT_RETRACTION_FROM_SEATED_M = 0.0405
SEAT_LAST_INCH_M = 0.090
SEAT_LEVEL = SEAT_RETRACTION_FROM_SEATED_M / SEAT_LAST_INCH_M

WEDGE_NUDGE_M = 0.75e-3
WEDGE_UNSTICK_PROGRESS_M = 2.0e-3
WEDGE_STRAIGHT_PROGRESS_MAX_M = 1.0e-3
WEDGE_LOW_FORCE_MAX_N = 10.0
WEDGE_STALL_WINDOW_S = 1.2
WEDGE_NUDGE_PROBE_STEPS = 40
WEDGE_MAX_RESET_ATTEMPTS = 12
WEDGE_SHALLOW_RANDOM_ATTEMPTS = 8

# The final deploy stage is an explicit mixture of the three clean Phase-1
# contact variants.  These reset levels were measured after settling (the
# commanded level alone is not an accurate proxy for delivered depth): they
# reproduce the script's shallow, middle, and deep hand-offs at roughly
# 5.5/26.8/40.5 mm while retaining straight-stall/lateral-unstick validation.
FULL_HANDOFF_LEVEL_BY_COMPILED_SEED = {
    20260715: 0.44,
    20260740: 0.20,
    20260731: 0.06,
}

# Contact-scale action: saturated deploy commands become at most 0.30 mm x/y,
# 0.70 mm axial, and 0.92/0.92/1.38 deg rotation per policy step.  This retains
# useful chamfer-relief wiggles without exposing the seat policy to gross motion.
SEAT_ACTION_GAIN = 0.20

# Reward weights.  Progress uses physical meters, forces use newtons.
# The 1200 weight still produced 8/8 crooked-deep bad collisions in the first
# squareness re-smoke.  At 800, a saturated 0.7 mm depth step is worth ~0.56,
# well below the -1.47 gate-depth cost of a 0.3 rad axis error.
W_DEPTH = 800.0
W_LAT_PROGRESS = 400.0
K_LAT_HOLD = 20.0
# Match SceneEnvConfig's crooked-at-depth bad-collision gate.  Squareness is
# free at the mouth and becomes important only as the plug approaches the gate.
W_SQUARE_AXIS = 4.0
W_SQUARE_ROLL = 4.0
SQUARE_AXIS_REF_RAD = 0.35
SQUARE_ROLL_REF_RAD = 0.35
SQUARE_DEPTH_ONSET = 0.15
SQUARE_DEPTH_GATE = 0.45
W_FORCE_DIRECTION = 0.50
FORCE_RELIEF_DEPTH_REF_M = 0.5e-3
FORCE_LATERAL_RELIEF_REF_N = 5.0
K_FORCE_LATERAL = 0.03
FORCE_SOFT_START_N = 15.0
FORCE_SOFT_RANGE_N = 5.0
W_FORCE_SOFT = 3.0
K_ACTION = 0.02
K_ACTION_RATE = 0.05
SEATED_SUCCESS_BONUS = 50.0
FAIL_PENALTY = 20.0
FAIL_STATUSES = frozenset(("force_abort", "bad_collision", "off_limit"))


@dataclass(frozen=True)
class SeatStage:
    name: str
    level: float
    level_range: tuple[float, float]
    lateral_offset_range_m: tuple[float, float]
    accepted_lateral_range_m: tuple[float, float]
    tilt_range_rad: tuple[float, float]
    perception_noise: float
    grasp_noise: float
    max_steps: int


_CANONICAL_STAGES = {
    "near_seated": SeatStage(
        "near_seated", 0.08, (0.07, 0.10), (0.65e-3, 0.85e-3),
        (0.25e-3, 0.55e-3),
        (np.radians(0.50), np.radians(0.75)), 0.0, 0.0, 300),
    "mid": SeatStage(
        "mid", 0.15, (0.14, 0.18), (0.90e-3, 1.20e-3),
        (0.25e-3, 0.75e-3),
        (np.radians(0.70), np.radians(1.20)), 0.0, 0.0, 350),
    # Hidden grasp perturbation remains confined to the hardest stage.
    "wedge": SeatStage(
        "wedge", 0.40, (0.05, 0.45), (0.65e-3, 1.00e-3),
        (0.30e-3, 1.00e-3),
        (np.radians(0.50), np.radians(1.00)), 0.0, 0.10, 400),
}

# Trainer compatibility: the deploy ABI and trainer core remain unchanged.
STAGES = {
    **_CANONICAL_STAGES,
    "tight": _CANONICAL_STAGES["near_seated"],
    "band": _CANONICAL_STAGES["mid"],
    "full": _CANONICAL_STAGES["wedge"],
}


def _lat_err(rel: np.ndarray) -> float:
    return float(np.linalg.norm(rel[:2]))


def _rot_err(rel: np.ndarray) -> float:
    return float(np.linalg.norm(rel[3:6]))


class SeatEnv(gym.Wrapper):
    """Stage-A wrapper retaining AlignEnv's 8x34 actor and 32-D critic ABI."""

    def __init__(self, env: gym.Env, stage: SeatStage,
                 history: int = HISTORY, reset_seed: int | None = None):
        super().__init__(env)
        self.contract_env = env
        self.scene = env.unwrapped
        self.history = int(history)
        self.stage = stage
        ridge_depth = self.scene._compiled_variant_diag.get(
            "contact_ridge_depth_m")
        if ridge_depth is None:
            self._reset_level = float(stage.level)
        else:
            self._reset_level = float(np.clip(
                (self.scene.cfg.seated_depth_m - float(ridge_depth))
                / self.scene.cfg.last_inch_m,
                *stage.level_range,
            ))
        compiled_seed = int(self.scene._compiled_variant_diag.get(
            "compiled_variant_seed", 0))
        if stage.name == "wedge":
            # The final policy must not specialize to the last curriculum
            # slice.  Every full-stage worker selects one of the calibrated
            # shallow/middle/deep hand-off anchors, and the vector ensemble
            # therefore retains the complete deployment range.
            self._reset_level = FULL_HANDOFF_LEVEL_BY_COMPILED_SEED[compiled_seed]
        self._validation_seed_base = {
            20260715: 1112,
            20260721: 20264753,
            20260740: 102,
            20260731: 6155,
        }.get(compiled_seed, compiled_seed)
        if self.history != HISTORY:
            raise ValueError(f"seat env requires history={HISTORY}")
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-np.inf, np.inf, (HISTORY, ACTOR_FRAME_DIM), np.float32),
            "privileged": gym.spaces.Box(-np.inf, np.inf, (PRIVILEGED_DIM,), np.float32),
        })
        self._frames: deque[np.ndarray] = deque(maxlen=HISTORY)
        self._wrench_ema = np.zeros(6, dtype=np.float64)
        self._prev_action = np.zeros(6, dtype=np.float64)
        self._last_dt = 0.05
        self._last_info: dict = {}
        self._current_obs69 = np.zeros(69, dtype=np.float32)
        self._prev_f_lateral = 0.0
        self._last_reward_terms: dict[str, float] = {}
        self._reset_rng = np.random.default_rng(reset_seed)
        self._reset_count = 0
        self._reset_resample_count = 0

    def _privileged(self) -> np.ndarray:
        teacher = np.asarray(build_teacher_obs21(self.scene), dtype=np.float64)
        info = self._last_info
        randomization = info.get("domain_randomization", {}) or {}
        extra = np.array([
            float(info.get("contact_force_norm", 0.0)),
            float(info.get("f_z", 0.0)),
            float(info.get("f_lateral", 0.0)),
            float(info.get("insertion_depth_m", self.scene._insertion_depth_m())),
            float(info.get("plug_port_contacts", 0.0)),
            float(randomization.get("friction_scale", 1.0)),
            float(randomization.get("controller_scale", 1.0)),
            float(randomization.get("wrench_limit_scale", 1.0)),
            float(info.get("policy_dt_s", self._last_dt)),
            float(randomization.get("action_delay_steps", 0.0)),
            float(randomization.get("pair_contact_margin_m", 0.0)),
        ], dtype=np.float64)
        out = np.concatenate([teacher, extra])
        if out.shape != (PRIVILEGED_DIM,):
            raise RuntimeError(f"privileged shape drifted: {out.shape}")
        return out.astype(np.float32)

    def _frame(self, obs69: np.ndarray) -> np.ndarray:
        obs69 = np.asarray(obs69, dtype=np.float64).reshape(69)
        rel = obs69[32:38].copy()
        wrench = obs69[51:57].copy()
        self._wrench_ema = 0.8 * self._wrench_ema + 0.2 * wrench
        frame = np.concatenate([
            rel, obs69[19:25], wrench, self._wrench_ema, self._prev_action,
            np.array([_lat_err(rel), _rot_err(rel), rel[2], max(self._last_dt, 1e-4)]),
        ])
        if frame.shape != (ACTOR_FRAME_DIM,):
            raise RuntimeError(f"actor frame shape drifted: {frame.shape}")
        return frame.astype(np.float32)

    def _observation(self, obs69: np.ndarray, *, append: bool = True) -> dict:
        if append:
            self._frames.append(self._frame(obs69))
        return {"actor": np.stack(tuple(self._frames), axis=0).astype(np.float32),
                "privileged": self._privileged()}

    @staticmethod
    def _direction(name: str) -> np.ndarray:
        return {
            "plus_x": np.array([1.0, 0.0]),
            "minus_x": np.array([-1.0, 0.0]),
            "plus_y": np.array([0.0, 1.0]),
            "minus_y": np.array([0.0, -1.0]),
        }[name].copy()

    def _candidate(self) -> dict:
        direction_name = tuple(("plus_x", "minus_x", "plus_y", "minus_y"))[
            int(self._reset_rng.integers(0, 4))]
        direction = self._direction(direction_name)
        magnitude = float(self._reset_rng.uniform(
            *self.stage.lateral_offset_range_m))
        tilt_magnitude = float(self._reset_rng.uniform(*self.stage.tilt_range_rad))
        tilt_sign = -1.0 if self._reset_rng.uniform() < 0.5 else 1.0
        # Tilt about the orthogonal in-plane axis so the displaced side remains
        # loaded against the cage instead of merely yawing around the port axis.
        tilt = np.zeros(2, dtype=np.float64)
        tilt[1 if direction[0] else 0] = tilt_sign * tilt_magnitude
        return {
            "direction": direction_name,
            "offset_xy_m": direction * magnitude,
            "commanded_offset_m": magnitude,
            "tilt_xy_rad": tilt,
            "commanded_tilt_rad": tilt_magnitude,
        }

    @staticmethod
    def _shallow_fallback_candidate() -> dict:
        """Known-good shallow pose, still subject to the full wedge validator.

        This is the measured seed-3303 candidate at the Phase-1 shallow ridge.
        It is used only after eight randomized shallow candidates fail, avoiding
        a training/evaluation crash from an unlucky finite rejection sequence.
        """
        magnitude = 0.0008643056179418527
        tilt = -0.010790705455643949
        return {
            "direction": "plus_y",
            "offset_xy_m": np.array([0.0, magnitude], dtype=np.float64),
            "commanded_offset_m": magnitude,
            "tilt_xy_rad": np.array([tilt, 0.0], dtype=np.float64),
            "commanded_tilt_rad": abs(tilt),
        }

    def _prepare_candidate(self, base_seed: int, candidate: dict):
        _obs69, reset_info = self.env.reset(
            seed=int(base_seed),
            options={"level": self._reset_level, "jitter": False},
        )
        scene = self.scene
        retract = self._reset_level * float(scene.cfg.last_inch_m)
        target_tip = (scene._inserted_tip + retract * scene._retract_dir
                      + scene._lat_x * candidate["offset_xy_m"][0]
                      + scene._lat_y * candidate["offset_xy_m"][1])
        tilt = candidate["tilt_xy_rad"]
        dq = scene._qmul(
            scene._axis_angle(scene._lat_x, float(tilt[0])),
            scene._axis_angle(scene._lat_y, float(tilt[1])),
        )
        quat = scene._qmul(dq, scene._goal_quat)
        tcp = scene._tcp_for_tip(target_tip, quat)
        q_seed = scene._ik(tcp, quat, scene._home)
        q_seed = scene._ik_tcp_position(tcp, q_seed)
        q_arm, ik_tip_err, ik_axis_err, ik_roll_err = scene._ik_tip_axis(
            target_tip, scene._insert_axis, q_seed)
        scene._rigid_home(q_arm)
        for _ in range(int(scene.cfg.settle_steps)):
            scene._apply_episode_wrench()
            scene.data.ctrl[:6] = scene._base_torque(q_arm)
            scene.data.ctrl[6] = scene.cfg.gripper_ctrl
            mujoco.mj_step(scene.model, scene.data)
        diag = scene._geometry_diag(
            target_tip=target_tip,
            ik_tip_err=ik_tip_err,
            ik_axis_err=ik_axis_err,
            ik_roll_err=ik_roll_err,
        )
        scene._last_reset_diag = diag
        scene._reset_arm_target = q_arm.copy()
        scene._arm_target = q_arm.copy()
        scene._cart_init_from_state()
        raw = scene._obs()
        obs69 = self.env._build_obs69(raw)
        info = {
            **diag,
            "f_z": abs(float(diag["f_axial"])),
            "policy_dt_s": float(scene._policy_dt_s),
            "term_status": None,
            "domain_randomization": dict(scene._randomization_diag),
        }
        invalid = bool(
            not np.isfinite(diag["tip_error_m"])
            or diag["contact_force_norm"] > scene.cfg.reset_contact_abort_n)
        return obs69, info, dict(reset_info), invalid

    def _straight_probe(self, base_seed: int, candidate: dict) -> dict:
        obs69, info, _reset_info, ended = self._prepare_candidate(
            base_seed, candidate)
        start_depth = float(self.scene._insertion_depth_m())
        max_depth = start_depth
        action = np.zeros(6, dtype=np.float32)
        action[2] = float(0.5e-3 / DEPLOY_POS_SCALE[2])
        steps = max(1, int(np.ceil(
            WEDGE_STALL_WINDOW_S / max(self.scene._policy_dt_s, 1e-6))))
        for _ in range(steps):
            if ended:
                break
            obs69, _reward, terminated, truncated, info = self.env.step(action)
            max_depth = max(max_depth, float(info["insertion_depth_m"]))
            ended = bool(terminated or truncated)
        return {
            "depth_progress_m": max_depth - start_depth,
            "end_status": info.get("term_status"),
        }

    def _nudge_probe(self, base_seed: int, candidate: dict,
                     direction_name: str) -> dict:
        obs69, info, _reset_info, ended = self._prepare_candidate(
            base_seed, candidate)
        start_depth = float(self.scene._insertion_depth_m())
        max_depth = start_depth
        direction = self._direction(direction_name)
        action = guided_action(obs69).astype(np.float64)
        action[:2] = direction * (
            WEDGE_NUDGE_M / np.asarray(DEPLOY_POS_SCALE[:2], dtype=np.float64))
        if not ended:
            obs69, _reward, terminated, truncated, info = self.env.step(
                np.clip(action, -1.0, 1.0).astype(np.float32))
            max_depth = max(max_depth, float(info["insertion_depth_m"]))
            ended = bool(terminated or truncated)
        for _ in range(WEDGE_NUDGE_PROBE_STEPS):
            if ended:
                break
            obs69, _reward, terminated, truncated, info = self.env.step(
                guided_action(obs69))
            max_depth = max(max_depth, float(info["insertion_depth_m"]))
            ended = bool(terminated or truncated)
        progress = max_depth - start_depth
        return {
            "direction": direction_name,
            "commanded_nudge_m": WEDGE_NUDGE_M,
            "depth_progress_m": progress,
            "end_status": info.get("term_status"),
            "unstick_success": bool(
                progress >= WEDGE_UNSTICK_PROGRESS_M
                or info.get("term_status") == "success"),
        }

    def _validate_candidate(self, base_seed: int, candidate: dict,
                            prepared_info: dict) -> dict:
        lateral = float(prepared_info.get("lateral_error_m", np.inf))
        force = float(prepared_info.get("contact_force_norm", np.inf))
        contacts = int(prepared_info.get("plug_port_contacts", 0))
        delivered_lo, delivered_hi = self.stage.accepted_lateral_range_m
        contact_wedge = bool(
            contacts > 0
            and np.isfinite(force) and force <= WEDGE_LOW_FORCE_MAX_N
            and delivered_lo <= lateral <= delivered_hi
        )
        if not contact_wedge:
            return {
                "true_lateral_wedge": False,
                "reason": "contact_or_offset",
                "contact_count": contacts,
                "force_n": force,
                "lateral_offset_m": lateral,
            }

        straight = self._straight_probe(base_seed, candidate)
        if straight["depth_progress_m"] > WEDGE_STRAIGHT_PROGRESS_MAX_M:
            return {
                "true_lateral_wedge": False,
                "reason": "not_stalled_axially",
                "contact_count": contacts,
                "force_n": force,
                "lateral_offset_m": lateral,
                "straight_probe": straight,
            }

        probes = []
        for direction_name in ("plus_x", "minus_x", "plus_y", "minus_y"):
            probe = self._nudge_probe(base_seed, candidate, direction_name)
            probes.append(probe)
            if probe["unstick_success"]:
                return {
                    "true_lateral_wedge": True,
                    "reason": "lateral_nudge_unstuck",
                    "contact_count": contacts,
                    "force_n": force,
                    "lateral_offset_m": lateral,
                    "straight_probe": straight,
                    "accepted_probe": probe,
                    "probes": probes,
                }
        return {
            "true_lateral_wedge": False,
            "reason": "flat_or_dead_stall",
            "contact_count": contacts,
            "force_n": force,
            "lateral_offset_m": lateral,
            "straight_probe": straight,
            "probes": probes,
        }

    def _normalize_accepted_reset(self, obs69: np.ndarray) -> np.ndarray:
        self.scene._step_count = 0
        self.scene._last_action = np.zeros(self.scene._action_dim, np.float32)
        self.scene._prev_depth_norm = self.scene._depth_norm()
        self.scene._f_ax_buf = []
        self.scene._off_limit_event_fired = False
        self.scene._force_sustain_event_fired = False
        self.scene._force_over_count = 0
        self.scene._action_queue = []
        self.scene._wrench_obs_queue = []
        self.scene._reset_score_state()
        self.env._last_deploy_action = np.zeros(6, dtype=np.float32)
        raw = self.scene._obs()
        return self.env._build_obs69(raw)

    def reset(self, **kwargs):
        requested_seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if kwargs:
            raise TypeError(f"unsupported reset arguments: {sorted(kwargs)}")
        if requested_seed is not None:
            self._reset_rng = np.random.default_rng(int(requested_seed))

        accepted = None
        rejected = []
        for attempt in range(WEDGE_MAX_RESET_ATTEMPTS):
            use_fallback = bool(
                self.stage.name == "wedge"
                and self._compiled_seed == 20260715
                and attempt == WEDGE_SHALLOW_RANDOM_ATTEMPTS
            )
            candidate = (
                self._shallow_fallback_candidate()
                if use_fallback else self._candidate()
            )
            # Each stage uses a Phase-1-proven runtime variant as its first
            # candidate; later attempts still sweep independently randomized
            # dynamics while preserving the same immutable ridge geometry.
            base_seed = (
                self._validation_seed_base
                if use_fallback
                else self._validation_seed_base + 1009 * attempt
            )
            obs69, prepared_info, reset_info, ended = self._prepare_candidate(
                base_seed, candidate)
            validation = (
                {"true_lateral_wedge": False, "reason": "terminated_while_settling"}
                if ended else self._validate_candidate(
                    base_seed, candidate, prepared_info))
            if validation["true_lateral_wedge"]:
                # Reconstruct the accepted candidate once more: none of the
                # straight/nudge validation actions leak into the learner start.
                obs69, prepared_info, reset_info, ended = self._prepare_candidate(
                    base_seed, candidate)
                if not ended:
                    accepted = (obs69, prepared_info, reset_info, candidate,
                                validation, attempt, base_seed, use_fallback)
                    break
            rejected.append({"attempt": attempt + 1, **validation})
        if accepted is None:
            raise RuntimeError(
                f"failed to construct a solvable {self.stage.name} lateral wedge "
                f"after {WEDGE_MAX_RESET_ATTEMPTS} attempts: {rejected}")

        (obs69, prepared_info, reset_info, candidate, validation, attempt,
         base_seed, used_fallback) = accepted
        obs69 = self._normalize_accepted_reset(obs69)
        self._reset_count += 1
        self._reset_resample_count += attempt
        info = dict(reset_info)
        info.update(prepared_info)
        info["reset_diag"] = dict(prepared_info)
        info.update({
            "term_status": None,
            "curriculum_level": self._reset_level,
            "seat_reset_nominal_level": self.stage.level,
            "seat_reset_stage": self.stage.name,
            "seat_reset_seed": base_seed,
            "seat_reset_attempts": attempt + 1,
            "seat_reset_resample_count": attempt,
            "seat_reset_used_fallback": used_fallback,
            "seat_reset_rejected": rejected,
            "seat_reset_commanded_offset_m": candidate["commanded_offset_m"],
            "seat_reset_commanded_tilt_rad": candidate["commanded_tilt_rad"],
            "seat_reset_direction": candidate["direction"],
            "seat_reset_true_lateral_wedge": True,
            "seat_reset_probe": validation,
            "seat_reset_cumulative_resample_rate": (
                self._reset_resample_count
                / max(self._reset_resample_count + self._reset_count, 1)),
        })
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        self._last_info = dict(info)
        self._last_dt = float(getattr(self.scene, "_policy_dt_s", 0.05))
        self._wrench_ema[:] = 0.0
        self._prev_action[:] = 0.0
        self._prev_f_lateral = 0.0
        self._last_reward_terms = {}
        first = self._frame(obs69)
        self._frames = deque((first.copy() for _ in range(HISTORY)), maxlen=HISTORY)
        return self._observation(obs69, append=False), info

    def step(self, action):
        before_rel = np.asarray(self._current_obs69[32:38], dtype=np.float64).copy()
        commanded = (SEAT_ACTION_GAIN * np.clip(
            np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0))
        obs69, _base_reward, terminated, truncated, info = self.env.step(commanded.astype(np.float32))
        self._current_obs69 = np.asarray(obs69, dtype=np.float32).copy()
        self._last_info = dict(info)
        self._last_dt = float(info.get("policy_dt_s", self._last_dt))
        rel = np.asarray(obs69[32:38], dtype=np.float64)
        reward = self._seat_reward(before_rel, rel, commanded, info)
        self._prev_action = commanded.copy()
        info = dict(info)
        info.update({
            "seat_lat_err_m": _lat_err(rel),
            "seat_rot_err_rad": _rot_err(rel),
            "seat_commanded_action": commanded.astype(np.float32),
            "seat_reward_terms": dict(self._last_reward_terms),
        })
        return self._observation(obs69), reward, terminated, truncated, info

    @staticmethod
    def _finite(value, default: float = 0.0) -> float:
        value = float(value)
        return value if np.isfinite(value) else float(default)

    def _seat_reward(self, before_rel, rel, commanded, info) -> float:
        """FORGE-style contact reward: make progress while keeping force gentle.

        Force direction is rewarded only while moving inward.  Its score is the
        axial share of contact force plus a bonus when lateral force drops from
        the previous step.  A separate lateral-force magnitude cost means that
        depth-up + side-load-down is strictly better than shoving with the same
        depth gain.  Absolute axial force is never rewarded without progress.

        Lateral pose shaping is explicit: reducing ``norm(rel[:2])`` is positive
        and remaining error is a small cost.  We intentionally do not reuse
        ``breakdown.xy``; that term is already non-positive in ``RL.reward`` and
        subtracting it would reward larger lateral error.

        Squareness shaping uses the same axis, roll, and depth scales as the
        environment's crooked-at-depth safety gate.  A quadratic depth ramp is
        zero near the mouth, then makes normalized axis/roll error increasingly
        costly as depth approaches the gate.  It shapes behavior only; the base
        environment remains responsible for termination.
        """
        before = np.asarray(before_rel, dtype=np.float64).reshape(6)
        after = np.asarray(rel, dtype=np.float64).reshape(6)
        act = np.asarray(commanded, dtype=np.float64).reshape(6)

        depth_progress = self._finite(after[2] - before[2])
        depth_term = W_DEPTH * depth_progress

        lat_before = _lat_err(before)
        lat_now = _lat_err(after)
        lateral_term = (W_LAT_PROGRESS * (lat_before - lat_now)
                        - K_LAT_HOLD * lat_now)

        depth_norm = max(0.0, self._finite(info.get("depth_norm", 0.0)))
        depth_ramp = float(np.clip(
            (depth_norm - SQUARE_DEPTH_ONSET)
            / (SQUARE_DEPTH_GATE - SQUARE_DEPTH_ONSET), 0.0, 1.0)) ** 2
        axis_error = abs(self._finite(info.get("plug_axis_error_rad", 0.0)))
        roll_error = abs(self._finite(info.get("plug_roll_error_rad", 0.0)))
        axis_ratio = float(np.clip(axis_error / SQUARE_AXIS_REF_RAD, 0.0, 1.0))
        roll_ratio = float(np.clip(roll_error / SQUARE_ROLL_REF_RAD, 0.0, 1.0))
        squareness_term = -depth_ramp * (
            W_SQUARE_AXIS * axis_ratio * axis_ratio
            + W_SQUARE_ROLL * roll_ratio * roll_ratio)

        f_z = abs(self._finite(info.get("f_z", 0.0)))
        f_lateral = max(0.0, self._finite(info.get("f_lateral", 0.0)))
        f_norm = max(0.0, self._finite(info.get("contact_force_norm", 0.0)))
        progress_gate = float(np.clip(
            max(depth_progress, 0.0) / FORCE_RELIEF_DEPTH_REF_M, 0.0, 1.0))
        force_sum = f_z + f_lateral
        axial_share = f_z / force_sum if force_sum > 1e-9 else 0.0
        lateral_drop = max(0.0, self._prev_f_lateral - f_lateral)
        relief = float(np.clip(
            lateral_drop / FORCE_LATERAL_RELIEF_REF_N, 0.0, 1.0))
        force_direction_term = (
            W_FORCE_DIRECTION * progress_gate * (axial_share + relief))
        lateral_force_term = -K_FORCE_LATERAL * f_lateral

        force_excess = max(0.0, f_norm - FORCE_SOFT_START_N)
        force_ratio = force_excess / FORCE_SOFT_RANGE_N
        force_soft_term = -W_FORCE_SOFT * force_ratio * force_ratio

        effort_term = -K_ACTION * float(np.linalg.norm(act))
        action_rate_term = -K_ACTION_RATE * float(np.linalg.norm(
            act - self._prev_action))

        term_status = info.get("term_status")
        success_term = SEATED_SUCCESS_BONUS if term_status == "success" else 0.0
        fail_term = -FAIL_PENALTY if term_status in FAIL_STATUSES else 0.0

        terms = {
            "depth": depth_term,
            "lateral": lateral_term,
            "squareness": squareness_term,
            "force_direction": force_direction_term,
            "lateral_force": lateral_force_term,
            "force_soft_cap": force_soft_term,
            "effort": effort_term,
            "action_rate": action_rate_term,
            "success": success_term,
            "failure": fail_term,
        }
        reward = float(sum(terms.values()))
        self._prev_f_lateral = f_lateral
        self._last_reward_terms = {name: float(value) for name, value in terms.items()}
        if not np.isfinite(reward):
            raise FloatingPointError(f"non-finite seat reward: {self._last_reward_terms}")
        return reward


def _compiled_seed_for_stage(seed: int, stage: SeatStage) -> int:
    """Select only immutable variants Phase 1 proved are lateral wedges."""
    proven = {
        "near_seated": (20260731,),       # 38.93 mm clean wedge
        "mid": (20260740,),               # 30.77 mm clean wedge
        # Hard workers span the clean Phase-1 wedge distribution rather than
        # collapsing back to a single mouth-depth geometry.
        "wedge": (20260715, 20260740, 20260731),
    }[stage.name]
    return int(proven[int(seed) % len(proven)])


def make_seat_env(stage: str = "full", *, seed: int | None = None,
                  domain_randomization: bool = True) -> SeatEnv:
    """Build a seat-only environment at a validated wedged hand-off pose."""
    if stage not in STAGES:
        raise ValueError(f"unknown seat stage: {stage!r} (have {list(STAGES)})")
    s = STAGES[stage]
    requested_seed = int(seed if seed is not None else 0)
    compiled_seed = _compiled_seed_for_stage(requested_seed, s)
    base = make_student_env_a(
        perception_noise=s.perception_noise,
        grasp_noise=s.grasp_noise,
        level=s.level,
        start_jitter_xy_m=0.0,
        start_jitter_yaw_rad=0.0,
        start_jitter_tilt_rad=0.0,
        start_curriculum_band=0.0,
        start_curriculum_easy_frac=0.0,
        action_convention="deploy",
        wrench_mode="baseline",
        # A compiled Phase-1 ridge is part of the seat task even when episode
        # randomization is disabled for deterministic acceptance tests.
        domain_randomization=True,
        max_episode_steps=s.max_steps,
        seed=compiled_seed,
    )
    if not domain_randomization:
        base.scene.cfg.domain_randomization = False
    env = SeatEnv(base, s, reset_seed=requested_seed)
    env._requested_seed = requested_seed
    env._compiled_seed = compiled_seed
    return env


__all__ = [
    "ACTOR_FRAME_DIM", "HISTORY", "PRIVILEGED_DIM", "SEAT_LEVEL",
    "SEAT_ACTION_GAIN", "SEAT_RETRACTION_FROM_SEATED_M", "STAGES",
    "SeatEnv", "make_seat_env",
]
