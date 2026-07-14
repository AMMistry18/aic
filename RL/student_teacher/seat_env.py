"""Force-reactive seat environment starting from a validated lateral wedge."""
from __future__ import annotations

import copy
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
WEDGE_RANDOM_ATTEMPTS_BEFORE_FALLBACK = 8
VALIDATED_START_POOL_SIZE = 2
VALIDATED_START_POOL_MAX_CANDIDATES = 24
UNSAFE_PROBE_STATUSES = frozenset((
    "bad_collision", "force_abort", "off_limit",
))

# Validated pose specifications are deterministic for a stage, compiled contact
# variant, and reset level.  Keep one process-local copy so repeated evaluator
# construction does not rerun the expensive physical nudge sweep.  Subprocess
# training workers build their own pools once, then use only reconstruction plus
# a cheap safety check at episode reset.
_VALIDATED_START_SPEC_CACHE: dict[tuple, dict] = {}

# Measured fallback candidates for the finite-rejection path.  They have each
# passed the same contact, straight-stall, and lateral-nudge validator used for
# randomized starts.  The stage is part of the key because near_seated and the
# deepest full-handoff variant share compiled seed 20260731 but need distinct
# delivered offsets/tilts.
_VALIDATED_FALLBACKS = {
    # candidate seed 3101, measured near-seated wedge (37.67 mm inserted)
    ("near_seated", 20260731): (
        "minus_x", (-0.0007031737131406934, 0.0),
        (0.0, -0.010026035269526843), 6155,
    ),
    # candidate seed 3303, shallow full-handoff wedge
    ("wedge", 20260715): (
        "plus_y", (0.0, 0.0008643056179418527),
        (-0.010790705455643949, 0.0), 1112,
    ),
    # candidate seed 3301, middle full-handoff wedge
    ("wedge", 20260740): (
        "minus_x", (-0.0009651352335727138, 0.0),
        (0.0, 0.01666155638998779), 1111,
    ),
    # candidate seed 3302, deep full-handoff wedge
    ("wedge", 20260731): (
        "minus_x", (-0.0007578445580473826, 0.0),
        (0.0, 0.013444755345908511), 6155,
    ),
}

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
        self._validated_start_pool: list[dict] = []
        self._pool_build_attempts = 0
        self._pool_build_rejections = 0
        self._steps_since_reset = 0
        self._current_reset_metrics: dict = {}

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

    def _validated_fallback_candidate(self) -> tuple[dict, int]:
        """Return a measured pose/dynamics pair for this stage/variant.

        Each pair previously passed the same contact, straight-stall, and
        lateral-nudge validator.  It is still revalidated on use and is reached
        only after eight randomized candidates fail, preventing an unlucky
        finite rejection sequence from crashing training or evaluation.
        """
        fallback_key = (self.stage.name, self._compiled_seed)
        direction, offset, tilt, base_seed = _VALIDATED_FALLBACKS[fallback_key]
        offset_xy = np.asarray(offset, dtype=np.float64)
        tilt_xy = np.asarray(tilt, dtype=np.float64)
        return ({
            "direction": direction,
            "offset_xy_m": offset_xy,
            "commanded_offset_m": float(np.linalg.norm(offset_xy)),
            "tilt_xy_rad": tilt_xy,
            "commanded_tilt_rad": float(np.linalg.norm(tilt_xy)),
        }, base_seed)

    @staticmethod
    def _copy_candidate(candidate: dict) -> dict:
        return {
            "direction": str(candidate["direction"]),
            "offset_xy_m": np.asarray(
                candidate["offset_xy_m"], dtype=np.float64).copy(),
            "commanded_offset_m": float(candidate["commanded_offset_m"]),
            "tilt_xy_rad": np.asarray(
                candidate["tilt_xy_rad"], dtype=np.float64).copy(),
            "commanded_tilt_rad": float(candidate["commanded_tilt_rad"]),
        }

    def _start_safety_reason(self, info: dict) -> str | None:
        """Return why a reconstructed start is unsafe, without stepping it."""
        cfg = self.scene.cfg
        force = float(info.get("contact_force_norm", np.inf))
        lateral = float(info.get("lateral_error_m", np.inf))
        contacts = int(info.get("plug_port_contacts", 0))
        delivered_lo, delivered_hi = self.stage.accepted_lateral_range_m
        if not np.isfinite(force) or force > WEDGE_LOW_FORCE_MAX_N:
            return "force_out_of_range"
        if contacts <= 0 or not (delivered_lo <= lateral <= delivered_hi):
            return "contact_or_offset"
        if int(info.get("off_limit_contacts", 0)) > 0:
            return "off_limit"

        penetration = float(info.get(
            "plug_port_penetration_excess_m", np.inf))
        overinsert = float(info.get("overinsert_m", np.inf))
        depth_norm = float(info.get("depth_norm", np.inf))
        axis_error = abs(float(info.get("plug_axis_error_rad", np.inf)))
        roll_error = abs(float(info.get("plug_roll_error_rad", np.inf)))
        if penetration > cfg.bad_collision_penetration_excess_m:
            return "bad_collision_penetration"
        if overinsert > cfg.bad_collision_overinsert_m:
            return "bad_collision_overinsert"
        if (depth_norm >= cfg.bad_collision_depth_gate
                and (axis_error > cfg.bad_collision_axis_rad
                     or roll_error > cfg.bad_collision_roll_rad)):
            return "bad_collision_crooked_deep"
        return None

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
        unsafe_start = self._start_safety_reason(prepared_info)
        if unsafe_start is not None:
            return {
                "true_lateral_wedge": False,
                "reason": unsafe_start,
            }
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
        if straight.get("end_status") in UNSAFE_PROBE_STATUSES:
            return {
                "true_lateral_wedge": False,
                "reason": "unsafe_straight_probe",
                "contact_count": contacts,
                "force_n": force,
                "lateral_offset_m": lateral,
                "straight_probe": straight,
            }
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
            if probe.get("end_status") in UNSAFE_PROBE_STATUSES:
                continue
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

    def _pool_cache_key(self) -> tuple:
        return (
            self.stage.name,
            int(self._compiled_seed),
            round(float(self._reset_level), 8),
        )

    def _build_validated_start_pool(self) -> list[dict]:
        """Validate a small pose pool once; never run probes in hot resets."""
        pool: list[dict] = []
        rejected = 0
        signatures = set()
        fallback_key = (self.stage.name, self._compiled_seed)
        for attempt in range(VALIDATED_START_POOL_MAX_CANDIDATES):
            use_fallback = bool(
                fallback_key in _VALIDATED_FALLBACKS
                and attempt == WEDGE_RANDOM_ATTEMPTS_BEFORE_FALLBACK
            )
            if use_fallback:
                candidate, base_seed = self._validated_fallback_candidate()
            else:
                candidate = self._candidate()
                base_seed = self._validation_seed_base + 1009 * attempt

            obs69, prepared_info, reset_info, ended = self._prepare_candidate(
                base_seed, candidate)
            if ended:
                rejected += 1
                continue
            validation = self._validate_candidate(
                base_seed, candidate, prepared_info)
            if not validation["true_lateral_wedge"]:
                rejected += 1
                continue

            # Reconstruct once after all probe actions and apply the same cheap
            # check used at runtime.  No validation action may leak into the
            # cached learner start.
            obs69, prepared_info, reset_info, ended = self._prepare_candidate(
                base_seed, candidate)
            safety_reason = (
                "terminated_while_settling" if ended
                else self._start_safety_reason(prepared_info)
            )
            if safety_reason is not None:
                rejected += 1
                continue

            signature = (
                int(base_seed),
                tuple(np.round(candidate["offset_xy_m"], 12)),
                tuple(np.round(candidate["tilt_xy_rad"], 12)),
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            pool.append({
                "candidate": self._copy_candidate(candidate),
                "base_seed": int(base_seed),
                "validation": copy.deepcopy(validation),
                "used_fallback": bool(use_fallback),
            })
            if len(pool) >= VALIDATED_START_POOL_SIZE:
                self._pool_build_attempts = attempt + 1
                break
        else:
            self._pool_build_attempts = VALIDATED_START_POOL_MAX_CANDIDATES

        self._pool_build_rejections = rejected
        if not pool:
            raise RuntimeError(
                f"failed to build any safe validated {self.stage.name} starts "
                f"from {self._pool_build_attempts} candidates")
        return pool

    def _ensure_validated_start_pool(self) -> None:
        if self._validated_start_pool:
            return
        cache_key = self._pool_cache_key()
        cached = _VALIDATED_START_SPEC_CACHE.get(cache_key)
        if cached is None:
            pool = self._build_validated_start_pool()
            cached = {
                "pool": pool,
                "build_attempts": self._pool_build_attempts,
                "build_rejections": self._pool_build_rejections,
            }
            _VALIDATED_START_SPEC_CACHE[cache_key] = copy.deepcopy(cached)
        self._validated_start_pool = copy.deepcopy(cached["pool"])
        self._pool_build_attempts = int(cached["build_attempts"])
        self._pool_build_rejections = int(cached["build_rejections"])

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

        self._ensure_validated_start_pool()
        accepted = None
        rejected = []
        order = self._reset_rng.permutation(len(self._validated_start_pool))
        for attempt, pool_index in enumerate(order):
            item = self._validated_start_pool[int(pool_index)]
            candidate = self._copy_candidate(item["candidate"])
            base_seed = int(item["base_seed"])
            use_fallback = bool(item["used_fallback"])
            obs69, prepared_info, reset_info, ended = self._prepare_candidate(
                base_seed, candidate)
            safety_reason = (
                "terminated_while_settling" if ended
                else self._start_safety_reason(prepared_info)
            )
            if safety_reason is None:
                accepted = (
                    obs69, prepared_info, reset_info, candidate,
                    copy.deepcopy(item["validation"]), attempt, base_seed,
                    use_fallback,
                )
                break
            rejected.append({
                "attempt": attempt + 1,
                "true_lateral_wedge": False,
                "reason": safety_reason,
            })
        if accepted is None:
            raise RuntimeError(
                f"all {len(self._validated_start_pool)} cached "
                f"{self.stage.name} starts failed the runtime safety check: "
                f"{rejected}")

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
            "seat_reset_pool_size": len(self._validated_start_pool),
            "seat_reset_pool_build_attempts": self._pool_build_attempts,
            "seat_reset_pool_build_rejections": self._pool_build_rejections,
            "seat_reset_cache_hit": True,
            "seat_reset_validation_mode": "cached_pose_safety_check",
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
        self._steps_since_reset = 0
        self._current_reset_metrics = {
            "seat_reset_attempts": int(attempt + 1),
            "seat_reset_used_fallback": bool(used_fallback),
            "seat_reset_pool_size": int(len(self._validated_start_pool)),
        }
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
        first_step = self._steps_since_reset == 0
        info["seat_reset_first_step"] = first_step
        if first_step:
            info.update(self._current_reset_metrics)
        self._steps_since_reset += 1
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
