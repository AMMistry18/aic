# Wiring the SC plug-pose estimator into `sc_controller.py` — proposed patch

**Status: NOT APPLIED.** `aic_model/aic_model/sc_controller.py` is being edited
in parallel on the seating path, so this change is written up rather than
committed. Apply it when that work lands. Line numbers are from the working
copy at the time of writing and will have moved; anchor on the symbol names.

Everything this patch depends on is already in the tree and tested:

| Piece | Where |
|---|---|
| SC keypoint geometry | `aic_model/aic_model/sc_plug_pose_geometry.py` |
| Fail-closed estimator | `aic_model/aic_model/sc_plug_pose.py` (`ScPlugPoseEstimator`) |
| Tests | `aic_model/test/test_sc_plug_pose.py` (15 pass) |
| Trainer | `train_sc_plug_pose.py` |
| Validation | `validate_sc_plug_pose.py` |

## 1. What is wrong today

`SC_TIP_IN_TCP_POS` / `_QUAT` default to the **SFP** grasp transform:

```python
SC_TIP_IN_TCP_POS  = _env_vector("RL_INSERT_SC_TIP_IN_TCP_POS",  SFP_TIP_IN_TCP_POS)
SC_TIP_IN_TCP_QUAT = _env_vector("RL_INSERT_SC_TIP_IN_TCP_QUAT", SFP_TIP_IN_TCP_QUAT)
```

so `sc_tip_pose_from_tcp()` returns the tip of the *wrong connector*. That is
the source of the +7 mm phantom depth reading that blocks seating, and a
hardcoded grasp constant is disallowed by the competition rules regardless of
whether it is accurate. The module docstring already flags this as
"UNCALIBRATED item 1".

Re-solving the constant with `RL_INSERT_CALIB_DUMP=1` is **not** the fix. It
would still be a fixed-grasp assumption, it would still be hardcoded, and
`dump_sc_grasp_calibration()` itself reports that the simulator publishes no
frame tracking the grasped plug, so there is nothing to solve it against.

## 2. The seam

`ScSeatAction._tip_pose()` is the single place the seating loop asks "where is
the plug tip?". Everything downstream — depth, alignment, seat detection —
flows from it. Replace its body and the rest of the loop inherits the
measurement.

```python
    # ------------------------------------------------------------- geometry
    def _tip_pose(self):
        tcp_pos, tcp_quat = self.policy._tcp()
        return sc_tip_pose_from_tcp(tcp_pos, tcp_quat)
```

## 3. Patch

### 3a. Build the estimator once, during configure

Mirrors `configure_v50()` in `v50_controller.py`. Add near the other SC
configuration:

```python
def configure_sc_plug_pose(policy) -> None:
    """Load the SC plug-pose model.  No fixed-grasp fallback is permitted."""

    if getattr(policy, "_sc_plug_estimator", None) is not None:
        return
    from .sc_plug_pose import ScPlugPoseEstimator, default_sc_plug_pose_weights

    weights = default_sc_plug_pose_weights()
    if weights is None:
        raise FileNotFoundError(
            "SC plug-pose weights missing (set AIC_SC_PLUG_POSE_WEIGHTS); "
            "no fixed-grasp fallback is allowed"
        )
    policy._sc_plug_estimator = ScPlugPoseEstimator(
        str(weights),
        imgsz=_env_int("RL_INSERT_SC_PLUG_IMGSZ", 960),
        conf_threshold=_env_float("RL_INSERT_SC_PLUG_CONF", 0.25),
    )
    # Pay the first-inference cost now, not inside the seating loop.
    from .sfp_plug_pose import PlugPoseView

    policy._sc_plug_estimator.detect_views([
        PlugPoseView(
            camera_name="sc_warmup",
            image_bgr=np.zeros((640, 640, 3), dtype=np.uint8),
            K=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 320.0], [0.0, 0.0, 1.0]]),
            T_world_from_camera=np.eye(4),
            stamp_s=0.0,
            frame_id="sc-warmup",
        )
    ])
```

`imgsz=960` is not optional. Ultralytics defaults to 640 at inference; the
cameras deliver 1152x1024 and the model trains at 960, and leaving the default
in place is leak #1 in `docs/SC_PERCEPTION_ACCURACY_PLAYBOOK.md`.

### 3b. Replace `_tip_pose()`

`_plug_views_from_observation()` in `v50_controller.py` is already generic — it
builds `PlugPoseView`s from `policy._build_views(observation)` and has nothing
SFP-specific in it. Reuse it rather than writing a second copy.

```python
    # ------------------------------------------------------------- geometry
    def _tip_pose(self, observation=None):
        """Measured ``sc_tip_link`` pose, or ``None`` if it cannot be measured.

        Returns ``(tip_pos, R_tip)`` exactly as the old fixed-grasp helper did,
        so call sites keep their shape -- but this one can answer "I do not
        know", and callers MUST stop rather than substitute a constant.
        """
        from .sc_plug_pose import ScPlugPoseEstimator  # noqa: F401
        from .sfp_plug_pose import stamp_to_seconds
        from .v50_controller import _plug_views_from_observation

        if observation is None:
            return None
        views = _plug_views_from_observation(self.policy, observation)
        if len(views) < 2:
            return None
        now_s = stamp_to_seconds(self.policy._parent_node.get_clock().now())
        return self.policy._sc_plug_estimator.estimate_tip_pose(
            views,
            now_s=now_s,
            max_age_s=_env_float("RL_INSERT_SC_PLUG_MAX_AGE_S", 0.35),
        )
```

### 3c. Delete the constant

Once no caller reads them, remove `SC_TIP_IN_TCP_POS`, `SC_TIP_IN_TCP_QUAT`,
`sc_tip_pose_from_tcp` and `tcp_pose_for_sc_tip`'s dependence on them, and drop
"UNCALIBRATED item 1" from the module docstring. Leaving a working
fixed-grasp path in the tree is how it gets used again by accident.

`tcp_pose_for_sc_tip()` still needs the inverse transform to convert a desired
*tip* pose into a TCP command. Derive it per-run from the measurement instead:
with a measured `T_world_from_tip` and the concurrent `T_world_from_tcp`,
`T_tcp_from_tip = inv(T_world_from_tcp) @ T_world_from_tip`. Cache it per grasp
(this is what `_v50_grasp_transform` does for SFP) and refresh it whenever a
fresh pose arrives.

## 4. The contract change is the risky part

`_tip_pose()` currently **always** returns a pose. After the patch it can
return `None`, and the four call sites differ in what that must mean:

| Call site | Today | After |
|---|---|---|
| `ScSeatAction._tip_pose()` (~L874) | fixed grasp | measured; `None` ⇒ **abort the seating attempt** |
| `dump_sc_grasp_calibration()` (~L758) | logs the constant | diagnostic only — keep, but also log the measured pose so the two can be compared |
| SC opening candidate tie-break (~L1610) | picks candidate nearest the assumed tip | already inside `try/except` falling back to `clean[0]`; on `None` keep that ordering fallback — it degrades ranking, it does not fabricate a pose |
| Handoff diagnostics (~L1741) | `dist` / `handoff_delta` / `handoff_rot` | `None` ⇒ do not enter the seating loop |

The one thing that must not happen anywhere is falling back to
`sc_tip_pose_from_tcp` when the estimate is missing. That would reintroduce
both the phantom depth and the rules violation, and it would do so exactly in
the frames where vision was least reliable.

## 5. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `AIC_SC_PLUG_POSE_WEIGHTS` | in-repo `weights/best_sc_plug_pose.pt` | trained checkpoint |
| `RL_INSERT_SC_PLUG_IMGSZ` | `960` | inference size; must match training |
| `RL_INSERT_SC_PLUG_CONF` | `0.25` | detector confidence floor |
| `RL_INSERT_SC_PLUG_MAX_AGE_S` | `0.35` | stale-frame guard, ROS clock domain |

## 6. Verification before trusting it

1. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .pixi/envs/default/bin/python -m pytest \
   aic_model/test/test_sc_plug_pose.py aic_model/test/test_sc_controller.py \
   aic_model/test/test_v50_controller.py -q`
2. `python validate_sc_plug_pose.py --mode dataset --weights <ckpt> --split test`
   and read `position_error_mm.median` against the 0.4 mm working target and
   `lateral_error_mm.p95` against the 0.725 mm binding clearance.
3. Run one SC trial with the patch and confirm the reported depth no longer
   carries the +7 mm offset, and that a deliberately blinded camera (cover two
   of three) makes the controller stop rather than seat.
4. Re-diff the dual copies: `diff aic_model/aic_model/sc_controller.py
   docker/aic_model/v50_overlay/aic_model/sc_controller.py`.
