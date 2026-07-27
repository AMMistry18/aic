# Deterministic Stage 1 policy for Check Board Visibility v4

Updated: 2026-07-27

This policy replaces the runtime use of `AdaptiveViewpointPlanner`. Stage 2 is
unchanged.

## Runtime behavior

```text
fresh triplet
  -> complete unobstructed Stage-2 insignia?
       yes: unchanged geometric Stage 2
       no:  validate fixed observation path
            -> execute guarded joint segments
            -> fresh triplet
            -> unchanged geometric Stage 2 or normal done=false
```

The observation posture is:

```text
degrees:
[-28.177, -70.126, -72.796, -109.782, 93.185, 19.934]
```

It is not sent blindly. The skill first reads one coherent six-joint state,
autocalibrates the existing UR5e model from the timestamped TCP, derives the
three real camera extrinsics, and validates the entire interpolation.

Stage-1 limits:

- worst physical joint travel: 185 degrees;
- total physical joint travel: 250 degrees;
- direct-joint speed: at most 0.20 rad/s;
- TCP height throughout the path: at least 0.245 m;
- base-origin TCP reach: at most 1.20 m;
- endpoint wrist/forearm self-clearance: at least the existing 0.140 m gate.

An invocation that safely reaches the observation posture but still lacks a
complete Stage-2 landmark returns:

```text
success=true
done=false
target_valid=false
last_action=deterministic_observation_exhausted
```

A force, controller, feedback, or settling failure returns `success=false`.
The current joint segment reverses to its measured start before returning.

## Backtest

Run:

```powershell
cd C:\Users\anshu\College\aic\aic\flowstate\aic_perception
python test/acquisition_sweep_runner.py --workers 8
```

The required result is:

```text
deterministic Stage 1: 144/144 acquired, 144/144 safe
```

The matrix is 8 board yaws x 2 tilts x 3 placements x 3 live starts. The
scoring condition is a fully framed `INSIGNIA_RECT_CORNERS` projection in at
least one production-calibrated camera after applying the shipped gripper
masks.

## Flowstate test

Keep the existing sequence and v4 skill node:

```text
Move Robot to pre-survey/start
-> Switch To AIC Controller
-> Check Board Visibility v4
-> Switch To Default Controller
-> require success && done && target_valid
-> Move Robot to result.target
```

For the first hardware run, capture:

- initial and post-observation joint vectors;
- `last_action`, `moves_executed`, `angular_travel_rad`;
- force norm and `force_abort`;
- all three initial and post-observation camera frames;
- the Stage-2 pose-search diagnostics.

Do not send a `done=false` result to Move Robot.
