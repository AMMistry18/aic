# Training Change Summary

This note tracks the current AIC Phase 1 RL training changes so future runs are
easy to reason about.

## Curriculum

- The reverse curriculum still starts at seated insertion for level `0.0`.
- The high end now starts from a real free-space approach instead of a nearly
  inserted pose.
- `SceneEnvConfig.last_inch_m` is now `0.090 m`.
- With `seated_depth_m ~= 0.0458 m`, level `0.8` places the plug tip about
  `0.026 m` outside the port entrance, roughly one inch of clearance.
- Level `1.0` starts farther back, giving room for pose-estimation error and
  lateral/yaw/tilt distortion before acquisition.
- `approach_gap_m` is now logged in episode metrics to make this visible.

## Reward

- Image reward is default-off: the policy can still observe camera images, but
  the reward renderer is skipped unless `--image-reward-weight > 0`.
- The reward is geometry-first:
  - positive reward for insertion-depth progress,
  - sparse success bonus,
  - stronger penalties for collision, force aborts, off-limit events, timeout,
    lateral error, and unsafe force.
- The intent is to reward seating/partial insertion while making "cable flies
  away" and hard shoving clearly bad.

## Video and Evaluation

- W&B videos are step-scheduled rather than episode-count scheduled.
- Periodic videos can be current-level only via `--wandb-video-episodes 1`.
- Curriculum transition videos are recorded before advancing to the next level
  under `eval/curriculum_advance_video`.

## Current-Run Caveat

Python training processes load the code once at startup. Any already-running run
will not pick up these code changes until it is stopped and a new or resumed run
is started.
