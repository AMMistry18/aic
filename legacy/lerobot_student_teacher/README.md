# Legacy LeRobot student-teacher tooling

This folder preserves the earlier LeRobot imitation-learning path for AIC
policies. It is not part of the deployed board-search, pose-store, or cable
insertion runtime, but remains available for teleoperation, demonstration
recording, and future imitation-learning experiments.

- `lerobot_robot_aic/` provides the AIC robot interface for LeRobot teleoperation, demo recording, and policy training.
- `lerobot-record` collects teacher demonstrations.
- `lerobot-train` trains the student policy, such as ACT, from the recorded dataset.

See `lerobot_robot_aic/README.md` for runnable commands.
