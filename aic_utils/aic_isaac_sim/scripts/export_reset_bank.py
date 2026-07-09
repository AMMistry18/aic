#!/usr/bin/env python3
"""Export MuJoCo reverse-curriculum reset states for the Isaac task."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from RL.scene_env import SceneEnvConfig, SceneInsertEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aic_utils/aic_isaac_sim/assets/reset_bank.npz"),
    )
    parser.add_argument("--levels", type=int, default=21)
    parser.add_argument("--samples-per-level", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    if args.levels < 2 or args.samples_per_level < 1:
        parser.error("--levels must be >= 2 and --samples-per-level must be >= 1")

    cfg = SceneEnvConfig(include_images=False)
    env = SceneInsertEnv(cfg)
    env.reset(seed=args.seed, options={"level": 0.0, "jitter": False})

    qpos: list[np.ndarray] = []
    levels: list[float] = []
    cable_root_pose: list[np.ndarray] = []
    diagnostics: list[np.ndarray] = []
    try:
        for level in np.linspace(0.0, 1.0, args.levels):
            for _ in range(args.samples_per_level):
                env._reset_to_level(float(level), jitter=True)
                qpos.append(env.data.qpos[env._arm_qadr].copy())
                cable_root_pose.append(
                    env.data.qpos[env._cfree_adr : env._cfree_adr + 7].copy()
                )
                diag = env._last_reset_diag
                diagnostics.append(
                    np.asarray(
                        [
                            diag["tip_error_m"],
                            diag["plug_axis_error_rad"],
                            diag["plug_roll_error_rad"],
                            diag["lateral_error_m"],
                            diag["contact_force_norm"],
                            diag["plug_port_penetration_m"],
                        ],
                        dtype=np.float64,
                    )
                )
                levels.append(float(level))
    finally:
        env.close()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene_path = Path(cfg.scene_path).resolve()
    source_files = [scene_path.parent / "aic_robot.xml", scene_path.parent / "aic_world.xml", scene_path]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
    np.savez_compressed(
        output,
        schema_version=np.asarray(1, dtype=np.int64),
        qpos=np.asarray(qpos, dtype=np.float32),
        level=np.asarray(levels, dtype=np.float32),
        cable_root_pose=np.asarray(cable_root_pose, dtype=np.float32),
        diagnostics=np.asarray(diagnostics, dtype=np.float32),
        diagnostic_names=np.asarray(
            [
                "tip_error_m",
                "plug_axis_error_rad",
                "plug_roll_error_rad",
                "lateral_error_m",
                "contact_force_norm",
                "plug_port_penetration_m",
            ]
        ),
        scene_path=np.asarray(str(scene_path)),
        scene_sha256=np.asarray(hashlib.sha256(scene_path.read_bytes()).hexdigest()),
        source_sha256=np.asarray(source_hash),
        seed=np.asarray(args.seed, dtype=np.int64),
    )
    print(f"wrote {len(qpos)} reset states to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
