# MuJoCo contact-physics calibration — 2026-07-12

## Final decision

The guided-only contact sentinel now matches the required Flowstate regime well
enough to use as the physics baseline. Do not treat this as an RL result: no
policy, reward, replay, teacher, snapshot, or Pixi change was made for this
calibration.

Final active parameters in `RL/scene_env.py`:

```text
ridge solref:                    (0.006, 1.0)      # positive, stable format
ridge friction:                  5.0
random contact timeconst range:  (0.006, 0.040) s  # above 2*dt edge
ridge solimp:                    (0.99, 0.999, 0.0001, 0.5, 2.0) unchanged
```

The previous ridge direct-format `solref=(-3.0e5, -3.0e3)` is not active.

## Guided-only sentinel results

All three loops used the same controller, seed, level, and 10 nominal + 40
randomized episodes. Existing snapshots were left intact; each test ran from a
separate disposable TACC source copy.

| loop | one parameter changed | QACC | nominal success | random success | peak force | peak lateral | max one-step lateral | result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | ridge solref: explosive direct format -> `(0.006, 1.0)` | 0 | 10/10 | 13/40 | 60.9 N | 26.2 mm | 16.0 mm | stable, borderline force maximum |
| 2 | ridge friction: `5.0 -> 2.0` | 0 | 10/10 | 12/40 | 39.4 N | 44.3 mm | 44.2 mm | rejected: lateral slip/jump exceeds bound |
| 3 | contact-timeconst floor: `0.004 -> 0.006` s | 0 | 10/10 | 16/40 | 31.6 N | 26.2 mm | 9.1 mm | selected |

The selected run is TACC job `3300263`:

```text
report: /scratch/11590/satya_a/aic/gate0_ridge_timeconst006_20260712/gate0_ridge_fix.json
summary: /scratch/11590/satya_a/aic/gate0_ridge_timeconst006_20260712/physics_summary.json
```

Selected-run distributions:

```text
nominal:    force mean/p95/max 4.51 / 4.53 / 4.53 N
            lateral mean/p95/max 0.79 / 0.79 / 0.79 mm
randomized: force mean/p95/max 11.27 / 18.87 / 31.61 N
            lateral mean/p95/max 16.91 / 24.29 / 26.23 mm
```

There were three realistic guided jams: all were sustained stalls at about
5.7--6.5 mm depth, with 8.15--8.45 N peak force and 0.44--0.69 mm lateral
error. There were no QACC warnings, kN force spikes, or >30 mm one-step
lateral ejections.

## Acceptance status

The selected physics run satisfies the sentinel requirements:

- zero QACC warnings;
- nominal guided insertion succeeds 10/10;
- all contact forces remain in a gentle 1--60 N regime (31.6 N worst case,
  18.9 N p95 under randomization);
- lateral stays below 30 mm and per-step lateral change below 30 mm;
- randomized failure is a bounded stall/jam, not a numerical ejection.

No scaling or training should be launched as part of this physics calibration.
