# Current repository handoff

Updated: 2026-07-22

`main` is the authoritative development branch. The previous insertion,
board-search, deployment, and TACC handoff files were consolidated into this
document and the two focused handoffs below:

- [Insertion handoff](INSERTION_HANDOFF.md)
- [Board-search handoff](BOARD_SEARCH_HANDOFF.md)

## Current baseline

The repository intentionally combines two explicit baselines:

- **Insertion:** the latest local SFP V50 plug-relative controller and pose
  model. Learned insertion is disabled. `RLInsert.py` forces script mode and
  does not load an RL actor.
- **Board search:** the source snapshot from commit
  `b269872eb6f0a4a49edc6334c6985e4b00238a5b` (`Record three-camera v4
  deployment`). Later board-search experiments are not part of this baseline.

Do not infer the active implementation from an old feature branch, deployment
note, W&B run, or versioned handoff. Start from `main` and these three files.

## Repository hygiene

Generated training and evaluation state is not versioned. In particular,
`wandb/`, `runs/`, `outputs/`, `results/`, `RL/output/`, visual validation
captures, and `.DS_Store` files are ignored. Store long-running experiment
artifacts outside the checkout or in an artifact service.

Model files that are required to build the active runtime remain versioned.
The SFP plug-pose weight is:

```text
aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt
```

Frozen SFP V50 scenario configurations remain under
`aic_engine/config/validation/` because they are reproducibility fixtures, not
run output.

## Development workflow

Create new work directly from the current `main` unless a short-lived feature
branch is needed for review:

```bash
git switch main
git pull --ff-only origin main
git status --short --branch
```

Before pushing insertion or board-search changes, run the focused commands in
the corresponding handoff. If the board-search implementation changes, update
its pin and explain the departure from `b269872` in the same commit.
