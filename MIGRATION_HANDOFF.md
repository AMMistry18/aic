# Research machine migration — November 14, 2026

Repository: https://github.com/AMMistry18/aic
Branch: `research/gazebo-rap-evaluation`
Release: `research-migration-20261114`

The release is the complete evidence/weight/runtime backup. The Git branch
contains the canonical research code, configs, paper and compact metadata.
Download the release as well as cloning Git. Large raw evidence and the copied
development workspace deliberately live in release archives instead of Git.
Only redundant caches, nested Git databases and credential files are omitted.

## Restore

Use Python 3.12+ and Docker with NVIDIA GPU support. A Windows host can use
Docker Desktop/WSL2 as before. Linux can use Docker Engine/NVIDIA Container
Toolkit. Keep at least 70 GB free for expanded evidence and Docker layers;
more is useful for additional trials. Release parts are at most 1 GiB each.

```sh
git clone --branch research/gazebo-rap-evaluation https://github.com/AMMistry18/aic.git
cd aic
gh release download research-migration-20261114 --repo AMMistry18/aic --dir ../migration-assets
python experiments/restore_migration58.py --assets ../migration-assets --load-docker --create-containers
```

The restore script verifies SHA256 and size of every archive part before
extraction/loading, then checks each restored workspace file. Archive parts
must remain together in numbered order. Images preserve the actual working
ROS 2 Kilted, Gazebo, trained-model dependencies and custom float64 Bullet
plugin. The original runtime metadata is in `docker-runtime.json`; host-specific
bind paths are replaced by the new checkout path. Inspect GPU passthrough
with `docker start research-aic-eval` and `docker exec research-aic-eval nvidia-smi`,
then stop that container before starting any experiment executor.

Optionally add `--restore-legacy` to recover older Flowstate sources and model
checkpoints into `.migration_legacy58/flowstate-build/`. Obsolete generated
Flowstate image tar bundles are omitted; their source/model files are preserved.
The script verifies the archived snapshot, then restores the newest committed
migration bootstrap tools from Git. The scientific source freeze is unchanged.
All 65,737 workspace files and 1,262 legacy files passed a full streamed
archive SHA256 check before any local evidence was eligible for cleanup.

A faster GPU helps only GPU-enabled inference. The frozen baseline also uses
software camera rendering and sequential contact measurements. Do not change
its runtime/model selection halfway through the fixed cohort and call that
the same experiment. Record the host migration as a resource change, preserve
the method/physics contract, and evaluate optimized policies as a new cohort.

Install host dependencies needed by the coordinator, for example:
`python -m pip install numpy scipy psutil pyyaml opencv-python`.
Inference runs in Docker; host PyTorch is unnecessary for the trial coordinator.
Paper building has additional dependencies listed under `paper/`.

The existing coordinator's final entry point uses Windows sleep prevention.
On Linux run its unchanged `main` function without that Windows-only wrapper:
`python -c "from experiments.resume_full_evaluation44 import main; main()"`.
On Windows use `python -u experiments/resume_full_evaluation44.py`.
Review the ledger first; do not start two coordinators. Linux container resource
paths remain `/research`. Existing relative evidence paths may use backslashes;
normalize them when writing new analysis code. Do not edit frozen scientific
sources to adjust host compatibility; use an external wrapper.

## State at shutdown

`experiments/migration58/stop_progress.json` and `progress_before_migration.json`
record the precise boundary and all completed attempts. The coordinator was
stopped after a completed, archived trial; no selected attempt was interrupted,
retried or replaced. The main ledger was marked `stopped_requires_inspection`
with the migration reason so the existing resume tool can validate its prefix.
Original manifests/source freezes remain intact. Primary study is 100 cases
(50 SC, 50 SFP), followed by 80 paired ablations. Read its live ledger for exact
counts; no unfinished tests are described as complete.

The fixed study uses recovered baseline policies. Optimized development results
are separate and must never be pooled with it. `experiments/speed_reliability54.md`
describes the speed work. The latest v6 and v7 SFP development trials both
verified: 46.5445 sim /494.4684 wall s at 8.641645 N raw peak, and 45.9865 sim
/490.9214 wall s at 8.417083 N. Each is one reused case, not robustness evidence.
The queued crop57 perception diagnostic was cancelled BEFORE execution for
migration; its source is preserved. Historical followup/lease files describe
the old machine, not a queue to start automatically. Do not rerun continuation
scripts that intentionally refuse existing output directories.

## Research objective and evidence rules

Make complete last-inch insertion faster than 30 seconds while retaining SC
accuracy and raising SFP reliability to a comparable level. SC has one verified
26.189 sim-second development run; SFP remains above 30 seconds. Simulated task
time differs from laptop wall time. Count perception, contact registration,
travel and verified dwell; separately report setup and phase times. Include
all attempts in success-within-30-seconds, not just successful-case medians.

Use BOTH SC and SFP, original learned/confidence-weighted image processing,
actual public meshes/full fixtures and contact physics, a prescribed rigid
grasp that is visually estimated, and upstream angular alignment. This is
conditional last-inch insertion, not full grasping or general angular recovery.
Simulator truth is only for setup, diagnostics and independent scoring. Never
feed actual object/target poses or the prescribed grasp to the deployed policy.
Keep original visual thresholds, seating/contact/dwell criteria and complete
2000 Hz raw force coverage. The raw resultant ceiling is 20 N. Do not revive
the stopped speed43 profile: it produced a 109.9 N raw spike.

Final44 outcomes must not tune candidates. Develop on separate development
cases, qualify changes, freeze a new policy and then predeclare fresh seeds.
SC baseline reliability is high; SFP still has visual rejection, sensor-freshness
and waypoint failures. Do not call both connectors robust based on a successful
single development run. Pose optimizer improvements alone did not fix a known
development SFP rejection (16.19 to 13.44 px, still over the 6 px gate).

## Models and paper

All original weights plus the three active recovery/adaptation checkpoints,
their hashes, training provenance, datasets, raw trial recordings, camera frames,
logs, candidate sources and frozen snapshots are in the workspace archive.
The original SC plug checkpoint is unchanged; the other three active models
were newly adapted in the November 12 recovery, not byte-identical old models.
Never change hardlinked checkpoint files in place.

Read `EXPERIMENT_PLAN.md`, `EXPERIMENTS.md`, `PAPER_RESULTS.md`,
`RESEARCH_PROGRESS.md`, `paper/ORIGINAL_PIPELINE_TRACEABILITY.md`,
`paper/NOVELTY_DECISION.md`, and the actual study ledger. The four-page paper
draft is incomplete; its small historical results table is not the latest
partial study. Do not silently substitute development results for held-out data.
Simulation-only eligibility is accepted per the user. Main novelty hypothesis:
connector-specific contact datums for uncertain coordinates, plus independent
physical verification. Matched ablations/baselines and uncertainty are essential.

Original approach: https://satyaa27.github.io/intrinsic-ai-for-industry-challenge-team-tar-2-sol/
Pinned public Phase 1 source: `7148df9ab62b4b550b101189098f05bc0dba8444`.
Writing skill: `paper/writing_support/empirical-paper-writer/SKILL.md`.
User requested about four pages and provided a December 1, 2026 collection deadline.

After restoring, inspect source-freeze hashes and raw evidence before any trials.
Read `experiments/migration58/` and release indexes for exact archive provenance.
