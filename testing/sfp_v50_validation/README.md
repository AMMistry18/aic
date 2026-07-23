# SFP v50 validation gate

This directory keeps validation separate from the controller and perception
implementation. The only physical-success oracle is the correct
`/scoring/insertion_event`; an action result or a partial-insertion score does
not count.

## Safety and truth contract

- Scope is SFP plug to SFP port only.
- Event timing starts when `/insert_cable` becomes active and uses
  `time.monotonic_ns`, not ROS/simulation time.
- A passing trial emits the correct module/port event within 45 wall seconds.
- Any wrong-port event, off-limit contact, or scorer-equivalent force penalty
  fails the trial.
- Force penalty tracking mirrors `ScoringTier2`: controller-state tare is
  subtracted, wrench header timestamps accumulate time above 20 N, and the
  trial fails above one second. The 45-second policy limit separately uses the
  monotonic wall clock.
- Every scenario uses `cable_0`. This is intentional: the checked-in bridge
  maps `/cable_0/insertion_event` but old generated configs named cables such
  as `cable_101`, whose insertion events never reached ROS.

## Scenario generation

The default grasp distribution varies all six TCP-to-plug pose dimensions
inside the qualification envelope. To use measurements instead, pass a schema
compatible YAML whose `sampling.kind` is `empirical_csv`; the CSV columns must
be exactly `x,y,z,roll,pitch,yaw`.

Generate a development suite with a non-held-out seed while tuning:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/generate_heldout.py \
  --suite-id sfp_v50_dev_v1 \
  --seed 2026071802 \
  --trials 100 \
  --shard-size 20 \
  --output-dir aic_engine/config/validation/sfp_v50_dev_v1
```

Generate the frozen scenarios once, without running them during tuning:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/generate_heldout.py \
  --suite-id sfp_v50_frozen_holdout_v1 \
  --seed 2026071801 \
  --trials 300 \
  --shard-size 30 \
  --output-dir aic_engine/config/validation/sfp_v50_frozen_holdout_v1
```

The manifest records every scenario seed, exact 6DoF grasp, expected event,
config SHA-256, bridge SHA-256, and distribution SHA-256.

## Staged ablation

`ablation_matrix.yaml` fixes the approved order:

1. Relative plug-to-port pose, with no fixed-bias fallback.
2. Visual contrast rescue as the first response to a wedge.
3. Persistent axial seating after visual rescue.
4. Lift plus fresh re-perception if seating still fails; a new wedge loops back
   through visual rescue.
5. The untouched 300-trial frozen gate.

Summarize completed stages with one JSONL evidence stream per stage:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/evaluate_ablation.py \
  --results visual_then_persistent_seating=results/seating.jsonl \
  --results recovery_with_visual_reentry=results/recovery.jsonl
```

## Freeze exact artifacts

Only freeze after the development stages pass. The four artifact names are
mandatory so the controller, both pose models, and runtime recipe cannot drift:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/freeze_gate.py \
  --scenario-manifest aic_engine/config/validation/sfp_v50_frozen_holdout_v1/sfp_v50_frozen_holdout_v1.manifest.json \
  --artifact controller_source=PATH_TO_CONTROLLER \
  --artifact plug_pose_model=PATH_TO_PLUG_MODEL \
  --artifact port_pose_model=PATH_TO_PORT_MODEL \
  --artifact runtime_recipe=PATH_TO_RUNTIME_RECIPE \
  --model-image my-solution:student-flowstate-v50 \
  --eval-image ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  --output results/sfp_v50_frozen_gate.json
```

Freezing resolves and records both Docker image IDs (`sha256:...`) in addition
to the four file hashes. Later capture refuses to run if either tag points to a
different image.

## Capture and evaluate

Start one observer before each engine trial. It begins timing from the next
active action goal and writes one JSON evidence record:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/observe_trial.py \
  --trial-id sfp_v50_frozen_holdout_v1_0001 \
  --frozen-gate results/sfp_v50_frozen_gate.json \
  --runtime-image-id evaluator=SHA256_FROM_GATE \
  --runtime-image-id model=SHA256_FROM_GATE \
  --output results/trial_0001.json
```

For the full local run, use the Docker orchestrator instead of invoking the
observer manually. It mounts each frozen shard into the official evaluator,
starts the frozen v50 model on a private Docker network, and connects the host
observer to that shard's unique published Zenoh port. It runs shards
sequentially to avoid oversubscribing a laptop:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/run_docker_gate.py \
  --gate results/sfp_v50_frozen_gate.json \
  --results-dir results/sfp_v50_300 \
  --shards all \
  --base-router-port 17447 \
  --platform linux/amd64 \
  --startup-poll-s 10
```

Shard `00` uses host router port `17447`, shard `01` uses `17448`, through
shard `09` on `17456`; every shard also receives unique network and container
names. The only periodic check is the 10-second router-startup probe. Trial
progress is event-driven and blocking, so the script does not poll training or
evaluation every few seconds. Evaluator/model logs, per-trial observer logs,
raw JSON evidence, shard JSONL, the combined `all_trials.jsonl`, and the final
gate report remain under the results directory. After each shard exits, the
runner also joins every record with the evaluator's official `scoring.yaml`;
full Tier 3 (`75`, correct event), insertion-force category `0`, and contacts
category `0` are independently required in addition to the live observer.

Inspect the exact Docker commands without starting any containers:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/run_docker_gate.py \
  --gate results/sfp_v50_frozen_gate.json \
  --results-dir results/sfp_v50_300 \
  --shards all \
  --base-router-port 17447 \
  --dry-run
```

After all 300 trials have run, evaluate them strictly:

```bash
.pixi/envs/default/bin/python testing/sfp_v50_validation/evaluate_gate.py \
  --gate results/sfp_v50_frozen_gate.json \
  --results results/all_trials.jsonl \
  --output results/frozen_gate_report.json
```

The command exits successfully only for exactly 300 unique results with
300/300 correct events within 45 seconds and zero prohibited penalties.
