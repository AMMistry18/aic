> **ARCHIVED / SUPERSEDED (2026-07-12).** Its premise (the `/aic_model/change_state` lifecycle service was unavailable) is RESOLVED. Deployment now works; see `docs/FLOWSTATE_DEPLOY_RECIPE.md` and `docs/FLOWSTATE_STATUS.md`.
> Kept for history only — do not follow as current instructions.

# Claude handoff: Flowstate lifecycle failure and insertion policy

Updated: 2026-07-11

## Mission

Continue debugging and finish the AIC student-policy deployment in Flowstate.
The immediate blocker is not insertion quality: Flowstate cannot configure the
ROS lifecycle node because `/aic_model/change_state` is unavailable. First make
the `aic_model` service start reliably and prove the lifecycle service is
visible. Only then run and score insertion trials.

Work autonomously where possible. Ask the user only for interactive credentials
or a Flowstate UI action that cannot be performed through `inctl`. Never put
passwords, MFA codes, Flowstate tokens, or generated credentials in Git.

## Repository state

```text
local repo: /Users/satya_anandh/Developer/aic
branch: main
remote: https://github.com/AMMistry18/aic.git
handoff commit: c83a56e8085fcab4d1fa69cd2823dc6763049202
commit subject: Fix Flowstate aic_model deployment
```

The worktree was clean when this handoff was written. Pull or inspect status
before editing; preserve any later user changes.

Read these first:

```text
docs/CLAUDE_FLOWSTATE_TACC_HANDOFF_20260711.md   this file
docs/FLOWSTATE_STUDENT_POLICY_HANDOFF.md         policy and deployment history
docs/FLOWSTATE_MUJOCO_PARITY_20260711.md         69-value contract diagnosis
docs/FLOWSTATE_GUIDED_V2_20260711.md             guided controller and tests
RL/student_teacher/TACC_NEXT_AGENT_HANDOFF.md    TACC jobs, paths, and login
```

Important correction: lines 7-8 of `docs/FLOWSTATE_STUDENT_POLICY_HANDOFF.md`
say that the service started and loaded the policy. That was true for an earlier
deployment, but it is not true of the latest Flowstate run described below.

## Current failure: confirmed evidence

The newest user-provided log is local at:

```text
/Users/satya_anandh/.codex/attachments/526e0ea3-60bb-4d81-823f-9c8af4f35c0e/pasted-text.txt
```

It is a log from `lifecycle_transition_skill`, not an `aic_model` container log.
At 2026-07-11 18:21-18:22 it repeatedly attempts transition 1 (configure) and
every attempt fails after five seconds:

```text
[lifecycle_transition_skill_node]: Transitioning node aic_model to transition 1
UNAVAILABLE: Service '/aic_model/change_state' not available
```

Earlier in the same log, Zenoh reports peer scouting and failures to connect to
several `tcp/172.26.0.x:<port>` locators. There is no `aic_model` startup line,
no `on_configure`, and no `aic_model.RLInsert` policy-load line.

Interpretation: the behavior tree and lifecycle helper are doing their jobs,
but no reachable lifecycle node named `aic_model` exists in their ROS graph.
This must be resolved before policy math, perception, or insertion motion can be
evaluated. The peer warnings support a ROS/Zenoh wiring problem, but because the
attached log belongs to the lifecycle helper they do not prove what environment
the `aic_model` container received.

## Most likely deployment problem

The local image was rebuilt with a router-aware entrypoint and then installed
using the same service manifest identity. Flowstate continued to display the
same immutable asset version:

```text
display name: Participant Policy Node
asset id: ai.intrinsic.aic_model
version: 0.0.1+c84d8e248aa372bfa959e0e0b790f6150d96ffd1900226879d6da3798741d393
```

The manifest contains no explicit version, and its content did not change even
though `aic_model.tar` changed. Therefore the first hypothesis to test is that
reinstalling the same asset ID/version did not replace the image actually used
by the running service. Do not assume the new image is deployed merely because
`inctl asset install` returned success.

The local image that was intended for deployment is:

```text
tag: my-solution:student-flowstate-guided-v5
image id: sha256:b2a56d8387dfb727c5895c2a9d76986d7fdfdeb84db427294e33750888ebaccd
entrypoint: ["/entrypoint.sh"]
cmd: ["--ros-args", "-p", "policy:=aic_model.RLInsert", "-p", "use_sim_time:=true"]
bundle: /private/tmp/aic-flowstate-guided-v5/images/aic_model/aic_model.bundle.tar
manifest: /private/tmp/aic-flowstate-guided-v5/aic_model.manifest.textproto
```

The entrypoint in `docker/aic_model/Dockerfile.student_flowstate` requires
`AIC_MODEL_ROUTER_ADDR` (or `AIC_ROUTER_ADDR`) and `AIC_MODEL_PASSWD`, writes a
Zenoh credential file, sets a direct router endpoint, and runs:

```text
pixi run --as-is ros2 run aic_model aic_model \
  --ros-args -p policy:=aic_model.RLInsert -p use_sim_time:=true
```

Other plausible causes, in priority order:

1. Flowstate is still launching the old image because the asset version did not
   change.
2. The new entrypoint is running but exits immediately because one of the two
   required runtime variables is absent or named differently in this service.
3. Flowstate overrides the OCI entrypoint/command, so `/entrypoint.sh` is never
   executed.
4. The service instance exists in the solution catalog but its pod/container is
   not scheduled or is crash-looping (the manifest requests one NVIDIA GPU).
5. A namespace/domain mismatch allows the model to run but hides its lifecycle
   service from `lifecycle_transition_skill`.

Get the actual `aic_model` pod/service startup logs before changing ROS code.
The lifecycle-helper log alone cannot distinguish these cases. In particular,
look for either `AIC_MODEL_ROUTER_ADDR must be provided`, `AIC_MODEL_PASSWD must
be provided`, the process command, image digest, scheduling/GPU errors, or an
`aic_model` Python traceback.

## Recommended next actions

1. Refresh Flowstate auth and verify the current cluster, solution, service
   instance, and asset version. Cluster IDs are snapshot-sensitive.
2. Obtain logs/status for the `aic_model` service itself, including prior pod or
   crash-loop logs if available. Do not treat `resource not found` as evidence
   that the service is healthy.
3. Inspect the service's resolved image digest and command. Compare them with
   the local image ID and `/entrypoint.sh` above.
4. Create a genuinely new asset version/manifest identity for the rebuilt image
   rather than reinstalling the `c84d8e...` version. Install it, delete and
   recreate only the `aic_model` service instance, and restart the sim solution.
   Keep the ROS node name `aic_model` even if the asset version changes.
5. Prove `/aic_model/change_state` exists before running the process tree. Then
   configure and activate once and capture `aic_model` logs.
6. Verify there is exactly one `/insert_cable` action server. The obsolete
   `aic_insertion_policy` service must remain absent.
7. Run one insertion with Cable End 2 disabled. Confirm closest-port selection,
   guided control, and the final insertion result before running more trials.
8. Report Flowstate outcomes separately from MuJoCo results.

Do not delete the solution, unrelated services, TACC datasets, teacher weights,
or training outputs. Deleting and recreating the single `aic_model` service
instance is acceptable when needed to bind a new asset version.

## Flowstate access

Snapshot from the last successful CLI session:

```text
organization: tar-2@xfa-prod-aic-us
solution id: 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
solution name: Copy of AIC Phase 1 Template
last active cluster/VM: vmp-f5ed-053nou72
operation mode: sim
service instance: aic_model
correct catalog asset: Participant Policy Node / ai.intrinsic.aic_model
wrong old asset: AIC Insertion Policy / ai.tar2.aic_insertion_policy
```

The cluster may have changed. Verify it in the Flowstate UI or with `inctl`
instead of blindly using the snapshot.

The Linux AMD64 CLI is run through Docker on this Mac:

```text
inctl binary: /private/tmp/inctl-linux-amd64
persisted inctl home: /private/tmp/aic-inctl-home
macOS CA file: /etc/ssl/cert.pem
```

Base command pattern:

```bash
docker run --rm --platform linux/amd64 \
  -v /etc/ssl/cert.pem:/etc/ssl/certs/ca-certificates.crt:ro \
  -v /private/tmp/inctl-linux-amd64:/inctl:ro \
  -v /private/tmp/aic-inctl-home:/root \
  debian:bookworm-slim \
  /inctl <command>
```

If auth has expired, run the same container with:

```text
/inctl auth login --no_browser --org tar-2@xfa-prod-aic-us
```

Give the resulting login link to the user and ask only for the fresh token when
the CLI requests it. Do not reuse tokens copied into old chat context and do not
write a token into this handoff, scripts, shell history, or Git.

Useful read-only checks after login:

```text
/inctl cluster list --org tar-2@xfa-prod-aic-us
/inctl solution list --org tar-2@xfa-prod-aic-us
/inctl asset list --org tar-2@xfa-prod-aic-us --cluster <current-cluster>
/inctl service state list --org tar-2@xfa-prod-aic-us \
  --solution 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH
/inctl logs --org tar-2@xfa-prod-aic-us \
  --solution 582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH \
  --service aic_model --since 10m --tail 300
```

Use `/inctl <subcommand> --help` if this CLI build differs. Previous `inctl
logs --service aic_model` calls returned `resource not found`; that remains an
unresolved clue, not a successful verification. Flowstate's UI owns execution
of the behavior tree, so the user may need to click Execute after the service is
repaired.

In the Flowstate asset picker, select **Participant Policy Node** with ID
`ai.intrinsic.aic_model`. Do not select **AIC Insertion Policy**. The service
instance name and the ROS lifecycle node must both remain `aic_model`.

## TACC access

TACC is needed for MuJoCo training/evaluation, not for diagnosing the immediate
Flowstate lifecycle-service absence.

```text
user: satya_a
host: stampede3.tacc.utexas.edu
login: ssh satya_a@stampede3.tacc.utexas.edu
```

TACC prompts for the account password and then a six-digit TACC MFA code. Ask
the user to enter them interactively. Never store either value. An older SSH
ControlMaster used `/tmp/codex-tacc-%r@%h:%p`, but assume it is stale and
re-authenticate unless a direct `ssh -S ... -O check` proves otherwise.

After login:

```bash
printf 'WORK=%s\nSCRATCH=%s\n' "$WORK" "$SCRATCH"
squeue -u satya_a
```

Expected roots:

```text
WORK=/work2/11590/satya_a/stampede3
SCRATCH=/scratch/11590/satya_a
```

Current relevant TACC artifacts:

```text
Flowstate-v1 source: /work2/11590/satya_a/stampede3/aic-flowstate-v1-5900b41
selected run: /scratch/11590/satya_a/aic/student_flowstate_v1_seed0_5900b41
parity fixtures: /scratch/11590/satya_a/aic/flowstate_parity_20260711
guided eval: /scratch/11590/satya_a/aic/guided_flowstate_v2_20260711
Slurm logs: /scratch/11590/satya_a/aic/slurm
```

Training and evaluation use Pixi directly on TACC with headless MuJoCo
(`MUJOCO_GL=egl`). Do not use Distrobox there. Local AIC Engine + Gazebo
validation is a separate `aic_eval` Distrobox workflow.

Do not modify or delete the frozen teacher:

```text
RL/student_teacher/weights/teacher_level1.zip
sha256=fac418a62bacab6c3ab39877e9a8b6f83db881ca41634fde9443a73630bd62b4
```

## Policy and evaluation state

The deployed learned checkpoint was epoch 25:

```text
/scratch/11590/satya_a/aic/student_flowstate_v1_seed0_5900b41/student_a_ep025.pt
MuJoCo held-out: 210/300 success, 88 timeout, 2 bad_collision
```

This is not a Flowstate score and is not 100% success. The guided exact-pose
controller report is `300/300` in MuJoCo at calibrated poses, but that also is
not proof of Flowstate success. Its machine-readable report is:

```text
RL/student_teacher/parity/guided_exact_pose_evaluation_300.json
```

The current Flowstate image selects `RL_INSERT_CONTROL_MODE=guided`, closest SFP
port (`AIC_SFP_TARGET_MODE=nearest_tip`), 26 mm preposition handoff, baseline
wrench subtraction after prepositioning, 6 mm lateral safety, 0.20 rad rotation
safety, and 18 N force abort. Relevant implementation:

```text
docker/aic_model/Dockerfile.student_flowstate
aic_model/aic_model/RLInsert.py
aic_model/aic_model/rl_insert_contract.py
aic_example_policies/aic_example_policies/ros/PerceptionInsert.py
models/final_insert_sfp_flowstate_v1.ts
models/final_insert_sfp_flowstate_v1.ts.contract.json
```

An earlier Flowstate run did reach the policy but drifted laterally and aborted.
That led to the 69-observation parity work and guided controller. The present
failure happens earlier: the lifecycle node cannot even be configured. Keep
these two failure phases separate.

## Definition of done

The task is complete only when all of the following are evidenced in fresh
logs or UI results:

1. The intended new asset version and image digest are actually bound to the
   single `aic_model` service instance.
2. The model container remains running and logs its startup.
3. `/aic_model/change_state` is reachable; configure and activate succeed.
4. Exactly one `/insert_cable` server accepts the goal.
5. A single-cable Flowstate run reaches policy execution and records selected
   closest port, depth progression, completion/timeout/abort, and any safety
   reason.
6. The final report states a Flowstate result without substituting MuJoCo or
   guided exact-pose evaluation numbers.

