# Flowstate upload pipeline — working state as of 2026-07-26

Handoff for whoever picks this up next (human or agent). Everything below was
executed and verified on 2026-07-25/26, not inferred.

## TL;DR

- The model **works**. It is live in the `satya` solution and reaches `on_activate()`.
- The `Copy of satya` solution is **broken in a way that has nothing to do with our
  artifact** — the identical bundle bytes run fine on `satya` and never produce a pod
  on `Copy of satya`. Do not burn time on it; use `satya`.
- Build → bundle → upload is fully scripted and takes ~5 min for a code-only change.

## Key identifiers

| Thing | Value |
|---|---|
| Repo root | `/home/Anshul/AIC_Phase_1/aic_0/aic` (NOT the top level; that has a stray empty `.git`) |
| Org | `tar-2@xfa-prod-aic-us` |
| **Working solution** | `60b15a74-0049-46b3-97e4-b2f2db410eee_BRANCH` — display name **`satya`** |
| Broken solution | `bfb431f8-4ba8-48a3-a32f-017c06d827f8_BRANCH` — `Copy of satya` (see below) |
| Asset id | `ai.intrinsic.aic_model` |
| `inctl` / `inbuild` | `/home/Anshul/AIC_Phase_1/flowstate_ws/{inctl,inbuild}` |
| Bundle artifacts | `~/aic-model-images/aic_model/<UTC-stamp>/aic_model.bundle.tar` |

**Three solutions share the display name "Copy of satya".** Always target by id, never
by name.

## The pipeline

### Deployment code lives in the overlay, not the dev tree

The Docker build fetches a **pinned upstream commit**
(`1c1e018eaecadf1829e18603bb531af30e02fce2`) and then copies
`docker/aic_model/v50_overlay/` over it. Files under `aic_model/aic_model/` are **not**
shipped. Every deployment change must be present in the overlay.

Before every build, check that no dev-side change was left out:

```bash
for f in $(git diff --name-only <PREV_SHA> HEAD -- aic_model/aic_model/); do
  b=$(basename "$f")
  git diff --quiet <PREV_SHA> HEAD -- "docker/aic_model/v50_overlay/aic_model/$b" \
    && echo "*** $b changed in dev but NOT in overlay -- would NOT ship"
done
```

`aic_model.py`, and sometimes `RLInsert.py`, **intentionally differ** between the dev tree
and the overlay. Do not blindly sync them.

### Fast build (code-only changes) — ~1 second

Valid only when nothing outside `v50_overlay/aic_model/*.py` changed. Verify first:
`docker/aic_model/Dockerfile`, `pixi.lock`, `pixi.toml`, `perception_core.py`,
`v50_overlay/aic_perception/`, `*.pt` weights, and the pinned upstream commit must all be
unchanged.

```bash
cd /home/Anshul/AIC_Phase_1/aic_0/aic
docker buildx build --load \
  --file docker/aic_model/Dockerfile.clock_skew_fix \
  --build-arg BASE_IMAGE=my-solution:requirements-base \
  --tag my-solution:v1 .
```

**`my-solution:requirements-base` must never be deleted** — it is what makes this 1 second
instead of 12 minutes. Likewise never run `docker builder prune`: it holds the rattler/conda
cache, and losing it forces a full dependency re-download on the next full build.

### Full build — ~12 min cold, ~11 s with warm cache

```bash
docker buildx build --load --file docker/aic_model/Dockerfile --tag my-solution:v1 .
docker tag my-solution:v1 my-solution:requirements-base
```

Required if weights, `perception_core.py`, `aic_perception/`, pixi files, the base
Dockerfile, or the pinned commit change. Never pass `--no-cache`.

### Verification (do not skip — this is what catches thin-build drift)

The strongest check is comparing every overlay module against **both** locations inside the
image; a thin build can silently omit newly added modules.

```bash
(cd docker/aic_model/v50_overlay/aic_model && sha256sum *.py) | sort -k2 > /tmp/t.txt
docker run --rm --entrypoint /bin/sh my-solution:v1 -c \
  'cd /ws_aic/src/aic/aic_model/aic_model && sha256sum *.py' | sort -k2 > /tmp/i.txt
docker run --rm --entrypoint /bin/sh my-solution:v1 -c \
  'SP=$(/ws_aic/src/aic/.pixi/envs/default/bin/python -c "import site; print(site.getsitepackages()[0])"); cd "$SP/aic_model" && sha256sum *.py' | sort -k2 > /tmp/s.txt
# all three must agree, line for line
```

Then imports, tests, and an actual run of the packaged image:

```bash
docker run --rm -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -v "$PWD:/work:ro" -w /tmp \
  --entrypoint /ws_aic/src/aic/.pixi/envs/default/bin/python my-solution:v1 \
  -m pytest /work/aic_model/test/test_sc_controller.py \
            /work/aic_model/test/test_v50_controller.py -q -p no:cacheprovider

timeout 90 docker run --rm flowstate:aic_model > /tmp/run.log 2>&1
grep -E 'Loading policy module|Loaded policy module|Using policy' /tmp/run.log
```

Expect the three lines `Loading policy module: aic_model.RLInsert` → `Loaded policy module`
→ `Using policy: RLInsert`. A trailing
`rclpy.executors.ExternalShutdownException` is **normal** — it is rclpy responding to the
SIGTERM from `timeout`, not a failure.

### Wrapper, bundle, upload

`deploy/flowstate/Dockerfile.aic_model_service` adds the zenoh/NVIDIA env vars.
`deploy/flowstate/aic_model.manifest.textproto` publishes `ai.intrinsic.aic_model` — do
**not** reuse `aic_model_v38.manifest.textproto`, which publishes a different asset id.

Scripts used for the last several rounds live in the session scratchpad; the essentials are:

```bash
docker save --output "$DIR/aic_model.tar" flowstate:aic_model
"$INBUILD" service bundle --manifest deploy/flowstate/aic_model.manifest.textproto \
  --oci_image "$DIR/aic_model.tar" --output "$DIR/aic_model.bundle.tar"
"$INCTL" asset install "$DIR/aic_model.bundle.tar" --org "$ORG" --solution "$SOLUTION"
```

`docker save` and bundling always stream the full ~6 GB regardless of Docker cache — budget
~4 min. Only the upload dedupes.

## Gotchas that cost real time

1. **`updater already finalized` on upload.** Transient. Always appears on *first contact
   with a new cluster* and typically clears on **attempt 4**. Retry the same bundle; the
   retry loop handles it. Uploads to a cluster that has already seen the layers succeed on
   attempt 1 in ~20-50 s. `--policy update_compatible` was needed once (2026-07-08) but not
   at any point on 07-25/26.
2. **`asset install` replaces the catalog version** — only one version of an id exists at a
   time. Installing a new version removes the previous one from the catalog.
3. **Sim VMs are ephemeral.** `inctl cluster list` shows only the current one; restarts land
   on a fresh `vmp-*` and rebuild from the solution snapshot. Assets/instances that are not
   persisted into the snapshot are lost on restart.
4. **`service state list`/`get` only report *running* instances.** `NotFound` there does not
   prove the instance is unregistered. The reliable probe for registration is `service add`,
   which returns `code:6 instance already exists`. The reliable probe for a **running pod**
   is `inctl logs --service <name>` — `resource not found` means no pod. Always run a control
   against a known-good service (e.g. `perception_service`) to confirm the command works.
5. **`service add --cluster` and `--solution` behave identically.** They write to the same
   registry. This was tested; there is no design-time/runtime split between them.
6. Build context: run everything from the repo root. `.dockerignore` only excludes `.pixi`,
   but buildx only transfers `COPY`-referenced paths (~127 MB), so context size is a non-issue.
7. **`asset install` restarts the pod on its own** (observed on `satya`: install at 01:00:02,
   pod back up and `Using policy: RLInsert` by 01:00:37). No manual restart needed.
8. **Always timestamp-check logs before attributing behaviour to a build.** `inctl logs`
   returns whatever is in the buffer, which routinely predates the install you just did.
   Convert the ROS epoch and compare against the install time:
   `date -d @1785044146` vs the `Finished installing` line. This bit us once — a depth trace
   from the *previous* build was read as evidence that a new change worked. The fix is
   mechanical: confirm `log_ts > install_ts` before drawing any conclusion, and remember the
   pod needs an actual insertion attempt after the restart before there is anything to judge.

## `Copy of satya` — known broken, do not chase

Symptom: the asset installs, the instance registers (`service add` → "already exists"), and
**no pod is ever created** (`inctl logs` → `resource not found`, while controls stream fine).

Ruled out by direct experiment:
- Not the image, bundle, or manifest — the **byte-identical bundle** (`a5e5f435…`,
  installed as `…+9ef63b37`) runs fine on `satya` and produces no pod on `Copy of satya`.
- Not the GPU limit — `satya` schedules the same `nvidia.com/gpu: 1` manifest and its
  perception service reports `NVIDIA L4/PCIe/SSE2` with CUDA/OpenGL interop.
- Not version pinning — re-added pinned to the exact installed `idVersion`; no change.
- Not `--cluster` vs `--solution`; not a stale definition entry (deleted and re-added
  cleanly); not a restart (waited 10 min after; still nothing).

Distinguishing observation: `Copy of satya` cycled through **four** clusters
(`9d6dvlrb → 2wgzs2be → 27ckdlqx → y96u29o9`) in ~3 hours while `satya` stayed on
`sba8jozm` the whole time. Its deployment state looks unhealthy in a way `inctl` cannot
reach. Fix it in the Flowstate UI or abandon it.

The one thing never tried: deleting the instance and re-adding it *after* the
`…+9ef63b37` install (every earlier add short-circuited on the pre-existing registration).

## Live state at handoff

| Solution | Installed version | Pod |
|---|---|---|
| `satya` (`60b15a74`) | `ai.intrinsic.aic_model.0.0.1+c88ac6a9…` | ✅ running |
| `Copy of satya` (`bfb431f8`) | `ai.intrinsic.aic_model.0.0.1+9ef63b37…` | ❌ none |

Healthy startup on `satya` looks like:

```
on_configure(LifecycleState(label='unconfigured', state_id=1))
Instantiating policy... / Policy.__init__()
[rl] loading SFP YOLO-pose weights: .../best.pt
[rl] SC pose weights available: .../best_sc_pose.pt
[v50] port and plug YOLO first-inference warmup completed during configure
[v50] plug-relative controller ready; force_target=8.0N force_abort=18.0N
[script] SCRIPT-ONLY: RL actor loading and inference are disabled
[rl] using training home qpos: [-0.1597 -1.3542 -1.6648 -1.6933 1.571 1.411]
on_activate()
```

## Open items

- **`RL_INSERT_CALIB_DUMP=1` is baked into `deploy/flowstate/Dockerfile.aic_model_service`**
  and is live in every uploaded version. The file's own comment marks it debug-only.
  **Remove it before a scored submission**, along with the commented-out
  `RL_INSERT_SC_TIP_IN_TCP_POS` / `_QUAT` lines once the median re-grasp value is known.
- Uploading to `satya` **replaced** the template's `ai.intrinsic.aic_model`
  (`…+42c2c39d`, "Participant Policy Node."). Nothing has depended on it so far.
- Disk: bundles are 6 GB each under `~/aic-model-images/aic_model/`. The intermediate
  `aic_model.tar` in each directory is safe to delete once its `.bundle.tar` exists, and
  deleting bundles does not slow future uploads.
