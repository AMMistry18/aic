# Wseto fresh AIC deployment handoff — 2026-07-27

## Requested outcome

Deploy the current repo's AIC model to Flowstate solution **`Wseto Copy of
satya robert`** with a fresh image/build path and without retaining the
previous deployed AIC model.

## Target

| Item | Value |
|---|---|
| Org | `tar-2@xfa-prod-aic-us` |
| Solution display name | `Wseto Copy of satya robert` |
| Solution ID | `2b7f0a8e-1995-4b16-b3d7-6770735736df_BRANCH` |
| Asset ID | `ai.intrinsic.aic_model` |
| CLI tools | `$HOME/.local/bin/inctl`, `$HOME/.local/bin/inbuild` |
| Repo root | `/home/rschnurr/satya/aic` |

## Current Flowstate state

The old deployment was deliberately removed before the fresh build:

```text
service delete aic_model                 -> completed
asset uninstall ai.intrinsic.aic_model   -> completed
```

As of this handoff, `inctl asset list` returns **no**
`ai.intrinsic.aic_model` record in the target solution. The active upload was
intentionally stopped at the user's request; it did not complete an install.

Because the `aic_model` service instance was also deleted to permit asset
uninstall, a resumed deployment must add it back *after* installing the asset.

## Fresh artifact already built and validated

This build was created with Docker build layer cache disabled and the ROS base
explicitly refreshed:

```bash
docker buildx build --no-cache --pull --load \
  --file docker/aic_model/Dockerfile --tag my-solution:v1 .
docker buildx build --no-cache --load \
  --file deploy/flowstate/Dockerfile.aic_model_service \
  --build-arg CONTAINER_IMAGE=my-solution:v1 --tag flowstate:aic_model .
```

The old local `my-solution` and `flowstate:aic_model` image tags were removed
before the new image completed, so this run did not reuse the old local AIC
image.

| Item | Value |
|---|---|
| Model image ID | `sha256:ecf5097f71158ef4bb37ed9005f89167f692618ce97cf2ad892356b986d44e54` |
| Flowstate wrapper image ID | `sha256:c86dc875e19f38216114bbe3c2cc698d94d8ea33a081b4c06e26dce38e90e0e2` |
| Artifact directory | `/home/rschnurr/aic-model-images/aic_model/20260727T202307Z` |
| OCI tar | `aic_model.tar` (6.0 GB) |
| Bundle | `aic_model.bundle.tar` (6.0 GB) |
| Bundle SHA-256 | `948c37c9e218900ecc9346702aeaf50d55f815b2e8b45bf8ed5f24f29ed410af` |

Validation already completed successfully:

- Every `docker/aic_model/v50_overlay/aic_model/*.py` file matches both the
  source-tree and installed-package locations inside `my-solution:v1`.
- Packaged imports passed for `RLInsert`, `run_v50_script`,
  `VisualGapRecoveryMixin`, and the five-keypoint SC mouth geometry module.
- Targeted packaged-image test suite: **188 passed**.
- Flowstate wrapper startup loaded `aic_model.RLInsert` successfully.
  The later `ExternalShutdownException` is the expected response to the
  90-second smoke-test timeout.

## Important source-layout note

The deployable policy lives in `docker/aic_model/v50_overlay/`, not necessarily
the development source tree. `docker/aic_model/v50_overlay/aic_model/RLInsert.py`
intentionally differs from `aic_model/aic_model/RLInsert.py`; it contains the
new V50 visual-gap implementation and imports the supporting overlay modules.
The Dockerfile copies all overlay Python modules and SC mouth-pose weights into
both the upstream source tree and the installed package. Do **not** overwrite
the overlay with the development copy without a separate review.

## Resume from the existing bundle

The stopped upload had reached the large 5.2 GB environment layer when stopped.
The artifact is valid; reuse the exact bundle rather than rebuilding unless the
repo changes again.

```bash
cd /home/rschnurr/satya/aic

ORG='tar-2@xfa-prod-aic-us'
SOLUTION='2b7f0a8e-1995-4b16-b3d7-6770735736df_BRANCH'
ARTIFACT='/home/rschnurr/aic-model-images/aic_model/20260727T202307Z'
INCTL="$HOME/.local/bin/inctl"

# Retry only the install. `updater already finalized` is a Flowstate-side
# transient; retry the same bundle after a short delay.
for attempt in 1 2 3 4 5 6; do
  if "$INCTL" asset install "$ARTIFACT/aic_model.bundle.tar" \
      --org "$ORG" --solution "$SOLUTION"; then
    break
  fi
  sleep 60
done

# The old instance was deleted before uninstall, so recreate it after a
# successful asset install. This is required for a service pod to exist.
"$INCTL" service add ai.intrinsic.aic_model \
  --name aic_model --org "$ORG" --solution "$SOLUTION"

# Confirm catalog version and fresh service startup.
"$INCTL" asset list --org "$ORG" --solution "$SOLUTION" --output json | \
  rg '"id":"ai\\.intrinsic\\.aic_model"'
sleep 30
"$INCTL" logs --org "$ORG" --solution "$SOLUTION" --service aic_model \
  --since 5m --tail 300 --timeout 60s | \
  rg 'Loading policy module|Loaded policy module|Using policy|on_activate'
```

If `service add` returns an "already exists" error, that is harmless: continue
to the log verification. If `asset install` returns `updater already finalized`,
wait and rerun the same install command; no rebuild is needed.

## Fresh-rebuild command, only if sources change again

```bash
cd /home/rschnurr/satya/aic
docker buildx build --no-cache --pull --load \
  --file docker/aic_model/Dockerfile --tag my-solution:v1 .
docker buildx build --no-cache --load \
  --file deploy/flowstate/Dockerfile.aic_model_service \
  --build-arg CONTAINER_IMAGE=my-solution:v1 --tag flowstate:aic_model .
```
