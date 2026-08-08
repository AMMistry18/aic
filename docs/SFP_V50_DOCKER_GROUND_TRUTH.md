# SFP V50 Docker ground truth

This is the reproducible Phase 1 image used to build and install the SFP V50
last-inch insertion policy.

The default `docker/aic_model/Dockerfile` now:

- fetches validated upstream Phase 1 commit
  `1c1e018eaecadf1829e18603bb531af30e02fce2`;
- builds its frozen Pixi environment with Pixi 0.67.2;
- installs Ultralytics 8.4.48;
- applies the V50 policy, perception code, and all three model weights; and
- starts `aic_model.CableInsertionPolicy`.

Dependency installation precedes the policy overlay, so policy-only edits reuse
the expensive Pixi and Ultralytics layers.

## Build

Run from an Ubuntu 24 WSL shell. The repository in the example is mounted from
Windows, but the commands run in WSL:

```bash
cd /mnt/c/tmp/ws_aic_phase1/src/aic
./docker/aic_model/build_sfp_v50.sh
```

The equivalent standard Compose command is:

```bash
cd docker
docker compose build model
```

Do not pass `--no-cache` for normal rebuilds. Docker will reuse the Pixi,
Ultralytics, and unchanged policy layers.

## Verify before bundling

```bash
docker image inspect my-solution:v1 \
  --format 'id={{.Id}} os={{.Os}} arch={{.Architecture}} entrypoint={{json .Config.Entrypoint}} cmd={{json .Config.Cmd}}'

docker run --rm --entrypoint /bin/sh my-solution:v1 \
  -c 'head -1 /entrypoint.sh | od -An -t x1'

docker run --rm --entrypoint /bin/bash my-solution:v1 -lc '
  cd /ws_aic/src/aic
  .pixi/envs/default/bin/python -c "
from ultralytics import YOLO
from aic_model.CableInsertionPolicy import CableInsertionPolicy
from aic_model.v50_controller import run_v50_script
from aic_perception.gripper_masks import GripperMaskBank
YOLO(\"/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/weights/best.pt\")
YOLO(\"/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/weights/best_sfp_plug_pose.pt\")
GripperMaskBank()
print(\"SFP V50 imports and weights OK\")
"'
```

The entrypoint bytes must end in `0a`, not `0d 0a`. The configured policy must
be `aic_model.CableInsertionPolicy`.

## Bundle

Run from the workspace root, which is the parent of this repository:

```bash
cd /mnt/c/tmp/ws_aic_phase1
./src/aic/flowstate/scripts/build_aic_model.sh \
  --container_image my-solution:v1

sha256sum images/aic_model/aic_model.bundle.tar
tar -tf images/aic_model/aic_model.bundle.tar
tar -xOf images/aic_model/aic_model.bundle.tar aic_model.tar |
  tar -tf - >/dev/null
echo "inner tar exit codes: ${PIPESTATUS[*]}"
```

## Install

The install target is solution-specific:

```bash
cd /mnt/c/tmp/ws_aic_phase1
./inctl asset install images/aic_model/aic_model.bundle.tar \
  --org tar-2@xfa-prod-aic-us \
  --solution YOUR_SOLUTION_ID_BRANCH
```

If Intrinsic returns `updater already finalized` or a missing remote content
digest, retry the same command after two minutes. Do not rebuild a locally
validated bundle for those server-side transfer errors.

## Provenance and current validation status

The reference build produced:

- V50 image ID:
  `sha256:e8069e8baa8ca7220b0d6097b68ba8efe90137d1dccc11cb049b538264819752`
- Bundle SHA-256:
  `5d1526ff0c9aa17eb62291a70b41c48a0ca3b156785b13571c08b2530d95a469`
- Installed asset:
  `ai.intrinsic.aic_model.0.0.1+cc1f992aafc7e5f332616c26f250660775bb30f01ea8135c3a19b607f5ffe036`
- Verified platform: Linux/amd64 on Ubuntu 24.04
- Verified entrypoint, policy command, Python imports, model weight loading,
  image export, bundle integrity, and Flowstate installation

The first V50 `InsertCable` execution reached the policy but aborted after about
seven seconds. That physical insertion result is not yet a validated success;
the `aic_model` service logs must be used to identify the exact V50 abort
condition. This repository state is therefore the build/deployment ground
truth, not a claim that the insertion behavior is fully tuned.

Never commit generated image tarballs, bundles, upload logs, API keys, or login
credentials.
