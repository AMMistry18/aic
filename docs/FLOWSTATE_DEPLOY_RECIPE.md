# Flowstate `aic_model` full-image deployment

Last verified: 2026-07-14 (v38). Use the full Dockerfile path below for v39 and
later. Do not use the old thin-overlay recipe: inherited images can contain the
strict router entrypoint that exits before the ROS lifecycle node starts.

The Intrinsic command-line tools and authentication are not committed to this
repository. Keep them in `~/.aic-flowstate`, not `/private/tmp`, so a reboot does
not erase the deployment environment.

## 1. Install the command-line tools persistently

Sign into the organization console:

<https://flowstate.intrinsic.ai/o/tar-2@xfa-prod-aic-us>

Use **Set up development environment** / developer tools to download the Linux
AMD64 builds of both `inctl` and `inbuild`. No stable public direct binary URL is
available; do not guess one.

```bash
mkdir -p ~/.aic-flowstate/bin ~/.aic-flowstate/inctl-home
cp ~/Downloads/<inctl-download> ~/.aic-flowstate/bin/inctl-linux-amd64
cp ~/Downloads/<inbuild-download> ~/.aic-flowstate/bin/inbuild
chmod 755 ~/.aic-flowstate/bin/inctl-linux-amd64 ~/.aic-flowstate/bin/inbuild
```

This repository includes a macOS-to-Linux wrapper:

```bash
scripts/flowstate/inctl.sh version
scripts/flowstate/inctl.sh auth login --no_browser \
  --org tar-2@xfa-prod-aic-us
```

The login requires an interactive terminal. Open the URL it prints, approve the
login, and paste the token into the CLI prompt. Never put the token in Git, a
script, a handoff document, or chat. Authentication persists under
`~/.aic-flowstate/inctl-home`.

`inbuild` is separately required. The recovered `inctl` does not implement
bundle creation (`inctl bundle --help` returned `unknown command`).

## 2. Fixed deployment target

```bash
export ORG='tar-2@xfa-prod-aic-us'
export SOL='582bcf0b-e30d-43b4-ad4c-6388e7b03719_BRANCH'
export FLOWSTATE_HOME="${AIC_FLOWSTATE_HOME:-$HOME/.aic-flowstate}"
```

Keep the service instance name exactly `aic_model`. Flowstate process lifecycle
nodes target `/aic_model/change_state`; changing `--name` breaks that wiring.

The cluster ID can change. Discover it after authenticating:

```bash
scripts/flowstate/inctl.sh cluster list --org "$ORG"
```

## 3. Choose a new immutable asset version

Flowstate keys services by manifest identity. Reinstalling the same asset name
does not replace the running image. v38 is current, so the next build is v39:

```bash
export VERSION=v39
export ASSET_NAME="aic_model_${VERSION}"
export IMAGE="my-solution:student-flowstate-script-${VERSION}"
export WORK="/private/tmp/aic-script-${VERSION}"
mkdir -p "$WORK/images/aic_model"
```

Copy the known-good tracked manifest and change **all** v38 identity/display
occurrences to v39. Do not edit the archive filename:

```bash
cp deploy/flowstate/aic_model_v38.manifest.textproto \
  "$WORK/${ASSET_NAME}.manifest.textproto"
sed -i '' 's/aic_model_v38/aic_model_v39/g; s/v38/v39/g' \
  "$WORK/${ASSET_NAME}.manifest.textproto"
rg -n 'v38|v39|archive_filename' "$WORK/${ASSET_NAME}.manifest.textproto"
```

For versions after v39, adjust both replacements. The manifest must identify
`ai.intrinsic.aic_model_vNN` and must reference `aic_model.tar`.

## 4. Build the full image

Build from the repository root. `BASE_IMAGE=my-solution:v8` must already exist
locally; it contains the validated AMD64 ROS/Pixi environment and model weights.

```bash
docker build --platform linux/amd64 \
  -f docker/aic_model/Dockerfile.student_flowstate \
  --build-arg BASE_IMAGE=my-solution:v8 \
  -t "$IMAGE" .
```

Do not replace this with `FROM <old-guided-image>` plus one `ENV`. The full
Dockerfile installs the current source and the router-safe `/entrypoint.sh`.

Verify the baked control mode, router fallback, board-search module, and the
preserved scripted `+Y` nudge before creating a 7 GB tar:

```bash
docker inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | rg 'RL_INSERT_CONTROL_MODE|RL_INSERT_BOARD_SEARCH|AIC_ROUTER_ADDR'

docker run --rm --platform linux/amd64 --entrypoint /bin/bash "$IMAGE" -lc '
  rg -n "class BoardSearch" \
    /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_model/board_search.py
  rg -n "_nudge_to_unstick|SCRIPT_NUDGE_DIRS" \
    /ws_aic/src/aic/.pixi/envs/default/lib/python3.12/site-packages/aic_model/RLInsert.py
  rg -n "falling back to rmw_zenoh peer scouting" /entrypoint.sh
'
```

Expected environment for the board-only asset:

```text
RL_INSERT_CONTROL_MODE=script
RL_INSERT_BOARD_SEARCH=0
RL_INSERT_BOARD_SEARCH_ONLY_TASK_ID=board_search_only
AIC_ROUTER_ADDR=zenoh-router.app-intrinsic-base.svc.cluster.local:7447
```

## 5. Save and bundle the image

```bash
docker save "$IMAGE" -o "$WORK/images/aic_model/aic_model.tar"
shasum -a 256 "$WORK/images/aic_model/aic_model.tar"
```

Run the Linux/AMD64 `inbuild` binary in Debian. The `$WORK` bind mount is writable;
the image directory is also mounted at the path supplied to `--oci_image`:

```bash
docker run --rm --platform linux/amd64 \
  -v "$FLOWSTATE_HOME/bin/inbuild:/inbuild:ro" \
  -v "$WORK:/work" \
  -v "$WORK/images/aic_model:/img:ro" \
  debian:bookworm-slim \
  /inbuild service bundle \
    --manifest "/work/${ASSET_NAME}.manifest.textproto" \
    --oci_image /img/aic_model.tar \
    --output "/work/${ASSET_NAME}.bundle.tar"
```

Verify the bundle before uploading. It must contain exactly two root members and
no macOS `._*` AppleDouble entries:

```bash
tar -tf "$WORK/${ASSET_NAME}.bundle.tar"
# aic_model.tar
# service_manifest.binarypb
shasum -a 256 "$WORK/${ASSET_NAME}.bundle.tar"
```

If a hand-packed tar is ever necessary, use
`COPYFILE_DISABLE=1 tar --no-mac-metadata`; prefer `inbuild` because it compiles
the textproto correctly.

## 6. Install the asset

The wrapper must be able to see the bundle. Because its default container only
mounts the CLI and auth home, stage the large bundle in a Docker volume and run
the equivalent `inctl` container with that volume mounted:

```bash
docker volume create aic-flowstate-upload
docker run --rm \
  -v aic-flowstate-upload:/upload \
  -v "$WORK:/host:ro" \
  debian:bookworm-slim \
  cp "/host/${ASSET_NAME}.bundle.tar" /upload/

docker run --rm --platform linux/amd64 \
  -v /etc/ssl/cert.pem:/etc/ssl/certs/ca-certificates.crt:ro \
  -v "$FLOWSTATE_HOME/bin/inctl-linux-amd64:/inctl:ro" \
  -v "$FLOWSTATE_HOME/inctl-home:/root" \
  -v aic-flowstate-upload:/upload:ro \
  debian:bookworm-slim \
  /inctl asset install "/upload/${ASSET_NAME}.bundle.tar" \
    --org "$ORG" --solution "$SOL"
```

The multi-gigabyte upload may be quiet for several minutes. If it fails with
401/XSRF or says the upload did not finish, reauthenticate and rerun the same
install; installation is content-addressed/idempotent.

## 7. Rebind the live service

Echo and then run the literal positional asset ID. Do not place the asset ID in
an unset shell variable: that produces `Asset ID cannot be empty`.

For v39, the required add command is literally:

```bash
scripts/flowstate/inctl.sh service delete aic_model \
  --org "$ORG" --solution "$SOL"

scripts/flowstate/inctl.sh service add ai.intrinsic.aic_model_v39 \
  --name aic_model --org "$ORG" --solution "$SOL"
```

Only delete/rebind the `aic_model` service. Do not delete the solution or
unrelated services. Keep old versioned assets until the new one is proven so
rollback remains possible.

## 8. Verify startup and runtime separately

Check recent logs:

```bash
scripts/flowstate/inctl.sh logs \
  --org "$ORG" --solution "$SOL" \
  --service aic_model --since 10m --tail 300
```

Startup proof is all three lines:

```text
Loading policy module: aic_model.RLInsert
Loaded policy module aic_model.RLInsert
Using policy: RLInsert
```

`service state list/get` has occasionally omitted a service whose logs prove it
is running, so do not treat the omission alone as authoritative. Conversely,
successful `asset install` is not runtime proof.

For board search, the Flowstate process must run serially:

```text
tare FT
  -> switch_to_aic_controller
  -> configure aic_model
  -> activate aic_model
  -> Insert Cable Skill (id = board_search_only)
  -> switch_to_default_controller
```

Runtime success requires `[board_search] board-only task complete: True` and
visible arm motion. See `docs/BOARD_SEARCH_V38_HANDOFF.md` for the current
controller/session blocker. Do not report a deploy as fully successful from
startup logs alone.

## 9. v38 rollback/provenance

```text
asset:             ai.intrinsic.aic_model_v38
installed version: ai.intrinsic.aic_model_v38.0.0.1+144f4ebff271742bd2c971ae709011b6ba279a45a7f32b31a5dc6026c91c619b
image tag:         my-solution:student-flowstate-script-v38
bundle SHA-256:    127e3338ab8ceca17029b1cb78c44450deafae7a0ca7c0db3abf6fd5182ab15a
image SHA-256:     2fb125ac454731d99b3a9974c430dd067fa8d458cb4c14859699da3393d0752b
```

To roll back the binding, delete `aic_model` and add the desired already
installed asset with the same `--name aic_model` convention.
