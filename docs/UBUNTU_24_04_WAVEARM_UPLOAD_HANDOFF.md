# Ubuntu 24.04 WaveArm Upload Handoff

This is the reproducible WaveArm-only Flowstate upload requested by Intrinsic.
Run it on a native **Ubuntu 24.04 x86_64 host**, outside Docker, for both `inbuild`
and `inctl`.

## What was reproduced locally

- Upstream source: `https://github.com/intrinsic-dev/aic`, branch `phase_1`.
- Commit used: `bf39564` (`Phase 1 scoring (#577)`).
- Host Pixi: `0.67.2`.
- Docker platform: `linux/amd64`.
- Dockerfile changes: WaveArm as the policy and `pixi install --frozen`.

The macOS build completed successfully when forced to `linux/amd64`. Its locally
generated bundle is deliberately not checked into Git (it is 5.1 GB):

```text
/Users/satya_anandh/ws_aic_phase1/images/aic_model/aic_model.bundle.tar
SHA-256: d5c3c8e80032992f1e59f232dbe54e41ce4e18811a4928a633209875b79c4f2a
```

The base image produced by that run was:

```text
my-solution:v1
sha256:571955caff164e974c245f5109f777375c694bf919ede7092ff334f485c3ed0e
architecture: amd64
```

## Build on Ubuntu

```bash
pixi self-update --version 0.67.2
pixi --version

mkdir -p ~/ws_aic_phase1/src
cd ~/ws_aic_phase1/src
git clone --branch phase_1 https://github.com/intrinsic-dev/aic.git
cd aic
```

Make exactly these two Dockerfile edits in `docker/aic_model/Dockerfile`:

```diff
-    cd /ws_aic/src/aic && pixi install --locked
+    cd /ws_aic/src/aic && pixi install --frozen
...
-CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.CheatCode"]
+CMD ["--ros-args", "-p", "policy:=aic_example_policies.ros.WaveArm"]
```

Build and record the requested verification:

```bash
cd ~/ws_aic_phase1/src/aic/docker
docker compose build model --no-cache |& tee ~/wavearm-build.log
tail -n 1 aic_model/Dockerfile
docker image ls | grep my-solution
docker image inspect my-solution:v1
```

## Bundle with the official script

```bash
cd ~/ws_aic_phase1
./src/aic/flowstate/scripts/build_aic_model.sh --container_image my-solution:v1 |& tee ~/wavearm-bundle.log
docker image ls | grep flowstate:aic_model
docker image inspect flowstate:aic_model
tree -L 3
sha256sum images/aic_model/aic_model.bundle.tar
```

## Upload from Ubuntu (outside Docker)

Create and start a new AIC Phase 1 Template solution, note its cluster ID, and
share the solution with the organization. Then:

```bash
cd ~/ws_aic_phase1
curl -fL "https://github.com/intrinsic-ai/sdk/releases/download/v1.31.20260427.1/inctl-linux-amd64" -o inctl
chmod +x inctl
./inctl auth login --org tar-2@xfa-prod-aic-us

./inctl asset install images/aic_model/aic_model.bundle.tar \
  --org tar-2@xfa-prod-aic-us \
  --cluster <CLUSTER-ID> |& tee ~/wavearm-upload.log
```

Use `asset install` (not `install asset`). Keep the terminal open until it prints
`Finished installing ...`. If the server returns `updater already finalized`, retry
the identical upload command once and retain `~/wavearm-upload.log`.

Finally, add an `aic_model` service instance in Flowstate, save the solution, run
its lifecycle configure transition, and capture the `aic_model` text logs.

## macOS limitation observed

On Apple Silicon, the unmodified Compose build defaults to `linux/arm64`, while the
Phase 1 Pixi workspace supports only `linux-64`; the exact Linux AMD64 `inctl`
binary also hangs under Docker Desktop emulation. Native Ubuntu x86_64 is therefore
the recommended environment for this diagnostic.
