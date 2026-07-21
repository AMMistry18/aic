# Flowstate v4 skill upload handoff

This document describes the exact workflow currently used to build, package,
install, and verify `ai.tar2.check_board_visibility_skill_v4` in the TAR-2
Flowstate solution.

## Current deployment

As of 2026-07-21, Flowstate reports:

```text
organization: tar-2@xfa-prod-aic-us
solution:     9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH
cluster:      vmp-efe2-cf8sn65n
state:        RUNNING_IN_SIM
asset:        ai.tar2.check_board_visibility_skill_v4.0.0.1+0daadf2edf3d773a20714a45a7119702cca3b23243bad23e7de6ab96c10824b3
image tag:    flowstate:check-board-visibility-v4-r39-one-crossing-sfp-guard
image ID:     sha256:a088163701b2ad74449adff4eee0c8b3cb7d323cf7cf75e7ba1a029537edc966
image tar:    e8e9675ddcdf46d47b691aa2f9d6fda2b26c8f392f48a16d63fbb0255575ae72
descriptor:   f030faf3d3c09a825c0520988416cfd59bb37f1debcea50354bd280c18603ccf
bundle:       26df83063a2d63a452a1999ed9c9a7547e440b96e032f356533461f464300f65
```

The long `+...` suffix is assigned from the installed bundle content. Do not
try to choose or reuse that suffix manually.

## Workspace layout

The Git repository is:

```text
C:\Users\anshu\College\aic\aic
```

Docker build context and local bundle artifacts live one level above it:

```text
C:\Users\anshu\College\aic
└── .flowstate-build\images\check_board_visibility_skill
```

`.flowstate-build` intentionally remains outside the Git repository because
each image tar/bundle is about 435 MB.

The deployed image overlays these current source files onto the last verified
complete base image:

```text
aic/flowstate/aic_perception/aic_perception/board_visibility.py
aic/flowstate/aic_perception/aic_perception/viewpoint_search.py
aic/flowstate/aic_perception/check_board_visibility_skill.py
```

The verified base tag is:

```text
flowstate:check-board-visibility-v4-r32-component-authority
```

Do not silently substitute an older policy image as the base. The base carries
the ROS/Intrinsic runtime, generated protos, skill service configuration, and
entrypoint; the overlay supplies the current Python controller.

## Prerequisites

- Docker Desktop is running in Linux-container mode.
- The verified r32 base image exists locally.
- An `inctl` toolbox container is running and authenticated. On this machine
  it is currently named `focused_lovelace`; confirm with `docker ps` instead of
  assuming the name will never change.
- The solution is already running in simulation.
- The previous staging directory contains the compatible binary descriptor and
  `skill_manifest.binpb`.

Never place API tokens, Flowstate credentials, or Docker registry credentials
in this repository or in this document.

## 1. Validate the source

Run from the Git repository:

```powershell
Set-Location -LiteralPath 'C:\Users\anshu\College\aic\aic'
$env:PYTHONPATH = (Resolve-Path 'flowstate/aic_perception').Path

python -m pytest flowstate/aic_perception/test -q
python -m py_compile `
  flowstate/aic_perception/aic_perception/board_visibility.py `
  flowstate/aic_perception/aic_perception/viewpoint_search.py `
  flowstate/aic_perception/check_board_visibility_skill.py
git diff --check
```

The r39 build passed 184 tests. Do not upload a revision with a failing test,
compile error, or whitespace/conflict-marker error.

## 2. Create the overlay Dockerfile

Choose a new monotonically increasing revision label, for example `r40`, and a
short descriptive suffix. Create the Dockerfile under:

```text
C:\Users\anshu\College\aic\.flowstate-build\images\check_board_visibility_skill
```

Use this structure:

```dockerfile
FROM flowstate:check-board-visibility-v4-r32-component-authority

COPY aic/flowstate/aic_perception/aic_perception/board_visibility.py /workspace/install/lib/python3.12/site-packages/aic_perception/board_visibility.py
COPY aic/flowstate/aic_perception/aic_perception/viewpoint_search.py /workspace/install/lib/python3.12/site-packages/aic_perception/viewpoint_search.py
COPY --chmod=0755 aic/flowstate/aic_perception/check_board_visibility_skill.py /workspace/install/lib/python3.12/site-packages/aic_perception/check_board_visibility_skill.py
COPY --chmod=0755 aic/flowstate/aic_perception/check_board_visibility_skill.py /workspace/install/lib/aic_perception/check_board_visibility_skill_main

RUN /workspace/.pixi/envs/skill-runtime/bin/python -m py_compile \
      /workspace/install/lib/python3.12/site-packages/aic_perception/board_visibility.py \
      /workspace/install/lib/python3.12/site-packages/aic_perception/viewpoint_search.py \
      /workspace/install/lib/python3.12/site-packages/aic_perception/check_board_visibility_skill.py

LABEL "ai.intrinsic.asset-id"="ai.tar2.check_board_visibility_skill_v4"
ENV SKILL_NAME=check_board_visibility_skill_v4

CMD ["/skills/skill_service", "--skill_service_config_filename=/skills/skill_service_config.proto.bin"]
```

If the proto, manifest, generated protobuf, installed dependencies, or skill
service configuration changes, a three-file overlay is no longer sufficient.
Rebuild the complete skill base/bundle instead of assuming the old base remains
compatible.

## 3. Build and smoke-test the image

Run from the parent directory because the Dockerfile copies paths beginning in
`aic/`:

```powershell
Set-Location -LiteralPath 'C:\Users\anshu\College\aic'

$revision = 'r40-example-change'
$dockerfile = ".flowstate-build/images/check_board_visibility_skill/Dockerfile.v4-$revision"
$image = "flowstate:check-board-visibility-v4-$revision"

docker build -f $dockerfile -t $image .
docker image inspect $image --format '{{.Id}}'
```

Smoke-test imports using the runtime Python inside the final image:

```powershell
docker run --rm `
  --entrypoint /workspace/.pixi/envs/skill-runtime/bin/python `
  $image `
  -c "from aic_perception.board_visibility import survey_view_requirements; from aic_perception.viewpoint_search import AdaptiveViewpointPlanner; print('policy_import_ok')"
```

For parameter changes, print the relevant requirement values in this smoke
command so the final container—not merely the working tree—proves what will be
deployed.

## 4. Assemble the flat bundle

The working binary descriptor and manifest are carried forward from the most
recent compatible staging directory. For r39 that directory is:

```text
.flowstate-build/images/check_board_visibility_skill/stage_r39_one_crossing_sfp_guard
```

Create a new directory rather than modifying a previous release in place:

```powershell
$assetDir = '.flowstate-build/images/check_board_visibility_skill'
$previousStage = "$assetDir/stage_r39_one_crossing_sfp_guard"
$stage = "$assetDir/stage_$revision"
$bundle = "$assetDir/check_board_visibility_skill_v4_$revision.bundle.tar"

if (Test-Path -LiteralPath $stage) {
  throw "Refusing to overwrite existing staging directory: $stage"
}

New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -LiteralPath `
  "$previousStage/descriptors-transitive-descriptor-set.proto.bin" `
  -Destination $stage
Copy-Item -LiteralPath `
  "$previousStage/skill_manifest.binpb" `
  -Destination $stage
```

Save the new image using this exact historical archive filename:

```powershell
$imageArchive = 'check_board_visibility_skill_v4_r32_component_authority.tar'
docker save $image -o "$stage/$imageArchive"
```

The filename is intentionally still `...r32_component_authority.tar`. The
compatible binary skill manifest/bundle contract expects that archive name;
the tar contents are the newly built image. Renaming only the archive without
regenerating the manifest can make installation fail or select no image.

Create a flat tar with exactly three root entries:

```powershell
tar -cf $bundle -C $stage `
  descriptors-transitive-descriptor-set.proto.bin `
  $imageArchive `
  skill_manifest.binpb

tar -tf $bundle
```

Expected listing:

```text
descriptors-transitive-descriptor-set.proto.bin
check_board_visibility_skill_v4_r32_component_authority.tar
skill_manifest.binpb
```

Do not wrap these entries in an extra staging-directory level.

## 5. Upload and install with `inctl`

Confirm the toolbox container name:

```powershell
docker ps --format '{{.Names}}  {{.Image}}'
$inctlContainer = 'focused_lovelace'
```

Copy the bundle into it, then perform an in-place compatible update:

```powershell
$bundleName = Split-Path -Leaf $bundle
docker cp $bundle "${inctlContainer}:/tmp/$bundleName"

docker exec $inctlContainer inctl skill install "/tmp/$bundleName" `
  --org tar-2@xfa-prod-aic-us `
  --solution 9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH `
  --policy update_compatible `
  --timeout 180s
```

`update_compatible` updates the existing v4 skill ID. Do not use a new asset
ID for routine policy revisions because the saved Flowstate nodes reference
`ai.tar2.check_board_visibility_skill_v4`.

## 6. Verify the actual installed asset

Do not treat an upload log alone as success. Query the solution:

```powershell
docker exec $inctlContainer inctl skill list `
  --org tar-2@xfa-prod-aic-us `
  --solution 9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH |
  Select-String 'check_board_visibility_skill_v4'

docker exec $inctlContainer inctl solution list `
  --org tar-2@xfa-prod-aic-us |
  Select-String '9b9e6784-583b-4d03-905e-98735b9aaa40_BRANCH'
```

The first command must show a new exact asset version, and the second must show
`RUNNING_IN_SIM` on `vmp-efe2-cf8sn65n`.

Record reproducibility hashes:

```powershell
$hashTargets = @(
  $bundle,
  "$stage/$imageArchive",
  "$stage/descriptors-transitive-descriptor-set.proto.bin"
)

foreach ($path in $hashTargets) {
  $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $path
  Write-Output "$($hash.Hash.ToLower())  $path"
}
```

Update `docs/BOARD_SEARCH_POLICY_HANDOFF.md` with the asset version, Docker
tag/ID, hashes, behavioral change, test count, and verification result.

## Flowstate process contract

The required runtime order is:

```text
check_board_visibility_skill_v4
  -> Switch To Default Controller
  -> require result.success == true AND result.done == true
  -> IVM / Move Robot
```

The skill intentionally returns an unsuccessful normal result after a deadline
or recoverable search failure so the next node can release the AIC controller.
Throwing a skill error before `Switch To Default Controller` previously left
the bridge's ICON session holding `arm`, causing `Part: 'arm' is already in
use`. Therefore the controller switch must run before the result condition.

## Common failure modes

- **New code is not visible:** the Docker build was run from the repository
  rather than its parent, or Docker reused the wrong tag. Inspect the final
  image and run the smoke import.
- **Install rejects the bundle:** verify the flat three-entry tar layout and
  preserve the manifest-compatible image archive filename.
- **Flowstate still lists the old hash:** wait for `inctl skill install` to
  finish, confirm the correct solution/org, and query `inctl skill list` again.
- **Skill node has no new parameter:** source/proto changed but the old binary
  descriptor/manifest was reused. Rebuild the complete skill package; an
  overlay cannot update the interface contract.
- **Move Robot reports `arm` already in use:** ensure `Switch To Default
  Controller` executes immediately after every normal v4 result.
- **Policy reaches IVM after unsuccessful search:** place the `success && done`
  condition after controller cleanup, not before it and not inside IVM.

## Release checklist

- [ ] Full perception test suite passes.
- [ ] Python byte compilation passes locally and in the image.
- [ ] Docker smoke output proves the intended policy values.
- [ ] Bundle contains exactly the descriptor, image tar, and manifest.
- [ ] `inctl skill install --policy update_compatible` completes.
- [ ] `inctl skill list` reports the new exact asset hash.
- [ ] The target solution remains `RUNNING_IN_SIM` on the expected cluster.
- [ ] Bundle/image/descriptor SHA-256 hashes are recorded.
- [ ] `BOARD_SEARCH_POLICY_HANDOFF.md` is updated.
- [ ] Flowstate tree retains controller cleanup before the success gate.
