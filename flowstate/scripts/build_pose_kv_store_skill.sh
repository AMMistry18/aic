#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIC_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=${AIC_WORKSPACE_ROOT:-$(cd -- "${AIC_ROOT}/../.." && pwd)}
SDK_ROOT="${WORKSPACE_ROOT}/src/sdk-ros"
DOCKERFILE="${AIC_ROOT}/flowstate/resources/Dockerfile.skill.cv"
OUTPUT_ROOT=${AIC_SKILL_OUTPUT_ROOT:-"${WORKSPACE_ROOT}/images"}
OUTPUT_DIR="${OUTPUT_ROOT}/pose_kv_store_skill"
IMAGE_TAG=${AIC_SKILL_IMAGE_TAG:-"flowstate:pose-kv-store"}
INBUILD_BIN=${INBUILD_BIN:-$(command -v inbuild || true)}
SKILL_PACKAGE=aic_kv_store
SKILL_NAME=pose_kv_store_skill
SKILL_EXECUTABLE_NAME=pose_kv_store_skill_main
SKILL_ASSET_ID=ai.tar2.pose_kv_store_skill_v1
SKILL_IMAGE_NAME=pose_kv_store_skill_v1

if [[ ! -f "${WORKSPACE_ROOT}/src/aic/pixi.toml" ]]; then
  echo "ERROR: workspace must contain src/aic (set AIC_WORKSPACE_ROOT)." >&2
  exit 2
fi
if [[ ! -d "${SDK_ROOT}" ]]; then
  echo "ERROR: missing compatible Intrinsic ROS SDK checkout: ${SDK_ROOT}" >&2
  exit 2
fi
if [[ -z "${INBUILD_BIN}" || ! -x "${INBUILD_BIN}" ]]; then
  echo "ERROR: missing executable inbuild (set INBUILD_BIN): ${INBUILD_BIN}" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not available." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

docker build --platform linux/amd64 \
  --file "${DOCKERFILE}" \
  --build-arg SKILL_PACKAGE="${SKILL_PACKAGE}" \
  --build-arg SKILL_NAME="${SKILL_NAME}" \
  --build-arg SKILL_EXECUTABLE_NAME="${SKILL_EXECUTABLE_NAME}" \
  --build-arg SKILL_ASSET_ID="${SKILL_ASSET_ID}" \
  --build-arg SKILL_IMAGE_NAME="${SKILL_IMAGE_NAME}" \
  --tag "${IMAGE_TAG}" \
  "${WORKSPACE_ROOT}"

# Flowstate reconciles skill images by these two labels. They must match the
# manifest id package.name and the image name the cluster expects, or a
# freshly installed pod can start while a later solution stop/start leaves the
# skill unreachable.
actual_asset_id=$(docker image inspect \
  --format '{{ index .Config.Labels "ai.intrinsic.asset-id" }}' \
  "${IMAGE_TAG}")
actual_image_name=$(docker image inspect \
  --format '{{ index .Config.Labels "ai.intrinsic.skill-image-name" }}' \
  "${IMAGE_TAG}")
if [[ "${actual_asset_id}" != "${SKILL_ASSET_ID}" ]]; then
  echo "ERROR: skill image asset-id label is '${actual_asset_id}', expected '${SKILL_ASSET_ID}'." >&2
  exit 2
fi
if [[ "${actual_image_name}" != "${SKILL_IMAGE_NAME}" ]]; then
  echo "ERROR: skill image name label is '${actual_image_name}', expected '${SKILL_IMAGE_NAME}'." >&2
  exit 2
fi

# A C++ skill cannot reach "Skill service listening" offline: the generated
# entrypoint blocks on the in-cluster world, geometry, motion-planner and
# skill-registry addresses first. The cold-start check therefore proves the
# packaging instead: every shared library resolves, and the binary starts and
# loads this skill's own config before it fails on those addresses.
docker run --rm --platform linux/amd64 \
  --entrypoint /bin/bash "${IMAGE_TAG}" --noprofile --norc -c '
    set -eo pipefail
    source /pixi_hook.sh
    source /workspace/install/local_setup.bash
    test -x "$SKILL_EXEC_PATH"
    test -f "/workspace/install/$SKILL_CONFIG"
    if ldd "$SKILL_EXEC_PATH" | grep -q "not found"; then
      echo "ERROR: unresolved shared libraries:" >&2
      ldd "$SKILL_EXEC_PATH" | grep "not found" >&2
      exit 1
    fi
    set +e
    timeout --kill-after=2s 40s "$SKILL_EXEC_PATH" \
      --skill_service_config_filename=/skills/skill_service_config.proto.bin \
      --grpc_connect_timeout_secs=5 \
      >/tmp/skill-smoke.log 2>&1
    status=$?
    set -e
    cat /tmp/skill-smoke.log
    case "$status" in
      126|127|139) echo "ERROR: skill binary failed to start ($status)" >&2; exit 1 ;;
    esac
    if grep -q "error while loading shared libraries" /tmp/skill-smoke.log; then
      echo "ERROR: skill binary is missing a shared library" >&2
      exit 1
    fi
    grep -q "pose_kv_store_skill_v1" /tmp/skill-smoke.log
  '

docker save "${IMAGE_TAG}" --output "${OUTPUT_DIR}/${SKILL_NAME}.tar"

container_id=$(docker create --platform linux/amd64 "${IMAGE_TAG}")
trap 'docker rm -f "${container_id}" >/dev/null 2>&1 || true' EXIT
docker cp \
  "${container_id}:/opt/ros/overlay/install/share/${SKILL_PACKAGE}/${SKILL_NAME}_protos.desc" \
  "${OUTPUT_DIR}/${SKILL_NAME}_protos.desc"
docker cp \
  "${container_id}:/opt/ros/overlay/install/share/intrinsic_sdk_cmake/intrinsic_proto.desc" \
  "${OUTPUT_DIR}/intrinsic_proto.desc"
docker rm -f "${container_id}" >/dev/null
trap - EXIT

# Skill desc already includes point/pose/quaternion; intrinsic_proto.desc also
# has them. Passing both raw to inbuild duplicates point.proto. Merge by file
# name so the platform skill services (e.g. Projector) remain available.
# Prefer AIC pixi python (system python3 often lacks protobuf).
PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${AIC_ROOT}/.pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN="${AIC_ROOT}/.pixi/envs/default/bin/python"
  else
    PYTHON_BIN=$(command -v python3)
  fi
fi
export OUTPUT_DIR SKILL_NAME
"${PYTHON_BIN}" - <<'PY'
from google.protobuf import descriptor_pb2
from pathlib import Path
import os
out = Path(os.environ["OUTPUT_DIR"])
skill = os.environ["SKILL_NAME"]
merged = descriptor_pb2.FileDescriptorSet()
seen = set()
for name in [f"{skill}_protos.desc", "intrinsic_proto.desc"]:
  fds = descriptor_pb2.FileDescriptorSet()
  fds.ParseFromString((out / name).read_bytes())
  for f in fds.file:
    if f.name in seen:
      continue
    seen.add(f.name)
    merged.file.append(f)
(out / f"{skill}_protos.merged.desc").write_bytes(merged.SerializeToString())
PY

"${INBUILD_BIN}" skill manifest \
  --manifest "${AIC_ROOT}/flowstate/${SKILL_PACKAGE}/${SKILL_NAME}.manifest.textproto" \
  --file_descriptor_sets "${OUTPUT_DIR}/${SKILL_NAME}_protos.merged.desc" \
  --file_descriptor_set_out "${OUTPUT_DIR}/${SKILL_NAME}_protos.augmented.desc" \
  --output "${OUTPUT_DIR}/${SKILL_NAME}.manifest.bin"

"${INBUILD_BIN}" skill bundle \
  --augmented_file_descriptor_set "${OUTPUT_DIR}/${SKILL_NAME}_protos.augmented.desc" \
  --augmented_manifest "${OUTPUT_DIR}/${SKILL_NAME}.manifest.bin" \
  --oci_image "${OUTPUT_DIR}/${SKILL_NAME}.tar" \
  --output "${OUTPUT_DIR}/${SKILL_NAME}.bundle.tar"

sha256sum \
  "${OUTPUT_DIR}/${SKILL_NAME}.tar" \
  "${OUTPUT_DIR}/${SKILL_NAME}_protos.desc" \
  "${OUTPUT_DIR}/${SKILL_NAME}.bundle.tar"
tar -tf "${OUTPUT_DIR}/${SKILL_NAME}.bundle.tar"

echo "Bundle: ${OUTPUT_DIR}/${SKILL_NAME}.bundle.tar"
