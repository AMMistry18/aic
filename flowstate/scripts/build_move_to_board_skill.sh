#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIC_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=${AIC_WORKSPACE_ROOT:-$(cd -- "${AIC_ROOT}/../.." && pwd)}
SDK_ROOT="${WORKSPACE_ROOT}/src/sdk-ros"
DOCKERFILE="${AIC_ROOT}/flowstate/resources/Dockerfile.skill.cv"
OUTPUT_ROOT=${AIC_SKILL_OUTPUT_ROOT:-"${WORKSPACE_ROOT}/images"}
OUTPUT_DIR="${OUTPUT_ROOT}/move_to_board_skill"
IMAGE_TAG=${AIC_SKILL_IMAGE_TAG:-"flowstate:move-to-board"}
INBUILD_BIN=${INBUILD_BIN:-$(command -v inbuild || true)}
SKILL_NAME=move_to_board_skill
SKILL_EXECUTABLE_NAME=move_to_board_skill_main
SKILL_ASSET_ID=ai.tar2.move_to_board_skill_v1
SKILL_IMAGE_NAME=move_to_board_skill_v1

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
  --build-arg SKILL_PACKAGE=aic_perception \
  --build-arg SKILL_NAME="${SKILL_IMAGE_NAME}" \
  --build-arg SKILL_CONFIG_NAME="${SKILL_NAME}" \
  --build-arg SKILL_EXECUTABLE_NAME="${SKILL_EXECUTABLE_NAME}" \
  --build-arg SKILL_ASSET_ID="${SKILL_ASSET_ID}" \
  --build-arg SKILL_IMAGE_NAME="${SKILL_IMAGE_NAME}" \
  --tag "${IMAGE_TAG}" \
  "${WORKSPACE_ROOT}"

docker run --rm --platform linux/amd64 \
  --entrypoint /bin/bash "${IMAGE_TAG}" --noprofile --norc -c '
    source /pixi_hook.sh
    source /workspace/install/local_setup.bash
    python -c "from aic_perception import move_to_board_skill, move_to_board_skill_pb2; print(\"move_to_board imports OK\")"
    test -x "$SKILL_EXEC_PATH"
    test -f "/workspace/install/$SKILL_CONFIG"
    set +e
    timeout --kill-after=2s 8s /run_skill.sh \
      --skill_service_config_filename=/skills/skill_service_config.proto.bin \
      >/tmp/skill-smoke.log 2>&1
    status=$?
    set -e
    cat /tmp/skill-smoke.log
    case "$status" in
      124|137) ;;
      *) exit "$status" ;;
    esac
    grep -q "gRPC server listening" /tmp/skill-smoke.log
  '

docker save "${IMAGE_TAG}" --output "${OUTPUT_DIR}/${SKILL_NAME}.tar"

container_id=$(docker create --platform linux/amd64 "${IMAGE_TAG}")
trap 'docker rm -f "${container_id}" >/dev/null 2>&1 || true' EXIT
docker cp \
  "${container_id}:/opt/ros/overlay/install/share/aic_perception/${SKILL_NAME}_protos.desc" \
  "${OUTPUT_DIR}/${SKILL_NAME}_protos.desc"
docker cp \
  "${container_id}:/opt/ros/overlay/install/share/intrinsic_sdk_cmake/intrinsic_proto.desc" \
  "${OUTPUT_DIR}/intrinsic_proto.desc"
docker rm -f "${container_id}" >/dev/null
trap - EXIT

# Newer inbuild: textproto + skill/SDK descs -> binary manifest, then bundle.
"${INBUILD_BIN}" skill manifest \
  --manifest "${AIC_ROOT}/flowstate/aic_perception/${SKILL_NAME}.manifest.textproto" \
  --file_descriptor_sets "${OUTPUT_DIR}/${SKILL_NAME}_protos.desc,${OUTPUT_DIR}/intrinsic_proto.desc" \
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
