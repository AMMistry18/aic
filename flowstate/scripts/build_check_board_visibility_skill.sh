#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIC_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=${AIC_WORKSPACE_ROOT:-$(cd -- "${AIC_ROOT}/../.." && pwd)}
SDK_ROOT="${WORKSPACE_ROOT}/src/sdk-ros"
DOCKERFILE="${AIC_ROOT}/flowstate/resources/Dockerfile.skill.cv"
OUTPUT_ROOT=${AIC_SKILL_OUTPUT_ROOT:-"${WORKSPACE_ROOT}/images"}
OUTPUT_DIR="${OUTPUT_ROOT}/check_board_visibility_skill"
IMAGE_TAG=${AIC_SKILL_IMAGE_TAG:-"flowstate:check-board-visibility"}
INBUILD_BIN=${INBUILD_BIN:-$(command -v inbuild || true)}

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
  --build-arg SKILL_NAME=check_board_visibility_skill \
  --build-arg SKILL_EXECUTABLE_NAME=check_board_visibility_skill_main \
  --tag "${IMAGE_TAG}" \
  "${WORKSPACE_ROOT}"

docker run --rm --platform linux/amd64 \
  --entrypoint /bin/bash "${IMAGE_TAG}" --noprofile --norc -c '
    source /pixi_hook.sh
    source /workspace/install/local_setup.bash
    python -c "import cv2; from aic_perception import board_visibility, check_board_visibility_skill_pb2; print(cv2.__version__)"
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

docker save "${IMAGE_TAG}" --output "${OUTPUT_DIR}/check_board_visibility_skill.tar"

container_id=$(docker create --platform linux/amd64 "${IMAGE_TAG}")
trap 'docker rm -f "${container_id}" >/dev/null 2>&1 || true' EXIT
docker cp \
  "${container_id}:/opt/ros/overlay/install/share/aic_perception/check_board_visibility_skill_protos.desc" \
  "${OUTPUT_DIR}/check_board_visibility_skill_protos.desc"
docker rm -f "${container_id}" >/dev/null
trap - EXIT

# The generated skill descriptor already includes the transitive descriptors
# for intrinsic_proto.Pose.  Passing intrinsic_proto.desc as well duplicates
# point.proto, quaternion.proto, and pose.proto, which inbuild rejects.
"${INBUILD_BIN}" skill bundle \
  --manifest "${AIC_ROOT}/flowstate/aic_perception/check_board_visibility_skill.manifest.textproto" \
  --file_descriptor_set "${OUTPUT_DIR}/check_board_visibility_skill_protos.desc" \
  --oci_image "${OUTPUT_DIR}/check_board_visibility_skill.tar" \
  --output "${OUTPUT_DIR}/check_board_visibility_skill.bundle.tar"

sha256sum \
  "${OUTPUT_DIR}/check_board_visibility_skill.tar" \
  "${OUTPUT_DIR}/check_board_visibility_skill_protos.desc" \
  "${OUTPUT_DIR}/check_board_visibility_skill.bundle.tar"
tar -tf "${OUTPUT_DIR}/check_board_visibility_skill.bundle.tar"

echo "Bundle: ${OUTPUT_DIR}/check_board_visibility_skill.bundle.tar"
