#!/usr/bin/env bash
# Build the intentionally stateless Flowstate lifecycle probe.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIC_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=${AIC_WORKSPACE_ROOT:-$(cd -- "${AIC_ROOT}/../.." && pwd)}
SDK_ROOT="${WORKSPACE_ROOT}/src/sdk-ros"
DOCKERFILE="${AIC_ROOT}/flowstate/resources/Dockerfile.skill.cv"
OUTPUT_ROOT=${AIC_SKILL_OUTPUT_ROOT:-"${WORKSPACE_ROOT}/images"}
INBUILD_BIN=${INBUILD_BIN:-$(command -v inbuild || true)}
SKILL_PACKAGE=aic_perception
SKILL_SOURCE_NAME=test_skill
SKILL_ASSET_ID=ai.tar2.test_skill_v1
SKILL_IMAGE_NAME=test_skill_v1
OUTPUT_DIR="${OUTPUT_ROOT}/${SKILL_IMAGE_NAME}"
IMAGE_TAG=${AIC_SKILL_IMAGE_TAG:-"${SKILL_PACKAGE}:${SKILL_IMAGE_NAME}"}
IMAGE_TAR="${OUTPUT_DIR}/${SKILL_IMAGE_NAME}.tar"
DESCRIPTOR_SET="${OUTPUT_DIR}/${SKILL_IMAGE_NAME}_protos.desc"
BUNDLE="${OUTPUT_DIR}/${SKILL_IMAGE_NAME}.bundle.tar"

if [[ ! -f "${WORKSPACE_ROOT}/src/aic/pixi.toml" ]]; then
  echo "ERROR: workspace must contain src/aic (set AIC_WORKSPACE_ROOT)." >&2
  exit 2
fi
if [[ ! -d "${SDK_ROOT}" ]]; then
  echo "ERROR: missing compatible Intrinsic ROS SDK checkout: ${SDK_ROOT}" >&2
  exit 2
fi
if [[ -z "${INBUILD_BIN}" || ! -x "${INBUILD_BIN}" ]]; then
  echo "ERROR: missing executable inbuild (set INBUILD_BIN)." >&2
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
  --build-arg SKILL_NAME="${SKILL_IMAGE_NAME}" \
  --build-arg SKILL_CONFIG_NAME="${SKILL_SOURCE_NAME}" \
  --build-arg SKILL_EXECUTABLE_NAME="${SKILL_SOURCE_NAME}_main" \
  --build-arg SKILL_ASSET_ID="${SKILL_ASSET_ID}" \
  --build-arg SKILL_IMAGE_NAME="${SKILL_IMAGE_NAME}" \
  --tag "${IMAGE_TAG}" \
  "${WORKSPACE_ROOT}"

actual_asset_id=$(docker image inspect \
  --format '{{ index .Config.Labels "ai.intrinsic.asset-id" }}' \
  "${IMAGE_TAG}")
actual_image_name=$(docker image inspect \
  --format '{{ index .Config.Labels "ai.intrinsic.skill-image-name" }}' \
  "${IMAGE_TAG}")
if [[ "${actual_asset_id}" != "${SKILL_ASSET_ID}" || \
      "${actual_image_name}" != "${SKILL_IMAGE_NAME}" ]]; then
  echo "ERROR: probe image identity labels do not match its manifest." >&2
  exit 2
fi

docker run --rm --platform linux/amd64 \
  --entrypoint /bin/bash "${IMAGE_TAG}" --noprofile --norc -c '
    source /pixi_hook.sh
    source /workspace/install/local_setup.bash
    python -c "from aic_perception import test_skill, test_skill_pb2; print(\"test skill imports OK\")"
    test -x "$SKILL_EXEC_PATH"
    test -f "/workspace/install/$SKILL_CONFIG"
    set +e
    timeout --kill-after=2s 8s /run_skill.sh \
      --skill_service_config_filename=/skills/skill_service_config.proto.bin \
      >/tmp/test-skill-smoke.log 2>&1
    status=$?
    set -e
    cat /tmp/test-skill-smoke.log
    case "$status" in 124|137) ;; *) exit "$status" ;; esac
    grep -q "gRPC server listening" /tmp/test-skill-smoke.log
    grep -q "TestSkill stopped" /tmp/test-skill-smoke.log
  '

docker save "${IMAGE_TAG}" --output "${IMAGE_TAR}"
container_id=$(docker create --platform linux/amd64 "${IMAGE_TAG}")
trap 'docker rm -f "${container_id}" >/dev/null 2>&1 || true' EXIT
docker cp \
  "${container_id}:/opt/ros/overlay/install/share/aic_perception/${SKILL_SOURCE_NAME}_protos.desc" \
  "${DESCRIPTOR_SET}"
docker rm -f "${container_id}" >/dev/null
trap - EXIT

"${INBUILD_BIN}" skill bundle \
  --manifest "${AIC_ROOT}/flowstate/aic_perception/${SKILL_SOURCE_NAME}.manifest.textproto" \
  --file_descriptor_set "${DESCRIPTOR_SET}" \
  --oci_image "${IMAGE_TAR}" \
  --output "${BUNDLE}"

bundle_contents=$(tar -tf "${BUNDLE}")
printf '%s\n' "${bundle_contents}"
if ! grep -Fxq "${SKILL_IMAGE_NAME}.tar" <<<"${bundle_contents}"; then
  echo "ERROR: bundle does not embed ${SKILL_IMAGE_NAME}.tar." >&2
  exit 2
fi
sha256sum "${IMAGE_TAR}" "${DESCRIPTOR_SET}" "${BUNDLE}"
echo "Bundle: ${BUNDLE}"
