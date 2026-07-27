"""Static guards for the Flowstate skill-container lifecycle contract."""

from pathlib import Path
import re


FLOWSTATE_DIR = Path(__file__).resolve().parents[2]
DOCKERFILE = FLOWSTATE_DIR / "resources" / "Dockerfile.skill.cv"
BUILD_SCRIPT = (
    FLOWSTATE_DIR / "scripts" / "build_check_board_visibility_skill.sh"
)
MOVE_BUILD_SCRIPT = FLOWSTATE_DIR / "scripts" / "build_move_to_board_skill.sh"
TEST_BUILD_SCRIPT = FLOWSTATE_DIR / "scripts" / "build_test_skill.sh"
MANIFEST = (
    FLOWSTATE_DIR
    / "aic_perception"
    / "check_board_visibility_skill.manifest.textproto"
)

SKILL_ASSET_ID = "ai.tar2.check_board_visibility_skill_v4"
SKILL_IMAGE_NAME = "check_board_visibility_skill_v4"


def test_v4_image_declares_both_flowstate_identity_labels():
    source = DOCKERFILE.read_text(encoding="utf-8")
    runtime = source.split("FROM ubuntu:24.04 AS runtime", maxsplit=1)[1]

    assert "ARG SKILL_CONFIG_NAME=check_board_visibility_skill" in runtime
    assert "ARG SKILL_ASSET_ID" in runtime
    assert "ARG SKILL_IMAGE_NAME" in runtime
    assert 'test -n "${SKILL_ASSET_ID}"' in runtime
    assert 'test -n "${SKILL_IMAGE_NAME}"' in runtime
    assert 'LABEL "ai.intrinsic.asset-id"="${SKILL_ASSET_ID}"' in runtime
    assert (
        'LABEL "ai.intrinsic.skill-image-name"="${SKILL_IMAGE_NAME}"'
        in runtime
    )
    assert (
        'LABEL "ai.intrinsic.asset-id"="ai.tar2.check_board_visibility_skill"'
        not in source
    )


def test_v4_build_identity_matches_the_manifest():
    manifest = MANIFEST.read_text(encoding="utf-8")
    package = re.search(r'package:\s*"([^"]+)"', manifest).group(1)
    name = re.search(r'name:\s*"([^"]+)"', manifest).group(1)

    assert f"{package}.{name}" == SKILL_ASSET_ID
    assert name == SKILL_IMAGE_NAME


def test_build_rejects_identity_drift_and_smokes_two_cold_starts():
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert f"SKILL_ASSET_ID={SKILL_ASSET_ID}" in source
    assert f"SKILL_IMAGE_NAME={SKILL_IMAGE_NAME}" in source
    assert 'IMAGE_TAG=${AIC_SKILL_IMAGE_TAG:-"${SKILL_PACKAGE}:${SKILL_IMAGE_NAME}"}' in source
    assert '--build-arg SKILL_NAME="${SKILL_IMAGE_NAME}"' in source
    assert '--build-arg SKILL_CONFIG_NAME="${SKILL_SOURCE_NAME}"' in source
    assert 'OUTPUT_DIR="${OUTPUT_ROOT}/${SKILL_IMAGE_NAME}"' in source
    assert 'IMAGE_TAR="${OUTPUT_DIR}/${SKILL_IMAGE_NAME}.tar"' in source
    assert 'BUNDLE="${OUTPUT_DIR}/${SKILL_IMAGE_NAME}.bundle.tar"' in source
    assert '--oci_image "${IMAGE_TAR}"' in source
    assert 'grep -Fxq "${SKILL_IMAGE_NAME}.tar"' in source
    assert "actual_asset_id=$(docker image inspect" in source
    assert "actual_image_name=$(docker image inspect" in source
    assert "for boot in 1 2; do" in source
    assert 'grep -q "gRPC server listening"' in source
    assert '! grep -q "Exception in thread"' in source
    assert '! grep -q "RCLError"' in source


def test_shared_dockerfile_receives_the_move_skill_identity_too():
    source = MOVE_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "SKILL_ASSET_ID=ai.tar2.move_to_board_skill_v1" in source
    assert "SKILL_IMAGE_NAME=move_to_board_skill_v1" in source
    assert '--build-arg SKILL_NAME="${SKILL_IMAGE_NAME}"' in source
    assert '--build-arg SKILL_CONFIG_NAME="${SKILL_NAME}"' in source


def test_stateless_lifecycle_probe_has_an_independent_identity_and_clean_smoke():
    source = TEST_BUILD_SCRIPT.read_text(encoding="utf-8")
    manifest = (FLOWSTATE_DIR / "aic_perception" / "test_skill.manifest.textproto").read_text(
        encoding="utf-8"
    )
    implementation = (FLOWSTATE_DIR / "aic_perception" / "test_skill.py").read_text(
        encoding="utf-8"
    )

    assert 'name: "test_skill_v1"' in manifest
    assert "SKILL_ASSET_ID=ai.tar2.test_skill_v1" in source
    assert "SKILL_IMAGE_NAME=test_skill_v1" in source
    assert "\nimport rclpy" not in implementation
    assert "\nimport cv2" not in implementation
    assert "RobotMotion" not in implementation
    assert 'grep -q "TestSkill stopped"' in source
    assert '--build-arg SKILL_ASSET_ID="${SKILL_ASSET_ID}"' in source
    assert '--build-arg SKILL_IMAGE_NAME="${SKILL_IMAGE_NAME}"' in source
