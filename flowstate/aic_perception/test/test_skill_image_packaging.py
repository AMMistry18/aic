"""Static guards for the Flowstate skill-container lifecycle contract."""

from pathlib import Path
import re


FLOWSTATE_DIR = Path(__file__).resolve().parents[2]
DOCKERFILE = FLOWSTATE_DIR / "resources" / "Dockerfile.skill.cv"
BUILD_SCRIPT = (
    FLOWSTATE_DIR / "scripts" / "build_check_board_visibility_skill.sh"
)
MOVE_BUILD_SCRIPT = FLOWSTATE_DIR / "scripts" / "build_move_to_board_skill.sh"
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
    assert "actual_asset_id=$(docker image inspect" in source
    assert "actual_image_name=$(docker image inspect" in source
    assert "for boot in 1 2; do" in source
    assert 'grep -q "gRPC server listening"' in source


def test_shared_dockerfile_receives_the_move_skill_identity_too():
    source = MOVE_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "SKILL_ASSET_ID=ai.tar2.move_to_board_skill_v1" in source
    assert "SKILL_IMAGE_NAME=move_to_board_skill_v1" in source
    assert '--build-arg SKILL_ASSET_ID="${SKILL_ASSET_ID}"' in source
    assert '--build-arg SKILL_IMAGE_NAME="${SKILL_IMAGE_NAME}"' in source
