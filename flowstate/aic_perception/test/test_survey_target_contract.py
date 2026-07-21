"""Source contracts for the Flowstate target-survey selector.

These checks intentionally avoid importing generated Intrinsic protobuf modules,
which are only present in the packaged skill image.
"""

from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROTO_PATH = PACKAGE_ROOT / "check_board_visibility_skill.proto"
MANIFEST_PATH = PACKAGE_ROOT / "check_board_visibility_skill.manifest.textproto"
WIRING_DOC_PATH = (
    PACKAGE_ROOT.parents[1] / "docs" / "V4_TARGET_SURVEY_FLOWSTATE_WIRING.md"
)


def test_survey_target_enum_is_stable_for_saved_behavior_trees():
    proto = PROTO_PATH.read_text(encoding="utf-8")

    enum_match = re.search(r"enum SurveyTarget\s*\{(?P<body>.*?)\}", proto, re.S)
    assert enum_match is not None
    entries = dict(
        (name, int(number))
        for name, number in re.findall(
            r"([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*;", enum_match.group("body")
        )
    )
    assert entries == {
        "UNSPECIFIED": 0,
        "STAGED_SFP_MODULE": 1,
        "NIC_SFP_DESTINATION": 2,
        "SC_DESTINATION_PORT": 3,
    }

    # Field number 31 must not change after a v4 node has been saved.
    assert re.search(r"\bSurveyTarget\s+survey_target\s*=\s*31\s*;", proto)


def test_manifest_exposes_backward_compatible_targeted_survey():
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")

    assert "survey_target dropdown" in manifest
    assert "UNSPECIFIED preserves legacy" in manifest
    assert "STAGED_SC_PLUG" not in manifest


def test_wiring_doc_reuses_the_saved_sfp_survey_pose_for_sc_plug_ivm():
    wiring = WIRING_DOC_PATH.read_text(encoding="utf-8")

    assert "STAGED_SC_PLUG" not in wiring
    assert "saved_sfp_survey_pose" in wiring
    assert "Parameter type | `Pose` (`intrinsic_proto.Pose`)" in wiring
    assert 'robot = world.get_kinematic_object("robot")' in wiring
    assert 'tool0 = robot.get_frame("tool0")' in wiring
    assert "node_a=world.root" in wiring
    assert 'world.get_object("root")' not in wiring
    assert "proto_conversion.pose_to_proto(root_t_tool0)" in wiring
    assert "moving_frame | `robot/tool0`" in wiring
    assert "target_frame | `root`" in wiring
    assert "SC-plug model; no new v4 search" in wiring
    assert wiring.count("Switch To Default Controller") >= 4
