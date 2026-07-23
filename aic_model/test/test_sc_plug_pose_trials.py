from copy import deepcopy

from generate_sc_plug_pose_trials import (
    RandomizationRanges,
    eligible_sc_trials,
    generate_config,
)


def _trial(*, cable_asset="sfp_sc_cable_reversed", target="sc_port_0"):
    return {
        "scene": {
            "task_board": {
                "pose": {
                    "x": 0.15,
                    "y": -0.2,
                    "z": 1.14,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 3.14,
                },
                "sc_rail_0": {"entity_present": True},
                "sc_rail_1": {"entity_present": True},
                "sc_mount_rail_0": {"entity_present": True},
                "sc_mount_rail_1": {"entity_present": True},
            },
            "cables": {
                "cable_0": {
                    "pose": {
                        "gripper_offset": {"x": 0.0, "y": 0.015, "z": 0.039},
                        "roll": 0.44,
                        "pitch": -0.48,
                        "yaw": 1.33,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": cable_asset,
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": "cable_0",
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": target,
                "time_limit": 180,
            }
        },
    }


def _template():
    regular = _trial(cable_asset="sfp_sc_cable")
    pure_sc = _trial(cable_asset="sc_cable")
    missing_port = _trial(target="sc_port_1")
    missing_port["scene"]["task_board"]["sc_rail_1"]["entity_present"] = False
    return {
        "scoring": {"topics": []},
        "trials": {
            "valid": _trial(),
            "wrong_mixed_cable_end": regular,
            "wrong_pure_sc_cable": pure_sc,
            "missing_target_port": missing_port,
        },
    }


def test_generator_accepts_only_canonical_gripped_sc_asset_and_real_port_rail():
    template = _template()
    original = deepcopy(template)
    generated, manifest = generate_config(
        template,
        trial_count=12,
        start_index=21,
        seed=99,
        ranges=RandomizationRanges(),
        time_limit=20,
    )

    assert template == original
    assert [name for name, _ in eligible_sc_trials(template)] == ["valid"]
    assert list(generated["trials"])[0] == "trial_0021"
    assert list(generated["trials"])[-1] == "trial_0032"
    assert len(manifest) == 12
    for index, (trial, row) in enumerate(zip(generated["trials"].values(), manifest)):
        task = next(iter(trial["tasks"].values()))
        cable = trial["scene"]["cables"][task["cable_name"]]
        assert len(trial["tasks"]) == 1
        assert list(trial["scene"]["cables"]) == [task["cable_name"]]
        assert cable["cable_type"] == "sfp_sc_cable_reversed"
        assert cable["attach_cable_to_gripper"] is True
        assert task["plug_name"] == "sc_tip"
        assert task["port_name"] == "sc_port_base"
        assert task["target_module_name"] in {"sc_port_0", "sc_port_1"}
        assert task["time_limit"] == 20
        expected_slot = (21 + index - 1) % 5
        assert task["cable_name"] == f"cable_{expected_slot}"
        assert row["physical_cable_slot"] == expected_slot
        assert row["plug_frame"].endswith("/sc_tip_link")
        assert "/sc_port_" in row["port_frame"]


def test_sc_generator_is_seed_reproducible():
    kwargs = {
        "trial_count": 4,
        "start_index": 1,
        "seed": 7,
        "ranges": RandomizationRanges(),
        "time_limit": 20,
    }
    first, first_manifest = generate_config(_template(), **kwargs)
    second, second_manifest = generate_config(_template(), **kwargs)
    assert first == second
    assert first_manifest == second_manifest


def test_resumed_range_matches_monolithic_generation():
    kwargs = {
        "seed": 7,
        "ranges": RandomizationRanges(),
        "time_limit": 20,
    }
    full, full_manifest = generate_config(
        _template(), trial_count=30, start_index=1, **kwargs
    )
    resumed, resumed_manifest = generate_config(
        _template(), trial_count=10, start_index=21, **kwargs
    )

    for offset, (name, trial) in enumerate(resumed["trials"].items()):
        assert trial == full["trials"][name]
        assert resumed_manifest[offset] == full_manifest[20 + offset]


def test_distractor_mount_does_not_substitute_for_missing_sc_port():
    trial = _trial()
    trial["scene"]["task_board"]["sc_rail_0"]["entity_present"] = False
    trial["scene"]["task_board"]["sc_mount_rail_0"]["entity_present"] = True
    assert eligible_sc_trials({"trials": {"distractor_only": trial}}) == []
