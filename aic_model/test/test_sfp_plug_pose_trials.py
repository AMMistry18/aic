from copy import deepcopy

from generate_sfp_plug_pose_trials import (
    RandomizationRanges,
    eligible_sfp_trials,
    generate_config,
)


def _template():
    return {
        "scoring": {"topics": []},
        "trials": {
            "sfp_source": {
                "scene": {
                    "task_board": {
                        "pose": {
                            "x": 0.15,
                            "y": -0.2,
                            "z": 1.14,
                            "roll": 0.0,
                            "pitch": 0.0,
                            "yaw": 3.14,
                        }
                    },
                    "cables": {
                        "cable_0": {
                            "pose": {
                                "gripper_offset": {"x": 0.0, "y": 0.015, "z": 0.042},
                                "roll": 0.44,
                                "pitch": -0.48,
                                "yaw": 1.33,
                            },
                            "attach_cable_to_gripper": True,
                            "cable_type": "sfp_sc_cable",
                        }
                    },
                },
                "tasks": {
                    "task_1": {
                        "cable_name": "cable_0",
                        "plug_type": "sfp",
                        "port_type": "sfp",
                        "time_limit": 180,
                    },
                    "task_2": {
                        "cable_name": "cable_0",
                        "plug_type": "sc",
                        "port_type": "sc",
                        "time_limit": 180,
                    },
                },
            },
            "irrelevant": {
                "scene": {},
                "tasks": {"task_1": {"plug_type": "sc", "port_type": "sc"}},
            },
        },
    }


def test_generator_keeps_only_sfp_tasks_and_randomizes_all_six_grasp_axes():
    template = _template()
    original = deepcopy(template)
    ranges = RandomizationRanges()

    generated, manifest = generate_config(
        template,
        trial_count=12,
        seed=99,
        ranges=ranges,
        time_limit=20,
    )

    assert template == original
    assert len(eligible_sfp_trials(template)) == 1
    assert len(generated["trials"]) == 12
    assert len(manifest) == 12
    poses = []
    for trial in generated["trials"].values():
        assert len(trial["tasks"]) == 1
        task = next(iter(trial["tasks"].values()))
        assert task["plug_type"] == "sfp"
        assert task["port_type"] == "sfp"
        assert task["time_limit"] == 20
        poses.append(trial["scene"]["cables"]["cable_0"]["pose"])
    for key in ("x", "y", "z"):
        assert len({round(pose["gripper_offset"][key], 8) for pose in poses}) > 1
    for key in ("roll", "pitch", "yaw"):
        assert len({round(pose[key], 8) for pose in poses}) > 1


def test_generator_is_seed_reproducible():
    kwargs = {
        "trial_count": 4,
        "seed": 7,
        "ranges": RandomizationRanges(),
        "time_limit": 20,
    }
    first, first_manifest = generate_config(_template(), **kwargs)
    second, second_manifest = generate_config(_template(), **kwargs)

    assert first == second
    assert first_manifest == second_manifest
