from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from common import validate_content_sha256
from generate_heldout import (
    CABLE_NAME,
    FIELDS,
    GraspSampler,
    bridge_supports_cable_zero,
    generate_suite,
)


REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "aic_engine/config/base_config.yaml"
BRIDGE = REPO / "aic_bringup/config/ros_gz_bridge_config.yaml"
GRASP = Path(__file__).resolve().parents[1] / "grasp_distribution_qualification.yaml"


def _generate(output: Path, count: int = 40) -> dict:
    manifest_path = generate_suite(
        base_path=BASE,
        bridge_path=BRIDGE,
        grasp_path=GRASP,
        output_dir=output,
        suite_id="test_sfp",
        master_seed=12345,
        trial_count=count,
        shard_size=13,
    )
    return json.loads(manifest_path.read_text())


def test_generated_suite_is_sfp_only_bridged_and_6dof(tmp_path: Path) -> None:
    manifest = _generate(tmp_path)
    validate_content_sha256(manifest, "test manifest")
    assert bridge_supports_cable_zero(BRIDGE)
    assert manifest["task_scope"] == "sfp_only"
    assert manifest["bridge_strategy"]["cable_name"] == "cable_0"
    assert len(manifest["trials"]) == 40
    rail_counts = {
        rail: sum(item["target_rail"] == rail for item in manifest["trials"])
        for rail in range(5)
    }
    port_counts = {
        port: sum(item["target_port"] == port for item in manifest["trials"])
        for port in range(2)
    }
    nic_counts = {
        count: sum(item["nic_count"] == count for item in manifest["trials"])
        for count in range(1, 6)
    }
    assert rail_counts == {rail: 8 for rail in range(5)}
    assert port_counts == {0: 20, 1: 20}
    assert nic_counts == {count: 8 for count in range(1, 6)}

    dimensions = {field: set() for field in FIELDS}
    for trial_meta in manifest["trials"]:
        for field in FIELDS:
            dimensions[field].add(trial_meta["grasp_6dof"][field])
    assert all(len(values) > 1 for values in dimensions.values())

    for shard in manifest["shards"]:
        config = yaml.safe_load((tmp_path / shard["path"]).read_text())
        for trial in config["trials"].values():
            entity_names = [
                value["entity_name"]
                for value in trial["scene"]["task_board"].values()
                if isinstance(value, dict) and value.get("entity_present")
            ]
            assert len(entity_names) == len(set(entity_names))
            assert list(trial["scene"]["cables"]) == [CABLE_NAME]
            task = trial["tasks"]["task_1"]
            assert task["cable_name"] == CABLE_NAME
            assert task["plug_type"] == "sfp"
            assert task["port_type"] == "sfp"
            assert task["time_limit"] == 45
            target_rail = int(task["target_module_name"].rsplit("_", 1)[1])
            assert trial["scene"]["task_board"][f"nic_rail_{target_rail}"][
                "entity_present"
            ]


def test_generation_is_deterministic(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first", count=20)
    second = _generate(tmp_path / "second", count=20)
    assert [item["sha256"] for item in first["shards"]] == [
        item["sha256"] for item in second["shards"]
    ]
    assert first["trials"] == second["trials"]


def test_missing_cable_zero_bridge_is_rejected(tmp_path: Path) -> None:
    broken_bridge = tmp_path / "bridge.yaml"
    broken_bridge.write_text(
        yaml.safe_dump(
            [
                {
                    "ros_topic_name": "/scoring/insertion_event",
                    "gz_topic_name": "/cable_101/insertion_event",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="does not bridge"):
        generate_suite(
            base_path=BASE,
            bridge_path=broken_bridge,
            grasp_path=GRASP,
            output_dir=tmp_path / "out",
            suite_id="bad",
            master_seed=1,
            trial_count=1,
            shard_size=1,
        )


def test_empirical_measured_distribution_is_supported(tmp_path: Path) -> None:
    csv_path = tmp_path / "measured.csv"
    csv_path.write_text(
        "x,y,z,roll,pitch,yaw\n"
        "0.001,0.016,0.043,0.45,-0.49,1.34\n"
        "-0.001,0.014,0.041,0.42,-0.46,1.31\n"
    )
    config_path = tmp_path / "measured.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "fields": list(FIELDS),
                "sampling": {
                    "kind": "empirical_csv",
                    "samples_file": csv_path.name,
                    "jitter_std": [0.0] * 6,
                },
            }
        )
    )
    sampler = GraspSampler(config_path)
    import random

    sample = sampler.sample(random.Random(7))
    assert tuple(sample) == FIELDS
    assert sample["z"] in {0.041, 0.043}
