from __future__ import annotations

from pathlib import Path

import pytest

from run_docker_gate import (
    _observer_environment,
    _parse_shards,
    build_eval_command,
    build_model_command,
    _scorer_truth,
)


def test_eval_and_model_commands_use_isolated_network_and_router_port(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("trials: {}\n")
    results = tmp_path / "results"
    results.mkdir()
    eval_command = build_eval_command(
        image="eval@sha256:test",
        platform="linux/amd64",
        container_name="eval-unique",
        network_name="net-unique",
        router_port=18447,
        config_path=config,
        results_path=results,
    )
    assert "127.0.0.1:18447:7447" in eval_command
    assert f"{config.resolve()}:/validation/config.yaml:ro" in eval_command
    assert "aic_engine_config_file:=/validation/config.yaml" in eval_command
    assert "shutdown_on_aic_engine_exit:=true" in eval_command

    model_command = build_model_command(
        image="model@sha256:test",
        platform="linux/amd64",
        container_name="model-unique",
        network_name="net-unique",
        eval_container_name="eval-unique",
    )
    assert "AIC_ROUTER_ADDR=eval-unique:7447" in model_command
    assert model_command[model_command.index("--network") + 1] == "net-unique"


def test_observer_uses_only_the_unique_tcp_router() -> None:
    environment = _observer_environment(19447)
    assert 'tcp/127.0.0.1:19447' in environment["ZENOH_CONFIG_OVERRIDE"]
    assert "shared_memory/enabled=false" in environment["ZENOH_CONFIG_OVERRIDE"]
    assert "ZENOH_ROUTER_CONFIG_URI" not in environment
    assert "ZENOH_SESSION_CONFIG_URI" not in environment


def test_shard_selection_is_exact_and_bounded() -> None:
    assert _parse_shards("all", 10) == list(range(10))
    assert _parse_shards("7,2,7", 10) == [2, 7]
    with pytest.raises(ValueError):
        _parse_shards("10", 10)


def test_official_scoring_truth_requires_full_event_and_zero_penalties() -> None:
    scoring = {
        "trial_001": {
            "tier_2": {
                "categories": {
                    "insertion force": {"score": 0, "message": "No excessive force detected"},
                    "contacts": {"score": 0, "message": "No contact detected."},
                }
            },
            "tier_3": {"score": 75, "message": "Cable insertion successful."},
        }
    }
    truth = _scorer_truth(scoring, "trial_001")
    assert truth["correct_full_insertion"]
    assert not truth["force_penalty"]
    assert not truth["offlimit_penalty"]
