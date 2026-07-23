#!/usr/bin/env python3
"""Run frozen SFP shards with Docker evaluator/model and capture 300 records.

The runner is intentionally sequential for laptop use.  Each shard gets its
own Docker network, container names, and host router port.  Startup readiness
is polled only at the requested coarse interval; trials themselves use blocking
observer/process waits rather than frequent status polling.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from common import (
    docker_image_id,
    load_json,
    sha256_file,
    validate_content_sha256,
    write_json,
)
from evaluate_gate import evaluate_gate


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _docker(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=capture,
        text=True,
    )


def build_eval_command(
    *,
    image: str,
    platform: str,
    container_name: str,
    network_name: str,
    router_port: int,
    config_path: Path,
    results_path: Path,
) -> list[str]:
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "--platform",
        platform,
        "--shm-size",
        "2g",
        "--init",
        "-p",
        f"127.0.0.1:{router_port}:7447",
        "-e",
        "AIC_RESULTS_DIR=/aic_results",
        "-v",
        f"{config_path.resolve()}:/validation/config.yaml:ro",
        "-v",
        f"{results_path.resolve()}:/aic_results",
        image,
        "gazebo_gui:=false",
        "launch_rviz:=false",
        "ground_truth:=false",
        "start_aic_engine:=true",
        "shutdown_on_aic_engine_exit:=true",
        "model_discovery_timeout_seconds:=600",
        "model_configure_timeout_seconds:=180",
        "aic_engine_config_file:=/validation/config.yaml",
    ]


def build_model_command(
    *,
    image: str,
    platform: str,
    container_name: str,
    network_name: str,
    eval_container_name: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--network",
        network_name,
        "--platform",
        platform,
        "--shm-size",
        "1g",
        "--init",
        "-e",
        "RMW_IMPLEMENTATION=rmw_zenoh_cpp",
        "-e",
        "ZENOH_ROUTER_CHECK_ATTEMPTS=-1",
        "-e",
        f"AIC_ROUTER_ADDR={eval_container_name}:7447",
        image,
    ]


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_for_router(port: int, timeout_s: float, poll_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0):
                return
        except OSError:
            time.sleep(poll_s)
    raise TimeoutError(f"Zenoh router did not open 127.0.0.1:{port}")


def _container_running(name: str) -> bool:
    process = _docker("inspect", "--format", "{{.State.Running}}", name, check=False)
    return process.returncode == 0 and process.stdout.strip() == "true"


def _save_logs(container_name: str, output: Path) -> None:
    process = _docker("logs", "--timestamps", container_name, check=False)
    output.write_text(process.stdout + process.stderr, encoding="utf-8")


def _cleanup_container(name: str) -> None:
    if _docker("inspect", name, check=False).returncode != 0:
        return
    _docker("stop", "--time", "10", name, check=False)
    _docker("rm", "-f", name, check=False)


def _observer_environment(router_port: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment["RMW_IMPLEMENTATION"] = "rmw_zenoh_cpp"
    environment["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"
    environment["ZENOH_CONFIG_OVERRIDE"] = (
        f'connect/endpoints=["tcp/127.0.0.1:{router_port}"];'
        "transport/shared_memory/enabled=false"
    )
    environment.pop("ZENOH_ROUTER_CONFIG_URI", None)
    environment.pop("ZENOH_SESSION_CONFIG_URI", None)
    return environment


def _start_observer(
    *,
    trial_id: str,
    gate_path: Path,
    image_ids: dict[str, str],
    output_path: Path,
    log_stream: BinaryIO,
    router_port: int,
    start_timeout_s: float,
) -> subprocess.Popen[bytes]:
    observer = Path(__file__).resolve().parent / "observe_trial.py"
    command = [
        sys.executable,
        str(observer),
        "--trial-id",
        trial_id,
        "--frozen-gate",
        str(gate_path.resolve()),
        "--output",
        str(output_path.resolve()),
        "--start-timeout-s",
        str(start_timeout_s),
        "--terminal-timeout-s",
        "70",
    ]
    for name, image_id in sorted(image_ids.items()):
        command.extend(("--runtime-image-id", f"{name}={image_id}"))
    return subprocess.Popen(
        command,
        env=_observer_environment(router_port),
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )


def _read_evidence(path: Path, trial_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Observer did not write evidence for {trial_id}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("trial_id") != trial_id:
        raise RuntimeError(f"Observer evidence trial mismatch for {trial_id}")
    return evidence


def _scorer_truth(scoring: dict[str, Any], trial_id: str) -> dict[str, Any]:
    if trial_id not in scoring:
        raise ValueError(f"Official scoring.yaml is missing {trial_id}")
    trial = scoring[trial_id]
    tier2 = trial.get("tier_2", {})
    categories = tier2.get("categories", {})
    insertion_force = categories.get("insertion force")
    contacts = categories.get("contacts")
    tier3 = trial.get("tier_3", {})
    if insertion_force is None or contacts is None or "score" not in tier3:
        raise ValueError(f"Official scoring.yaml is incomplete for {trial_id}")
    tier3_score = float(tier3["score"])
    tier3_message = str(tier3.get("message", ""))
    force_score = float(insertion_force["score"])
    contacts_score = float(contacts["score"])
    return {
        "source": "aic_engine/scoring.yaml",
        "tier3_score": tier3_score,
        "tier3_message": tier3_message,
        "correct_full_insertion": tier3_score == 75.0
        and tier3_message == "Cable insertion successful.",
        "wrong_port_penalty": tier3_score < 0.0
        and "Incorrect Port" in tier3_message,
        "insertion_force_score": force_score,
        "insertion_force_message": str(insertion_force.get("message", "")),
        "force_penalty": force_score < 0.0,
        "contacts_score": contacts_score,
        "contacts_message": str(contacts.get("message", "")),
        "offlimit_penalty": contacts_score < 0.0,
    }


def _attach_official_scoring(
    *,
    scoring_path: Path,
    evidence_dir: Path,
    evidence_jsonl: Path,
    trial_ids: list[str],
) -> None:
    if not scoring_path.is_file():
        raise RuntimeError(f"Evaluator did not write official scoring output: {scoring_path}")
    scoring = yaml.safe_load(scoring_path.read_text(encoding="utf-8"))
    enriched = []
    for trial_id in trial_ids:
        evidence_path = evidence_dir / f"{trial_id}.json"
        evidence = _read_evidence(evidence_path, trial_id)
        evidence["scorer_truth"] = _scorer_truth(scoring, trial_id)
        write_json(evidence_path, evidence)
        enriched.append(evidence)
    with evidence_jsonl.open("w", encoding="utf-8") as aggregate:
        for evidence in enriched:
            aggregate.write(json.dumps(evidence, sort_keys=True) + "\n")


def _run_shard(
    *,
    gate: dict[str, Any],
    gate_path: Path,
    scenario_manifest_path: Path,
    shard_index: int,
    router_port: int,
    results_dir: Path,
    platform: str,
    poll_s: float,
    router_timeout_s: float,
    observer_start_timeout_s: float,
    keep_failed: bool,
) -> Path:
    shard = gate["scenario_suite"]["shards"][shard_index]
    config_path = _resolve(scenario_manifest_path.parent, shard["path"])
    if sha256_file(config_path) != shard["sha256"]:
        raise ValueError(f"Frozen shard hash changed: {config_path}")
    if not _port_is_available(router_port):
        raise RuntimeError(f"Unique router port is already in use: {router_port}")

    short_gate = gate["content_sha256"][:8]
    run_token = f"{short_gate}-s{shard_index:02d}-{os.getpid()}"
    network_name = f"aic-v50-{run_token}"
    eval_name = f"aic-v50-eval-{run_token}"
    model_name = f"aic-v50-model-{run_token}"
    shard_dir = results_dir / f"shard_{shard_index:02d}"
    evidence_dir = shard_dir / "evidence"
    observer_dir = shard_dir / "observer_logs"
    eval_results = shard_dir / "aic_results"
    for path in (evidence_dir, observer_dir, eval_results):
        path.mkdir(parents=True, exist_ok=True)
    evidence_jsonl = shard_dir / "evidence.jsonl"
    if evidence_jsonl.exists():
        raise FileExistsError(
            f"Refusing to overwrite prior shard evidence: {evidence_jsonl}"
        )

    images = gate["runtime_images"]
    image_ids = {name: metadata["id"] for name, metadata in images.items()}
    eval_command = build_eval_command(
        image=images["evaluator"]["id"],
        platform=platform,
        container_name=eval_name,
        network_name=network_name,
        router_port=router_port,
        config_path=config_path,
        results_path=eval_results,
    )
    model_command = build_model_command(
        image=images["model"]["id"],
        platform=platform,
        container_name=model_name,
        network_name=network_name,
        eval_container_name=eval_name,
    )
    run_record = {
        "schema_version": 1,
        "gate_sha256": gate["content_sha256"],
        "shard_index": shard_index,
        "shard_sha256": shard["sha256"],
        "router_host_port": router_port,
        "network": network_name,
        "eval_container": eval_name,
        "model_container": model_name,
        "eval_command": eval_command,
        "model_command": model_command,
        "runtime_image_ids": image_ids,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(shard_dir / "run.json", run_record)

    failed = True
    active_observer: subprocess.Popen[bytes] | None = None
    observer_log: BinaryIO | None = None
    try:
        _docker("network", "create", network_name)
        subprocess.run(eval_command, check=True, capture_output=True, text=True)
        _wait_for_router(router_port, router_timeout_s, poll_s)

        trial_ids = shard["trial_ids"]
        first_id = trial_ids[0]
        first_output = evidence_dir / f"{first_id}.json"
        observer_log = (observer_dir / f"{first_id}.log").open("wb")
        active_observer = _start_observer(
            trial_id=first_id,
            gate_path=gate_path,
            image_ids=image_ids,
            output_path=first_output,
            log_stream=observer_log,
            router_port=router_port,
            start_timeout_s=observer_start_timeout_s,
        )
        subprocess.run(model_command, check=True, capture_output=True, text=True)

        with evidence_jsonl.open("w", encoding="utf-8") as aggregate:
            for index, trial_id in enumerate(trial_ids):
                if index > 0:
                    output_path = evidence_dir / f"{trial_id}.json"
                    observer_log = (observer_dir / f"{trial_id}.log").open("wb")
                    active_observer = _start_observer(
                        trial_id=trial_id,
                        gate_path=gate_path,
                        image_ids=image_ids,
                        output_path=output_path,
                        log_stream=observer_log,
                        router_port=router_port,
                        start_timeout_s=observer_start_timeout_s,
                    )
                else:
                    output_path = first_output
                return_code = active_observer.wait(
                    timeout=observer_start_timeout_s + 90.0
                )
                observer_log.close()
                observer_log = None
                if return_code != 0:
                    raise RuntimeError(
                        f"Observer failed for {trial_id} with exit code {return_code}"
                    )
                evidence = _read_evidence(output_path, trial_id)
                aggregate.write(json.dumps(evidence, sort_keys=True) + "\n")
                aggregate.flush()
                if index < len(trial_ids) - 1 and not _container_running(eval_name):
                    raise RuntimeError(
                        f"Evaluator exited before completing shard {shard_index}"
                    )

        # Engine exits itself after the final trial; block instead of polling.
        wait_result = subprocess.run(
            ["docker", "wait", eval_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if wait_result.stdout.strip() != "0":
            raise RuntimeError(
                f"Evaluator container exited with code {wait_result.stdout.strip()!r}"
            )
        _attach_official_scoring(
            scoring_path=eval_results / "scoring.yaml",
            evidence_dir=evidence_dir,
            evidence_jsonl=evidence_jsonl,
            trial_ids=trial_ids,
        )
        failed = False
        return evidence_jsonl
    finally:
        if active_observer is not None and active_observer.poll() is None:
            active_observer.terminate()
            try:
                active_observer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                active_observer.kill()
        if observer_log is not None:
            observer_log.close()
        _save_logs(eval_name, shard_dir / "evaluator.log")
        _save_logs(model_name, shard_dir / "model.log")
        if not (failed and keep_failed):
            _cleanup_container(model_name)
            _cleanup_container(eval_name)
            _docker("network", "rm", network_name, check=False)
        run_record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        run_record["completed"] = not failed
        write_json(shard_dir / "run.json", run_record)


def _load_gate_and_paths(gate_path: Path) -> tuple[dict[str, Any], Path]:
    gate = load_json(gate_path)
    validate_content_sha256(gate, "frozen gate")
    scenario_path = _resolve(gate_path.resolve().parent, gate["scenario_manifest"]["path"])
    if sha256_file(scenario_path) != gate["scenario_manifest"]["sha256"]:
        raise ValueError("Frozen scenario manifest file hash changed")
    for name, metadata in gate["runtime_images"].items():
        actual = docker_image_id(metadata["ref"])
        if actual != metadata["id"]:
            raise ValueError(
                f"Docker image drift for {name}: frozen={metadata['id']}, actual={actual}"
            )
    return gate, scenario_path


def _parse_shards(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    selected = sorted({int(token) for token in value.split(",") if token.strip()})
    if not selected or selected[0] < 0 or selected[-1] >= count:
        raise ValueError(f"Shard selection must be within 0..{count - 1}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--shards", default="all", help="all or comma-separated indices")
    parser.add_argument("--base-router-port", type=int, default=17447)
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--startup-poll-s", type=float, default=10.0)
    parser.add_argument("--router-timeout-s", type=float, default=180.0)
    parser.add_argument("--observer-start-timeout-s", type=float, default=300.0)
    parser.add_argument("--keep-failed-containers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gate, scenario_path = _load_gate_and_paths(args.gate.resolve())
    shard_count = len(gate["scenario_suite"]["shards"])
    selected = _parse_shards(args.shards, shard_count)
    if args.base_router_port <= 1024 or args.base_router_port + max(selected) > 65535:
        parser.error("Router port range must stay within 1025..65535")
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        plans = []
        for shard_index in selected:
            shard = gate["scenario_suite"]["shards"][shard_index]
            config = _resolve(scenario_path.parent, shard["path"])
            port = args.base_router_port + shard_index
            token = f"{gate['content_sha256'][:8]}-s{shard_index:02d}-PID"
            network = f"aic-v50-{token}"
            eval_name = f"aic-v50-eval-{token}"
            plans.append(
                {
                    "shard": shard_index,
                    "router_port": port,
                    "eval_command": build_eval_command(
                        image=gate["runtime_images"]["evaluator"]["id"],
                        platform=args.platform,
                        container_name=eval_name,
                        network_name=network,
                        router_port=port,
                        config_path=config,
                        results_path=results_dir / f"shard_{shard_index:02d}/aic_results",
                    ),
                    "model_command": build_model_command(
                        image=gate["runtime_images"]["model"]["id"],
                        platform=args.platform,
                        container_name=f"aic-v50-model-{token}",
                        network_name=network,
                        eval_container_name=eval_name,
                    ),
                }
            )
        print(json.dumps(plans, indent=2))
        return

    shard_outputs = []
    for shard_index in selected:
        shard_outputs.append(
            _run_shard(
                gate=gate,
                gate_path=args.gate.resolve(),
                scenario_manifest_path=scenario_path,
                shard_index=shard_index,
                router_port=args.base_router_port + shard_index,
                results_dir=results_dir,
                platform=args.platform,
                poll_s=args.startup_poll_s,
                router_timeout_s=args.router_timeout_s,
                observer_start_timeout_s=args.observer_start_timeout_s,
                keep_failed=args.keep_failed_containers,
            )
        )

    if selected == list(range(shard_count)):
        records = []
        aggregate_path = results_dir / "all_trials.jsonl"
        with aggregate_path.open("w", encoding="utf-8") as aggregate:
            for path in shard_outputs:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        records.append(record)
                        aggregate.write(json.dumps(record, sort_keys=True) + "\n")
        report = evaluate_gate(gate, records)
        write_json(results_dir / "frozen_gate_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
