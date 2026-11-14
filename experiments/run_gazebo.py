"""Prepare and run bounded, resumable AIC interface smoke trials on Linux/Kilted.

Starts a fresh evaluator, model, and Zenoh router for every scene. This validates
plumbing only. No physical-success result is inferred from process exit or score.
Full insertion/recovery ablations are deliberately gated until a close-handoff
adapter and evaluator-only contact/fixture recorder have been validated.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
PIN = "7148df9ab62b4b550b101189098f05bc0dba8444"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(config_path, out):
    config_bytes = config_path.read_bytes()
    spec = json.loads(config_bytes)
    if spec["upstream_commit"] != PIN:
        raise ValueError("Review an upstream revision change before preparing trials")
    if spec["policy"] != "aic_example_policies.ros.WaveArm":
        raise ValueError("This runner currently supports interface smoke trials only")
    if spec["purpose"] != "interface_smoke_not_insertion_evaluation":
        raise ValueError("Insertion runs need the handoff/recorder acceptance gates first")
    seeds = spec["seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Use unique nonnegative integer scene seeds")
    if spec["task_time_limit_s"] <= 0 or spec["trial_wall_timeout_s"] <= spec["task_time_limit_s"]:
        raise ValueError("Wall timeout must exceed positive task time limit")
    template_path = ROOT / spec["template"]
    template = yaml.safe_load(template_path.read_text())
    candidates = [trial for trial in template["trials"].values()
                  if len(trial["tasks"]) == 1
                  and next(iter(trial["tasks"].values()))["plug_type"] == spec["connector"]]
    if not candidates:
        raise ValueError("Official template does not have a single-task connector trial")
    template_hash = digest(template_path.read_bytes())
    identity = digest(config_bytes + template_path.read_bytes())
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["identity_sha256"] != identity:
            raise ValueError("Existing output belongs to a different experiment; use a new directory")
        for row in existing["trials"]:
            if digest((out / row["engine_config"]).read_bytes()) != row["engine_config_sha256"]:
                raise ValueError("Prepared scene was modified; use a new experiment directory")
        return existing
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"identity_sha256": identity, "spec": spec,
                "template_sha256": template_hash, "seed_scope": "scene_configuration_only",
                "simulator_seed": None, "trials": []}
    for seed in seeds:
        scene = copy.deepcopy(template)
        scene["trials"] = {"trial_1": copy.deepcopy(candidates[0])}
        trial = scene["trials"]["trial_1"]
        next(iter(trial["tasks"].values()))["time_limit"] = spec["task_time_limit_s"]
        pose = trial["scene"]["task_board"]["pose"]
        rng = random.Random(seed)
        for axis, bound in (("x", "x_m"), ("y", "y_m"), ("yaw", "yaw_rad")):
            amplitude = spec["scene_perturbation"][bound]
            if amplitude < 0:
                raise ValueError("Perturbation amplitude cannot be negative")
            pose[axis] += rng.uniform(-amplitude, amplitude)
        name = f"scene_{seed}.yaml"
        text = yaml.safe_dump(scene, sort_keys=False)
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        manifest["trials"].append({"scene_seed": seed, "engine_config": name,
                                    "engine_config_sha256": digest(text.encode())})
    write_json(manifest_path, manifest)
    return manifest


def preflight():
    errors = []
    if os.name != "posix":
        errors.append("Run inside Linux (WSL/container), not native Windows")
    if os.environ.get("ROS_DISTRO") != "kilted":
        errors.append("Source the Kilted evaluator workspace first; ROS_DISTRO must be kilted")
    if not shutil.which("ros2"):
        errors.append("ros2 is unavailable")
    else:
        for pkg in ("aic_bringup", "aic_model", "aic_example_policies", "rmw_zenoh_cpp"):
            p = subprocess.run(["ros2", "pkg", "prefix", pkg], capture_output=True, timeout=20)
            if p.returncode:
                errors.append(f"Missing ROS package: {pkg}")
        # Reject a qualification/latest image masquerading as phase_1 assets.
        p = subprocess.run(["ros2", "pkg", "prefix", "--share", "aic_assets"],
                           capture_output=True, text=True, timeout=20)
        if p.returncode:
            errors.append("Missing installed aic_assets")
        else:
            expected = ROOT / ".artifacts/upstream-phase1/aic_assets/models"
            installed = Path(p.stdout.strip()) / "models"
            for source in expected.rglob("*.sdf"):
                target = installed / source.relative_to(expected)
                if not target.exists() or digest(target.read_bytes()) != digest(source.read_bytes()):
                    errors.append(f"Installed model differs from pinned source: {source.relative_to(expected)}")
                    break
    return errors


def stop_processes(processes):
    for proc in reversed(processes):
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 8
    for proc in reversed(processes):
        try:
            proc.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_one(row, spec, out):
    root = out / f"trial_{row['scene_seed']}"
    root.mkdir()
    env = os.environ.copy()
    env.update(RMW_IMPLEMENTATION="rmw_zenoh_cpp", AIC_RESULTS_DIR=str(root / "official_results"),
               RL_INSERT_CALIB_DUMP="0")
    # Trial-level RNG is not exposed by the pinned launch file: never claim it is set.
    commands = [
        ["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
        ["ros2", "launch", "aic_bringup", "aic_gz_bringup.launch.py",
         "gazebo_gui:=false", "launch_rviz:=false", "ground_truth:=false",
         "start_aic_engine:=true", "shutdown_on_aic_engine_exit:=true",
         f"aic_engine_config_file:={out / row['engine_config']}"],
        ["ros2", "run", "aic_model", "aic_model", "--ros-args", "-p", "use_sim_time:=true",
         "-p", f"policy:={spec['policy']}"],
    ]
    processes = []
    started = time.monotonic()
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "commands": commands,
                "scene_seed": row["scene_seed"], "simulator_seed": None,
                "verified_success": None, "status": "running", "purpose": spec["purpose"]}
    metadata["ros_distro"] = os.environ.get("ROS_DISTRO")
    metadata["repository_commit"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        capture_output=True, timeout=10).stdout.strip() or None
    metadata["research_source_sha256"] = {path.name: digest(path.read_bytes())
                                          for path in (ROOT / "experiments").glob("*.py")}
    write_json(root / "run.json", metadata)
    try:
        with ExitStack() as stack:
            for name, command in zip(("router", "eval", "model"), commands):
                log = stack.enter_context((root / f"{name}.log").open("wb"))
                processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   env=env, start_new_session=True))
                time.sleep(1)
            while processes[1].poll() is None:
                if processes[0].poll() is not None or processes[2].poll() is not None:
                    metadata["status"] = "infrastructure_process_exit"
                    break
                if time.monotonic() - started > spec["trial_wall_timeout_s"]:
                    metadata["status"] = "infrastructure_wall_timeout"
                    break
                time.sleep(.25)
            else:
                metadata["status"] = ("completed_unverified" if processes[1].returncode == 0
                                       else "evaluator_error")
    except Exception as exc:
        metadata.update(status="infrastructure_exception", error=repr(exc))
    finally:
        stop_processes(processes)
        metadata["wall_time_s"] = time.monotonic() - started
        metadata["process_returncodes"] = [p.returncode for p in processes]
        write_json(root / "run.json", metadata)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--config", type=Path, default=ROOT / "experiments/configs/interface_smoke.json")
    p.add_argument("--out", type=Path, default=ROOT / ".artifacts/gazebo_interface_smoke")
    args = p.parse_args()
    out = args.out.resolve()
    manifest = prepare(args.config.resolve(), out)
    if args.action == "prepare":
        print(f"Prepared {len(manifest['trials'])} immutable scenes: {out}")
        return
    errors = preflight()
    write_json(out / "preflight.json", {"errors": errors})
    if errors:
        raise SystemExit("Preflight blocked: " + "; ".join(errors))
    for row in manifest["trials"]:
        previous = out / f"trial_{row['scene_seed']}"
        if previous.exists():
            # Keep failed/interrupted attempts as well; reruns need a new output directory.
            print(f"Preserving existing attempt: {previous}")
            continue
        result = run_one(row, manifest["spec"], out)
        print(f"Scene {row['scene_seed']}: {result['status']}")
        if result["status"] != "completed_unverified":
            raise SystemExit("Inspect infrastructure failure before starting additional scenes")


if __name__ == "__main__":
    main()
