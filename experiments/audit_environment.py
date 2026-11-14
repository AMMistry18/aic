"""Capture bounded, read-only environment probes without installing packages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def probe(argv, timeout=25):
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout)
        # wsl.exe uses UTF-16 for its inventory on this Windows host.
        def decode(data):
            return data.decode("utf-16-le" if b"\0" in data[:100] else "utf-8", errors="replace").strip()
        return {"argv": argv, "returncode": p.returncode,
                "stdout": decode(p.stdout), "stderr": decode(p.stderr)}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "experiments/evidence/environment.json")
    args = p.parse_args()
    commands = [["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                ["git", "-C", str(ROOT), "status", "--short"],
                ["docker", "info", "--format", "{{.OSType}}"],
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                [sys.executable, "-m", "pip", "show", "numpy", "opencv-python-headless", "matplotlib", "pytest"]]
    if os.name == "nt":
        commands += [["wsl", "--list", "--verbose"],
                     ["wsl", "-d", "Ubuntu-24.04", "--", "gz", "sim", "--versions"],
                     ["wsl", "-d", "Ubuntu-24.04", "--", "ls", "/opt/ros"],
                     ["wsl", "-d", "Ubuntu-24.04", "--", "free", "-h"]]
    else:
        commands += [["gz", "sim", "--versions"], ["ros2", "pkg", "prefix", "aic_bringup"]]
    checkpoint_root = ROOT / "aic_example_policies/aic_example_policies/ros/weights"
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoint_root.glob("*.pt")}
    result = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
              "tools": {name: shutil.which(name) for name in ("docker", "pixi", "ros2", "gz", "wsl")},
              "checkpoint_sha256": hashes, "probes": [probe(c) for c in commands]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
