"""Native Gazebo physics smoke check; not an AIC policy or insertion experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wsl", help="Existing WSL distribution, e.g. Ubuntu-24.04")
    p.add_argument("--out", type=Path, default=Path("experiments/evidence/gazebo_smoke"))
    args = p.parse_args()
    prefix = ["wsl", "-d", args.wsl, "--"] if args.wsl else []
    argv = prefix + ["gz", "sim", "-s", "-r", "-v", "3", "--iterations", "100", "shapes.sdf"]
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=45)
        code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        code, out, err = 124, exc.stdout or b"", exc.stderr or b""
    (args.out / "stdout.log").write_bytes(out)
    (args.out / "stderr.log").write_bytes(err)
    result = {"evidence_type": "gazebo_shapes_smoke_not_aic_baseline",
              "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "argv": argv, "returncode": code, "wall_time_s": time.monotonic() - start,
              "passed": code == 0, "insertion_success": None}
    (args.out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if code == 0 else 1)


if __name__ == "__main__":
    main()
