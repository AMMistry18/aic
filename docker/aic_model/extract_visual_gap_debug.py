#!/usr/bin/env python3
"""Reconstruct visual-gap diagnostic JPEGs from aic_model service logs."""

from __future__ import annotations

import argparse
import base64
import re
import sys
from collections import defaultdict
from pathlib import Path


IMAGE_RE = re.compile(
    r"\[visual-gap-debug-image\] "
    r"id=(?P<image_id>\S+) camera=(?P<camera>\S+) step=(?P<step>\d+) "
    r"part=(?P<part>\d+)/(?P<total>\d+) data=(?P<data>[A-Za-z0-9+/=]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log", nargs="?", type=Path,
        help="saved inctl log file; stdin is used when omitted")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("visual_gap_debug"))
    args = parser.parse_args()

    text = args.log.read_text(errors="replace") if args.log else sys.stdin.read()
    images: dict[tuple[str, str, int], dict[int, str]] = defaultdict(dict)
    totals: dict[tuple[str, str, int], int] = {}
    for match in IMAGE_RE.finditer(text):
        key = (
            match.group("image_id"),
            match.group("camera"),
            int(match.group("step")),
        )
        images[key][int(match.group("part"))] = match.group("data")
        totals[key] = int(match.group("total"))

    if not images:
        print("no visual-gap diagnostic images found", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for (image_id, camera, step), parts in sorted(images.items()):
        total = totals[(image_id, camera, step)]
        missing = [index for index in range(1, total + 1) if index not in parts]
        if missing:
            print(
                f"skipping incomplete {camera} step {step}: missing {missing}",
                file=sys.stderr,
            )
            continue
        payload = base64.b64decode("".join(parts[index] for index in range(1, total + 1)))
        if not payload.startswith(b"\xff\xd8"):
            print(f"skipping invalid JPEG for {camera} step {step}", file=sys.stderr)
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", image_id)
        output = args.output_dir / f"{safe_id}_{camera}_step{step}.jpg"
        output.write_bytes(payload)
        print(output)
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
