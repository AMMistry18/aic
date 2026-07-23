"""Small, dependency-light helpers shared by the validation tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


_ARTIFACT_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def content_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def attach_content_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = content_sha256(result)
    return result


def validate_content_sha256(payload: dict[str, Any], label: str) -> None:
    recorded = payload.get("content_sha256")
    calculated = content_sha256(payload)
    if recorded != calculated:
        raise ValueError(
            f"{label} content_sha256 mismatch: recorded={recorded!r}, "
            f"calculated={calculated}"
        )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_artifact(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Artifact must use NAME=PATH, got {value!r}")
    name, raw_path = value.split("=", 1)
    if not _ARTIFACT_NAME.fullmatch(name):
        raise ValueError(f"Invalid artifact name {name!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Artifact does not exist or is not a file: {path}")
    return name, path


def relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=base_dir.resolve())


def docker_image_id(image_ref: str) -> str:
    process = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image_ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ValueError(f"Unable to inspect Docker image {image_ref!r}: {detail}")
    image_id = process.stdout.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError(f"Unexpected Docker image id for {image_ref!r}: {image_id!r}")
    return image_id
