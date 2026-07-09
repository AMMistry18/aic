"""Generated-asset manifest loading and fail-fast validation."""

from __future__ import annotations

import json
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PACKAGE_ROOT / "usd" / "asset_manifest.json"
DEFAULT_RESET_BANK = PACKAGE_ROOT / "assets" / "reset_bank.npz"


def manifest_path() -> Path:
    return Path(os.environ.get("AIC_ISAAC_ASSET_MANIFEST", DEFAULT_MANIFEST)).expanduser().resolve()


def load_asset_manifest() -> dict:
    path = manifest_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Isaac asset manifest not found: {path}. Run "
            "scripts/import_mjcf_to_usd.py with Isaac Sim's python first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "robot_usd",
        "world_usd",
        "robot_tool_relpath",
        "robot_tcp_relpath",
        "robot_tcp_offset_pos",
        "robot_tcp_offset_quat_wxyz",
        "world_plug_relpath",
        "world_tip_relpath",
        "world_tail_relpath",
        "world_target_relpath",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"asset manifest {path} is missing keys: {missing}")
    for key in ("robot_usd", "world_usd"):
        candidate = Path(data[key])
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        data[key] = str(candidate.resolve())
    return data


def reset_bank_path() -> Path:
    return Path(os.environ.get("AIC_ISAAC_RESET_BANK", DEFAULT_RESET_BANK)).expanduser().resolve()


__all__ = ["load_asset_manifest", "manifest_path", "reset_bank_path"]
