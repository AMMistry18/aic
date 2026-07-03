"""
Run caching for the last-inch residual-SAC trainer.

A "run" is uniquely identified by a SHA256 hash of:
    - port type
    - all reward weights (REWARD_SPEC §3, §7b)
    - all env hyperparameters (start-pose distribution, dt, etc.)
    - the observation spec (image H/W/n_cams)
    - the training hyperparameters (lr, batch size, buffer size, warmup)
    - the git rev of the RL folder (so any code change invalidates)
    - the contents of the MJCF XML (so scene changes invalidate)

`cache_hit(out_dir, key)` returns the existing checkpoint path if a prior
run with the same key has produced a model.zip in `out_dir`. Otherwise
None. This lets train.py skip re-training when nothing has changed.

`write_config(out_dir, cfg_dict, key)` freezes the config + key into the
output dir so future runs can be diffed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional


def _rl_root() -> Path:
    return Path(__file__).resolve().parent


def _git_rev(root: Path) -> str:
    """Best-effort git rev. Returns 'unknown-<ts>' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2.0,
        ).decode().strip()
    except Exception:
        return f"unknown-{os.environ.get('AIC_NO_GIT', '1')}"


def _mjcf_xml() -> str:
    """Return the current MJCF XML content for both port types so the
    cache key tracks per-port scene changes.
    """
    try:
        from RL.env import _PORT_GEOMETRY, _build_mjcf  # type: ignore
        return (_build_mjcf("sc", "", 32) + "::" + _build_mjcf("sfp", "", 32) +
                "::" + str(sorted(_PORT_GEOMETRY.items())))
    except Exception:
        return ""


def make_cache_key(cfg: Mapping[str, Any]) -> str:
    """Compute a stable SHA256 hash over the config dict.

    The dict should contain only JSON-friendly primitives. The order of
    keys matters for reproducibility, so we sort on dump.
    """
    enriched = dict(cfg)
    enriched["_git_rev"] = _git_rev(_rl_root())
    enriched["_mjcf_xml"] = _mjcf_xml()
    payload = json.dumps(enriched, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_hit(out_dir: Path, key: str) -> Optional[Path]:
    """If `out_dir/cache_key.txt` matches `key` AND `model.zip` exists,
    return its path. Otherwise None.
    """
    key_file = out_dir / "cache_key.txt"
    model = out_dir / "model.zip"
    if not key_file.exists() or not model.exists():
        return None
    try:
        existing = key_file.read_text().strip()
    except Exception:
        return None
    return model if existing == key else None


def write_config(out_dir: Path, cfg: Mapping[str, Any], key: str) -> None:
    """Freeze the config + key into the output dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache_key.txt").write_text(key)
    (out_dir / "config.json").write_text(
        json.dumps(dict(cfg), indent=2, sort_keys=True, default=str)
    )


def read_config(out_dir: Path) -> Optional[dict]:
    """Read a previously-frozen config (or None if it doesn't exist)."""
    p = out_dir / "config.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


__all__ = [
    "make_cache_key",
    "cache_hit",
    "write_config",
    "read_config",
]