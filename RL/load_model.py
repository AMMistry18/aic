"""
Active-model registry for the AIC last-inch policy.

Reads ``RL/models.toml`` and surfaces the model currently selected by the
``AIC_MODEL`` environment variable (falling back to ``[defaults].active`` if
unset, or to the baseline if active model has no weights and the fallback
flag is set).

Typical uses:

    # Print the currently active policy class name (for the aic_model CLI):
    pixi run python RL/load_model.py --class

    # Print the weight path so you can sanity-check before training runs:
    pixi run python RL/load_model.py --weights

    # List all registered models:
    pixi run python RL/load_model.py --list

    # Quick smoke test inside a pixi shell:
    pixi run python -c "from RL.load_model import active_model; print(active_model())"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

_REGISTRY_PATH = Path(__file__).parent / "models.toml"


def _load_registry(path: Path = _REGISTRY_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_active_model_name(
    env: dict | None = None,
    registry: dict | None = None,
) -> str:
    """Resolve which model should be active, honouring AIC_MODEL env var,
    [defaults].active, and the fallback-to-baseline flag if weights are
    missing."""
    env = env if env is not None else os.environ
    registry = registry if registry is not None else _load_registry()

    defaults = registry.get("defaults", {})
    fallback_name = defaults.get("active", "perception_insert_baseline")
    explicit = env.get("AIC_MODEL", "").strip() or fallback_name

    models = registry.get("models", {})
    if explicit in models:
        return explicit
    if env.get("AIC_MODEL"):
        # user-requested model is missing — warn loudly
        print(
            f"[RL/load_model] WARNING: AIC_MODEL={env['AIC_MODEL']!r} "
            f"not found in {models.keys()} — using fallback {fallback_name!r}",
            file=sys.stderr,
        )
    return fallback_name


def active_model(name: str | None = None, registry: dict | None = None) -> dict:
    """Return the model dict for the currently active (or explicitly named) model."""
    registry = registry if registry is not None else _load_registry()
    if name is None:
        name = resolve_active_model_name(registry=registry)
    models = registry.get("models", {})
    model = models.get(name)
    if model is None:
        raise KeyError(f"Model {name!r} not in registry. Known: {sorted(models)}")
    # basic shape validation: every model needs policy_class + type
    for k in ("type", "policy_class"):
        if k not in model:
            raise ValueError(f"Model {name!r} missing required field {k!r}")
    return model


def weights_for_active_model(name: str | None = None) -> str:
    """Return the weight path for the currently active model.

    If ``[defaults].fallback_to_baseline_on_missing_weights`` is true and
    the requested model has no ``weight_path``, we silently substitute the
    baseline's policy class — useful for Phase-1 submissions where you want
    the evaluator to run with rule-based control if your trained weights
    disappear."""
    registry = _load_registry()
    defaults = registry.get("defaults", {})

    if name is None:
        name = resolve_active_model_name(registry=registry)
    model = active_model(name=name, registry=registry)

    weight_path = model.get("weight_path", "") or ""
    weight_exists = bool(weight_path) and Path(weight_path).exists()

    if not weight_exists:
        if defaults.get("fallback_to_baseline_on_missing_weights", True):
            fallback_name = defaults.get("active", "perception_insert_baseline")
            if name != fallback_name and fallback_name in registry.get("models", {}):
                fb = active_model(name=fallback_name, registry=registry)
                if not fb.get("weight_path"):  # baseline has no weights anyway
                    return ""
        return ""
    return weight_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description="Active-model registry tool")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--class", dest="policy_class", action="store_true",
                   help="print the policy class name for the active model")
    g.add_argument("--weights", dest="weights", action="store_true",
                   help="print the weight path for the active model")
    g.add_argument("--list", dest="list_models", action="store_true",
                   help="print the names of all registered models")
    g.add_argument("--show", dest="show", metavar="MODEL",
                   help="print the full metadata for MODEL")
    p.add_argument("--name", action="store_true",
                   help="print only the active model name")
    args = p.parse_args()

    if args.list_models:
        registry = _load_registry()
        for n, m in registry.get("models", {}).items():
            mark = "*" if n == resolve_active_model_name(registry=registry) else " "
            print(f"{mark} {n:40s}  {m.get('type','?'):14s}  {m.get('description','')[:60]}")
        return 0

    if args.show:
        m = active_model(name=args.show)
        for k, v in m.items():
            print(f"  {k:20s}: {v}")
        return 0

    if args.name:
        print(resolve_active_model_name())
        return 0

    if args.policy_class:
        m = active_model()
        print(m["policy_class"])
        return 0

    if args.weights:
        print(weights_for_active_model())
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "resolve_active_model_name",
    "active_model",
    "weights_for_active_model",
    "_load_registry",
]


# Allow ``from RL.load_model import active_model`` style imports to be a
# no-op outside of __main__ so CLI commands don't pollute sys.argv when
# used as a library.
def _noop_when_imported():
    return None
