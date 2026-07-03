#!/usr/bin/env python3
"""AIC-flavoured ``sdf2mjcf`` runner.

The upstream ``sdf2mjcf`` CLI does ``import sdformat`` and ``from gz.math import ...``
using the *unversioned* module names. Inside the ``aic_eval`` container those
unversioned modules either don't exist (``sdformat``) or resolve to a
*different* gz-math ABI (``gz.math`` -> gz-math9) than the one the SDFormat
bindings were linked against (``sdformat15`` -> gz-math8). Mixing the two blows
up with::

    ImportError: generic_type: type "Helpers" is already registered!

The ABI-consistent pair actually present in the container is
``sdformat15`` + ``gz.math8`` (both link ``libgz-math8``). This wrapper aliases
the unversioned names the converter imports onto that consistent pair, then
hands off to the real CLI. ``libsdformat15.so.15`` comes from the ROS overlay,
so source ``/opt/ros/kilted/setup.bash`` + ``/ws_aic/install/setup.bash`` before
running this.

Usage:
    python sdf2mjcf_aic.py <input.sdf> <output.xml>
"""
import importlib
import os
import sys

# The converter package lives in the gz-mujoco checkout; make it importable
# without a full install. Override with SDFORMAT_MJCF_SRC if it lives elsewhere.
_SRC = os.environ.get(
    "SDFORMAT_MJCF_SRC", "/ws_aic/src/gz-mujoco/sdformat_mjcf/src"
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Alias the unversioned bindings the converter imports onto the gz-math8 pair.
# Do this before importing the converter so its ``import sdformat`` / ``from
# gz.math import ...`` resolve to the consistent modules.
for _unversioned, _versioned in (("sdformat", "sdformat15"), ("gz.math", "gz.math8")):
    if _unversioned not in sys.modules:
        sys.modules[_unversioned] = importlib.import_module(_versioned)

from sdformat_mjcf.sdformat_to_mjcf.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
