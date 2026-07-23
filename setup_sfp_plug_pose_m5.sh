#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pose_venv="${AIC_SFP_POSE_VENV:-${HOME}/.venvs/aic-sfp-plug-pose-m5}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first, then rerun this script." >&2
  exit 1
fi

if [[ ! -x "${pose_venv}/bin/python" ]]; then
  uv venv "${pose_venv}" --python 3.12
fi
uv pip install \
  --python "${pose_venv}/bin/python" \
  --requirement "${repo_dir}/requirements-sfp-plug-pose-m5.txt"

"${pose_venv}/bin/python" - <<'PY'
from ultralytics import YOLO  # Load OpenCV before torch in this environment.
import torch

assert torch.backends.mps.is_built(), "PyTorch was not built with MPS support"
assert torch.backends.mps.is_available(), "MPS is not available on this Mac"
value = (torch.ones((8, 8), device="mps") @ torch.ones((8, 8), device="mps")).sum()
assert float(value.cpu()) == 512.0
print(f"SFP pose environment ready: torch={torch.__version__}, device=mps")
PY

echo "Python: ${pose_venv}/bin/python"
