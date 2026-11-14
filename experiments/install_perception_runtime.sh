#!/usr/bin/env bash
set -eo pipefail
# Isolated from the running simulator and its system Python packages.
runtime="${AIC_PERCEPTION_RUNTIME:-/opt/research-perception}"
evidence="${AIC_PERCEPTION_RUNTIME_EVIDENCE:-/research/.artifacts/original_perception28}"
wheel_index="${AIC_TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cpu}"
mkdir -p "$evidence"
python3 -m venv --without-pip --system-site-packages "$runtime"
if ! "$runtime/bin/python" -m pip --version >/dev/null 2>&1; then
  python3 -c "import sys, urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', sys.argv[1])" "$runtime/get-pip.py"
  sha256sum "$runtime/get-pip.py" > "$evidence/pip_bootstrap.sha256"
  "$runtime/bin/python" "$runtime/get-pip.py"
fi
"$runtime/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "$wheel_index"
"$runtime/bin/python" -m pip install numpy==1.26.4 scipy==1.12.0 opencv-python==4.11.0.86 ultralytics==8.4.48
"$runtime/bin/python" -m pip freeze > "$evidence/runtime_freeze.txt"
