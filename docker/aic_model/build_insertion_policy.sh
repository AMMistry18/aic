#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIC_TOP_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

OUTPUT_IMAGE=${OUTPUT_IMAGE:-my-solution:v1}

docker build \
  --file "$SCRIPT_DIR/Dockerfile" \
  --tag "$OUTPUT_IMAGE" \
  "$AIC_TOP_DIR"

echo "Built $OUTPUT_IMAGE with the canonical SFP + SC InsertionPolicy runtime"
