#!/usr/bin/env bash
# Run Intrinsic's Linux/AMD64 inctl binary from macOS without keeping any
# important state under /private/tmp. Download the binary from the authenticated
# Flowstate console and place it at ~/.aic-flowstate/bin/inctl-linux-amd64.
set -euo pipefail

flowstate_home="${AIC_FLOWSTATE_HOME:-$HOME/.aic-flowstate}"
inctl_bin="${INCTL_BIN:-$flowstate_home/bin/inctl-linux-amd64}"
inctl_home="${INCTL_HOME:-$flowstate_home/inctl-home}"

if [[ ! -f "$inctl_bin" ]]; then
  cat >&2 <<EOF
Missing inctl binary: $inctl_bin

Download the Linux AMD64 inctl binary from:
  https://flowstate.intrinsic.ai/o/tar-2@xfa-prod-aic-us
  Set up development environment -> developer tools

Then install it persistently (not under /private/tmp):
  mkdir -p "$flowstate_home/bin"
  cp ~/Downloads/<downloaded-inctl-file> "$inctl_bin"
  chmod 755 "$inctl_bin"
EOF
  exit 2
fi

mkdir -p "$inctl_home"

docker_args=(--rm --platform linux/amd64)
if [[ -t 0 ]]; then
  docker_args+=(-i)
fi
if [[ -t 0 && -t 1 ]]; then
  docker_args+=(-t)
fi

ca_args=()
if [[ -f /etc/ssl/cert.pem ]]; then
  ca_args=(-v /etc/ssl/cert.pem:/etc/ssl/certs/ca-certificates.crt:ro)
fi

exec docker run "${docker_args[@]}" \
  "${ca_args[@]}" \
  -v "$inctl_bin:/inctl:ro" \
  -v "$inctl_home:/root" \
  debian:bookworm-slim \
  /inctl "$@"
