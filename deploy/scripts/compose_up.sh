#!/usr/bin/env bash
# Start face-api with the correct ONNX Runtime wheel:
#   - GPU host + nvidia-container-toolkit → RUNTIME=gpu + docker-compose.gpu.yml
#   - otherwise → RUNTIME=cpu + docker-compose.cpu.yml
#
# Docker build cannot see the host GPU, so selection happens here (standard pattern).
# App still uses DEVICE=auto at runtime (CUDA EP if present, else CPU).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

detect_runtime() {
  if [[ -n "${RUNTIME:-}" ]]; then
    echo "${RUNTIME}"
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi

  # Prefer detecting the NVIDIA container runtime / toolkit
  if docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia|nvidia'; then
    echo "gpu"
    return
  fi

  # Fallback probe: one-shot GPU container
  if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    echo "gpu"
    return
  fi

  echo "cpu"
}

RUNTIME="$(detect_runtime)"
export RUNTIME

echo "==> Detected RUNTIME=${RUNTIME}"

if [[ "${RUNTIME}" == "gpu" ]]; then
  export DEVICE="${DEVICE:-auto}"
  DOCKER_BUILDKIT=0 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build "$@"
else
  export DEVICE=cpu
  DOCKER_BUILDKIT=0 docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build "$@"
fi

echo "==> face-api starting (DEVICE=${DEVICE:-auto}). Check: curl -s http://localhost:8000/v1/health"
