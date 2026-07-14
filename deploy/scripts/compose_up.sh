#!/usr/bin/env bash
# Start face-api with the correct ONNX Runtime wheel and compose mode:
#   - GPU host + nvidia-container-toolkit → RUNTIME=gpu + docker-compose.gpu.yml
#   - otherwise → RUNTIME=cpu + docker-compose.cpu.yml
#   - COMPOSE_MODE=dev  (default): bind-mount app + uvicorn --reload
#   - COMPOSE_MODE=prod: image-bundled app, no reload, production liveness policy
#
# Docker build cannot see the host GPU, so selection happens here (standard pattern).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

detect_docker_platform() {
  if [[ -n "${DOCKER_PLATFORM:-}" ]]; then
    echo "${DOCKER_PLATFORM}"
    return
  fi
  case "$(uname -m)" in
    x86_64|amd64) echo "linux/amd64" ;;
    aarch64|arm64) echo "linux/arm64" ;;
    *) echo "linux/amd64" ;;
  esac
}

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

  if docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia|nvidia'; then
    echo "gpu"
    return
  fi

  if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    echo "gpu"
    return
  fi

  echo "cpu"
}

COMPOSE_MODE="${COMPOSE_MODE:-dev}"
case "${COMPOSE_MODE}" in
  dev|prod) ;;
  *)
    echo "Invalid COMPOSE_MODE=${COMPOSE_MODE} (expected dev or prod)" >&2
    exit 1
    ;;
esac

RUNTIME="$(detect_runtime)"
export RUNTIME
DOCKER_PLATFORM="$(detect_docker_platform)"
export DOCKER_PLATFORM
export DOCKER_DEFAULT_PLATFORM="${DOCKER_PLATFORM}"

echo "==> Detected RUNTIME=${RUNTIME}  DOCKER_PLATFORM=${DOCKER_PLATFORM}  COMPOSE_MODE=${COMPOSE_MODE}"

mkdir -p "${ROOT}/data"

export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

COMPOSE_FILES=(-f docker-compose.yml)
if [[ "${COMPOSE_MODE}" == "prod" ]]; then
  COMPOSE_FILES+=(-f docker-compose.prod.yml)
else
  COMPOSE_FILES+=(-f docker-compose.dev.yml)
fi

if [[ "${RUNTIME}" == "gpu" ]]; then
  export DEVICE="${DEVICE:-auto}"
  COMPOSE_FILES+=(-f docker-compose.gpu.yml)
else
  export DEVICE=cpu
  COMPOSE_FILES+=(-f docker-compose.cpu.yml)
fi

docker compose "${COMPOSE_FILES[@]}" up -d --build "$@"

echo "==> face-api starting (DEVICE=${DEVICE:-auto}, COMPOSE_MODE=${COMPOSE_MODE})."
echo "    Health: curl -s http://localhost:${PORT:-8123}/v1/health"
