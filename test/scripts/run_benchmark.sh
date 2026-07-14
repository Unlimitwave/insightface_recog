#!/usr/bin/env bash
# 时延基准：注册 + 1:N identify + 1:1 verify
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INTEGRATION="${TEST_ROOT}/integration"

BASE_URL="${BASE_URL:-http://localhost:8123}"
API_KEY="${API_KEY:-}"
IMAGE="${IMAGE:-}"
ENROLL_IMAGE="${ENROLL_IMAGE:-}"
PROBE_IMAGE="${PROBE_IMAGE:-}"
ENROLL_DIR="${ENROLL_DIR:-}"
ENROLL_STRATEGY="${ENROLL_STRATEGY:-}"
MODE="${MODE:-both}"
RUNS="${RUNS:-20}"

source "${SCRIPT_DIR}/_ensure_deps.sh"
ensure_test_deps "${TEST_ROOT}/requirements.txt"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_env.sh"
resolve_runtime_env

if [[ -z "${IMAGE}" && -z "${ENROLL_IMAGE}" && -z "${PROBE_IMAGE}" && -z "${ENROLL_DIR}" ]]; then
  if IMAGE="$(pick_preferred_face_image)"; then
    echo "Using preferred enroll/probe image: ${IMAGE}"
  fi
fi

if [[ -z "${IMAGE}" && -z "${ENROLL_IMAGE}" && -z "${ENROLL_DIR}" ]]; then
  echo "Usage: IMAGE=/path/to/face.jpg [MODE=both|identify|verify] $0"
  echo "   or: ENROLL_IMAGE=... PROBE_IMAGE=... $0"
  exit 1
fi

ARGS=(
  --base-url "${BASE_URL}"
  --mode "${MODE}"
  --warmup 2
  --runs "${RUNS}"
  --output "${TEST_ROOT}/results/latency_report.json"
)

if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi
if [[ -n "${IMAGE}" ]]; then
  ARGS+=(--image "${IMAGE}")
fi
if [[ -n "${ENROLL_IMAGE}" ]]; then
  ARGS+=(--enroll-image "${ENROLL_IMAGE}")
fi
if [[ -n "${ENROLL_DIR}" ]]; then
  ARGS+=(--enroll-dir "${ENROLL_DIR}")
fi
if [[ -n "${ENROLL_STRATEGY}" ]]; then
  ARGS+=(--enroll-strategy "${ENROLL_STRATEGY}")
fi
if [[ -n "${ENROLL_COUNT:-}" ]]; then
  ARGS+=(--enroll-count "${ENROLL_COUNT}")
fi
if [[ -n "${PROBE_IMAGE}" ]]; then
  ARGS+=(--probe-image "${PROBE_IMAGE}")
fi
if [[ "${SKIP_LIVENESS}" == "true" ]]; then
  ARGS+=(--skip-liveness)
fi

echo "Running latency benchmark against ${BASE_URL} (mode=${MODE}) ..."
echo "  SKIP_LIVENESS=${SKIP_LIVENESS}"
"${TEST_PYTHON}" "${INTEGRATION}/latency_benchmark.py" "${ARGS[@]}"
