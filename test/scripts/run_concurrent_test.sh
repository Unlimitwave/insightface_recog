#!/usr/bin/env bash
# 并发 / QPS 基准测试
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "$(dirname "${SCRIPT_DIR}")" && pwd)"
INTEGRATION="${TEST_ROOT}/integration"

BASE_URL="${BASE_URL:-http://localhost:8123}"
API_KEY="${API_KEY:-}"
ENDPOINT="${ENDPOINT:-identify}"
WORKERS="${WORKERS:-4}"
REQUESTS="${REQUESTS:-100}"
PROBE_IMAGE="${PROBE_IMAGE:-}"

source "${SCRIPT_DIR}/_ensure_deps.sh"
ensure_test_deps "${TEST_ROOT}/requirements.txt"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_env.sh"
resolve_runtime_env

if [[ -z "${PROBE_IMAGE}" ]]; then
  if PROBE_IMAGE="$(pick_preferred_face_image)"; then
    echo "Using preferred probe image: ${PROBE_IMAGE}"
  fi
fi

if [[ -z "${PROBE_IMAGE}" ]]; then
  echo "Usage: PROBE_IMAGE=/path/to/face.jpg [WORKERS=4] [REQUESTS=100] $0"
  exit 1
fi

ARGS=(
  --base-url "${BASE_URL}"
  --endpoint "${ENDPOINT}"
  --probe-image "${PROBE_IMAGE}"
  --workers "${WORKERS}"
  --requests "${REQUESTS}"
  --output "${TEST_ROOT}/results/concurrent_report.json"
)

if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi
if [[ "${SKIP_LIVENESS}" == "true" ]]; then
  ARGS+=(--skip-liveness)
fi

echo "Running concurrent benchmark (${ENDPOINT}) against ${BASE_URL} ..."
echo "  workers=${WORKERS} requests=${REQUESTS} SKIP_LIVENESS=${SKIP_LIVENESS}"
"${TEST_PYTHON}" "${INTEGRATION}/concurrent_benchmark.py" "${ARGS[@]}"
