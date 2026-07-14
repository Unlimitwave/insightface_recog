#!/usr/bin/env bash
# 注册时延基准：单张 / 多张 batch / 多张 sequential
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INTEGRATION="${TEST_ROOT}/integration"

BASE_URL="${BASE_URL:-http://localhost:8123}"
API_KEY="${API_KEY:-}"
ENROLL_STRATEGY="${ENROLL_STRATEGY:-batch}"
OUTPUT="${OUTPUT:-${TEST_ROOT}/results/enroll_latency_report.json}"

# One of: ENROLL_DIR, ENROLL_IMAGES (space-separated), ENROLL_IMAGE, IMAGE
ENROLL_DIR="${ENROLL_DIR:-}"
ENROLL_IMAGES="${ENROLL_IMAGES:-}"
ENROLL_IMAGE="${ENROLL_IMAGE:-}"
IMAGE="${IMAGE:-}"

source "${SCRIPT_DIR}/_ensure_deps.sh"
ensure_test_deps "${TEST_ROOT}/requirements.txt"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_env.sh"
resolve_runtime_env

if [[ -z "${ENROLL_DIR}" && -z "${ENROLL_IMAGES}" && -z "${ENROLL_IMAGE}" && -z "${IMAGE}" ]]; then
  for name in ym wjr whd zjy tjc; do
    d="${TEST_ROOT}/enroll_images/${name}"
    if [[ -d "${d}" ]]; then
      ENROLL_DIR="${d}"
      echo "Using preferred enroll dir: ${ENROLL_DIR}"
      break
    fi
  done
fi

if [[ -z "${ENROLL_DIR}" && -z "${ENROLL_IMAGES}" && -z "${ENROLL_IMAGE}" && -z "${IMAGE}" ]]; then
  for d in "${TEST_ROOT}"/enroll_images/*/; do
    [[ -d "${d}" ]] || continue
    ENROLL_DIR="${d%/}"
    break
  done
fi

if [[ -z "${ENROLL_DIR}" && -z "${ENROLL_IMAGES}" && -z "${ENROLL_IMAGE}" && -z "${IMAGE}" ]]; then
  echo "Usage:"
  echo "  ENROLL_DIR=test/enroll_images/wjr $0"
  echo "  ENROLL_IMAGES='img1.jpg img2.jpg' ENROLL_STRATEGY=sequential $0"
  echo "  ENROLL_IMAGE=photo.jpg ENROLL_COUNT=3 $0   # same file repeated"
  exit 1
fi

ARGS=(
  --base-url "${BASE_URL}"
  --mode enroll
  --enroll-strategy "${ENROLL_STRATEGY}"
  --output "${OUTPUT}"
)

if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi
if [[ -n "${ENROLL_DIR}" ]]; then
  ARGS+=(--enroll-dir "${ENROLL_DIR}")
fi
if [[ -n "${ENROLL_IMAGES}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--enroll-images ${ENROLL_IMAGES})
fi
if [[ -n "${ENROLL_IMAGE}" ]]; then
  ARGS+=(--enroll-image "${ENROLL_IMAGE}")
fi
if [[ -n "${IMAGE}" ]]; then
  ARGS+=(--image "${IMAGE}")
fi
if [[ -n "${ENROLL_COUNT:-}" ]]; then
  ARGS+=(--enroll-count "${ENROLL_COUNT}")
fi
if [[ "${SKIP_LIVENESS}" == "true" ]]; then
  ARGS+=(--skip-liveness)
fi

echo "Running enrollment latency benchmark against ${BASE_URL} (strategy=${ENROLL_STRATEGY}) ..."
echo "  SKIP_LIVENESS=${SKIP_LIVENESS}"
"${TEST_PYTHON}" "${INTEGRATION}/latency_benchmark.py" "${ARGS[@]}"
