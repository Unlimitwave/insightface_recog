#!/usr/bin/env bash
# 功能冒烟测试：全量多人底库 + enroll_add + 全 API 主流程
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INTEGRATION="${TEST_ROOT}/integration"

BASE_URL="${BASE_URL:-http://localhost:8123}"
IDENTIFY_PERSONS="${IDENTIFY_PERSONS:-wjr,whd,zjy}"
STRANGER_DIR="${STRANGER_DIR:-cyt}"
API_KEY="${API_KEY:-}"
RESET_GALLERY="${RESET_GALLERY:-true}"

source "${SCRIPT_DIR}/_ensure_deps.sh"
ensure_test_deps "${TEST_ROOT}/requirements.txt"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_env.sh"
resolve_runtime_env

ARGS=(
  --base-url "${BASE_URL}"
  --stranger-dir "${STRANGER_DIR}"
  --identify-persons "${IDENTIFY_PERSONS}"
  --output "${TEST_ROOT}/results/feature_smoke_report.json"
)

if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi
if [[ "${SKIP_LIVENESS}" == "true" ]]; then
  ARGS+=(--skip-liveness)
fi
if [[ "${RESET_GALLERY}" == "false" ]]; then
  ARGS+=(--no-reset-gallery)
fi

echo "Running feature smoke test against ${BASE_URL} ..."
echo "  SKIP_LIVENESS=${SKIP_LIVENESS}  STRANGER_DIR=${STRANGER_DIR}  IDENTIFY_PERSONS=${IDENTIFY_PERSONS}"
"${TEST_PYTHON}" "${INTEGRATION}/feature_smoke_test.py" "${ARGS[@]}"
