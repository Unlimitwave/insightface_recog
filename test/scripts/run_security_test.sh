#!/usr/bin/env bash
# P0 安全与负向路径测试：鉴权、prod skip_liveness、错误码
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INTEGRATION="${TEST_ROOT}/integration"

BASE_URL="${BASE_URL:-http://localhost:8123}"
API_KEY="${API_KEY:-}"

source "${SCRIPT_DIR}/_ensure_deps.sh"
ensure_test_deps "${TEST_ROOT}/requirements.txt"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_env.sh"
resolve_runtime_env

ARGS=(
  --base-url "${BASE_URL}"
  --output "${TEST_ROOT}/results/security_smoke_report.json"
)

if [[ -n "${API_KEY}" ]]; then
  ARGS+=(--api-key "${API_KEY}")
fi
if [[ "${REQUIRE_PRODUCTION}" == "true" ]]; then
  ARGS+=(--require-production)
fi

echo "Running security smoke test against ${BASE_URL} ..."
echo "  API_KEY=${API_KEY:+set}  REQUIRE_PRODUCTION=${REQUIRE_PRODUCTION}"
"${TEST_PYTHON}" "${INTEGRATION}/security_smoke_test.py" "${ARGS[@]}"
