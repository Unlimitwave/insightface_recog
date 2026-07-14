#!/usr/bin/env bash
# 一键运行全部集成测试（功能 + 安全 + 时延 + 并发）
#
# 开发:
#   ./test/scripts/run_all.sh
#
# 生产（鉴权 + 禁止 skip_liveness）:
#   API_KEY=your-secret ./test/scripts/run_all.sh
#   （自动探测 ENVIRONMENT=production → SKIP_LIVENESS=false + REQUIRE_PRODUCTION=true）
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/_ensure_deps.sh"
ensure_test_deps "${TEST_ROOT}/requirements.txt"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_env.sh"

echo "========== Resolve runtime env =========="
resolve_runtime_env
export SKIP_ENV_RESOLVE=1
echo ""

FAILED=0

run_suite() {
  local title="$1"
  shift
  echo "========== ${title} =========="
  if "$@"; then
    echo "[OK] ${title}"
  else
    local rc=$?
    echo "[FAIL] ${title} (exit ${rc})"
    FAILED=1
  fi
  echo ""
}

run_suite "1/4 Feature smoke" "${SCRIPT_DIR}/run_feature_test.sh"
run_suite "2/4 Security smoke" "${SCRIPT_DIR}/run_security_test.sh"
run_suite "3/4 Latency benchmark" "${SCRIPT_DIR}/run_benchmark.sh"
run_suite "4/4 Concurrent benchmark" "${SCRIPT_DIR}/run_concurrent_test.sh"

if [[ "${FAILED}" -eq 0 ]]; then
  echo "All tests completed successfully. Reports under test/results/"
  exit 0
fi

echo "One or more suites failed. Reports under test/results/"
exit 1
