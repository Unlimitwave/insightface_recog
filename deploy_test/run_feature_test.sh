#!/usr/bin/env bash
# 功能冒烟测试：全量多人底库 + enroll_add 追加注册 + 全 API 覆盖
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8000}"
IDENTIFY_PERSONS="${IDENTIFY_PERSONS:-wjr,whd,zjy}"
STRANGER_DIR="${STRANGER_DIR:-cyt}"
API_KEY="${API_KEY:-}"
# 样例图常过不了活体，默认跳过；生产真实摄像头图可设 SKIP_LIVENESS=false
SKIP_LIVENESS="${SKIP_LIVENESS:-true}"
RESET_GALLERY="${RESET_GALLERY:-true}"

pip install -q -r "${SCRIPT_DIR}/requirements.txt"

ARGS=(
  --base-url "${BASE_URL}"
  --stranger-dir "${STRANGER_DIR}"
  --identify-persons "${IDENTIFY_PERSONS}"
  --output "${SCRIPT_DIR}/results/feature_smoke_report.json"
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
python3 "${SCRIPT_DIR}/feature_smoke_test.py" "${ARGS[@]}"
