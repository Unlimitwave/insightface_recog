#!/usr/bin/env bash
# Ensure test dependencies via project-local venv (test/.venv).
set -euo pipefail

ensure_test_deps() {
  local req_file="$1"
  local test_root
  test_root="$(cd "$(dirname "${req_file}")" && pwd)"
  local venv="${test_root}/.venv"

  if [[ ! -x "${venv}/bin/python" ]]; then
    echo "Creating test venv at ${venv} ..."
    python3 -m venv "${venv}"
    "${venv}/bin/pip" install -q -r "${req_file}"
  elif ! "${venv}/bin/python" -c "import requests" 2>/dev/null; then
    "${venv}/bin/pip" install -q -r "${req_file}"
  fi

  export TEST_PYTHON="${venv}/bin/python"
}
