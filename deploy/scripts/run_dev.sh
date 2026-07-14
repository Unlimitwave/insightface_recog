#!/usr/bin/env bash
# Start the API locally with GPU library paths configured (non-Docker dev).
# Paths come from .env (./models, ./data) — cwd is deploy/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/.."
cd "${ROOT}"

mkdir -p data

export PYTHON="${ROOT}/.venv/bin/python3"
# shellcheck source=nvidia_lib_path.sh
source "${SCRIPT_DIR}/nvidia_lib_path.sh"

exec uv run uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8123}" --reload "$@"
