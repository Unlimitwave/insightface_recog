#!/usr/bin/env bash
# Start the API locally with GPU library paths configured (non-Docker dev).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/.."
cd "${ROOT}"

export PYTHON="${ROOT}/.venv/bin/python3"
# shellcheck source=nvidia_lib_path.sh
source "${SCRIPT_DIR}/nvidia_lib_path.sh"

export INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-${HOME}/.insightface}"
export DATA_DIR="${DATA_DIR:-./data}"
export ANTISPOOF_MODEL_DIR="${ANTISPOOF_MODEL_DIR:-./models/antispoof}"

exec uv run uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload "$@"
