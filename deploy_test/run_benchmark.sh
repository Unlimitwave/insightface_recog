#!/usr/bin/env bash
# Run latency benchmark against a running face-api service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8000}"
IMAGE="${IMAGE:-}"

if [[ -z "${IMAGE}" ]]; then
  echo "Usage: IMAGE=/path/to/face.jpg [BASE_URL=http://localhost:8000] $0"
  echo ""
  echo "Example:"
  echo "  IMAGE=~/photos/alice.jpg ./deploy_test/run_benchmark.sh"
  exit 1
fi

pip install -q -r "${SCRIPT_DIR}/requirements.txt"

python3 "${SCRIPT_DIR}/latency_benchmark.py" \
  --base-url "${BASE_URL}" \
  --image "${IMAGE}" \
  --warmup 2 \
  --runs 20 \
  --output "${SCRIPT_DIR}/results/latency_report.json"
