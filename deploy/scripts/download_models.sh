#!/usr/bin/env bash
# Download passive RGB liveness models (MiniFASNet) for production gate deployment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-${SCRIPT_DIR}/../models/antispoof}"
mkdir -p "${DEST}"

download() {
  local name="$1"
  local url="$2"
  local dest="${DEST}/${name}"
  if [[ -f "${dest}" ]]; then
    echo "OK  ${name} already exists"
    return
  fi
  echo "GET ${url}"
  curl -fsSL -o "${dest}" "${url}"
  echo "OK  ${name}"
}

download "MiniFASNetV2.onnx" \
  "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx"

download "MiniFASNetV1SE.onnx" \
  "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx"

echo ""
echo "Liveness models ready in: ${DEST}"
echo "Ensure buffalo_l is at: ~/.insightface/models/buffalo_l/ (insightface auto-download or manual copy)"
