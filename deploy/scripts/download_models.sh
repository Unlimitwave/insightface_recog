#!/usr/bin/env bash
# Download face detection, recognition, and passive RGB liveness models into deploy/models/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_ROOT="${1:-${ROOT}/models}"

DET_DIR="${MODELS_ROOT}/detection"
RECOG_DIR="${MODELS_ROOT}/recog"
ANTISPOOF_DIR="${MODELS_ROOT}/antispoof"

INSIGHTFACE_BASE="https://github.com/deepinsight/insightface/releases/download/v0.7"
ANTISPOOF_BASE="https://github.com/yakhyo/face-anti-spoofing/releases/download/weights"

mkdir -p "${DET_DIR}" "${RECOG_DIR}" "${ANTISPOOF_DIR}"

download() {
  local dest_dir="$1"
  local name="$2"
  local url="$3"
  local dest="${dest_dir}/${name}"
  if [[ -f "${dest}" ]]; then
    echo "OK  ${name} already exists"
    return
  fi
  echo "GET ${url}"
  curl -fsSL -o "${dest}" "${url}"
  echo "OK  ${name}"
}

echo "==> Detection models -> ${DET_DIR}"
download "${DET_DIR}" "det_10g.onnx" "${INSIGHTFACE_BASE}/det_10g.onnx"

echo ""
echo "==> Recognition models -> ${RECOG_DIR}"
download "${RECOG_DIR}" "w600k_r50.onnx" "${INSIGHTFACE_BASE}/w600k_r50.onnx"

echo ""
echo "==> Liveness models -> ${ANTISPOOF_DIR}"
download "${ANTISPOOF_DIR}" "MiniFASNetV2.onnx" \
  "${ANTISPOOF_BASE}/MiniFASNetV2.onnx"
download "${ANTISPOOF_DIR}" "MiniFASNetV1SE.onnx" \
  "${ANTISPOOF_BASE}/MiniFASNetV1SE.onnx"

echo ""
echo "Models ready under: ${MODELS_ROOT}"
echo "  detection/  det_10g.onnx"
echo "  recog/      w600k_r50.onnx"
echo "  antispoof/  MiniFASNetV1SE.onnx, MiniFASNetV2.onnx"
