#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MEGAFACE_ROOT="${MEGAFACE_ROOT:-/home/cmsr/桌面/东风/数据集/megaface}"
MODEL_FILE="${MODEL_FILE:-$HOME/.insightface/models/buffalo_l/w600k_r50.onnx}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ENROLL_COUNTS="${ENROLL_COUNTS:-1,3,5,10}"
DISTRACTOR_SIZE="${DISTRACTOR_SIZE:-10000}"
MIN_IMAGES_PER_ID="${MIN_IMAGES_PER_ID:-11}"
MAX_IDENTITIES="${MAX_IDENTITIES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEGAFACE_ROOT}/results/enrollment_ablation}"

PYTHON="${PYTHON:-python}"
export ORT_LOGGING_LEVEL=3

setup_nvidia_runtime() {
  local nvidia_lib_path
  nvidia_lib_path="$("${PYTHON}" - <<'PY'
import glob
import site
paths = glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib')
print(':'.join(paths))
PY
)"
  if [[ -n "${nvidia_lib_path}" ]]; then
    export LD_LIBRARY_PATH="${nvidia_lib_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

verify_cuda_runtime() {
  "${PYTHON}" - <<'PY'
import onnxruntime as ort
import os

model = os.path.expanduser('~/.insightface/models/buffalo_l/w600k_r50.onnx')
if not os.path.isfile(model):
    raise SystemExit(0)
sess = ort.InferenceSession(
    model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
if sess.get_providers()[0] != 'CUDAExecutionProvider':
    raise SystemExit(1)
PY
}

setup_nvidia_runtime

if [[ "${GPU}" -ge 0 ]]; then
  if ! verify_cuda_runtime >/dev/null 2>&1; then
    echo "Warning: CUDAExecutionProvider unavailable, fallback to CPU."
    echo "If onnxruntime-gpu is installed, check NVIDIA driver and CUDA libs."
    GPU=-1
  else
    echo "CUDA runtime OK"
  fi
fi

DATA_DIR="${MEGAFACE_ROOT}/data"
if [[ ! -d "${DATA_DIR}/facescrub_images" || ! -d "${DATA_DIR}/megaface_images" ]]; then
  echo "Image folders not found under ${DATA_DIR}"
  exit 1
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "Model file not found: ${MODEL_FILE}"
  exit 1
fi

echo "Enrollment ablation experiment"
echo "dataset root:   ${MEGAFACE_ROOT}"
echo "model file:     ${MODEL_FILE}"
echo "enroll counts:  ${ENROLL_COUNTS}"
echo "distractor sz:  ${DISTRACTOR_SIZE}"
echo "max identities: ${MAX_IDENTITIES:-all}"
echo "output dir:     ${OUTPUT_DIR}"

ARGS=(
  --megaface-root "${MEGAFACE_ROOT}"
  --model-file "${MODEL_FILE}"
  --gpu "${GPU}"
  --enroll-counts "${ENROLL_COUNTS}"
  --distractor-size "${DISTRACTOR_SIZE}"
  --min-images-per-id "${MIN_IMAGES_PER_ID}"
  --max-identities "${MAX_IDENTITIES}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
)

"${PYTHON}" -u "${ROOT}/enrollment_ablation.py" "${ARGS[@]}"
