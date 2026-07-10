#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

MEGAFACE_ROOT="${MEGAFACE_ROOT:-/home/cmsr/桌面/东风/数据集/megaface}"
MODEL_FILE="${MODEL_FILE:-$HOME/.insightface/models/buffalo_l/w600k_r50.onnx}"
ALGO="${ALGO:-buffalo_l}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GALLERY_SIZE="${GALLERY_SIZE:-1000000}"
FORCE_REEXTRACT="${FORCE_REEXTRACT:-0}"

DATA_DIR="${MEGAFACE_ROOT}/data"
DEVKIT_ROOT="${MEGAFACE_ROOT}/devkit"
RESULTS_DIR="${MEGAFACE_ROOT}/results/${ALGO}"
FEATURE_OUT="${MEGAFACE_ROOT}/feature_out/${ALGO}"
FEATURE_OUT_CLEAN="${MEGAFACE_ROOT}/feature_out_clean/${ALGO}"

PYTHON="${PYTHON:-python}"
export ORT_LOGGING_LEVEL=3

setup_devkit_runtime() {
  # shellcheck source=/dev/null
  source "${ROOT}/setup_devkit_libs.sh"
}

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
setup_devkit_runtime

if [[ "${GPU}" -ge 0 ]]; then
  if ! verify_cuda_runtime >/dev/null 2>&1; then
    echo "Warning: CUDAExecutionProvider unavailable, fallback to CPU."
    echo "If onnxruntime-gpu is installed, check NVIDIA driver and CUDA libs."
    GPU=-1
  else
    echo "CUDA runtime OK"
  fi
fi

echo "MegaFace evaluation with ${ALGO}"
echo "dataset root: ${MEGAFACE_ROOT}"
echo "model file:   ${MODEL_FILE}"
echo "results dir:  ${RESULTS_DIR}"

if [[ ! -d "${DATA_DIR}/facescrub_images" || ! -d "${DATA_DIR}/megaface_images" ]]; then
  echo "Image folders not found. Prepare data first:"
  echo "  cd ${DATA_DIR}"
  echo "  rm -f facescrub_images megaface_images"
  echo "  unzip megaface_testpack_v1.0.zip"
  exit 1
fi

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "Model file not found: ${MODEL_FILE}"
  exit 1
fi

mkdir -p "${FEATURE_OUT}" "${FEATURE_OUT_CLEAN}" "${RESULTS_DIR}"

echo "[1/4] Extract features with ONNX model"
FEATURE_ARGS=(
  --gpu "${GPU}"
  --algo "${ALGO}"
  --batch-size "${BATCH_SIZE}"
  --model-file "${MODEL_FILE}"
  --facescrub-root "${DATA_DIR}/facescrub_images"
  --megaface-root "${DATA_DIR}/megaface_images"
  --facescrub-lst "${DATA_DIR}/facescrub_lst"
  --megaface-lst "${DATA_DIR}/megaface_lst"
  --output "${FEATURE_OUT}"
)
if [[ "${FORCE_REEXTRACT}" != "1" ]]; then
  FEATURE_ARGS+=(--skip-existing)
  echo "resume mode: skip existing features (set FORCE_REEXTRACT=1 to rebuild)"
else
  echo "force mode: re-extract all features"
fi
"${PYTHON}" -u "${ROOT}/gen_megaface_onnx.py" "${FEATURE_ARGS[@]}"

echo "[2/4] Remove noisy labels"
"${PYTHON}" -u "${ROOT}/remove_noises.py" \
  --algo "${ALGO}" \
  --feature-dir-input "${FEATURE_OUT}" \
  --feature-dir-out "${FEATURE_OUT_CLEAN}" \
  --facescrub-lst "${DATA_DIR}/facescrub_lst" \
  --megaface-lst "${DATA_DIR}/megaface_lst" \
  --facescrub-noises "${DATA_DIR}/facescrub_noises.txt" \
  --megaface-noises "${DATA_DIR}/megaface_noises.txt"

echo "[3/4] Run MegaFace devkit scoring"
"${PYTHON}" -u "${ROOT}/run_experiment_py3.py" \
  --devkit-root "${DEVKIT_ROOT}" \
  "${FEATURE_OUT_CLEAN}/megaface" \
  "${FEATURE_OUT_CLEAN}/facescrub" \
  "_${ALGO}.bin" \
  "${RESULTS_DIR}" \
  -s "${GALLERY_SIZE}" \
  -p "${DEVKIT_ROOT}/templatelists/facescrub_features_list.json"

echo "[4/4] Print metrics"
"${PYTHON}" -u "${ROOT}/parse_megaface_results.py" \
  --result-dir "${RESULTS_DIR}" \
  --algo "${ALGO}" \
  --gallery-size "${GALLERY_SIZE}" \
  --feature-dir-clean "${FEATURE_OUT_CLEAN}" \
  --facescrub-lst "${DATA_DIR}/facescrub_lst" \
  --devkit-root "${DEVKIT_ROOT}"

echo "Done. Raw json files are under ${RESULTS_DIR}"
