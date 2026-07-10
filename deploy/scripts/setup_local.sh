#!/usr/bin/env bash
# Install deploy dependencies for local (non-Docker) development.
# Prefer GPU wheel when CUDA is usable; otherwise install CPU onnxruntime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

want_gpu=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  want_gpu=1
fi

if [[ "${want_gpu}" -eq 1 ]]; then
  echo "==> Installing GPU requirements (onnxruntime-gpu)"
  uv pip install -r requirements-gpu.txt
  # insightface pulls CPU onnxruntime, which shadows onnxruntime-gpu (same import name).
  # Uninstall alone leaves a broken onnxruntime/ tree — reinstall the GPU wheel.
  uv pip uninstall onnxruntime 2>/dev/null || true
  uv pip install --reinstall --no-deps "onnxruntime-gpu>=1.19.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
else
  echo "==> Installing CPU requirements (onnxruntime)"
  uv pip install -r requirements-cpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

export PYTHON="${ROOT}/.venv/bin/python3"
# shellcheck source=nvidia_lib_path.sh
source "${SCRIPT_DIR}/nvidia_lib_path.sh"

echo ""
"${PYTHON}" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNXRuntime providers:", providers)
if "CUDAExecutionProvider" in providers:
    print("GPU ready: use ./scripts/run_dev.sh to start the server")
else:
    print("CPU mode: CUDAExecutionProvider not available — inference will use CPU.")
    print("  Start with: ./scripts/run_dev.sh")
PY
