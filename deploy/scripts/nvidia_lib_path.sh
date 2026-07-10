#!/usr/bin/env bash
# Export LD_LIBRARY_PATH with pip-installed NVIDIA CUDA libs (cudnn, cublas, etc.)
# so onnxruntime-gpu can load them. Safe to source multiple times.
if [[ -n "${NVIDIA_LIB_PATH_DONE:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

_nv_python="${PYTHON:-python3}"
NVIDIA_LIB="$(
"${_nv_python}" - <<'PY'
import glob
import site

paths = []
for base in site.getsitepackages():
    paths.extend(glob.glob(base + "/nvidia/*/lib"))
print(":".join(paths))
PY
)"

if [[ -n "${NVIDIA_LIB}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export NVIDIA_LIB_PATH_DONE=1
