#!/usr/bin/env bash
# Set NVIDIA pip library paths so onnxruntime-gpu can load libcudnn / libcublas.
# No-op when CUDA pip packages are absent (CPU image).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=nvidia_lib_path.sh
source "${SCRIPT_DIR}/nvidia_lib_path.sh"

exec "$@"
