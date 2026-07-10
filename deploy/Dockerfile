# syntax=docker/dockerfile:1
#
# Build-time runtime selection (host GPU is not visible during `docker build`):
#   docker build --build-arg RUNTIME=gpu --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 .
#   docker build --build-arg RUNTIME=cpu --build-arg BASE_IMAGE=ubuntu:22.04 .
# Prefer: ./scripts/compose_up.sh  (auto-detects host GPU + nvidia-container-toolkit)

ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG RUNTIME=gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNTIME=${RUNTIME} \
    INSIGHTFACE_ROOT=/models/insightface \
    DATA_DIR=/data \
    ANTISPOOF_MODEL_DIR=/models/antispoof \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-gpu.txt requirements-cpu.txt ./

# insightface depends on CPU `onnxruntime`, which shares the import path with
# `onnxruntime-gpu`. Uninstalling the CPU wheel alone leaves a broken package
# tree — always reinstall onnxruntime-gpu afterwards (same as setup_local.sh).
# Pin <1.27: PyPI 1.27+ wheels expect CUDA 13 (libcudart.so.13).
RUN python3.12 -m venv "${VIRTUAL_ENV}" && \
    pip install --upgrade pip && \
    if [ "${RUNTIME}" = "gpu" ]; then \
      pip install -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
      pip uninstall -y onnxruntime 2>/dev/null || true && \
      pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.19.0,<1.27.0" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    else \
      pip install -r requirements-cpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi

# Entrypoint only — app code is bind-mounted via compose for hot reload
COPY scripts/docker-entrypoint.sh scripts/nvidia_lib_path.sh /app/scripts/
RUN chmod +x /app/scripts/docker-entrypoint.sh && \
    mkdir -p /models/insightface /models/antispoof /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/v1/health || exit 1

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
