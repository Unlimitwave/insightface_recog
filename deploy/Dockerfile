# syntax=docker/dockerfile:1
#
# Build-time runtime selection (host GPU is not visible during `docker build`):
#   docker build --build-arg RUNTIME=gpu --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu24.04 .
#   docker build --build-arg RUNTIME=cpu --build-arg BASE_IMAGE=ubuntu:24.04 .
# Prefer: ./scripts/compose_up.sh  (auto-detects host GPU + nvidia-container-toolkit)

# TARGETPLATFORM is injected by BuildKit / compose `build.platform` (linux/arm64 or linux/amd64).
ARG TARGETPLATFORM=linux/amd64
ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu24.04
FROM --platform=$TARGETPLATFORM ${BASE_IMAGE}

ARG RUNTIME=gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNTIME=${RUNTIME} \
    VIRTUAL_ENV=/service/.venv \
    PATH="/service/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 curl tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service

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

# App code is copied for production; dev compose bind-mounts over /service/app.
COPY app/ /service/app/
COPY scripts/docker-entrypoint.sh scripts/nvidia_lib_path.sh /service/scripts/
RUN chmod +x /service/scripts/docker-entrypoint.sh && \
    mkdir -p /service/models/detection /service/models/recog /service/models/antispoof /service/data

EXPOSE 8123

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:8123/v1/health || exit 1

ENTRYPOINT ["/service/scripts/docker-entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8123"]
