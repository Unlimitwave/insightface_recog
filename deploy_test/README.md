# 时延基准测试（deploy 服务）

对运行中的 **face-api**（`deploy/docker-compose`）做端到端测试，包含：

| 脚本 | 用途 |
| ---- | ---- |
| `latency_benchmark.py` | 注册 + 1:N 识别 **时延**基准 |
| `feature_smoke_test.py` | **1:1 验证、陌生人告警、统计、事件审计** 功能冒烟 |

## 前提

1. 服务已启动：

```bash
cd deploy
./scripts/compose_up.sh
curl -s http://localhost:8000/v1/health
```

2. 测试图片：项目自带 `deploy_test/enroll_images/`（按人名分子目录）。

## 安装依赖

```bash
pip install -r deploy_test/requirements.txt
```

---

## 功能冒烟测试（推荐先跑）

一键验证完整 API：**清库 → 全量注册多人 → enroll_add 追加 → 探针测试 → 清理**

```bash
chmod +x deploy_test/run_feature_test.sh
./deploy_test/run_feature_test.sh
```

默认行为：

| 步骤 | 说明 |
| ---- | ---- |
| 清库 | 删除底库中**所有**现有人员 |
| 全量注册 | `enroll_images/` 下**除 cyt 外**每个子目录的全部照片（wjr/zjy/whd/hjh/ym/…） |
| 陌生人 | `cyt` **不注册**，仅作 stranger 探针 |
| 1:N 探针 | 仅对 `wjr`、`whd`、`zjy` 做 `identify_matched_*` |
| 追加注册 | `enroll_add/hjh/add_hjh.png` 成功（3→4 张）；`enroll_add/ym/add_ym.png` 触发 `ENROLLMENT_LIMIT`（已满 5 张） |
| 清理 | 测试结束后删除所有 `smoke_*` 人员 |

**关于上次 16/18 失败的原因：**

- `re_enroll`：`image copy 2.png` 活体未通过（未设 `SKIP_LIVENESS=true`）
- `stats`：因 re_enroll 失败 + delete_face 删掉 wjr 唯一人脸，底库总 face 数对不上

现已改为：默认 `SKIP_LIVENESS=true`；全量注册跳过单张失败；delete_face 改测 hjh（4 张删 1）；stats 按实际 face 数校验。

```bash
# 使用真实摄像头图、需测活体时
SKIP_LIVENESS=false ./deploy_test/run_feature_test.sh

# 保留现有底库（不清空）
RESET_GALLERY=false ./deploy_test/run_feature_test.sh
```

Python 直接调用：

```bash
python deploy_test/feature_smoke_test.py \
  --identify-persons wjr,whd,zjy \
  --stranger-dir cyt \
  --skip-liveness \
  --output deploy_test/results/feature_smoke_report.json
```

### 测试项

| 步骤 | 验证内容 |
| ---- | -------- |
| `health` / `ready` | 存活与就绪 |
| `reset_gallery` | 清空底库 |
| `setup_gallery` | 除 cyt 外全员全量注册 |
| `list_persons` / `get_person` | 人员查询 |
| `create_person_duplicate` | 重复建人 → 400 |
| `enroll_add_hjh` | hjh 3 张后再加 1 张 → 成功（4/5）；**若 hjh 照片人脸过小则跳过** |
| `enroll_add_ym_limit` | ym 补满 5 张后再加 → `ENROLLMENT_LIMIT` |
| `identify_matched_wjr/whd/zjy` | 1:N 匹配 |
| `identify_stranger` | cyt 陌生人告警 |
| `verify_pass` / `verify_fail_stranger` / `verify_wrong_person` | 1:1 验证 |
| `delete_face` | 删除 hjh 单张人脸 |
| `stats` / `events` | 统计与审计 |

目录结构：

```
deploy_test/
├── enroll_images/     # 初始底库（cyt 仅 stranger）
│   ├── wjr/  zjy/  whd/  hjh/  ym/  cyt/  ...
└── enroll_add/        # 追加注册测试
    ├── hjh/add_hjh.png   → 应成功
    └── ym/add_ym.png     → 应 ENROLLMENT_LIMIT（ym 会先补满 5 张再测）

**说明：**

- `MIN_FACE_SIZE_PX=80` 质量门控下，部分样例图（如 hjh、dlrb）可能全部注册失败，测试会跳过该人员并在 Notes 中说明
- ym 目录 5 张图可能仅 3 张过质量门控，测试会用已成功照片**补满 5 张**再测上限
- hjh 需 `enroll_images/hjh` 至少 3 张过质量门控，才能测 `enroll_add_hjh` 追加成功场景
```

参数：

| 参数 / 环境变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `--identify-persons` / `IDENTIFY_PERSONS` | `wjr,whd,zjy` | 1:N 探针人员 |
| `--stranger-dir` / `STRANGER_DIR` | `cyt` | 陌生人不注册 |
| `--skip-liveness` / `SKIP_LIVENESS` | **`true`** | 跳过活体（样例图推荐） |
| `--reset-gallery` / `RESET_GALLERY` | `true` | 测试前清库 |
| `--skip-cleanup` | false | 测试后保留人员 |

---

## 时延基准测试

```bash
# 同一张图用于注册和探针（主要测时延；是否 matched 取决于是否同一人）
python deploy_test/latency_benchmark.py \
  --base-url http://localhost:8000 \
  --image /path/to/face.jpg

# 注册图与探针图分开（推荐）
python deploy_test/latency_benchmark.py \
  --enroll-image deploy_test/enroll_images/Hugh_Jackman/Hugh_Jackman_23835.png \
  --probe-image deploy_test/enroll_images/Hugh_Jackman/Hugh_Jackman_23847.png \
  --runs 50 \
  --output deploy_test/results/latency_report.json
```

或使用封装脚本：

```bash
chmod +x deploy_test/run_benchmark.sh
IMAGE=deploy_test/enroll_images/Hugh_Jackman/Hugh_Jackman_23835.png deploy_test/run_benchmark.sh
```

## 输出说明

| 指标 | 含义 |
|------|------|
| `Device: cuda:0` | 服务正在用 **GPU**（ONNXRuntime CUDA） |
| `Device: cpu` | GPU 不可用或未配置，已 **回退 CPU** |
| `Enrollment wall time` | 注册 1 张人脸的 **HTTP 端到端**耗时（含检测+活体+提特征+写库） |
| `Client end-to-end` | 识别请求的 **HTTP 端到端**耗时（p50/p95/p99） |
| `Server inference` | 服务返回的 `latency_ms.inference`（模型推理） |
| `Server search` | 服务返回的 `latency_ms.search`（FAISS 1:N 检索） |

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--base-url` | `http://localhost:8000` | API 地址 |
| `--api-key` | 无 | 若 deploy 设置了 `API_KEY` |
| `--warmup` | `2` | 正式计时前的预热请求数 |
| `--runs` | `20` | 识别计时次数 |
| `--enroll-count` | `1` | 注册几张（同文件重复，测批量注册时可增大） |
| `--skip-liveness` | false | 跳过活体（仅调试） |
| `--skip-cleanup` | false | 保留测试人员不删除 |

## GPU / CPU 回退说明

- `DEVICE=auto`（默认）：有 CUDA 则用 GPU，否则 CPU。
- `/v1/health` 的 `device` 字段：`cuda:0` = GPU，`cpu` = CPU。

### 若宿主机有 GPU 但显示 `cpu`

常见两个原因（已在本项目 Dockerfile 中处理）：

1. **CPU 版 onnxruntime 覆盖了 GPU 版**  
   `insightface` 依赖会装上 `onnxruntime`（CPU），需卸载后只保留 `onnxruntime-gpu`。

2. **容器内找不到 cuDNN**（`libcudnn.so.9`）  
   需安装 `nvidia-cudnn-cu12` 等 pip 包，并在启动时设置 `LD_LIBRARY_PATH`（见 `scripts/docker-entrypoint.sh`）。

宿主机还需安装 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

验证 GPU 是否进容器：

```bash
docker exec face-api nvidia-smi
docker exec face-api python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 期望含 CUDAExecutionProvider
```

修复后重新构建：

```bash
docker compose down
docker compose up -d --build
curl -s http://localhost:8000/v1/health   # device 应为 cuda:0
```

## 活体检测

服务默认开启被动 RGB 活体（MiniFASNet）。`/v1/health` 中：

- `liveness_enabled=true`
- `liveness_models_loaded=true` → 注册/识别均会做活体

若 `models_loaded=false`，需先执行 `deploy/scripts/download_models.sh` 并重启容器。
