# InsightFace 门禁服务（Docker + HTTP）

面向生产门禁场景的 **1:N / 1:1 人脸识别微服务**：REST API、GPU 优先/CPU 回退、被动 RGB 活体检测、FAISS 向量检索、持久化底库、识别事件审计。

## 架构

```
业务系统（门禁/考勤/访客）
        │  HTTP/JSON + multipart
        ▼
┌───────────────────────────────────────────────┐
│  face-api (FastAPI + Docker)                  │
│  ├─ POST /v1/identify     探针 1:N 识别        │
│  ├─ POST /v1/verify       1:1 验证（探针 vs 指定人员）│
│  ├─ POST /v1/persons/…/faces  注册             │
│  ├─ GET  /v1/stats        底库 + 事件统计       │
│  ├─ GET  /v1/events       识别/验证审计日志     │
│  └─ GET  /v1/health       健康检查             │
│                                               │
│  InsightFace buffalo_l (检测+特征)             │
│  MiniFASNet V1SE+V2 (被动活体)                 │
│  SQLite + FAISS (底库与 1:N 检索)               │
│  SQLite events.db (识别/验证事件审计)            │
└───────────────────────────────────────────────┘
```

## 快速启动

### 1. 准备模型

**识别模型**（已有可跳过）：

```bash
# buffalo_l 通常在 ~/.insightface/models/buffalo_l/
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
```

**活体模型**（必下，用于注册/识别防照片攻击）：

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
# 输出: deploy/models/antispoof/MiniFASNetV*.onnx
```

### 2. Docker 启动（推荐：自动 GPU / CPU）

`docker build` **无法可靠探测宿主机 GPU**，因此依赖选择在 compose 启动时完成（标准做法）：


| 文件                                    | 作用                                                     |
| ------------------------------------- | ------------------------------------------------------ |
| `requirements.txt`                    | 公共依赖（不含 onnxruntime）                                   |
| `requirements-gpu.txt`                | `onnxruntime-gpu` + CUDA pip 库                         |
| `requirements-cpu.txt`                | `onnxruntime`（CPU）                                     |
| `Dockerfile`                          | `ARG RUNTIME` + `ARG BASE_IMAGE`：选依赖与基础镜像              |
| `docker-compose.gpu.yml` / `.cpu.yml` | GPU 透传 / CPU 覆盖                                        |
| `scripts/compose_up.sh`               | 探测 `nvidia-smi` + nvidia-container-toolkit，自动选 GPU/CPU |


```bash
cd deploy
cp .env.example .env
./scripts/compose_up.sh
```

开发模式：`./app` 挂载进容器，uvicorn `--reload` 热更新；**不要**在 Dockerfile 里 `COPY app`。

### 3. 手动指定 GPU / CPU

```bash
# GPU 镜像 + GPU 透传
DOCKER_BUILDKIT=0 RUNTIME=gpu docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

# 纯 CPU
DOCKER_BUILDKIT=0 docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

### 4. 宿主机有 GPU 但 health 显示 `device: cpu`

GPU 已映射进容器（`docker exec face-api nvidia-smi` 能成功）但仍用 CPU 时，通常是：


| 原因                       | 处理                                                                        |
| ------------------------ | ------------------------------------------------------------------------- |
| 构建时用了 CPU 依赖             | 用 `./scripts/compose_up.sh` 或 `RUNTIME=gpu` 重建                            |
| 装了 CPU 版 `onnxruntime`   | GPU 镜像会 `uninstall onnxruntime` 后 **再 force-reinstall** `onnxruntime-gpu` |
| 拉到了 ORT ≥1.27（要 CUDA 13） | `requirements-gpu.txt` 已钉 `onnxruntime-gpu<1.27`（配 CUDA 12.4 镜像）          |
| 缺 `libcudnn.so.9`        | `nvidia-cudnn-cu12` + entrypoint 设置 `LD_LIBRARY_PATH`                     |


**重新构建**（必须 `--build`）：

```bash
docker compose down
./scripts/compose_up.sh
```

验证：

```bash
docker exec face-api python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# GPU 镜像应包含 CUDAExecutionProvider

curl -s http://localhost:8000/v1/health | python3 -m json.tool
# "device": "cuda:0" 或 "cpu"
```

### 5. 验证

```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
curl -s http://localhost:8000/v1/ready | python3 -m json.tool
open http://localhost:8000/docs   # Swagger UI
```

## API 说明


| 方法     | 路径                       | 说明                        |
| ------ | ------------------------ | ------------------------- |
| GET    | `/v1/health`             | 存活探针                      |
| GET    | `/v1/ready`              | 就绪探针（含活体模型检查）             |
| POST   | `/v1/persons`            | 创建人员                      |
| GET    | `/v1/persons`            | 列表                        |
| GET    | `/v1/persons/{id}`       | 详情                        |
| DELETE | `/v1/persons/{id}`       | 删除人员及所有人脸                 |
| POST   | `/v1/persons/{id}/faces` | 注册人脸（multipart，建议 3~5 张）  |
| POST   | `/v1/identify`           | **1:N 识别**（探针图 multipart） |
| POST   | `/v1/verify`             | **1:1 验证**（探针 vs 指定 `person_id`） |
| GET    | `/v1/stats`              | 底库规模 + 事件统计（日趋势、通过率、活跃度） |
| GET    | `/v1/events`             | 识别/验证事件审计（支持分页与筛选）       |


### 注册流程（门禁底库）

```bash
# 1. 创建人员
curl -X POST http://localhost:8000/v1/persons \
  -H "Content-Type: application/json" \
  -d '{"person_id":"emp001","display_name":"张三","metadata":{"dept":"研发"}}'

# 2. 注册人脸（可多次上传，最多 MAX_FACES_PER_PERSON 张）
curl -X POST "http://localhost:8000/v1/persons/emp001/faces" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg"
```

**重复注册说明**（同一人再次 POST `…/faces`）：

| 场景 | 行为 |
| ---- | ---- |
| 人员已存在，再次上传照片 | **累加**新人脸模板，不覆盖旧模板；1:N 检索时取同人最高相似度 |
| 超过 `MAX_FACES_PER_PERSON`（默认 5） | 返回 `ENROLLMENT_LIMIT`，含 `current` / `requested` / `max` |
| 重复 POST `/v1/persons` 创建同 `person_id` | 返回 400 `person_id already exists` |
| 人员不存在就注册人脸 | 返回 404 `PERSON_NOT_FOUND` |

### 识别流程（闸机探针）

```bash
curl -X POST "http://localhost:8000/v1/identify" \
  -F "image=@probe.jpg"
```

响应示例：

```json
{
  "request_id": "...",
  "matched": true,
  "person_id": "emp001",
  "display_name": "张三",
  "similarity": 0.68,
  "threshold": 0.42,
  "is_stranger": false,
  "alert": false,
  "candidates": [
    {"rank": 1, "person_id": "emp001", "display_name": "张三", "similarity": 0.68, "matched": true}
  ],
  "quality": {"det_score": 0.91, "bbox": [...], "face_width_px": 120, "face_height_px": 145},
  "liveness": {"passed": true, "score": 0.87, "method": "rgb_passive_minifasnet", "model_scores": {...}},
  "latency_ms": {"inference": 45.2, "search": 0.8, "total": 46.0}
}
```

- `is_stranger=true`：检测到人脸，但底库无超过阈值的匹配（陌生人）
- `alert=true`：在 `STRANGER_ALERT_ENABLED=true` 且为陌生人时为 true，供业务侧触发闯入告警

### 1:1 验证流程（闸机先验身份）

适用于刷卡/扫码后已知 `person_id`、需确认「来人是否为该身份」的场景：

```bash
curl -X POST "http://localhost:8000/v1/verify?person_id=emp001" \
  -F "image=@probe.jpg"
```

响应示例：

```json
{
  "request_id": "...",
  "verified": true,
  "person_id": "emp001",
  "display_name": "张三",
  "similarity": 0.72,
  "threshold": 0.42,
  "matched_face_id": "uuid-of-best-template",
  "quality": {...},
  "liveness": {...},
  "latency_ms": {"inference": 42.1, "verify": 0.3, "total": 42.4}
}
```

### 统计与审计

```bash
# 近 7 日识别量、通过率、日趋势、人员活跃度
curl -s "http://localhost:8000/v1/stats?days=7" | python3 -m json.tool

# 事件历史（分页）
curl -s "http://localhost:8000/v1/events?limit=20" | python3 -m json.tool

# 仅陌生人（闯入告警）事件
curl -s "http://localhost:8000/v1/events?is_stranger=true" | python3 -m json.tool

# 按人员筛选
curl -s "http://localhost:8000/v1/events?person_id=emp001&event_type=identify" | python3 -m json.tool
```

事件持久化在 `DATA_DIR/events.db`（默认容器内 `/data/events.db`），与底库 `gallery.db` 同卷。

## 生产配置

复制 `.env.example` 为 `.env`，重点项：


| 变量                     | 默认     | 说明                      |
| ---------------------- | ------ | ----------------------- |
| `DEVICE`               | `auto` | `auto`/`cuda`/`cpu`     |
| `IDENTIFY_THRESHOLD`   | `0.42` | 1:N 相似度阈值，**须在私有数据上标定** |
| `VERIFY_THRESHOLD`     | `0.42` | 1:1 验证阈值，可与 1:N 分开标定   |
| `STRANGER_ALERT_ENABLED` | `true` | 识别响应中启用 `alert` 字段（陌生人告警） |
| `EVENT_LOG_ENABLED`    | `true` | 识别/验证事件持久化             |
| `EVENT_LOG_RETENTION_DAYS` | `90` | 事件保留天数，启动时自动清理过期记录   |
| `LIVENESS_ENABLED`     | `true` | 启用被动活体                  |
| `LIVENESS_ON_IDENTIFY` | `true` | 识别时强制活体（门禁推荐）           |
| `LIVENESS_ON_VERIFY`   | `true` | 1:1 验证时强制活体             |
| `API_KEY`              | 空      | 设置后要求请求头 `X-API-Key`    |


### 鉴权（生产必开）

```bash
# .env
API_KEY=your-production-secret
```

```bash
curl -H "X-API-Key: your-production-secret" ...
```

### 阈值标定

公开 IJB-C 分数 ≠ 业务可用。请在 **真实摄像头 + 真实底库** 上标定 `IDENTIFY_THRESHOLD`（目标 FAR/FPIR 见项目 `docs/buffalo_l生产应用评估报告.md`）。

## 工业级能力说明


| 能力      | 实现                                                   |
| ------- | ---------------------------------------------------- |
| 1:N 多模板 | 每人最多 5 张，检索时按 **同人最高相似度** 聚合                         |
| 1:1 验证  | 探针与指定人员所有注册模板取最高相似度，独立 `VERIFY_THRESHOLD`            |
| 陌生人告警  | identify 返回 `is_stranger` + `alert`，事件写入审计库              |
| 事件审计    | identify/verify 自动落库，支持 `/v1/events` 与 `/v1/stats` 查询   |
| 被动活体    | MiniFASNet V1SE + V2 双模型 ensemble                    |
| 质量门控    | 检测置信度、最小人脸尺寸                                         |
| GPU/CPU | ONNXRuntime `CUDAExecutionProvider` 优先，不可用自动回退 CPU   |
| 持久化     | SQLite 存元数据+向量，FAISS 内存索引，容器卷 `/data`                |
| 可观测     | `/v1/health`、`/v1/ready`、响应 `latency_ms`、标准错误码       |
| 可升级     | `FaceEngine` 抽象层，可换 InspireFace/TensorRT 而不改 HTTP 契约 |


### 活体检测说明

- **类型**：单帧 RGB **被动活体**（Silent-Face-Anti-Spoofing / MiniFASNet）
- **适用**：拦截打印照片、屏幕翻拍等常见攻击
- **局限**：不能替代高安全场景的 **动作活体**（眨眼/转头）或 **红外活体**；金融/支付级需额外模块
- **注册**：默认也做活体；管理员可信底库图可用 `skip_liveness=true`（慎用）

### 商用许可

InsightFace 开源模型默认 **非商业研究用途**。正式商用请联系 `recognition-oss-pack@insightface.ai`。

## 本地开发（不用 Docker）

```bash
cd deploy
./scripts/setup_local.sh          # 有 GPU 装 requirements-gpu，否则 requirements-cpu
./scripts/download_models.sh
./scripts/run_dev.sh              # 自动设置 LD_LIBRARY_PATH 并启动（GPU 优先）

# 测试
curl -s http://localhost:8000/v1/health | python3 -m json.tool
# "device": "cuda:0" 或 "cpu"
```

有 GPU 时 `setup_local.sh` 会 `uv pip uninstall onnxruntime`：`insightface` 会拉取 CPU 版 `onnxruntime`，与 `onnxruntime-gpu` 包名冲突。`run_dev.sh` 与 Docker 入口脚本一样，会把 pip 安装的 `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` 加入 `LD_LIBRARY_PATH`。

验证 GPU：

```bash
export PYTHON=./.venv/bin/python3
source ./scripts/nvidia_lib_path.sh
$PYTHON -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 应包含 CUDAExecutionProvider

```

## 性能扩展路径

1. **水平扩容**：`docker compose up --scale face-api=3` + Nginx 负载均衡（底库需共享 `/data` 卷或迁移 Milvus）
2. **推理加速**：实现新的 `FaceEngine` 后端（TensorRT / InspireFace C++）
3. **大库 1:N**：底库 >10 万时建议迁移 **Milvus** 向量库

## 错误码


| code                | 含义                             |
| ------------------- | ------------------------------ |
| `FACE_NOT_DETECTED` | 未检测到人脸                         |
| `LIVENESS_FAILED`   | 活体未通过                          |
| `LOW_FACE_QUALITY`  | 质量不达标                          |
| `NO_MATCH`          | 识别成功但无超过阈值的身份（`matched=false`） |
| `NO_ENROLLED_FACES` | 人员存在但未注册人脸（verify 时）          |
| `GALLERY_EMPTY`     | 底库为空                           |
| `EVENT_LOG_DISABLED`| 事件日志已关闭（`/v1/events` 不可用）     |
| `UNAUTHORIZED`      | API Key 无效                     |


完整 OpenAPI：`http://localhost:8000/docs`

## 测试

| 脚本 | 说明 |
| ---- | ---- |
| `[deploy_test/latency_benchmark.py](../deploy_test/latency_benchmark.py)` | 注册 + 1:N 识别时延基准 |
| `[deploy_test/feature_smoke_test.py](../deploy_test/feature_smoke_test.py)` | 1:1 验证、陌生人告警、统计、事件审计功能冒烟测试 |

快速功能测试（服务已启动后）：

```bash
./deploy_test/run_feature_test.sh
# 默认：清库 → 注册 wjr/zjy/whd 三人 → 全 API 冒烟 → 清理
SKIP_LIVENESS=true ./deploy_test/run_feature_test.sh   # 调试跳过活体
```

## 时延基准测试

见项目 `[deploy_test/](../deploy_test/README.md)`：对运行中的服务测注册/识别延迟，并报告实际使用的 GPU 或 CPU。