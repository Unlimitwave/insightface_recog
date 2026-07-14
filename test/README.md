# Face Access Control — 集成测试

对运行中的 **face-api** 服务（`deploy/docker-compose`）进行 **HTTP 集成测试**。  
测试依赖已启动的 API 进程，**不是** Python 单元测试（不测模块内部逻辑）。


| 层级   | 目录/脚本                                    | 用途                        |
| ---- | ---------------------------------------- | ------------------------- |
| 集成测试 | `test/integration/`                      | 通过 HTTP 验证 API 契约与 E2E 行为 |
| 测试数据 | `test/enroll_images/`、`test/enroll_add/` | 注册/探针样例图（gitignore）       |
| 报告输出 | `test/results/`                          | JSON 报告（gitignore）        |
| 入口脚本 | `test/scripts/`                          | 一键运行各类测试                  |


完整 API 说明见 `[deploy/docs/API.md](../deploy/docs/API.md)`。

---

## 快速开始

### 1. 启动服务

```bash
cd deploy
./scripts/compose_up.sh
curl -s http://localhost:8123/v1/health
```

### 2. 安装依赖

脚本首次运行会自动创建 `test/.venv` 并安装依赖，也可手动：

```bash
python3 -m venv test/.venv
test/.venv/bin/pip install -r test/requirements.txt
```



### 3. 运行测试（推荐顺序）

```bash
chmod +x test/scripts/*.sh

# 开发环境一键全套（自动 SKIP_LIVENESS=true）
./test/scripts/run_all.sh

# 生产环境一键全套（鉴权 + 真实活体）
# 脚本会探测 ENVIRONMENT=production，自动设 SKIP_LIVENESS=false、REQUIRE_PRODUCTION=true
API_KEY=your-secret ./test/scripts/run_all.sh

# 也可显式覆盖
API_KEY=your-secret SKIP_LIVENESS=false REQUIRE_PRODUCTION=true ./test/scripts/run_all.sh
```

单独跑某一项：

```bash
./test/scripts/run_feature_test.sh
./test/scripts/run_security_test.sh
./test/scripts/run_benchmark.sh
./test/scripts/run_concurrent_test.sh
```



---



## 测试分层



### 功能冒烟 — `feature_smoke_test.py`

验证门禁主流程 Happy Path。


| 步骤                              | 验证内容                       |
| ------------------------------- | -------------------------- |
| `health` / `ready`              | 存活与就绪探针                    |
| `reset_gallery`                 | 清空底库                       |
| `setup_gallery`                 | 除 stranger(cyt) 外全量注册      |
| `list_persons` / `get_person`   | 人员查询                       |
| `create_person_duplicate`       | 重复建人 → 400                 |
| `enroll_add_hjh`                | 追加注册成功（或 skip）             |
| `enroll_add_ym_limit`           | 5 张上限 → `ENROLLMENT_LIMIT` |
| `identify_matched_*`            | 1:N 匹配                     |
| `identify_stranger`             | 陌生人 + 告警                   |
| `verify_pass` / `verify_fail_*` | 1:1 验证                     |
| `delete_face`                   | 删除单张人脸                     |
| `stats` / `events`              | 统计与审计                      |


**报告**：`test/results/feature_smoke_report.json`

```bash
# 开发环境：未显式设置时自动探测；样例图可用 skip
./test/scripts/run_feature_test.sh
SKIP_LIVENESS=true ./test/scripts/run_feature_test.sh

# 生产 / 测真实活体（未设置时生产机自动 SKIP_LIVENESS=false）
API_KEY=your-secret ./test/scripts/run_feature_test.sh
RESET_GALLERY=false ./test/scripts/run_feature_test.sh
```

---



### 安全/负向 — `security_smoke_test.py`（P0）


| 步骤                            | 验证内容                                  | 备注                         |
| ----------------------------- | ------------------------------------- | -------------------------- |
| `auth_no_key`                 | 无 Key → 401 `UNAUTHORIZED`            | 服务端未配 `API_KEY` 时 **skip** |
| `auth_wrong_key`              | 错误 Key → 401                          | 同上                         |
| `auth_valid_key`              | 正确 Key → 200                          | 需 `API_KEY` 环境变量           |
| `prod_skip_liveness_identify` | identify 带 `skip_liveness=true` → 403 | dev 下 **skip**             |
| `prod_skip_liveness_enroll`   | 注册 skip → 403                         | dev 下 **skip**             |
| `gallery_empty_identify`      | 空库 1:N → 503 `GALLERY_EMPTY`          |                            |
| `person_not_found_get`        | GET 不存在 → 404                         |                            |
| `person_not_found_delete`     | DELETE 不存在 → 404                      |                            |
| `face_not_found_delete`       | DELETE 假 face_id → 404                |                            |
| `verify_no_enrolled_faces`    | 人员无人脸 → 422 `NO_ENROLLED_FACES`       |                            |
| `face_not_detected_identify`  | 无人脸图 → 400 `FACE_NOT_DETECTED`        |                            |


**报告**：`test/results/security_smoke_report.json`

```bash
# 开发环境（鉴权/prod 用例部分 skip）
./test/scripts/run_security_test.sh

# 生产验收：只需 API_KEY；脚本自动 REQUIRE_PRODUCTION=true
API_KEY=your-secret ./test/scripts/run_security_test.sh
```

---



### 时延基准 — `latency_benchmark.py`

测量注册 wall time，以及 identify / verify 的客户端端到端延迟与服务端 `latency_ms` 分位值。


| 指标                              | 含义                                         |
| ------------------------------- | ------------------------------------------ |
| `enroll_wall_ms`                | 注册 HTTP 端到端总耗时                             |
| `enroll.strategy`               | `batch`（一次 POST 多图）或 `sequential`（逐张 POST） |
| `enroll.wall_ms_per_request`    | 每次 HTTP 请求的 wall time                      |
| `enroll.wall_ms_per_face`       | 按成功注册人脸数均摊的近似单张耗时                          |
| `identify.wall` / `verify.wall` | 客户端 p50/p95/p99                            |
| `*.server_inference`            | `latency_ms.inference`                     |
| `identify.server_secondary`     | `latency_ms.search`（FAISS）                 |
| `verify.server_secondary`       | `latency_ms.verify`                        |


**报告**：`test/results/latency_report.json`（全链路）；仅注册见 `test/results/enroll_latency_report.json`

```bash
# 单张 + identify/verify（默认）
IMAGE=test/enroll_images/wjr/photo.jpg ./test/scripts/run_benchmark.sh
MODE=both RUNS=50 ./test/scripts/run_benchmark.sh

# 多张不同图片 — 一次请求批量注册
ENROLL_DIR=test/enroll_images/wjr ./test/scripts/run_enroll_benchmark.sh

# 多张不同图片 — 逐张注册，对比单次请求耗时分布
ENROLL_DIR=test/enroll_images/wjr ENROLL_STRATEGY=sequential ./test/scripts/run_enroll_benchmark.sh

# 同一张图重复 N 次（测批量上传同图）
ENROLL_IMAGE=test/enroll_images/wjr/photo.jpg ENROLL_COUNT=3 ./test/scripts/run_enroll_benchmark.sh
```

---



### 并发 / QPS — `concurrent_benchmark.py`

多线程并行发送 identify 或 verify 请求，统计吞吐量与延迟。


| 指标                      | 含义                   |
| ----------------------- | -------------------- |
| `qps`                   | 总请求数 / 总耗时           |
| `latency_wall_ms`       | 成功+失败请求的 p50/p95/p99 |
| `successful` / `failed` | 成功/失败计数              |


**报告**：`test/results/concurrent_report.json`

```bash
PROBE_IMAGE=test/enroll_images/wjr/photo.jpg ./test/scripts/run_concurrent_test.sh
ENDPOINT=verify WORKERS=8 REQUESTS=200 ./test/scripts/run_concurrent_test.sh
```

---



## 环境变量


| 变量                   | 默认                      | 说明                                                      |
| -------------------- | ----------------------- | ------------------------------------------------------- |
| `BASE_URL`           | `http://localhost:8123` | API 地址                                                  |
| `API_KEY`            | 空                       | 请求头 `X-API-Key`（服务端配了 `API_KEY` 时必填）                    |
| `SKIP_LIVENESS`      | **自动探测**                | 未设置时：dev→`true`，prod（禁止 skip）→`false`                   |
| `RESET_GALLERY`      | `true`                  | feature 测试前清库                                           |
| `REQUIRE_PRODUCTION` | **自动探测**                | 未设置时：prod→`true`（skip 未 403 则失败）                       |
| `IDENTIFY_PERSONS`   | `wjr,whd,zjy`           | 1:N 探针人员                                                |
| `STRANGER_DIR`       | `cyt`                   | 陌生人目录                                                   |
| `MODE`               | `both`                  | 时延测试：identify / verify / both / enroll                  |
| `ENROLL_STRATEGY`    | `batch`                 | 注册时延：batch / sequential                                 |
| `ENROLL_DIR`         | 空                       | 注册时延：人员目录下全部图片；未指定时优先 `ym/wjr/...`                     |
| `WORKERS`            | `4`                     | 并发线程数                                                   |
| `REQUESTS`           | `100`                   | 并发总请求数                                                  |
| `ENDPOINT`           | `identify`              | 并发测试端点                                                  |


---



## 报告目录


| 报告文件                                      | 产生脚本                      |
| ----------------------------------------- | ------------------------- |
| `test/results/feature_smoke_report.json`  | `run_feature_test.sh`     |
| `test/results/security_smoke_report.json` | `run_security_test.sh`    |
| `test/results/latency_report.json`        | `run_benchmark.sh`        |
| `test/results/enroll_latency_report.json` | `run_enroll_benchmark.sh` |
| `test/results/concurrent_report.json`     | `run_concurrent_test.sh`  |


---



## 生产部署验收清单

- [ ] `COMPOSE_MODE=prod`，`.env` 设置 `API_KEY` 与 `ENVIRONMENT=production`
- [ ] `API_KEY=... ./test/scripts/run_all.sh` 四套全过（自动 `SKIP_LIVENESS=false`）
- [ ] 或分项：`run_feature_test` / `run_security_test`（`REQUIRE_PRODUCTION` 自动）/ latency / concurrent
- [ ] `latency_report.json`：identify/verify p95 满足 SLA
- [ ] `concurrent_report.json`：`failed=0`，QPS 达标

---



## 目录结构

```
test/
├── README.md                 # 本文档
├── requirements.txt
├── enroll_images/            # 按人名分子目录（gitignore）
├── enroll_add/               # 追加注册测试图
├── results/                  # JSON 报告（gitignore）
├── integration/
│   ├── common.py             # 共享 HTTP 工具
│   ├── feature_smoke_test.py # 主流程 E2E
│   ├── security_smoke_test.py# P0 安全/负向
│   ├── latency_benchmark.py  # 时延（identify + verify）
│   └── concurrent_benchmark.py # 并发/QPS
└── scripts/
    ├── _ensure_deps.sh
    ├── _env.sh                 # 探测 prod / 优选人脸图
    ├── run_feature_test.sh
    ├── run_security_test.sh
    ├── run_benchmark.sh
    ├── run_concurrent_test.sh
    ├── run_enroll_benchmark.sh
    └── run_all.sh
```

---



## 与单元测试的区别


|     | 单元测试     | 本目录集成测试                 |
| --- | -------- | ----------------------- |
| 依赖  | 无外部服务    | 需运行中的 face-api          |
| 范围  | 函数/类     | HTTP API 契约 + E2E       |
| 运行  | `pytest` | `test/scripts/run_*.sh` |
| 数据  | mock     | 真实图片 + 真实底库             |


如需后续接入 CI，可将 `test/scripts/run_all.sh` 作为 pipeline 步骤，并在 job 中先 `compose up` 等待 `/v1/ready`。

---



## GPU / 设备验证

时延/并发报告中的 `device` 字段来自 `/v1/health`：

- `cuda:0` — GPU 推理
- `cpu` — CPU 回退

详见 `[deploy/README.md](../deploy/README.md)` 中的 GPU 排查说明。

---



## 从 deploy_test 迁移

原 `deploy_test/` 已合并为 `test/`：


| 旧路径                                 | 新路径                                               |
| ----------------------------------- | ------------------------------------------------- |
| `deploy_test/feature_smoke_test.py` | `test/integration/feature_smoke_test.py`          |
| `deploy_test/latency_benchmark.py`  | `test/integration/latency_benchmark.py`（含 verify） |
| `deploy_test/run_*.sh`              | `test/scripts/run_*.sh`                           |
| —                                   | `test/integration/security_smoke_test.py`（新增）     |
| —                                   | `test/integration/concurrent_benchmark.py`（新增）    |


