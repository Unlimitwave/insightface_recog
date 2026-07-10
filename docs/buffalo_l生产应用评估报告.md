# buffalo_l 生产应用评估报告

> 面向业务系统选型与上线决策的技术报告。  
> 基于 InsightFace Model Zoo 公开指标、MFR/FRVT 对标数据及本项目评测文档整理。  
> 生成日期：2026-07-03

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [公开评测集概览](#2-公开评测集概览)
3. [业务系统应跑哪些测试](#3-业务系统应跑哪些测试)
4. [IJB-C 评测实操指南](#4-ijb-c-评测实操指南)
5. [IJB-C 数据准备与存储](#5-ijb-c-数据准备与存储)
6. [buffalo_l 模型概况](#6-buffalo_l-模型概况)
7. [buffalo_l 能否用于生产](#7-buffalo_l-能否用于生产)
8. [与生产门槛的量化对比](#8-与生产门槛的量化对比)
9. [各业务场景指标要求](#9-各业务场景指标要求)
10. [公开基准到私有生产的落差](#10-公开基准到私有生产的落差)
11. [上线前检查清单](#11-上线前检查清单)
12. [参考资料](#12-参考资料)

---

## 1. 执行摘要

| 维度 | 结论 |
|------|------|
| **算法能力** | `buffalo_l` 是 InsightFace 默认推荐包，开源模型中综合表现第一梯队 |
| **工业 1:1 核心指标** | IJB-C(E4) = **97.25%**（TAR@FAR=1e-4），适合作为门禁/考勤类场景选型参考 |
| **国内用户风险** | MFR East Asian 子集仅 **74.96%**（TAR@FAR=1e-6），国内 C 端不能只看 MR-ALL 总分 |
| **商业许可** | **禁止直接用于商业生产**，仅限非商业研究；商用需联系 InsightFace 获取授权 |
| **生产验收** | 无统一「及格分数线」；必须在**私有真实数据**上按业务 FAR 测 TAR，不能只看公开基准 |

**一句话建议**：`buffalo_l` 适合 PoC、算法验证和一般门禁类能力评估；正式商用需解决许可问题，并在私有数据上完成 IJB-C + MFR East Asian + 场景回放验收。

---

## 2. 公开评测集概览

人脸识别领域有完整的公开评测体系，按难度与用途分层如下。

### 2.1 快速验证（1:1，易上手）

| 数据集 | 规模 | 特点 | 获取 |
|--------|------|------|------|
| **LFW** | 5749 人 / 6000 对 | 最经典，正面照为主 | [官网](http://vis-www.cs.umass.edu/lfw/) |
| **CFP-FP** | 500 人 / 7000 对 | 侧脸/姿态变化 | InsightFace 验证集 |
| **AgeDB-30** | 570 人 / 6000 对 | 跨年龄 | InsightFace 验证集 |
| **CALFW / CPLFW** | 同 LFW 人数 | 跨年龄 / 跨姿态变体 | 公开 |

**指标**：Accuracy。好模型普遍 99%+，**已饱和，区分度有限**。

### 2.2 工业级基准（推荐）

| 数据集 | 特点 | 常用指标 |
|--------|------|----------|
| **IJB-C** | 姿态、光照、低质量、多模板，工业界 1:1 金标准 | TAR@FAR=1e-4 / 1e-5 |
| **MegaFace** | 100 万 gallery 的 1:N 识别 | TAR@FAR=1e-6 / Rank-1 |
| **MFR-Ongoing / IFRT** | 非名人、与训练集重叠少，评测更公平 | TAR@FAR=1e-6（MR-ALL） |

### 2.3 官方与持续评测

| 平台 | 说明 |
|------|------|
| **MFR-Ongoing** | InsightFace 官方 ongoing 评测，含多种族、口罩、儿童子集 |
| **WebFace260M Benchmark** | 大规模训练与评测框架 |
| **NIST FRVT** | 工业/政府级认证（需正式提交） |

> 数据集列表详见 [`recognition/_datasets_/README.md`](../recognition/_datasets_/README.md)

---

## 3. 业务系统应跑哪些测试

### 3.1 按业务形态选择

| 业务类型 | 典型场景 | 核心协议 |
|----------|----------|----------|
| **1:1 验证** | 刷脸登录、实人认证、支付核验 | 两张脸是否同一人 |
| **1:N 识别** | 门禁、考勤、访客库检索 | 从 N 人底库中找最像的一个 |

### 3.2 推荐测试组合（按优先级）

#### 必跑（工业界通用）

| 测试 | 为什么 | 关注指标 |
|------|--------|----------|
| **IJB-C** | LFW 已饱和；IJB-C 覆盖真实难点 | TAR@FAR=**1e-4**（门禁）、**1e-5**（支付/高安全） |
| **MFR MR-ALL** | 多种族泛化，与 MS1M 等训练集重叠少 | TAR@FAR=1e-6；**重点看 East Asian** |

#### 按场景加测

| 测试 | 适用场景 |
|------|----------|
| **CFP-FP** | 闸机侧脸、非正面抓拍 |
| **AgeDB-30** | 证件照 vs 现场照、跨年龄 |
| **MegaFace** | 1:N 大库（上千~百万人） |
| **MFR Mask** | 口罩场景 |
| **MFR Children** | 儿童脸（校园、亲子） |

#### 一般不必作为主依据

| 测试 | 原因 |
|------|------|
| LFW | 仅作冒烟测试 |
| CALFW / CPLFW | 偏学术，除非业务明确跨年龄/姿态 |
| NIST FRVT | 政府/金融合规才需要，流程重 |

### 3.3 按场景的最小组合

| 业务类型 | 最少应跑 |
|----------|----------|
| 通用 1:1（登录/实人） | IJB-C + AgeDB-30 + CFP-FP |
| 面向国内 C 端 | 上述 + **MFR MR-ALL（East Asian）** |
| 1:N 考勤/门禁 | 上述 + MegaFace（或自建万级库） |
| 高安全/支付 | **IJB-C @ FAR=1e-5** 为主 |
| 口罩/儿童 | MFR Mask / Children |

### 3.4 三层评测策略

```
第一层：公开基准（选型）
  → LFW + CFP-FP + AgeDB + IJB-C + MFR 子集

第二层：场景专项（预验收）
  → 口罩 / 儿童 / 多种族 / 低照度 / 侧脸 / 老化

第三层：私有真实数据（最终验收）
  → 实际摄像头、用户群、底库规模
```

> 公开基准分数高 ≠ 业务一定好用。详见 [`人脸识别算法指标与评测指南.md`](./人脸识别算法指标与评测指南.md)。

---

## 4. IJB-C 评测实操指南

### 4.1 推荐方案：ONNX 模型 + onnx_ijbc.py

```bash
cd recognition/arcface_torch

pip install opencv-python numpy pandas torch mxnet onnxruntime-gpu \
  prettytable scikit-learn scikit-image insightface

# 准备模型目录（示例：buffalo_l）
mkdir -p ~/models/buffalo_l_onnx
cp ~/.insightface/models/buffalo_l/w600k_r50.onnx ~/models/buffalo_l_onnx/

CUDA_VISIBLE_DEVICES=0 python onnx_ijbc.py \
  --model-root ~/models/buffalo_l_onnx \
  --image-path /path/to/IJB_release/IJBC \
  --target IJBC
```

### 4.2 备选方案：单文件 ONNX + ijb_evals.py

```bash
cd recognition/_evaluation_/ijb

python ijb_evals.py \
  -m ~/.insightface/models/buffalo_l/w600k_r50.onnx \
  -d /path/to/IJB_release \
  -s IJBC \
  -b 128
```

> 注意：`onnx_ijbc.py` 的 `--image-path` 指向 `IJBC/` 目录；`ijb_evals.py` 的 `-d` 指向包含 `IJBC/` 的父目录。

### 4.3 PyTorch 自训模型

```bash
cd recognition/arcface_torch

CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix /path/to/backbone.pth \
  --image-path /path/to/IJB_release/IJBC \
  --result-dir ./results/my_model \
  --batch-size 128 \
  --target IJBC \
  --network iresnet50
```

### 4.4 结果解读

输出表格中关注列：

| 列名 | 含义 | 业务参考 |
|------|------|----------|
| **1e-04** | TAR@FAR=0.01% | 门禁、考勤 |
| **1e-05** | TAR@FAR=0.001% | 支付、高安全 |
| **1e-06** | TAR@FAR=0.0001% | 金融、MFR 多种族 |

### 4.5 在线评测（免本地数据）

导出 ONNX 提交 [MFR-Ongoing](http://iccv21-mfr.com/)（含 IJB-C 指标）。服务可能间歇下线，见 [`challenges/mfr/README.md`](../challenges/mfr/README.md)。

---

## 5. IJB-C 数据准备与存储

### 5.1 重要现状：NIST 已停止分发

**NIST 已于 2023 年 3 月 14 日停止 IJB-A/B/C 公开分发**，无法再向 NIST 新申请。

- 官方说明：[Face Challenges | NIST](https://www.nist.gov/programs-projects/face-challenges)
- 原申请页已标注停止：[IJB-C Dataset Request Form](https://www.nist.gov/itl/tted/btg/ijb-c-dataset-request-form)

### 5.2 推荐数据获取：InsightFace testsuite

| 来源 | 链接 |
|------|------|
| Google Drive | [testsuite](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view?usp=sharing) |
| 百度网盘 | [链接](https://pan.baidu.com/s/1oer0p4_mcOrs4cfdeWfbFg) |
| 更新版 meta | [百度 code:7g8o](https://pan.baidu.com/s/1x-ytzg4zkCTOTtklUgAhfg) / [GDrive](https://drive.google.com/file/d/1MXzrU_zUESSx_242pRUnVvW_wDzfU8Ky/view?usp=sharing) |

解压后目录结构：

```
IJB_release/
└── IJBC/
    ├── meta/
    │   ├── ijbc_face_tid_mid.txt
    │   ├── ijbc_template_pair_label.txt
    │   └── ijbc_name_5pts_score.txt
    └── loose_crop/          # 约 22.7 万张裁剪人脸
        └── *.jpg
```

### 5.3 存储需求

| 方案 | 内容 | 建议预留 |
|------|------|----------|
| **InsightFace testsuite（推荐）** | loose_crop + meta | **50 GB** |
| **NIST 完整原始包（历史备份）** | 原图 + 视频 + 帧 + 协议 | **300 GB+** |

### 5.4 NIST 原始数据 vs InsightFace 评测数据

| 项目 | NIST 官方 IJB-C | InsightFace 评测用 IJB-C |
|------|-----------------|--------------------------|
| 图像规模 | ~13.8 万脸图 + 11.7 万帧 + 视频 | 评测脚本用 **227,630** 张 |
| 体积 | ~200 GB+ | ~15–30 GB |
| 能否直接跑 InsightFace 脚本 | 否，需自行 crop | 是 |
| 与论文数字对齐 | 协议不同可能不一致 | 与 Model Zoo 报告一致 |

---

## 6. buffalo_l 模型概况

### 6.1 模型组成

| 模块 | 模型 | 说明 |
|------|------|------|
| 检测 | SCRFD-10GF | 精度高，算力需求较大 |
| 识别 | ResNet50@WebFace600K | 512 维特征 |
| 对齐 | 2d106 + 3d68 | 完整 pipeline |
| 属性 | 性别/年龄 | 可选 |
| **总体积** | **326 MB** | InsightFace 默认推荐包 |

### 6.2 公开基准成绩（Model Zoo）

| 指标 | buffalo_l | 含义 |
|------|-----------|------|
| **LFW** | 99.83% | 1:1 Accuracy（已饱和） |
| **CFP-FP** | 99.33% | 侧脸/大角度 |
| **AgeDB-30** | 98.23% | 跨年龄 |
| **IJB-C(E4)** | **97.25%** | TAR@FAR=1e-4 |
| **MR-ALL** | 91.25% | TAR@FAR=1e-6，多种族综合 |

### 6.3 多种族子集（MFR @ FAR=1e-6）

| 子集 | buffalo_l |
|------|-----------|
| African | 90.29% |
| Caucasian | 94.70% |
| South Asian | 93.16% |
| **East Asian** | **74.96%** |
| **MR-ALL** | **91.25%** |

### 6.4 同系列对比

| 模型 | IJB-C(E4) | MR-ALL | 体积 |
|------|-----------|--------|------|
| **buffalo_l** | **97.25%** | **91.25%** | 326MB |
| buffalo_m | 97.25%（同 buffalo_l） | 91.25% | 313MB |
| buffalo_s | 95.02% | 71.87% | 159MB |
| antelopev2 | 更高 | 更高 | 407MB |

---

## 7. buffalo_l 能否用于生产

### 7.1 许可限制（首要问题）

Model Zoo 与 python-package 明确声明：

> **ALL models are available for non-commercial research purposes only.**  
> **这些模型仅供非商业研究用途。**

| 场景 | 能否使用 buffalo_l |
|------|-------------------|
| 内部 PoC、研发验证 | ✅ 可以 |
| 对外商业产品（门禁、考勤、支付等） | ❌ **不可以** |
| 政府/金融合规项目 | 需商业授权或自研/采购合规模型 |

**商用路径**：

- 邮件联系：**contact@insightface.ai**
- 使用 InspireFace 商业 SDK / 更高精度模型
- 使用项目 [`commercial_evaluation.md`](../python-package/docs/commercial_evaluation.md) 在私有数据评测后洽谈授权

### 7.2 算法能力评估

| 业务场景 | 安全等级 | buffalo_l 公开基准评估 |
|----------|----------|------------------------|
| 普通门禁/考勤 | FAR≈1e-4 | **基本够用**（IJB-C 97.25%） |
| 刷脸登录/实人认证 | FAR≈1e-4~1e-5 | **可用，需压测** |
| 支付级核验 | FAR≈1e-5~1e-6 | **偏紧**，建议更强模型 |
| 大库 1:N（万人+） | — | **需单独测**，公开表无 MegaFace 数据 |
| 国内 C 端多种族 | FAR≈1e-6 | **East Asian 74.96% 不达标** |

---

## 8. 与生产门槛的量化对比

### 8.1 核心认知：没有统一「及格分数线」

生产验收逻辑：

```
业务先定 FAR 上限（安全线）
        ↓
在私有真实数据上测 TAR@该FAR
        ↓
TAR 是否满足体验 KPI（拒识率能否接受）
```

- **FAR=1e-4 / 1e-5** 是业务约束，不是模型「考分」
- **TAR@FAR** 才是模型能力指标
- **LFW 99.83%** 对生产决策几乎无参考价值

### 8.2 buffalo_l 相对各档参考线

| 指标 | buffalo_l | 勉强可用 | 较好生产参考 | 商业顶级（FRVT） | 相对「较好生产」 |
|------|-----------|----------|--------------|------------------|------------------|
| IJB-C @ 1e-4 | **97.25%** | ~95% | 96~98% | ~98%+ | 基本持平 |
| IJB-C @ 1e-5 | ~94~95%（估） | ~90~93% | 93~96% | ~97%+ | 接近或略低 |
| MR-ALL @ 1e-6 | **91.25%** | ~72% | 88~92% | **97.48%** | **低约 6 点** |
| East Asian @ 1e-6 | **74.96%** | ~51% | ≥80~85% | **87.69%** | **低约 10~12 点** |
| LFW | 99.83% | 99.5%+ | 无意义 | 无意义 | 无参考价值 |

> FRVT 商业顶级数据来自 MFR-Ongoing 基线表 `insightface-000 of frvt`（MR-ALL 97.481%，East Asian 87.694%）。

### 8.3 分场景结论

| 场景 | buffalo_l 相对生产门槛 |
|------|------------------------|
| 一般门禁（@1e-4） | 公开集已达较好水平；比 buffalo_s 高约 **2~2.5%** |
| 实名核验（@1e-5） | **接近及格线**，无明显富余 |
| 国内 C 端（@1e-6） | East Asian **低于生产期望约 10~12 点** |
| LFW | 比「能用」高约 0.3%，**对生产无意义** |

### 8.4 同一模型：FAR 越严，TAR 越低

| FAR 目标 | 典型 TAR（示例） | buffalo_l 级参考 | 业务场景 |
|----------|------------------|------------------|----------|
| **1e-4** | ~96% | **97.25%**（IJB-C） | 门禁、考勤 |
| **1e-5** | ~78%~94% | ~94~95%（估） | 实名核验 |
| **1e-6** | ~65%~91% | **91.25%**（MR-ALL） | 金融、MFR |

规律：**FAR 目标越严 → 阈值越高 → TAR 越低**。比较模型必须在**同一 FAR** 下进行。

---

## 9. 各业务场景指标要求

以下为**私有真实数据**上的经验目标（行业实践，非国标）：

### 9.1 普通门禁 / 考勤（FAR ≈ 1e-3 ~ 1e-4）

| 项目 | 建议 |
|------|------|
| 公开基准选型 | IJB-C @ 1e-4 ≥ **95%** |
| **私有数据验收** | TAR@FAR=1e-4 ≥ **90~93%** |
| buffalo_l | IJB-C 97.25%，私有预估 **89~94%** → **基本够用** |

### 9.2 刷脸登录 / 实人认证（FAR ≈ 1e-4 ~ 1e-5）

| 项目 | 建议 |
|------|------|
| 公开基准选型 | IJB-C @ 1e-5 ≥ **93%** |
| **私有数据验收** | TAR@FAR=1e-5 ≥ **85~90%** |
| buffalo_l | 估计公开 **94~95%**，私有 **87~92%** → **勉强达标** |

### 9.3 金融 / 支付级（FAR ≈ 1e-5 ~ 1e-6）

| 项目 | 建议 |
|------|------|
| 公开基准选型 | IJB-C @ 1e-5 ≥ **95%**，MR-ALL @ 1e-6 ≥ **90%** |
| **私有数据验收** | TAR@FAR=1e-6 ≥ **80~85%**（常配合活体） |
| buffalo_l | MR-ALL 91.25% 尚可，但 East Asian **74.96%** → **不建议单独使用** |

### 9.4 1:N 大库检索

| 项目 | 建议 |
|------|------|
| 公开基准 | MegaFace Rank-1 或 IJB-C 1:N |
| **私有验收** | 按实际底库规模建库，测 Top-1 / TPIR@FPIR |

### 9.5 私有验收数据集规模建议

| 阶段 | 身份数 | 每人图像数 | 负样本对规模 |
|------|--------|------------|--------------|
| POC | 50~200 | 3~10 | ≥ 10,000 |
| 预生产 | 500~2000 | 5~20 | ≥ 100,000 |
| 正式验收 | 接近生产底库量级 | 真实采集条件 | 支撑目标 FAR 统计 |

> 估计 FAR=1e-4 至少需要 **10,000** 次负样本比对；FAR=1e-6 需要 **100 万次以上**。

---

## 10. 公开基准到私有生产的落差

### 10.1 经验公式

```
私有 TAR ≈ 公开 IJB-C TAR − (3% ~ 10%)
```

落差取决于：摄像头质量、光照、距离、底库图质量、口罩等。

### 10.2 buffalo_l 粗算（IJB-C @ 1e-4 = 97.25%）

| 私有落差 | 预估私有 TAR | 门禁 | 实名 |
|----------|--------------|------|------|
| −3% | **94%** | ✅ 够 | 边缘 |
| −5% | **92%** | ✅ 够 | 边缘 |
| −8% | **89%** | 边缘 | ❌ 偏紧 |

### 10.3 经验底线

| 条件 | 判断 |
|------|------|
| 私有 TAR@FAR=1e-4 **< 88%** | 一般门禁也偏紧 |
| 私有 TAR@FAR=1e-5 **< 82%** | 登录/实人体验差 |
| 只看 LFW **> 99.5%** | **不能**作为上线依据 |

---

## 11. 上线前检查清单

- [ ] 公开基准：LFW + CFP-FP + AgeDB-30
- [ ] 工业基准：**IJB-C @ 1e-4 和 1e-5**
- [ ] 泛化基准：**MFR MR-ALL @ 1e-6**（重点 East Asian）
- [ ] 业务专项：口罩 / 儿童 / 多种族（按场景）
- [ ] **私有 1:1 或 1:N 数据集验收**
- [ ] **注册策略**：每人 3~5 张、1:N 用项目内置多模板，1:1 生产实现 Max 比对（见 [指标指南第 8 节](./人脸识别算法指标与评测指南.md#8-人脸注册与多模板实践)）
- [ ] 负样本规模支撑目标 FAR
- [ ] 阈值在私有集上标定并文档化
- [ ] 端到端延迟与并发压测
- [ ] 活体检测（高安全场景）
- [ ] 隐私合规：告知同意、数据留存、权限审计
- [ ] **模型许可证：商用授权确认**

---

## 12. 参考资料

### 项目内文档

| 文档 | 路径 |
|------|------|
| 人脸识别算法指标与评测指南 | [`docs/人脸识别算法指标与评测指南.md`](./人脸识别算法指标与评测指南.md) |
| Model Zoo 精度表 | [`model_zoo/README.md`](../model_zoo/README.md) |
| 数据集列表 | [`recognition/_datasets_/README.md`](../recognition/_datasets_/README.md) |
| IJB 评测说明 | [`recognition/_evaluation_/ijb/README.md`](../recognition/_evaluation_/ijb/README.md) |
| IJB-C 评测脚本 | [`recognition/arcface_torch/onnx_ijbc.py`](../recognition/arcface_torch/onnx_ijbc.py) |
| MFR / IFRT 评测 | [`challenges/mfr/README.md`](../challenges/mfr/README.md) |
| 商业私有评测 | [`python-package/docs/commercial_evaluation.md`](../python-package/docs/commercial_evaluation.md) |
| InspireFace 商业 SDK | [`cpp-package/inspireface/README.md`](../cpp-package/inspireface/README.md) |

### 外部链接

| 资源 | 链接 |
|------|------|
| NIST Face Challenges | https://www.nist.gov/programs-projects/face-challenges |
| NIST FRVT | https://pages.nist.gov/frvt/html/frvt11.html |
| MFR 在线评测 | http://iccv21-mfr.com/ |
| InsightFace 商业联系 | contact@insightface.ai |

---

*本报告基于 InsightFace 开源仓库公开资料整理，指标以 Model Zoo 最新公布值为准。生产决策请以私有数据验收结果及法务合规审查为最终依据。*
