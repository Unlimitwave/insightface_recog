# 多注册张数对比实验（1:N 识别 + 1:1 验证，MegaFace / FaceScrub）

本目录用于对比 **每人注册 1 / 3 / 5 / 10 张照片** 时，对 **1:N 识别** 与 **1:1 验证** 指标的影响。

两种评测共用同一数据划分与 Max 融合策略，但协议与指标含义不同：

| 场景 | 典型业务 | 核心问题 | 主要指标（本实验输出） |
|------|----------|----------|------------------------|
| **1:N 识别** | 门禁、考勤、黑名单 | 现场照在底库里认出了谁？ | Rank-1 / Rank-5 / mAP / TPIR@FPIR / FNIR / FPIR |
| **1:1 验证** | 刷脸登录、人证比对 | 现场照是不是已注册本人？ | TAR@FAR / FRR@FAR / EER / 最佳阈值 Accuracy |

> 每人多张注册照各自存一个向量，绑定同一 `person_id`，比对时取 **Max 相似度**（不是 Mean 融合）。

---

## 1. 实验目的

回答业务问题：

- **1:N**：注册照片越多，门禁/考勤类认人成功率是否显著提升？增加几张最划算？
- **1:1**：注册照片越多，刷脸登录/人证比对的通过率（TAR）是否提高？对误识（FAR）影响如何？

在百万级底库干扰（MegaFace distractor）下量化 1:N 影响；1:1 则在已注册身份间构造正负样本对，模拟「已知道是谁，再确认是不是本人」。

---

## 2. 评测协议

### 2.1 数据

| 组件 | 来源 | 作用 |
|------|------|------|
| **注册身份** | FaceScrub（`data/facescrub_images/`，80 人） | 目标底库，每人 27~50 张图 |
| **探针** | FaceScrub 同一人的剩余照片 | 模拟现场抓拍 |
| **Distractor** | MegaFace（默认 100 万条） | **仅 1:N 使用**，无身份干扰项 |

### 2.2 数据划分（两种评测共用）

对每个人、每个注册张数 `k`：

```
按文件名排序后：
  前 k 张  → 注册（gallery），每图一个向量，绑定同一 identity
  剩余     → 探针（probe），不参与注册
```

例如 `k=3` 时，Adam_Brody 的前 3 张作注册，第 4 张起作探针。

### 2.3 多模板融合（Max 策略）

与生产系统一致：

```
探针 vs 张三的 k 张注册照 → 得分 [0.52, 0.68, 0.55]
张三最终得分 = max(0.52, 0.68, 0.55) = 0.68
```

- **每张照片单独存一个向量**，不合并成单个模板
- **同一人取最高分**，对单张坏图/角度差异更鲁棒
- **不是 Mean 融合**（平均向量）

---

## 3. 1:N 识别协议

### 3.1 底库构成

```
底库 = FaceScrub 注册模板（80 人 × k 张/人）
     + MegaFace distractor（100 万条，无身份）
```

**Distractor 的作用**：模拟生产环境中「底库里有大量无关人员」的真实场景。Rank-1 判定要求正确身份得分不仅要在 80 人中排第一，还要 **≥ 100 万 distractor 中的最高分**，否则视为识别失败。

### 3.2 指标（生产上线推荐）

| 指标 | 英文 | 含义 | 上线用途 |
|------|------|------|----------|
| **Rank-1** | Top-1 | 正确身份在「80 人 + distractor」全局排序第 1 | 评估认人能力（无阈值） |
| **Rank-5** | Top-5 | 正确身份出现在全局 Top-5 | 人工复核场景、候选列表 |
| **mAP** | mean AP | 正确身份排序位置的综合质量 | 学术对比 / 检索质量 |
| **TPIR@FPIR** | True Pos. ID Rate | 在误识率 FPIR ≤ ε 下，本人被正确识别的比例 | **开集门禁核心指标** |
| **FNIR** | False Neg. ID Rate | 漏检率 = 1 − TPIR | 本人被拒识的概率（体验） |
| **FPIR** | False Pos. ID Rate | 误检率 = 陌生人被硬认成底库某人的概率 | **安全红线** |

**TPIR@FPIR 正负样本构造**（每条 enrolled 探针）：

```
正样本 = 探针 vs 正确身份的 Max 分
负样本 = max(最佳错误身份 Max 分, distractor 最高分)
```

在全部正负样本上扫阈值，在 FPIR ≤ 目标值（如 1e-4）约束下取 TPIR 最大 → 得到 TPIR@FPIR=1e-4 及对应 FNIR / FPIR / 阈值。

> Rank-1 / mAP 看排序能力；**上线阈值选型必须看 TPIR@FPIR + FNIR/FPIR**。

---

## 4. 1:1 验证协议

### 4.1 场景说明

1:1 验证模拟 **刷脸登录、人证比对**：系统**已经知道**要比对的是谁（例如登录账号对应「张三」），只需判断现场抓拍是不是张三本人。

生产中的单次比对：

```
用户登录账号 → 已知 person_id = 张三
现场抓拍 1 张 query
score = max(cos(query, 张三注册照1), cos(query, 张三注册照2), cos(query, 张三注册照3))
is_same = score >= threshold    # 是本人 → 通过；不是 → 拒识
```

**与 1:N 的关键区别**：

| | 1:N 识别 | 1:1 验证 |
|---|----------|----------|
| 问题 | 这张脸是底库里的谁？ | 这张脸是不是指定的某人？ |
| 比对对象 | 80 人 + 100 万 distractor | 仅 80 个已注册身份 |
| 是否含 distractor | ✅ 含 | ❌ 不含 |

---

### 4.2 数据怎么切？（以 k=3 为例）

以 **Adam_Brody** 为例，假设他共有 30 张照片（按文件名排序）：

```
Adam_Brody/
  001.jpg  ─┐
  002.jpg   ├─ 前 3 张 → 注册（gallery），各存 1 个向量
  003.jpg  ─┘
  004.jpg  ─┐
  005.jpg   │
  ...       ├─ 第 4 张起 → 探针（probe），模拟「现场抓拍」
  030.jpg  ─┘
```

**k=3 时，Adam_Brody 有：**

| 角色 | 数量 | 说明 |
|------|------|------|
| 注册照 | 3 张 | 001、002、003，各 1 个 embedding |
| 探针 | 27 张 | 004~030，每张当作一次「现场抓拍」 |

80 人全部按同样规则切分。k=3 时全库合计 **3266 张探针**（每人「总照片数 − 3」之和）。

> **要点**：不是只用 1 张 probe，而是**每人的剩余照片全部当作 probe**。每多 1 张注册照，该人可用 probe 就少 1 张。

---

### 4.3 每张 probe 怎么算分？（k=3 逐步示例）

取 Adam_Brody 的 **004.jpg** 作为一次现场抓拍（probe），与 80 人底库做 1:1 比对：

#### Step 1：probe 与 Adam 本人 3 张注册照分别算相似度

```
cos(004.jpg, 001.jpg) = 0.58
cos(004.jpg, 002.jpg) = 0.72   ← 最高
cos(004.jpg, 003.jpg) = 0.61

→ 正样本分数 = max(0.58, 0.72, 0.61) = 0.72
```

**1 张 probe 只产生 1 条正样本分数**（对本人 k 张注册照取 Max，不是 k 条分开算）。

#### Step 2：probe 与每个「错误身份」的 3 张注册照算 Max 分

```
vs 其他人_A：max(cos(004, A的001), cos(004, A的002), cos(004, A的003)) = 0.31
vs 其他人_B：max(...) = 0.28
...
vs 其他人_共79人：各得 1 个分数
```

**1 张 probe 产生 79 条负样本分数**（80 人底库去掉本人）。

#### Step 3：汇总到全库

```
Adam 的 27 张 probe  →  27 条正样本  +  27×79 = 2133 条负样本
其他 79 人各自的 probe 同理累加
─────────────────────────────────────────────────────────
全库 k=3：3266 条正样本  +  3266×79 = 258014 条负样本
```

---

### 4.4 和真实业务的对应

| 评测中的操作 | 真实业务场景 |
|-------------|-------------|
| 004.jpg 当作 probe | 用户刷脸登录时的现场抓拍 |
| 001~003.jpg 当作注册照 | 用户开户时录入的 3 张底库照 |
| 正样本分 0.72 | 系统算出的「与本人相似度」 |
| 负样本分 0.31 等 | 若误把现场照与其他人比对，得到的相似度（用于估 FAR） |
| TAR@FAR | 在「误识率 ≤ 某值」约束下，本人能通过的比例 |

**一次真实 1:1 登录** = 1 张 probe vs 1 个已知身份的 k 张注册照 → 得到 1 个分数 → 与阈值比。

**本实验**把每人的每张剩余照片都当作一次独立登录，批量重复上述过程，再统计 TAR@FAR。

---

### 4.5 指标（生产上线推荐）

| 指标 | 英文 | 含义 | 上线用途 |
|------|------|------|----------|
| **TAR@FAR** | True Accept Rate | 在 FAR ≤ ε 下，本人被 Accept 的比例 | **通过率**（核心） |
| **FRR@FAR** | False Reject Rate | 拒识率 = 1 − TAR | 本人刷脸失败概率（体验） |
| **FAR** | False Accept Rate | 误识率，陌生人被 Accept 的比例 | **安全红线** |
| **EER** | Equal Error Rate | FAR = FRR 时的错误率 | 模型选型、阈值初估 |
| **Accuracy** | — | 验证集上使 Accuracy 最大的阈值及对应 TAR/FAR/FRR | 阈值选型参考 |

**关系**：在同一 FAR 工作点，**TAR + FRR = 100%**（TAR 是本人通过率，FRR 是本人被拒识率）。

**TAR@FAR 计算**（k=3 时共 3266 正 + 258014 负样本）：

```
Step 1. 每张 probe 产生 1 条正样本（vs 本人 3 张注册照 Max 分）
Step 2. 每张 probe 产生 79 条负样本（vs 每个错误身份 Max 分）
Step 3. 在全部正负样本上扫阈值 threshold
Step 4. FAR = P(负样本 >= threshold)
Step 5. TAR = P(正样本 >= threshold)
Step 6. 在 FAR <= 1e-4 约束下取 TAR 最大 → TAR@FAR=1e-4，FRR = 1 - TAR
```

> 1:1 的 FAR 目标（1e-2~1e-5）与 1:N 的 FPIR 目标（1e-2~1e-6）**数值不可直接对比**，但语义类似。

---

### 4.6 预期结论（与文档第 8 节一致）

- **1 → 3 张**：TAR 提升最明显（现场照更容易与某张注册照匹配）
- **3 → 5 张**：仍有收益，但边际递减
- **对 FAR 影响较小**：陌生人需对多张模板都产生高相似度才易误过，通常更难
- **提升幅度小于 1:N**：1:1 不涉及百万 distractor 竞争，基线已经很高（buffalo_l 在 FaceScrub 上 TAR@1e-4 约 99.4%~99.8%）

---

## 5. 生产上线指标选用建议

| 业务场景 | 必看指标 | 说明 |
|----------|----------|------|
| **门禁 / 考勤（1:N）** | Rank-1 + **TPIR@FPIR=1e-4** + FNIR + FPIR | Rank-1 看认人；FPIR 控制陌生人误开；FNIR 看本人漏过 |
| **黑名单检索（1:N）** | Rank-1 / Rank-5 + TPIR@FPIR | Rank-5 可用于人工复核候选 |
| **MegaFace / 学术对比（1:N）** | Rank-1 + mAP | 与论文对齐 |
| **刷脸登录 / 人证比对（1:1）** | **TAR@FAR=1e-4~1e-5** + **FRR@FAR** | TAR 看通过率，FRR 看本人被拒概率 |
| **1:1 阈值选型** | EER + 最佳 Accuracy 阈值 | 上线前在私有验证集微调 |
| **模型选型** | Rank-1（1:N）+ EER（1:1） | 快速对比不同模型 |

**代码结构**：

```text
enroll_count_eval.py   # 主流程：数据划分 → 相似度矩阵 → 调用指标模块
enroll_metrics.py      # 标准化指标：Rank/mAP/TPIR/TAR/FRR/EER
```

---

## 6. 前置条件

需先完成 MegaFace 特征提取（GPU 阶段，只需跑一次）：

```bash
cd recognition/_evaluation_/megaface
bash run_buffalo_l.sh
```

特征输出目录：

```text
$MEGAFACE_ROOT/feature_out_clean/buffalo_l/
  facescrub/   # FaceScrub 特征
  megaface/    # MegaFace distractor 特征
```

本实验 **不再跑模型推理**，只读取已有 `.bin` 特征做 CPU 矩阵运算。

---

## 7. 运行方式

### 7.1 一键运行

```bash
cd recognition/_evaluation_/enroll_experiment_megaface
conda activate face_recog_env
bash run.sh
```

### 7.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MEGAFACE_ROOT` | `/home/cmsr/桌面/东风/数据集/megaface` | 数据集根目录 |
| `ALGO` | `buffalo_l` | 模型/特征后缀 |
| `GALLERY_SIZE` | `1000000` | distractor 数量（仅影响 1:N） |
| `ENROLL_COUNTS` | `1 3 5 10` | 对比的注册张数 |

### 7.3 直接调用 Python

```bash
conda activate face_recog_env
python enroll_count_eval.py \
  --megaface-root /path/to/megaface \
  --algo buffalo_l \
  --gallery-size 1000000 \
  --enroll-counts 1 3 5 10
```

快速验证 1:1（1:1 不依赖 distractor，可用任意 gallery-size 加速 1:N 部分）：

```bash
python enroll_count_eval.py --gallery-size 10000 --enroll-counts 1 3 5 10
```

---

## 8. 输出结果

### 8.1 文件位置

```text
results/buffalo_l/
  enroll_ablation.json   # 完整 JSON 结果（含 1:N + 1:1）
  run.log                # 运行日志
```

Distractor 特征缓存（加速重复运行，仅 1:N 使用）：

```text
$MEGAFACE_ROOT/feature_out_clean/buffalo_l/distractor_cache_1000000.npz
```

### 8.2 JSON 结构

```json
{
  "results": [
    {
      "enroll_count": 3,
      "identification_1n": {
        "rank1": 92.41,
        "rank5": 95.2,
        "map": 88.5,
        "tpir_at_fpir": { "1e-4": 1.47 },
        "fnir_at_fpir": { "1e-4": 98.53 },
        "fpir_at_fpir": { "1e-4": 0.01 }
      },
      "verification_11": {
        "positive_pairs": 3266,
        "negative_pairs": 258014,
        "tar_at_far": { "1e-4": 99.69 },
        "frr_at_far": { "1e-4": 0.31 },
        "eer": 0.5,
        "best_threshold": { "accuracy": 0.998, "threshold": 0.42 }
      }
    }
  ]
}
```

### 8.3 参考结果（buffalo_l，gallery=100 万）

#### 1:N 识别

| 注册张数 | Rank-1 | Rank-5 | mAP | 探针数 | TPIR@1e-4 | FNIR@1e-4 |
|---------|--------|--------|-----|--------|-----------|-----------|
| 1 张 | 82.98% | — | — | 3426 | 0.61% | 99.39% |
| 3 张 | 92.41% | — | — | 3266 | 1.47% | 98.53% |
| 5 张 | 94.95% | — | — | 3106 | 2.67% | 97.33% |
| 10 张 | 97.15% | — | — | 2706 | 4.77% | 95.23% |

> Rank-5 / mAP / 完整 TPIR 档位请运行 `bash run.sh` 后查看终端汇总表或 JSON。

**1:N 结论**：

- 1 → 3 张 Rank-1 提升最大（+9.4 个百分点）
- 3 → 5 张仍有明显收益（+2.5%）
- 5 → 10 张收益递减（+2.2%）
- 生产建议：**每人注册 3~5 张**，覆盖不同角度/光照

#### 1:1 验证

| 注册张数 | 正样本 | 负样本 | TAR@1e-4 | FRR@1e-4 | TAR@1e-3 | FRR@1e-3 | EER |
|---------|--------|--------|----------|----------|----------|----------|-----|
| 1 张 | 3426 | 270654 | 99.42% | 0.58% | 99.82% | 0.18% | — |
| 3 张 | 3266 | 258014 | 99.69% | 0.31% | 99.85% | 0.15% | — |
| 5 张 | 3106 | 245374 | 99.74% | 0.26% | 99.84% | 0.16% | — |
| 10 张 | 2706 | 213774 | 99.82% | 0.18% | 99.89% | 0.11% | — |

**1:1 结论**：

- TAR@FAR=1e-4 随注册张数上升：99.42% → 99.82%
- 对应 FRR@FAR=1e-4（拒识率）下降：0.58% → 0.18%
- 提升幅度远小于 1:N Rank-1（1:1 无 distractor 竞争，基线已很高）

---

## 9. 与生产系统的对应关系

### 9.1 1:N 识别

```text
生产 1:N 底库:
  person_id=张三 → [vec_001, vec_002, vec_003]   # 每张照片一个向量

现场探针 query:
  score(张三) = max(cos(query, vec_001), cos(query, vec_002), cos(query, vec_003))
  在所有人 + 底库干扰项里取得分最高者 → Top-1
```

### 9.2 1:1 验证

```text
生产 1:1 底库:
  user_id=张三 → [vec_001, vec_002, vec_003]

现场探针 query:
  score = max(cos(query, vec_001), cos(query, vec_002), cos(query, vec_003))
  is_same = score >= threshold   # 不涉及与其他人比较
```

本实验用 FaceScrub 80 人复现上述逻辑；1:N 额外加入 MegaFace 100 万 distractor。

---

## 10. 相关文档

- 多注册实践说明：`docs/人脸识别算法指标与评测指南.md` 第 8 节
- MegaFace 标准评测：`recognition/_evaluation_/megaface/README.md`
- 特征提取脚本：`recognition/_evaluation_/megaface/run_buffalo_l.sh`
