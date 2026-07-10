# 评测运行开始（IJB-C / MegaFace）

本文用于在本项目中快速启动常见人脸识别测评，并解释常用参数、评测场景与指标含义。

## 0. 你需要准备什么

- **模型（推荐 ONNX）**：例如 InsightFace `buffalo_l`，文件一般在 `~/.insightface/models/buffalo_l/w600k_r50.onnx`
- **数据集目录**：
  - **IJB-C**：包含 `IJBC/loose_crop` 与 `IJBC/meta/*.txt`
  - **MegaFace**：包含 `devkit/` 与 `data/` 等结构（本项目脚本有约定，见下文）

> 说明：IJB-C 的评测最耗时部分是 **Embedding 特征提取**（几十万张图），中途不要频繁中断。

### 0.1 配置 GPU 运行环境（ONNXRuntime CUDA）

使用 `onnxruntime-gpu` 时，除了安装包本身，还需要让系统能找到 CUDA/cuBLAS/cuDNN 等动态库。若通过 pip 安装了 `nvidia-cublas-cu12`、`nvidia-cudnn-cu12` 等包，**必须在启动 `python` 之前**把库路径加入 `LD_LIBRARY_PATH`（在 Python 进程内修改通常无效）。

**每次评测前执行（推荐）：**

```bash
conda activate face_recog_env

# face_recog_env 激活后自动加 nvidia lib 路径
export LD_LIBRARY_PATH="$(
python - <<'PY'
import glob, site
print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

**验证 CUDA provider 是否生效：**

```bash
python - <<'PY'
import os, onnxruntime as ort
model = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("providers:", sess.get_providers())
PY
```

期望输出：

```text
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

若只有 `CPUExecutionProvider`，常见原因：

- 未设置 `LD_LIBRARY_PATH`（报错如 `libcublasLt.so.12: cannot open shared object file`）
- 缺少 CUDA 12 / cuDNN 9 依赖（可用 pip 安装：`nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12`）
- 安装后仍缺库时，用 conda：`conda install -y -c nvidia -c conda-forge cuda-cublas=12.* cudnn=9.* cuda-cudart=12.*`（注意：conda 不支持 `-i` 参数）

**一劳永逸（可选）：** 将上述 `export LD_LIBRARY_PATH=...` 写入 `~/.bashrc`，或在 conda 环境的 `etc/conda/activate.d/` 下新建脚本，这样每次 `conda activate face_recog_env` 后自动生效。

> `recognition/_evaluation_/megaface/run_buffalo_l.sh` 已内置类似的 nvidia lib 路径配置；`ijb_evals.py` 跑 IJB-C 时需手动 export，或按上文配置 activate 脚本。

---

## 1. IJB-C（1:1 验证）怎么跑（推荐）

本项目推荐用 `recognition/_evaluation_/ijb/ijb_evals.py`，它能直接读取 IJB meta，并支持 ONNX 模型。

### 1.1 启动命令（IJBC 1:1）

```bash
conda activate face_recog_env
cd recognition/_evaluation_/ijb

# 先配置 GPU 库路径（见 0.1 节）
export LD_LIBRARY_PATH="$(
python - <<'PY'
import glob, site
print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=0 python ijb_evals.py \
  -m ~/.insightface/models/buffalo_l/w600k_r50.onnx \
  -d "/home/cmsr/桌面/东风/数据集/ijb的数据集/ijb" \
  -s IJBC \
  -b 128
```

### 1.2 关键参数解释（`ijb_evals.py`）

- `**-m, --model_file**`：模型路径  
  - `*.onnx`：使用 ONNXRuntime（推荐）
  - 也支持其他格式（取决于脚本分支），但实际复现一般用 ONNX 最省事
- `**-d, --data_path**`：数据集父目录（**必须包含 `IJBB/` 与/或 `IJBC/` 子目录**）  
  - 例如：`/path/to/IJB_release`  
  - 目录下应存在：`/path/to/IJB_release/IJBC/loose_crop` 与 `.../IJBC/meta/*.txt`
- `**-s, --subset`**：`IJBB` 或 `IJBC`
- `**-b, --batch_size**`：Embedding 批大小  
  - 越大通常越快，但越吃显存；OOM 就降到 64/32
- `**-N, --is_one_2_N**`：跑 **1:N**（识别）协议（不加则默认 1:1）
- `**-F, --force_reload`**：强制重读 meta 并重建 `*_backup.npz`（一般不需要）
- `**-B, --is_bunch**`：一次性跑多组合（N/D/F 的 8 种组合），耗时更久
- `**-E, --save_embeddings**`：保存中间 embeddings（是否用于续跑取决于脚本逻辑与用法）
- `**-P, --plot_only**`：只绘图（需要已有保存结果）

### 1.3 运行过程你会看到什么

典型输出阶段：

- **Loading templates / pairs / images**：读取 `meta/` 文件（会生成 `IJBC_backup.npz` 便于下次复用）
- `**Embedding: x/steps`**：正在对 `loose_crop` 图片做对齐+特征提取（最耗时）
- **Extract template feature**：把同一 template 下图片/视频帧聚合成模板特征
- **Verification**：对 meta 中的 pair 做相似度计算
- **最终表格**：输出多列 `TAR@FAR`（例如 `1e-04 / 1e-05 / 1e-06`）

> 常见 Warning（如 `FutureWarning: estimate is deprecated`）通常不影响结果，可忽略。

### 1.4 结果解读与数据查看

跑完 1:1 IJBC 后，若终端出现如下表格，说明 **评测已完整完成**（Embedding → 模板聚合 → Verification → 指标计算均结束）：

```text
>>>> plot roc and calculate tpr...
|                   |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |     0.1 |      AUC |
|:------------------|---------:|---------:|---------:|---------:|---------:|--------:|---------:|
| w600k_r50_IJBC_11 | 0.904331 | 0.956282 | 0.972593 | 0.981695 | 0.987626 | 0.99325 | 0.996721 |
```

末尾若出现 `matplotlib plot failed`，仅表示 **ROC 曲线图未画出**（常见于无图形界面的终端），**不影响上述数值结果**。

#### 这张表是什么？

这是 **IJBC 1:1 验证** 的指标汇总。默认协议为 **N0D1F1**（`use_norm_score=False`、`use_detector_score=True`、`use_flip_test=True`）。

脚本会输出 **两套表**（2025+ 生产验收版）：

1. **TAR / FRR / Threshold @ FAR**（推荐用于上线决策）
2. **Legacy TAR@FAR summary**（与旧版 / Model Zoo 对齐，便于横向对比）

| 列 | 含义 | 生产关注点 |
|----|------|-----------|
| **TAR@FAR** | 在 FAR 约束下的真接受率 | 本人通过率（越高越好） |
| **FRR@FAR** | 拒识率 = 1 − TAR | 本人被拒概率（体验） |
| **Threshold@FAR** | 满足 FAR 约束的相似度阈值 | **上线阈值参考**（需在私有集重搜） |
| **EER** | FAR = FRR 时的等错误率 | 模型选型、阈值初估 |
| **AUC** | ROC 曲线下面积 | 整体区分能力 |

旧版单列 TAR@FAR 对照（Model Zoo 常用 **0.0001 列**）：

| 列          | 含义                        | buffalo_l 参考（Model Zoo） |
| ---------- | ------------------------- | ----------------------- |
| **1e-06**  | TAR @ FAR=10⁻⁶            | —                       |
| **1e-05**  | TAR @ FAR=10⁻⁵            | —                       |
| **0.0001** | TAR @ FAR=10⁻⁴（**最常对比项**） | **97.25%**（IJB-C(E4)）   |
| **0.001**  | TAR @ FAR=10⁻³            | —                       |
| **0.01**   | TAR @ FAR=10⁻²            | —                       |
| **0.1**    | TAR @ FAR=10⁻¹            | —                       |
| **AUC**    | ROC 曲线下面积                 | —                       |


表中数值为小数（如 `0.972593` = **97.26%**）。与 Model Zoo 对比时，重点看 `**0.0001` 列**。

#### 完整结果保存在哪？

脚本默认将原始分数保存到：

```text
recognition/_evaluation_/ijb/IJB_result/{model_name}_{subset}_11.npz
recognition/_evaluation_/ijb/IJB_result/{model_name}_{subset}_11.metrics.json   # 生产指标 JSON（新增）
```

例如 `w600k_r50_IJBC_11.npz`（约 120MB），内容如下：


| 字段       | 形状                      | 说明                        |
| -------- | ----------------------- | ------------------------- |
| `scores` | `(1, 15658489)`         | 约 1565 万对 template 的相似度分数 |
| `names`  | `['w600k_r50_IJBC_11']` | 结果名称                      |


默认 **不保存** `label`（每对 genuine/impostor 标签）。需要保存时加 `-L`：

```bash
python ijb_evals.py ... -L
```

#### 如何重新查看结果（不重跑 Embedding）

```bash
cd recognition/_evaluation_/ijb

python ijb_evals.py \
  -P IJB_result/w600k_r50_IJBC_11.npz \
  -d "/path/to/IJB_release" \
  -s IJBC
```

会重新打印 TAR@FAR / FRR@FAR / Threshold / EER 表，并写 `.metrics.json`。终端会分阶段输出进度（`[1/5]`…`[5/5]`）；**ROC 一步通常需 1–3 分钟**（约 1565 万 pair）。

**加速 label 加载**：若存在 `{data_path}/IJBC_backup.npz`（完整跑过一次 1:1 后自动生成），`-P` 会直接从缓存读 label，秒级完成；否则会慢读 `meta/*_template_pair_label.txt`。

也可显式传入 label 文件（与 `-P` 并列）：

```bash
python ijb_evals.py -P IJB_result/w600k_r50_IJBC_11.npz \
  "/path/to/IJB_release/IJBC/meta/ijbc_template_pair_label.txt"
```

用 Python 查看原始分数：

```python
import numpy as np

d = np.load("IJB_result/w600k_r50_IJBC_11.npz", allow_pickle=True)
scores = d["scores"][0]  # 15658489 个相似度
print(scores.shape, scores.min(), scores.max(), scores.mean())
```

#### 本次跑完还缺什么？（生产上线检查清单）

| 项目 | 状态 | 生产说明 |
|------|------|----------|
| IJBC **1:1** 默认协议 | ✅ 跑 `-m` 即完成 | 刷脸登录 / 人证比对必跑 |
| TAR@FAR + **FRR@FAR** + **Threshold@FAR** | ✅ 自动输出 | FRR 看体验，Threshold 供阈值初估 |
| **EER / AUC** | ✅ 自动输出 | 模型选型 |
| `.metrics.json` 结构化报告 | ✅ 自动保存 | 归档、CI、验收对比 |
| IJBC **1:N** | ❌ 需加 `-N` | 门禁 / 考勤 / 检索必跑 |
| 1:N **Rank-1/5/10 + mAP** | ❌ 需加 `-N` | 闭集认人能力 |
| 1:N **TPIR@FPIR + FNIR + FPIR** | ❌ 需加 `-N` | 开集：漏检率 / 误检率 |
| MegaFace 百万底库 | ❌ 另跑 | 大规模底库 Rank-1 |
| 私有真实数据验收 | ❌ 另建 | **上线最终依据** |
| ROC 曲线图 | ❌ 可能未生成 | 无 GUI 可忽略，数值不受影响 |
| 8 种 N/D/F 组合对比 | ❌ 需加 `-B` | 学术复现，生产通常不需要 |

### 1.5 IJBC 1:1 评测原理详解

本节说明 IJB-C 1:1 验证的**完整数据流**、**协议文件含义**、以及 **TAR@FAR 如何从相似度分数计算出来**。理解这些有助于区分「学术 benchmark 协议」与「线上注册/查库」业务流程的差异。

#### 1.5.1 核心概念：Template / Media / Pair

IJB-C 不是「单张图 vs 单张图」直接比对，而是以 **Template（模板）** 为最小评测单位：

| 概念 | 含义 | 对应 meta 文件 |
|------|------|----------------|
| **Face crop** | 一张裁剪好的人脸图 | `ijbc_name_5pts_score.txt`（文件名 + 5 点关键点 + detector score） |
| **Media** | 一次采集来源（一张照片或一段视频） | `ijbc_face_tid_mid.txt` 第 3 列 `media_id` |
| **Template** | 一个人一次轨迹/一次采集的聚合单位，可含多张图或多段视频 | `ijbc_face_tid_mid.txt` 第 2 列 `template_id` |
| **Pair** | 两个 template 组成的一对，用于 1:1 验证 | `ijbc_template_pair_label.txt` |

**与线上业务的区别**：线上系统通常「每人存一个向量、probe 查库取 Top1」；IJB-C 1:1 **不做查库识别**，而是对官方给定的 **template pair 列表**逐对算相似度，再统计 ROC / TAR@FAR。

#### 1.5.2 协议文件：`ijbc_template_pair_label.txt`

这是 IJB 官方规定的 1:1 评测协议，每一行 3 列（空格分隔）：

```text
template_id_A  template_id_B  label
```

- **第 1、2 列**：两个 `template_id`（整数编号），**不是向量本身**
- **第 3 列 `label`**：官方标注，`1` = 同一人（genuine），`0` = 不同人（impostor）

示例（示意）：

```text
1 11065 1
1 11066 0
...
```

IJBC 规模大致为：**约 801 万对**，其中 genuine 约 1 万对，其余为 impostor（比例极不均衡，这是 IJB 协议的设计）。

`template_id` 与具体图片的对应关系在 `ijbc_face_tid_mid.txt` 中：每一行把一张 face crop 映射到 `(template_id, media_id)`。

#### 1.5.3 完整评测流程（对应 `ijb_evals.py`）

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 对每张 face crop 提 embedding                           │
│  loose_crop/*.jpg → 5点对齐 → 模型 → 512维向量                   │
│  （可选 flip test、detector score 加权）                          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 聚合成 Template 向量（image2template_feature）           │
│  同一 template 下：                                              │
│    - 同一 media 多帧 → 先取均值                                   │
│    - 各 media 向量 → 求和 → L2 normalize                          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 对 pair 表每一行取两个 template 向量，算余弦相似度         │
│  score[i] = dot(feat_A, feat_B)   （向量已归一化，dot = cosine）  │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 用全部 (score, label) 画 ROC，取 TAR@FAR               │
│  roc_curve(label, score) → 在不同阈值下统计 TPR/FPR               │
│  TPR = TAR，FPR = FAR                                            │
└─────────────────────────────────────────────────────────────────┘
```

**注意**：脚本**不会**先选一个固定阈值再逐行判对错；而是把所有 pair 的分数一次性交给 `roc_curve`，扫描所有可能阈值，得到整条 ROC 曲线，再在指定 FAR 点上读出 TAR。

#### 1.5.4 Pair 打分：`verification_11()` 在做什么？

对 pair 表第 `i` 行：

1. 读出 `p1[i]`、`p2[i]`（两个 `template_id`）
2. 通过 `template2id` 映射到 `template_norm_feats` 中的向量索引
3. 计算 `score[i] = dot(feat1, feat2)`（余弦相似度）

最终得到长度 = pair 行数的 `scores` 数组（IJBC 约 1565 万个分数），以及对应的 `label` 数组。

**`fpr` / `tpr` 不是「某一对 template 的结果」**，而是在某个阈值下、对**全部 pair 整体统计**得到的比例。

#### 1.5.5 TAR / FAR / FPR / TPR 的关系

在 1:1 验证中，通常约定：

- **Positive（正样本）**：同一人（label = 1，genuine pair）
- **Negative（负样本）**：不同人（label = 0，impostor pair）
- **判定规则**：`score >= 阈值 τ` → 预测为「同一人 / Accept」

| 指标 | 别名 | 含义 |
|------|------|------|
| **TPR** | **TAR** | 所有 genuine 对中，被正确 Accept 的比例 |
| **FPR** | **FAR** | 所有 impostor 对中，被误 Accept 的比例 |

给定阈值 τ 时的计算公式：

```
TPR(τ) = TAR(τ) = TP / (TP + FN)    # genuine 中被接受的比例
FPR(τ) = FAR(τ) = FP / (FP + TN)    # impostor 中被误接受的比例
```

其中：

- **TP**：label=1 且 score ≥ τ
- **FN**：label=1 且 score < τ
- **FP**：label=0 且 score ≥ τ
- **TN**：label=0 且 score < τ

**TAR@FAR = 1e-4** 的含义：在所有可能阈值中，找一个 τ 使得 FAR(τ) ≈ 10⁻⁴，然后报告对应的 TAR(τ)。

代码实现（`ijb_evals.py`）：

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(label, score)
fpr, tpr = np.flipud(fpr), np.flipud(tpr)  # 同一 FAR 下取最大 TAR
tar_at_far = tpr[np.argmin(abs(fpr - 1e-4))]  # 例如取 FAR=1e-4 时的 TAR
```

`roc_curve` 会按 `score` 的取值自动扫描大量阈值，每个阈值产生一个 (FPR, TPR) 点，连起来就是 ROC 曲线。

#### 1.5.6 手算示例：6 对 Pair 理解 TAR@FAR

假设只有 6 对 template-pair，已算出相似度：

| pair 编号 i | label（真值） | score（相似度） | 含义 |
|------------|-------------|----------------|------|
| 1 | 1 | 0.90 | 同一人 |
| 2 | 1 | 0.80 | 同一人 |
| 3 | 1 | 0.40 | 同一人 |
| 4 | 0 | 0.70 | 不同人 |
| 5 | 0 | 0.30 | 不同人 |
| 6 | 0 | 0.20 | 不同人 |

正样本（genuine）共 **P = 3** 个，负样本（impostor）共 **N = 3** 个。

**阈值 τ = 0.75 时：**

- score ≥ 0.75 的：i=1(0.90), i=2(0.80)
- TP = 2（i=1, 2），FN = 1（i=3）
- FP = 0（负样本最高 0.70 < 0.75），TN = 3（i=4, 5, 6）

```
TPR = TAR = 2/3 ≈ 0.667
FPR = FAR = 0/3 = 0.000
```

**阈值 τ = 0.65 时：**

- score ≥ 0.65 的：i=1, i=2, i=4（多了 impostor i=4 = 0.70）
- TP = 2，FN = 1，FP = 1，TN = 2

```
TPR = TAR = 2/3 ≈ 0.667
FPR = FAR = 1/3 ≈ 0.333
```

**阈值 τ = 0.35 时：**

- score ≥ 0.35 的：i=1, i=2, i=3, i=4（genuine i=3 也被接受）
- TP = 3，FN = 0，FP = 1，TN = 2

```
TPR = TAR = 3/3 = 1.000
FPR = FAR = 1/3 ≈ 0.333
```

把不同阈值下的 (FAR, TAR) 连起来就是 ROC 曲线上的点：

| 阈值 τ | FAR (FPR) | TAR (TPR) |
|--------|-----------|-----------|
| 0.85 | 0.000 | 0.333 |
| 0.75 | 0.000 | 0.667 |
| 0.65 | 0.333 | 0.667 |
| 0.35 | 0.333 | 1.000 |
| 0.25 | 0.667 | 1.000 |
| 0.15 | 1.000 | 1.000 |

若要求 **TAR@FAR = 0.333**：在 ROC 上找 FAR = 0.333 的点。同一 FAR 可能对应多个 TAR（如 0.667 或 1.000），脚本会取**最大 TAR**（`np.flipud` 后 `argmin` 取点）。

真实 IJBC 有 801 万对、FAR 可精细到 10⁻⁶，手算不现实，但逻辑与上述小例子完全一致。

#### 1.5.7 与「注册 + 查库」业务逻辑的对比

| | IJB-C 1:1 评测 | 线上 1:1 / 1:N 业务 |
|---|---------------|---------------------|
| 输入 | 官方 pair 表（template A vs template B） | probe 图 + 已注册底库 |
| 比对方式 | 对**固定 pairs** 算相似度 | probe 与库中**所有/TopK** 候选比 |
| 输出 | 每条 pair 一个 score，再算 ROC / TAR@FAR | 预测身份 ID 或 Accept/Reject |
| 是否取 Top1 | 否 | 1:N 会取最相似身份 |
| 阈值 | 从 ROC 反推（满足 FAR 约束） | 通常业务固定或调参确定 |

IJB-C 1:1 评测的是：**模型能否把「同一人 template 对」与「不同人 template 对」在相似度上拉开**，并在极低 FAR 下仍保持高 TAR。这是算法能力评估，不是复刻线上检索流程。

---

## 2. IJB-C（1:N 识别）怎么跑

在 `ijb_evals.py` 基础上加 `-N`：

```bash
conda activate face_recog_env
cd recognition/_evaluation_/ijb

# 先配置 GPU 库路径（见 0.1 节）
export LD_LIBRARY_PATH="$(
python - <<'PY'
import glob, site
print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=0 python ijb_evals.py \
  -m ~/.insightface/models/buffalo_l/w600k_r50.onnx \
  -d "/path/to/IJB_release" \
  -s IJBC \
  -b 128 \
  -N
```

脚本默认将 1:N 结果保存到：

```text
recognition/_evaluation_/ijb/IJB_result/{model_name}_{subset}_1N.npz
recognition/_evaluation_/ijb/IJB_result/{model_name}_{subset}_1N.metrics.json
```

### 2.1 运行过程你会看到什么

典型输出阶段（Embedding 阶段与 1:1 相同，之后进入 1:N 专用逻辑）：

- **Loading gallery feature / probe feature**：读取 `meta/` 下 gallery / probe 协议 csv
- **Extract template feature**：分别聚合 gallery G1、G2、probe_mixed 的 template 向量
- **Gallery 1 / Gallery 2**：分别对两套 gallery 做检索评测
- **Mean**：G1 与 G2 的 Rank / mAP / TPIR 取平均
- **最终表格**：输出 **Rank-1/5/10、mAP** 及 **TPIR@FPIR / FNIR(漏检) / FPIR(误检)**

终端示例（数值因模型而异）：

```text
>>>> Gallery 1
Closed-set (Rank / mAP):
|           | Rank-1   | Rank-5   | Rank-10  | mAP      |
|:----------|---------:|---------:|---------:|---------:|
| Gallery 1 | 92.35%   | 96.72%   | 97.83%   | 93.10%   |

Open-set (TPIR / FNIR / FPIR):
| FPIR  | TPIR   | FNIR(Miss) | FPIR(achieved) | Threshold |
|-------|--------|------------|----------------|-----------|
| 1e-4  | 0.85.. | 0.14..     | 0.0001         | 0.42..    |

>>>> Mean
[Mean] Rank-1: 0.922345, Rank-5: 0.967216, Rank-10: 0.978277, mAP: 0.931045
| fpir  | g1_tpir | g1_fnir | g1_fpir | mean_tpir | mean_fnir | mean_fpir |
|-------|---------|---------|---------|-----------|-----------|-----------|
| 0.0001| 0.85... | 0.14... | 0.0001  | 0.84...   | 0.15...   | 0.0001    |
```

> **1:N 指标对照**
>
> | 指标 | 别名 | 含义 |
> |------|------|------|
> | **Rank-1 / Top-1** | — | 正确身份排第 1（闭集，不设阈值） |
> | **Rank-5 / Rank-10** | Top-5/10 | 正确身份进前 K |
> | **mAP** | mean AP | 正确身份排序位置的综合质量 |
> | **TPIR@FPIR** | 开集识别率 | 在 FPIR 约束下正确识别在库 probe 的比例 |
> | **FNIR / 漏检率** | Miss Rate | = 1 − TPIR，在库 probe 未被正确识别 |
> | **FPIR / 误检率** | False Positive Retrieval | 不在库 probe 被误认成库中某身份的比例 |

> 1:N 默认协议固定为 **N0D1F1**（`use_norm_score=False`、`use_detector_score=True`、`use_flip_test=True`），与 1:1 默认一致。

### 2.2 IJBC 1:N 评测原理详解

本节说明 IJB-C 1:N 识别的**完整数据流**、**协议文件含义**、**Top-K 与 TPIR@FPIR 如何计算**。1:N 比 1:1 更接近线上「底库 + 探针检索」的业务形态，但仍遵循 IJB 官方规定的 gallery / probe 划分。

#### 2.2.1 核心概念：Gallery / Probe / Subject

| 概念 | 含义 | 在评测中的角色 |
|------|------|----------------|
| **Gallery（底库）** | 已注册身份集合，每个身份对应一个或多个 template | 被检索的「候选库」 |
| **Probe（探针）** | 待识别样本，每个 probe 对应一个 template | 查询向量，与 gallery 全库比相似度 |
| **Subject ID** | 人的身份编号（同一人可有多 template） | 判断 Top-K / TPIR 是否正确的真值 |
| **Template** | 与 1:1 相同，IJB 的最小聚合单位 | gallery / probe 都以 template 向量参与检索 |

**与 1:1 的核心区别**：

| | 1:1 验证 | 1:N 识别 |
|---|---------|---------|
| 输入 | 官方 pair 表（template A vs template B） | 1 个 probe + 整个 gallery |
| 任务 | 这两张脸是不是同一个人？ | 这张脸在底库里是谁？ |
| 是否排序 | 否 | **是**（对 gallery 全库排序） |
| 典型指标 | TAR@FAR | Top-1/5/10、TPIR@FPIR |

#### 2.2.2 协议文件（官方规定）

IJBC 1:N 使用 3 个 csv 文件（位于 `IJBC/meta/`）：

| 文件 | 含义 | 列格式（跳过首行表头） |
|------|------|------------------------|
| `ijbc_1N_gallery_G1.csv` | 底库集合 1 | `template_id, subject_id` |
| `ijbc_1N_gallery_G2.csv` | 底库集合 2 | `template_id, subject_id` |
| `ijbc_1N_probe_mixed.csv` | 混合探针集合 | `template_id, subject_id` |

示例（示意）：

```text
template_id,subject_id
1001,42
1002,42
2001,87
...
```

**为什么有两套 Gallery（G1 / G2）？**  
IJB 官方协议要求对**不同 gallery 划分**各测一遍，再取平均，避免结果偶然依赖某一次底库构成。脚本会分别跑 G1、G2，最后输出 `[Mean]` 和 `mean_tpir`。

**`probe_mixed` 的含义**：probe 集合是**混合的（mixed）**，包含两类：

- **Enrolled probe（在库探针）**：该 probe 的 `subject_id` 存在于当前 gallery 中 → 用于算 **Top-K**、**TPIR**
- **Non-enrolled probe（不在库探针）**：该 probe 的 `subject_id` **不在** gallery 中（陌生人）→ 用于算 **FPIR**、定阈值

这正是**开集 1:N（open-set identification）**：系统既要「认对在库的人」，又要「不要把陌生人硬认成库里的某人」。

IJBC 规模大致为（以代码注释中的典型 shape 为例）：

- Gallery G1：约 1772 个 template
- Gallery G2：约 1759 个 template
- Probe mixed：约 19593 个 template

#### 2.2.3 完整评测流程（对应 `ijb_evals.py`）

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 对全部 face crop 提 embedding（与 1:1 相同）            │
│  loose_crop → 5点对齐 → 模型 → 512维向量                         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 按协议 csv 分别聚合 Template 向量                        │
│  - gallery G1 的 template → g1_templates_feature                 │
│  - gallery G2 的 template → g2_templates_feature                 │
│  - probe_mixed 的 template → probe_templates_feature             │
│  （聚合规则与 1:1 相同：media 内均值 → template 求和 → L2 norm）   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 矩阵乘法做全库检索                                       │
│  similarity = probe_feats @ gallery_feats.T                      │
│  形状：(num_probe, num_gallery)，每行是一个 probe 对全库的相似度   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 对每个 probe 分别统计                                    │
│  - 在库 probe → Top-1/5/10、pos_sim vs neg_sim                   │
│  - 不在库 probe → max_sim（用于 FPIR 定阈值）                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: 由 non-gallery probe 定阈值，算 TPIR@FPIR               │
│  G1、G2 各算一遍 → 取 mean_tpir                                   │
└─────────────────────────────────────────────────────────────────┘
```

**与线上业务的对应关系**：

- **Gallery template 向量** ≈ 底库里每个注册样本的向量
- **Probe template 向量** ≈ 现场抓拍/待识别图的向量
- **similarity 矩阵的一行** ≈ 一次 1:N 检索对所有候选的打分
- **Top-1** ≈ 取得分最高的 gallery 条目，看 `subject_id` 对不对

#### 2.2.4 检索打分：`evaluation_1N()` 在做什么？

核心代码逻辑：

```python
# 全库相似度矩阵
similarity = np.dot(query_feats, gallery_feats.T)  # (num_probe, num_gallery)

for index, query_id in enumerate(query_ids):
    if query_id in reg_ids:          # 在库 probe（enrolled）
        gallery_label = np.argwhere(reg_ids == query_id)[0, 0]
        index_sorted = np.argsort(similarity[index])[::-1]
        # Top-K：正确 gallery 条目的 index 是否排进前 K
        top_1_count += gallery_label in index_sorted[:1]
        ...
        pos_sims.append(similarity[index][reg_ids == query_id][0])  # 与正确身份的相似度
        neg_sims.append(similarity[index][reg_ids != query_id])      # 与所有错误身份的相似度
    else:                            # 不在库 probe（non-enrolled）
        non_gallery_sims.append(similarity[index])  # 整行相似度，后面取 max 用于 FPIR
```

几个要点：

1. **比对单位是 template，不是 subject**：gallery 里同一人若有多个 template，会占多个 slot；Top-K 看的是「正确 template 对应的 gallery index 是否进前 K」。
2. **Top-K 通常不设阈值**：纯看排序，得分最高的是否为正确 gallery 条目（闭集部分）。
3. **`pos_sims` / `neg_sims`**：对每个在库 probe，记录「与正确身份的相似度」和「与所有错误身份的相似度数组」，供 TPIR 判定使用。

#### 2.2.5 Top-1 / Top-5 / Top-10 怎么算？

**定义**（仅统计**在库 probe**，即 `subject_id ∈ gallery`）：

```
Top-1 = 正确 template 排在第 1 名的 probe 数 / 在库 probe 总数
Top-5 = 正确 template 排进前 5 名的 probe 数 / 在库 probe 总数
Top-10 = 正确 template 排进前 10 名的 probe 数 / 在库 probe 总数
```

**手算小例子**：

Gallery（3 个 template，2 个 subject）：

| gallery index | subject_id |
|---------------|------------|
| 0 | 张三 |
| 1 | 李四 |
| 2 | 张三（另一 template） |

某在库 Probe（真实 subject = 张三，正确 gallery index = 0）对全库相似度：

```text
index:      0     1     2
sim:      0.88  0.72  0.85
subject:  张三  李四  张三
```

排序后：`[0(0.88), 2(0.85), 1(0.72)]`

- **Top-1**：第 1 名 index=0（张三）→ ✅ 正确
- **Top-5**：index=0 在前 5 内 → ✅（这里 gallery 只有 3 个，Top-5 等价于 Top-3）

若相似度变成 `[0.70, 0.72, 0.85]`（index=2 的张三 template 最高）：

- 排序：`[2(0.85), 1(0.72), 0(0.70)]`
- Top-1 第 1 名是 index=2，也是张三 → ✅（同一 subject 的另一个 template 排第一也算对）
- 若第 1 名是 index=1（李四）→ ❌ Top-1 错误

脚本对 G1、G2 分别统计后，**Mean Top-1** = `(g1_top_1_count + g2_top_1_count) / probe_total`。

#### 2.2.6 TPIR / FPIR 与 TPIR@FPIR

1:N 开集评测的核心指标，与 1:1 的 TAR/FAR 对应：

| 1:1 | 1:N | 含义 |
|-----|-----|------|
| TAR | **TPIR** | 应该认出来的人，被正确识别的比例 |
| FAR | **FPIR** | 不在库的人（陌生人），被误认成库里某身份的比例 |
| TAR@FAR | **TPIR@FPIR** | 在 FPIR 约束下能达到的 TPIR |

**TPIR 判定比 Top-1 更严格**，对每个在库 probe 需同时满足：

1. **排序正确**：`pos_sim > max(neg_sims)`（正确身份相似度高于所有错误身份）
2. **超过阈值**：`pos_sim > threshold`（threshold 由 non-gallery probe 分布确定）

代码：

```python
correct_pos_cond = pos_sims > neg_sims.max(1)   # 条件 1：比所有 impostor 都高

# 用不在库 probe 的「对全库最高相似度」来定阈值
non_gallery_sims_sorted = np.sort(non_gallery_sims.max(1))[::-1]
thresh = non_gallery_sims_sorted[int(len * far) - 1]   # 使 FPIR ≈ far

recall = np.logical_and(correct_pos_cond, pos_sims > thresh).sum() / len(pos_sims)
# recall 即 TPIR@FPIR=far
```

**直觉**：

- **Non-enrolled probe** 理想情况下对 gallery 都不该有高相似度；取它们「对全库最高分」的分布，按 FPIR 目标（如 1%）定阈值。
- **Enrolled probe** 不仅要排第一，分数还要高过该阈值，才算 TPIR 成功。

#### 2.2.7 手算示例：理解 TPIR@FPIR

**Gallery**（2 人）：

| index | subject |
|-------|---------|
| 0 | 张三 |
| 1 | 李四 |

**Probe 集合**：

| probe | 是否在库 | 真实 subject | 与 gallery 相似度 [idx0, idx1] |
|-------|---------|-------------|-------------------------------|
| P1 | ✅ 在库 | 张三 | [0.90, 0.40] |
| P2 | ✅ 在库 | 李四 | [0.35, 0.88] |
| P3 | ✅ 在库 | 张三 | [0.55, 0.60] |
| U1 | ❌ 不在库 | — | [0.50, 0.45] |
| U2 | ❌ 不在库 | — | [0.30, 0.25] |

**Step 1：Top-1（只看在库 P1/P2/P3）**

| probe | 排序第 1 | Top-1 |
|-------|---------|-------|
| P1 | idx0 张三 0.90 | ✅ |
| P2 | idx1 李四 0.88 | ✅ |
| P3 | idx1 李四 0.60 | ❌（张三被李四超过） |

```
Top-1 = 2/3 ≈ 0.667
```

**Step 2：TPIR 条件 1（pos_sim > max(neg_sims)）**

| probe | pos_sim | max(neg_sims) | 排序正确？ |
|-------|---------|---------------|-----------|
| P1 | 0.90 | 0.40 | ✅ |
| P2 | 0.88 | 0.35 | ✅ |
| P3 | 0.55 | 0.60 | ❌ |

**Step 3：用 non-enrolled 定阈值（FPIR）**

Non-enrolled 对全库最高分：`U1 → 0.50`，`U2 → 0.30`  
降序：`[0.50, 0.30]`

若目标 **FPIR = 0.5**（2 个 non-enrolled 里允许 1 个误识）：

```
threshold = 0.30   # 第 int(2*0.5)-1 = 0 位
```

FPIR 检验：U1(0.50) ≥ 0.30 → 误识；U2(0.30) 未超 → 不误识 → FPIR = 1/2 = 0.5 ✅

**Step 4：TPIR@FPIR=0.5**

在库 probe 需同时满足「排序正确 AND pos_sim > 0.30」：

| probe | 排序正确 | pos_sim > 0.30 | TPIR 成功？ |
|-------|---------|----------------|------------|
| P1 | ✅ | 0.90 > 0.30 ✅ | ✅ |
| P2 | ✅ | 0.88 > 0.30 ✅ | ✅ |
| P3 | ❌ | — | ❌ |

```
TPIR@FPIR=0.5 = 2/3 ≈ 0.667
```

若目标 **FPIR = 0.0**（不允许陌生人误识）：

```
threshold = 0.50   # 必须高于 U1 的最高分
```

则 P1(0.90)、P2(0.88) 仍成功，P3 仍失败：

```
TPIR@FPIR=0.0 = 2/3 ≈ 0.667
```

FPIR 越严格（越小），阈值越高，TPIR 通常越低——与 1:1 里 FAR 越小、TAR 越难做高是同一逻辑。

#### 2.2.8 G1 / G2 双 Gallery 与 Mean 指标

`run_model_test_1N()` 对 G1、G2 **各跑一遍** `evaluation_1N()`，然后：

```python
top_1 = (g1_top_1_count + g2_top_1_count) / query_num
mean_tpirs = (g1_recalls + g2_recalls) / 2
```

最终表格里的 `mean_tpir` 列是 G1、G2 在相同 FPIR 点上的 TPIR 平均，这是 IJB 1:N 论文/Model Zoo 常引用的汇总方式。

#### 2.2.9 与「注册 + 查库」业务逻辑的对比

| | IJB-C 1:N 评测 | 线上 1:N 业务 |
|---|---------------|--------------|
| 底库构成 | 官方 G1/G2 csv 规定的 template | 业务注册入库的向量 |
| Probe | 官方 probe_mixed（含在库 + 不在库） | 现场抓拍 / 通行记录 |
| 检索 | probe 向量 × gallery 矩阵 | 同样（FAISS / Milvus 等） |
| Top-1 | 取得分最高 gallery 条目是否正确 | 同上 |
| 陌生人 | non-enrolled probe 专门测 FPIR | 未注册访客不应被认成员工 |
| 阈值 | 由 non-gallery 分布按 FPIR 反推 | 通常业务调参或 FAR 目标确定 |
| 身份单位 | template + subject_id | 通常按「人」或「注册记录」 |

IJB-C 1:N 比 1:1 **更接近真实检索**，但仍有两点差异需注意：

1. Gallery 按 **template** 存条目，同一人多 template 占多个 slot（线上可能按人聚合）。
2. Top-1 与 TPIR@FPIR 是两套指标：前者纯排序，后者还要控陌生人误识。

#### 2.2.10 1:1 vs 1:N 指标对照速查

| 问题 | 1:1 | 1:N |
|------|-----|-----|
| 输入是什么？ | 官方 pair 表 | Gallery csv + Probe csv |
| 怎么比？ | 两个 template 算相似度 | Probe 对全 Gallery 检索排序 |
| 核心输出 | 每对一条 score | 每个 probe 一条排序 + 最高分 |
| 闭集指标 | TAR@FAR | Top-1 / Top-5 / Top-10 |
| 开集指标 | （pair 里 impostor 比例固定） | TPIR@FPIR |
| 是否取 Top1 | 否 | 是（Top-K 指标） |
| 误识含义 | 不同人 pair 被 Accept | 陌生人被认成库中某人 |

---

## 3. 另一种 IJBC 跑法：`ijb_onnx.py`（单文件 onnx）

适用于你希望用“单模型文件 + 指定 IJBC 目录”的方式启动：

```bash
conda activate face_recog_env
cd recognition/_evaluation_/ijb

# 先配置 GPU 库路径（见 0.1 节）
export LD_LIBRARY_PATH="$(
python - <<'PY'
import glob, site
print(":".join(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib")))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES=0 python ijb_onnx.py \
  --model-file ~/.insightface/models/buffalo_l/w600k_r50.onnx \
  --image-path "/path/to/IJB_release/IJBC" \
  --result-dir ./results/buffalo_l \
  --target IJBC
```

参数说明：

- `**--model-file**`：ONNX 模型路径
- `**--image-path**`：**直接指向 `IJBC/` 目录本身**（不要传它的父目录）
- `**--result-dir`**：结果输出目录
- `**--target**`：`IJBC` 或 `IJBB`

---

## 4. MegaFace（1:N 百万底库）怎么跑（buffalo_l）

项目提供了一键脚本：`recognition/_evaluation_/megaface/run_buffalo_l.sh`。

```bash
conda activate face_recog_env
cd recognition/_evaluation_/megaface

# 常用覆盖项（按需设置）
# MEGAFACE_ROOT=... MODEL_FILE=... GPU=... BATCH_SIZE=... GALLERY_SIZE=...
./run_buffalo_l.sh
```

常用环境变量：

- `**MEGAFACE_ROOT**`：MegaFace 数据根目录
- `**MODEL_FILE**`：ONNX 模型路径（默认 `~/.insightface/models/buffalo_l/w600k_r50.onnx`）
- `**GPU**`：GPU id（脚本内部会做 CUDA provider 检查）
- `**BATCH_SIZE**`：批大小
- `**GALLERY_SIZE**`：底库规模，常见 `1000000`

---

## 5. 评测场景有哪些（怎么选）

- **1:1 验证（Verification）**
  - **场景**：刷脸登录、人证比对、门禁“是不是本人”
  - **协议/数据**：IJB-C 1:1（pair 协议）
  - **核心指标**：`TAR@FAR`、`FRR@FAR`、`Threshold@FAR`、`EER`
- **1:N 识别（Identification / Search）**
  - **场景**：底库检索、黑名单、考勤“TA是谁”
  - **协议/数据**：IJB-C 1:N、MegaFace
  - **核心指标**：`Rank-1/5/10`、`mAP`、`TPIR@FPIR`、`FNIR(漏检)`、`FPIR(误检)`
- **大规模底库鲁棒性**
  - **场景**：百万底库误识控制、检索边界
  - **协议/数据**：MegaFace（gallery=1e6）
  - **核心指标**：Rank-1@1e6（以及误识约束下的识别率）

---

## 6. 常见指标解释（如何解读）

> 指标实现统一在 `recognition/_evaluation_/metrics/production_metrics.py`，IJB / MegaFace 多注册实验共用同一套算法。

### 6.1 1:1：TAR / FAR / FRR

- **FAR（False Accept Rate）**：不同人被误判为同一人的概率（越低越安全）
- **TAR（True Accept Rate）**：同一人被正确接受的概率（越高越好）
- **FRR（False Reject Rate）**：同一人被错误拒绝的概率 = **1 − TAR**（体验敏感场景必看）
- **`TAR@FAR=1e-4 / 1e-5 / 1e-6`**：在误识率不超过 10⁻⁴/10⁻⁵/10⁻⁶ 的约束下，系统能达到的通过率
- **`Threshold@FAR`**：满足 FAR 约束的相似度阈值（**上线前必须在私有集重搜，不可直接抄 IJB 值**）
- **`EER`**：FAR = FRR 时的错误率，用于模型选型
  - 业务直觉：
    - **`1e-4`**：门禁/考勤常用参考
    - **`1e-5`**：更严格（支付/高安全）
    - **`1e-6`**：极严格（金融/强风控）

### 6.2 1:N：Rank-K / mAP / TPIR@FPIR / FNIR / FPIR

- **Rank-1 / Top-1**：返回第一名就是正确身份的比例（闭集，通常不设阈值）
- **Rank-5 / Rank-10**：正确身份出现在前 K 名的比例
- **mAP**：正确身份在排序列表中位置的综合质量（MegaFace 等大规模检索常用）
- **TPIR**：在给定 FPIR 约束下的正确识别率（1:N 开集，对应 1:1 的 TAR）
- **FNIR / 漏检率（Miss Rate）**：= 1 − TPIR，在库 probe 未被正确识别
- **FPIR / 误检率（False Positive Retrieval）**：不在库 probe 被误认成库中某身份的比例（对应 1:1 的 FAR）

### 6.3 生产上线指标选用速查

| 业务场景 | 必看 IJB 指标 | 说明 |
|----------|--------------|------|
| 刷脸登录 / 人证比对（1:1） | TAR@FAR=1e-4~1e-5 + **FRR@FAR** + Threshold | 安全 + 体验 |
| 门禁 / 考勤（1:N） | **Rank-1** + TPIR@FPIR=1e-4 + **FNIR + FPIR** | 认人 + 控陌生人误开 |
| 模型选型 | EER（1:1）+ Rank-1 + mAP（1:N） | 快速对比 |
| 最终上线 | **私有真实数据** 复测上述指标 | 公开基准 ≠ 业务可用 |

---

## 7. 常见坑与排查

- **路径传错（最常见）**
  - `ijb_evals.py -d`：传 **包含 `IJBC/` 的父目录**
  - `ijb_onnx.py --image-path`：传 `**IJBC/` 目录本身**
- **中断与续跑**
  - `ijb_evals.py` 会写 `IJBC_backup.npz` 用于加速下次“加载 meta”，但 **Embedding 可能仍要重算**（除非你明确保存/恢复 embeddings 并按脚本参数使用）
- **显存不足**
  - 降低 `-b`（比如 128 → 64 → 32）
- **GPU provider 没启用 / 评测很慢**
  - 先按 **0.1 节** 设置 `LD_LIBRARY_PATH` 并验证 `providers` 包含 `CUDAExecutionProvider`
  - 确认安装 `onnxruntime-gpu`，并且 `CUDA_VISIBLE_DEVICES` 设置正确
  - 若 `nvidia-smi` 显示 GPU 利用率接近 0%，多半是 ONNX 在 CPU 上跑
- **运行很慢属于正常**
  - IJBC 大约 47 万张图，Embedding 阶段通常需要数小时（视 GPU/IO 而定）
- **表格打印报错 `Import tabulate`**
  - 安装：`pip install tabulate`（用于 `pandas.to_markdown()` 输出结果表）

---

## 8. 延伸阅读

- `docs/人脸识别算法指标与评测指南.md`：指标体系、IJB-C / MegaFace 的业务解释
- `docs/buffalo_l生产应用评估报告.md`：`buffalo_l` 的 IJB-C / MegaFace 参考值与上线建议

