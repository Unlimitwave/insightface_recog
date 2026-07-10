# 人脸特征向量与 PartialFC 训练原理详解

> 面向需要理解 InsightFace 训练机制的开发者。本文解释「特征向量」「标签」「PartialFC」三者的关系，以及训练与推理为何不同。结合本项目 `recognition/arcface_torch` 中的实际实现编写。

---

## 目录

1. [一句话结论](#1-一句话结论)
2. [特征向量与标签分别是什么](#2-特征向量与标签分别是什么)
3. [训练流程：Backbone + PartialFC + ArcFace](#3-训练流程backbone--partialfc--arcface)
4. [逐步数值演算示例](#4-逐步数值演算示例)
5. [PartialFC 是什么](#5-partialfc-是什么)
6. [训练 vs 推理](#6-训练-vs-推理)
7. [评测时标签怎么用](#7-评测时标签怎么用)
8. [与大模型词表的类比](#8-与大模型词表的类比)
9. [常见误区](#9-常见误区)
10. [本项目代码索引](#10-本项目代码索引)

---

## 1. 一句话结论

| 阶段 | 模型输出 | 「标签」是什么 |
|------|----------|----------------|
| **训练** | 512 维特征向量（Embedding） | **身份 ID**（整数，如 0、85742） |
| **评测（1:1）** | 两张图各一个向量 | **是否同一人**（True / False） |
| **评测（1:N）** | Probe 与 Gallery 各一个向量 | **身份 ID**（业务系统中的用户编号） |

**核心要点**：

- 损失永远是**向量和向量**算出来的，不是靠整数 ID 直接参与数学运算。
- 整数 ID 的作用是**索引**：告诉网络「weight 表里第几行是正确答案」。
- 训练用 PartialFC，推理时**只保留 Backbone**，扔掉分类头。

---

## 2. 特征向量与标签分别是什么

### 2.1 特征向量（Embedding）

InsightFace 把一张对齐后的人脸图像映射为固定维度的浮点向量，通常为 **512 维**：

```
人脸图像 (112×112) → Backbone (ResNet / ViT 等) → f = [0.12, -0.05, ..., 0.33]
```

这个向量刻画的是「这张脸在特征空间中的位置」。推理时所有比对都建立在这个向量之上。

### 2.2 标签（Label）——不是向量

标签是**整数身份编号**，表示「这张图属于训练集里的第几个人」：

```
训练集:
  person_0/   ← 张三，几百张不同角度照片，标签都是 0
  person_1/   ← 李四，标签都是 1
  ...
  person_85742/ ← 标签都是 85742
```

数据集每张图只附带一个整数，**不提供任何 512 维向量**。对应代码见 `recognition/arcface_torch/dataset.py` 中 `MXFaceDataset.__getitem__`：

```python
header, img = mx.recordio.unpack(s)
label = header.label          # 整数，如 85742
return sample, label          # 返回 (图像, 整数标签)
```

### 2.3 三者关系一览

| 概念 | 是什么 | 谁提供 | 训练后保留吗 |
|------|--------|--------|-------------|
| **标签 label** | 整数 ID | 数据集标注 | 不需要（仅训练时用） |
| **类中心 w_i** | 512 维向量（weight 表第 i 行） | PartialFC 随机初始化、训练中学习 | **通常扔掉** |
| **人脸特征 f** | 512 维向量 | Backbone 对图像计算 | **保留，推理用这个** |

---

## 3. 训练流程：Backbone + PartialFC + ArcFace

### 3.1 整体结构

```
┌─────────────────────────────────────────────────────────┐
│  训练时才有的部分（推理时扔掉）                              │
│                                                         │
│  人脸图 → Backbone (ResNet) → 512 维向量 f               │
│                                    ↓                    │
│                         PartialFC（类中心表 + ArcFace 损失）│
│                                    ↑                    │
│                         标签 label（整数，来自数据集）       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  推理 / 上线时保留的部分                                    │
│                                                         │
│  人脸图 → Backbone → 512 维向量 → 与底库 / 另一张脸比相似度  │
└─────────────────────────────────────────────────────────┘
```

训练循环（`recognition/arcface_torch/train_v2.py`）：

```python
for _, (img, local_labels) in enumerate(train_loader):
    local_embeddings = backbone(img)                        # 图像 → 向量
    loss = module_partial_fc(local_embeddings, local_labels) # 向量 + 标签 → 损失
    loss.backward()
    optimizer.step()
```

### 3.2 PartialFC 内部的向量运算

`recognition/arcface_torch/partial_fc_v2.py` 中的核心逻辑：

```python
norm_embeddings = normalize(embeddings)       # 人脸向量 f，L2 归一化
norm_weight = normalize(weight)             # 类中心表，每行归一化
logits = linear(norm_embeddings, norm_weight) # f 与每个 w_i 算内积（余弦相似度）
logits = margin_softmax(logits, labels)     # ArcFace：对 label 对应那一维加角度间隔
loss = cross_entropy(logits, labels)        # 交叉熵：正确 ID 那一维分数要最高
```

| 变量 | 含义 | 形状 |
|------|------|------|
| `embeddings` | 当前 batch 每张脸的特征 | `(batch, 512)` |
| `weight` | 每个身份一个类中心 | `(num_classes, 512)` |
| `logits` | f 与每个 w_i 的相似度 | `(batch, num_classes)` |
| `labels` | 整数，选出正确的那一维 | `(batch,)` |

**ID 只出现在两处**：ArcFace 指定修改 `logits[label]` 那一维；交叉熵取 `probs[label]` 计算损失。

### 3.3 交叉熵损失的向量含义

损失公式本质上要求：人脸向量 f 与正确类中心 w_label 的相似度，要远大于与其他类中心的相似度。

```
Loss = -log( exp(sim(f, w_label)) / Σ_j exp(sim(f, w_j)) )
         \_________________/   \________________________/
           和目标向量要近          和其他向量要远（分母里所有 j）
```

---

## 4. 逐步数值演算示例

以下用 **3 个身份、4 维特征** 的极小例子，完整走一遍从输入到损失的计算（实际 InsightFace 为 512 维，算法相同）。

**设定**：

- 3 个身份：ID=0（张三）、ID=1（李四）、ID=2（王五）
- 当前训练图：张三的照片，`label = 0`
- ArcFace 超参：角度间隔 `m = 0.5`（弧度），缩放 `s = 64`

### 第 1 步：Backbone 输出人脸向量

```
f_raw = [0.96,  0.28,  0.00,  0.00]
```

### 第 2 步：L2 归一化

```
|f_raw| = √(0.96² + 0.28²) = 1.0

f = [0.96, 0.28, 0.00, 0.00]    # 已是单位向量
```

### 第 3 步：类中心表 weight（训练中学出，此处为收敛后的理想值）

```
         维度0   维度1   维度2   维度3
w_0 =  [ 1.00,   0.00,   0.00,   0.00 ]   ← ID=0 张三
w_1 =  [ 0.00,   1.00,   0.00,   0.00 ]   ← ID=1 李四
w_2 =  [ 0.00,   0.00,   1.00,   0.00 ]   ← ID=2 王五
```

### 第 4 步：f 与每个 w_i 算内积 → logits

```
logit_0 = f · w_0 = 0.96×1.00 + 0.28×0.00 = 0.96
logit_1 = f · w_1 = 0.96×0.00 + 0.28×1.00 = 0.28
logit_2 = f · w_2 = 0.00

logits = [0.96,  0.28,  0.00]
```

### 第 5 步：ArcFace 对 label=0 那一维加角度惩罚

```
θ = arccos(0.96) ≈ 0.283 弧度
θ_new = θ + m = 0.283 + 0.5 = 0.783 弧度
logit_0_new = cos(0.783) ≈ 0.709

logits' = [0.709,  0.28,  0.00]    # 只改第 0 维
```

### 第 6 步：乘以缩放系数 s = 64

```
logits'' = [45.38,  17.92,  0.00]
```

### 第 7 步：Softmax → 概率

```
p_0 ≈ 0.9999999983   ← 张三（正确答案）
p_1 ≈ 0.0000000017
p_2 ≈ 0.0000000000
```

### 第 8 步：交叉熵损失

```
Loss = -log(p_0) ≈ 0    # 学得很好，损失极小
```

### 对比：学错了的情况

若 Backbone 未学好，输出 `f_bad = [0.30, 0.95, 0.00, 0.00]`，标签仍为 0：

```
logits = [0.30,  0.95,  0.00]    # 李四分数反而最高
经 ArcFace + 缩放后，p_1 ≈ 1.0
Loss = -log(p_0) → 非常大      # 反向传播修正 Backbone 和类中心
```

### 计算步骤汇总

| 步骤 | 做什么 | 是否用到 ID |
|------|--------|------------|
| 1 | Backbone 提特征 | 否 |
| 2 | L2 归一化 | 否 |
| 3 | 读类中心表 | 表按 ID 编号 |
| 4 | 内积算相似度 | 否 |
| 5 | ArcFace margin | **是，选第 label 维** |
| 6 | ×64 缩放 | 否 |
| 7 | Softmax | 否 |
| 8 | 交叉熵 | **是，取 p_label** |

---

## 5. PartialFC 是什么

### 5.1 定义

**PartialFC（Partial Fully Connected）** 是 InsightFace 训练时专用的分类头，一张形状为 `(num_classes, embedding_size)` 的可学习权重矩阵。例如 MS1M 数据集配置（`configs/ms1mv3_r50_onegpu.py`）：

```python
config.num_classes = 93431    # 93431 个身份
config.embedding_size = 512   # 每个类中心 512 维
```

初始化时**随机生成**，不是数据集自带的：

```python
self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (num_local, embedding_size)))
```

### 5.2 不是数据集自带的

| 来源 | 提供什么 |
|------|----------|
| **数据集** | 每张图对应的整数标签（如 85742） |
| **PartialFC** | 随机初始化、训练中学习的类中心向量表 |

```
数据集标签（标注）              PartialFC 类中心表（模型参数）
─────────────────              ─────────────────────────────
图1 → label=0        ──→      第 0 行 w_0（训练中学出）
图2 → label=0        ──→      同一人的不同照片，都指向 w_0
图3 → label=85742    ──→      第 85742 行 w_85742
```

数据集只告诉你「该用哪一行」；那一行向量是网络自己学出来的。

### 5.3 不算标签

- **标签**：整数 ID，是监督信号，表示「这是谁」。
- **类中心 w_i**：模型参数，类似分类网络最后一层权重，**不是标签**。

### 5.4 为什么叫「Partial」

当身份数达到几十万时，对全部类中心算 softmax 显存和算力吃不消。PartialFC 在每个 batch **只采样一部分负类中心**参与计算（`sample_rate < 1`），但整张表仍保留并逐步更新。`sample_rate = 1.0` 时等价于普通全连接。

### 5.5 训练过程中类中心如何变化

```
训练前（随机）:  w_0 = [0.003, -0.012, ...]   ← 无意义噪声
训练中:          损失要求 f 靠近 w_0、远离 w_1、w_2 → 反向传播更新
训练后（收敛）:  w_0 ≈ 张三所有人脸特征的平均方向
```

---

## 6. 训练 vs 推理

### 6.1 对比表

| | 训练 | 推理 / 上线 |
|---|------|-------------|
| 用的模块 | Backbone + PartialFC | **只用 Backbone** |
| 输出 | 512 维向量 f（再进 FC 算损失） | **512 维向量 f**（直接使用） |
| 是否关心身份个数 | 要，`num_classes = 93431` 等 | **不用**，与训练集身份数无关 |
| 标签 | 需要整数 label | **不需要** |
| PartialFC 类中心表 | 用于算损失、更新参数 | **扔掉** |

### 6.2 推理时在做什么

```
人脸图 → Backbone → 512 维向量 f → 结束
```

之后全是向量运算：

```
# 1:1 验证
sim = cosine(f_A, f_B)
if sim >= threshold: 判定为同一人

# 1:N 识别
对每个底库用户算 cosine(f_probe, f_注册用户)
取得分最高者 → 识别结果
```

### 6.3 为什么训练用了 FC，推理却可以扔掉

训练的目标不是「记住 MS1M 里 93431 个人」，而是学一个**好的特征空间**：

- 同一人不同照片 → 向量靠近
- 不同人 → 向量远离

PartialFC 是施加这一约束的**训练工具**。Backbone 学会之后，新用户、训练集里没见过的人，也能映射到同一特征空间中直接比对。

### 6.4 业务底库 vs 训练类中心

训练 PartialFC 里的 `w_85742` 表示 **MS1M 训练集里第 85742 号人**，不是业务系统中的用户「张三」。

```
注册: 张三拍照 → Backbone → f_张三 → 存入业务数据库
识别: 现场脸   → Backbone → f_probe → 与 f_张三、f_李四... 比相似度
```

业务底库向量来自**注册照提取**，不是来自训练时的类中心表。

---

## 7. 评测时标签怎么用

推理阶段不输出分类结果，评测时用标签判断**向量相似度判定是否正确**。

### 7.1 1:1 验证（LFW、IJB-C 等）

```
图 A → 向量 v_A
图 B → 向量 v_B
相似度 = cosine(v_A, v_B)
标签 = issame（True = 同一人，False = 不同人）
```

代码见 `recognition/arcface_torch/eval/verification.py`：`embeddings` 是特征向量，`issame_list` 是布尔标签（不是向量）。

### 7.2 1:N 识别

```
Gallery: 张三 → v_张三, 李四 → v_李四, ...
Probe:   现场脸 → v_probe
```

对每个 Gallery 身份算相似度，取得分最高者；标签是 Probe 的**真实身份 ID**，看 Top-1 是否正确。

更多指标定义见 [人脸识别算法指标与评测指南](./人脸识别算法指标与评测指南.md)。

---

## 8. 与大模型词表的类比

### 8.1 相似点

两者训练时都采用「大表 + 整数 ID + Softmax 交叉熵」：

| | 人脸识别 PartialFC | 大模型词表 |
|---|-------------------|-----------|
| 表的大小 | `num_classes` 行（几万～几百万） | `vocab_size` 行（几万～几十万） |
| ID 的作用 | `label=85742` 指定身份 | `token_id=1234` 指定词 |
| 损失 | 正确 ID 对应分数要最高 | 正确 token 对应分数要最高 |
| 规模优化 | PartialFC 采样负类 | Sampled Softmax 等 |

### 8.2 关键不同

**① 在网络中的位置不同**

```
大模型:
  token_id ──查表──→ 词嵌入向量 ──Transformer──→ hidden ──LM Head──→ 词表打分
  ↑ 输入侧查表                                              ↑ 输出侧打分

人脸识别:
  图像 ──Backbone──→ 向量 f ──PartialFC──→ 身份表打分
  ↑ 向量从图像算出，不是从 ID 查表
```

PartialFC 更像大模型的 **LM Head（输出词表头）**，而不是输入侧的 **Token Embedding（词嵌入表）**。

**② 推理时命运不同**

| | 大模型 | 人脸识别 |
|---|--------|----------|
| 推理还用这张表吗 | **用**（输入嵌入 + LM Head） | **不用**（PartialFC 扔掉） |
| 推理输入 | token ID | 图像 |
| 推理输出 | 词表上的概率分布 | 512 维向量 |

**③ 对应关系**

| 人脸 | 大模型 |
|------|--------|
| PartialFC 类中心表 | LM Head 权重矩阵 |
| Backbone 输出的 f | Transformer 的 hidden state |
| 数据集 label | 下一个 token 的 target id |
| 推理时底库向量 | 无直接对应（人脸有独立的注册底库） |

---

## 9. 常见误区

| 误区 | 正确理解 |
|------|----------|
| 标签是向量 | 标签是整数 ID；和向量比对的是类中心表里的 w_i |
| PartialFC 表是数据集自带的 | 随机初始化，训练中学习；数据集只提供整数标签 |
| 推理时要加载 num_classes 和 FC | 推理只要 Backbone；底库规模与训练集身份数无关 |
| 业务用户对应训练集类中心某一行 | 业务底库向量来自注册照提取，与训练类中心无直接映射 |
| 训练目标是记住训练集所有人 | 目标是学一个好的度量空间，支持未见过的身份 |

---

## 10. 本项目代码索引

| 内容 | 路径 |
|------|------|
| 训练主循环 | `recognition/arcface_torch/train_v2.py` |
| PartialFC 实现 | `recognition/arcface_torch/partial_fc_v2.py` |
| ArcFace 损失 | `recognition/arcface_torch/losses.py` |
| 数据集读取（整数标签） | `recognition/arcface_torch/dataset.py` |
| 训练配置示例 | `recognition/arcface_torch/configs/ms1mv3_r50_onegpu.py` |
| LFW 等 1:1 评测 | `recognition/arcface_torch/eval/verification.py` |
| 推理特征提取（GUI） | `python-package/insightface/gui/core/face_engine.py` |
| 指标与评测指南 | `docs/人脸识别算法指标与评测指南.md` |
| 训练数据集说明 | `recognition/_datasets_/README.md` |

---

## 参考资料

- ArcFace: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698v1.pdf)
- PartialFC: [Killing Two Birds with One Stone](https://arxiv.org/abs/2203.15565)
- InsightFace 训练文档: `recognition/arcface_torch/README.md`
