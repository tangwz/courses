---
course: "foundations-transformers-architecture"
chapter: "positional-encoding-embedding-layer"
lesson: "sinusoidal-encoding-properties"
sourceId: 1994
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer/sinusoidal-encoding-properties"
title: "正弦编码的特性"
description: "分析正弦函数为何适用：固定范围、相对位置表示、外推能力。"
order: 4
plots: ["plots/1994-0.json"]
sourceHash: "7c0480e1ea28517b994bc381bdaf784d1b9180706ed073fd4dfac1405eaa6397"
sourceCorrections: []
---

在上一节中，我们已经定义了正弦位置编码 (positional encoding)的数学结构，具体为：

正弦位置编码的数学结构定义如下：

现在，我们来分析*为何*这种特殊的公式有效且常用。这些固定的、非学习的编码具有几个理想的特性，与Transformer架构良好匹配。

### 确定性和唯一性编码

与需要训练的学习型位置嵌入 (embedding)不同，正弦编码由固定函数生成。这意味着它们是确定性的：对于给定的位置 `pos` 和维度索引 `i`，其值始终相同。这也不需要专门针对位置信息的额外可训练参数 (parameter)。

此外，正弦和余弦函数在不同频率（由 $10000^{2i/d_{model}}$ 项确定）上的组合，确保在合理的序列长度内，每个位置 `pos` 获得唯一的编码向量 (vector) $PE_{pos} \in \mathbb{R}^{d_{model}}$。虽然在极长序列中理论上可能发生冲突，但在典型的模型限制下，这种情况实际上不存在。

### 有界值

正弦和余弦函数自然生成值在固定范围 $[-1, 1]$ 内。当这些位置编码 (positional encoding)添加到词元 (token)嵌入 (embedding)中（词元嵌入通常也在可控范围内，通常通过归一化 (normalization)或初始化）时，这种有界性可以避免位置信息大幅改变组合嵌入的幅度。与可能无界的位置信号相比，这有助于更稳定的训练过程。

### 相对位置表示

正弦编码最重要的优点，或许是其固有的通过线性变换表示相对位置的能力。考虑位置 $pos+k$ 的编码。使用三角函数和角公式：

$\sin(a+b) = \sin(a)\cos(b) + \cos(a)\sin(b)$
$\cos(a+b) = \cos(a)\cos(b) - \sin(a)\sin(b)$

令 $\omega_i = 1 / 10000^{2i/d_{model}}$。$PE_{pos+k}$ 的分量可以表示为 $PE_{pos}$ 的形式：

$PE_{(pos+k, 2i)} = \sin((pos+k)\omega_i) = \sin(pos \cdot \omega_i)\cos(k \cdot \omega_i) + \cos(pos \cdot \omega_i)\sin(k \cdot \omega_i)$
$= PE_{(pos, 2i)}\cos(k \cdot \omega_i) + PE_{(pos, 2i+1)}\sin(k \cdot \omega_i)$

$PE_{(pos+k, 2i+1)} = \cos((pos+k)\omega_i) = \cos(pos \cdot \omega_i)\cos(k \cdot \omega_i) - \sin(pos \cdot \omega_i)\sin(k \cdot \omega_i)$
$= PE_{(pos, 2i+1)}\cos(k \cdot \omega_i) - PE_{(pos, 2i)}\sin(k \cdot \omega_i)$

这可以表示为每对维度 $(2i, 2i+1)$ 的矩阵乘法：


$$
\begin{pmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{pmatrix}
=
\begin{pmatrix} \cos(k \omega_i) & \sin(k \omega_i) \\ -\sin(k \omega_i) & \cos(k \omega_i) \end{pmatrix}
\begin{pmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{pmatrix}
$$


这表明 $PE_{pos+k}$ 的位置编码 (positional encoding)是 $PE_{pos}$ 的一个线性函数（具体来说，是一个旋转）。变换矩阵仅取决于偏移量 $k$，而不是绝对位置 $pos$。这个特性使得自注意力 (self-attention)机制 (attention mechanism)（它涉及线性投影（查询、键、值）和点积）更容易学习根据词元 (token)之间的相对距离进行注意力计算。模型不需要为位置 5 上的 +2 偏移量和位置 50 上的 +2 偏移量学习单独的规则；这种关系被一致地编码。

### 平滑插值

正弦函数随位置平滑变化。这意味着相邻位置 $pos$ 和 $pos+1$ 的位置编码 (positional encoding)是相似的，这体现了相邻词语通常具有紧密相关的上下文 (context)作用的直觉。这种平滑变化与结构性较差的编码方案可能发生的突变形成对比。

### 外推能力

由于正弦编码由固定函数生成，而非从固定序列长度范围内的数据中学习，它们在处理比训练期间遇到的更长序列时提供了优势。该函数可以为任何位置 $pos$ 生成编码。虽然模型在处理更长序列时的整体性能可能仍会因其他因素（如注意力模式无法泛化）而下降，但位置编码 (positional encoding)机制本身不会出现固有故障，也不会为未见过的位置产生未定义的值，这与学习型嵌入 (embedding)不同，后者会缺乏超出训练最大值的位置的表示。

### 编码模式可视化

频率选择（$\omega_i = 1 / 10000^{2i/d_{model}}$）产生的信号从高频（对于小 $i$，随位置快速变化）到极低频（对于大 $i$，在整个序列中缓慢变化）不等。这使得模型能够以不同粒度捕捉位置信息。



![正弦位置编码值（前6个维度）](plots/1994-0.json)



> 热力图显示了128维嵌入 (embedding)（`d_model=128`）中，前30个位置（`pos`）和前6个维度（`d=0`到`d=5`）的正弦位置编码 (positional encoding)值。请注意，较低维度（顶部行）的振荡速度更快，而较高维度（底部行）的振荡速度更慢。

总而言之，正弦位置编码提供了一种简单而有效的无参数 (parameter)方法，可将序列顺序信息注入Transformer。它们的数学特性与注意力机制 (attention mechanism)在词元 (token)间建模关系的能力良好匹配，尤其是在相对位置方面，同时保持稳定性并为泛化到更长序列提供了可能。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv preprint arXiv:1706.03762; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer模型和正弦位置编码的开创性论文，详细阐述了其数学公式和通过三角恒等式表示相对位置的能力。
- [CS224N: Natural Language Processing with Deep Learning, Lecture 10: Transformers and Pretraining](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF6FRRpsJWOpbF-TWcMxtCrX5MHQ8yy3tFUck-Ni_MSiO60UuXgBa-Bj8lMyEC7XH4wQf3e9ObmJdzbYYUwubW8ZLgHE1gc8tIraB5Md314-T43-mAexq2LuE0Z33dmAqnAh7cQHybTWdaECcSu12nVFPQptD7W8P8V5GxeuxVM0eHwkDQPxWmaQpDdtZQ3a1TYS0YatagqPoZq) — Abigail See, Kevin Clark, Yuval Pinter (2023)
  Publisher: Stanford University
  对Transformer架构提供了易于理解的解释，包括正弦位置编码的原理和特性，特别是通过三角恒等式表示相对位置的方面。
