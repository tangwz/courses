---
course: "foundations-transformers-architecture"
chapter: "multi-head-self-attention"
lesson: "concatenation-final-projection"
sourceId: 1963
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-3-multi-head-self-attention/concatenation-final-projection"
title: "拼接与最终线性投影"
description: "通过拼接和一个最终线性层组合多头的输出。"
order: 6
plots: []
sourceHash: "283d57eced67f8f55e2039c41a172cd99c89b51e9a90386c90a4d83a80c7c90a"
sourceCorrections: []
---

在为每个 $h$ 个头并行计算缩放点积注意力后，我们会得到 $h$ 个不同的输出向量 (vector)，通常表示为 $head_1, head_2, ..., head_h$。每个 $head_i$ 都是通过对原始查询、键和值使用投影版本来应用注意力机制 (attention mechanism)的结果。

$head_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$

其中 $X$ 表示输入序列嵌入 (embedding)（或来自上一层的输出），并且 $W_i^Q$、$W_i^K$、$W_i^V$ 是第 $i$ 个头的参数 (parameter)矩阵。每个 $head_i$ 的维度为 $d_v$，其中 $d_v = d_{model} / h$。这种划分使得所有头的计算成本与具有完整 $d_{model}$ 维度的单头注意力机制的计算成本相当。

然而，Transformer 中的后续层，例如逐位置前馈网络和残差连接，需要一个具有模型隐藏维度 $d_{model}$ 的单个输入张量。因此，我们需要一种机制将这些并行头的输出整合回一个单一的表示形式。

### 组合头输出：拼接

整合不同头所捕获信息的首要步骤是直接的：拼接。所有 $h$ 个头的输出向量 (vector)沿其特征维度进行拼接。

如果每个 $head_i$ 是一个形状为（序列长度，$d_v$）的矩阵，拼接操作会将这些矩阵并排堆叠，产生一个形状为（序列长度，$h \times d_v$）的新矩阵。由于我们将 $d_v$ 定义为 $d_{model} / h$，得到的拼接矩阵具有（序列长度，$d_{model}$）的维度。

$\text{拼接}(head_1, head_2, ..., head_h) \in \mathbb{R}^{\text{序列长度} \times (h \cdot d_v)} = \mathbb{R}^{\text{序列长度} \times d_{model}}$

此操作有效汇聚了每个头所关注的不同表征子空间中获取的见解。

> 流程图显示了单个注意力头的输出如何被拼接，然后通过最终线性层进行投影。

### 最终线性投影

拼接汇聚了头部的输出，但结果张量只是简单地将这些专用表示并置。为了让这些表示能够有效交互和组合，并确保输出维度与后续层（特别是残差连接）所需的 $d_{model}$ 相匹配，会应用一个最终线性投影。

这种投影通过另一个学习到的权重 (weight)矩阵 $W^O$ 实现，其维度为 $(h \times d_v, d_{model})$，简化后为 $(d_{model}, d_{model})$。拼接后的输出与此权重矩阵相乘：

$\text{多头输出} = \text{拼接}(head_1, ..., head_h) W^O$

这种最终线性变换作用显著：

1. **信息混合：** 它使得从不同头（代表不同子空间）中学习到的信息得以组合和整合。线性层充当一个学习到的组合函数。
2. **维度匹配：** 它确保输出张量具有Transformer块架构其余部分所需的精确 $d_{model}$ 维度，从而能够进行残差连接（加和归一化 (normalization)）等操作。

根本上，多头机制首先将模型的表示能力分解为多个子空间（$h$ 个头，每个维度为 $d_v$），允许每个头在其子空间内专门关注输入的不同方面，然后使用拼接，再进行线性投影（$W^O$），将这些专业化的视图合并回一个维度为 $d_{model}$ 的单一、更丰富的表示。

这种结构使模型能够同时捕获多种关系模式（例如，短距离依赖、长距离依赖、句法关系），这对于单一注意力机制 (attention mechanism)而言将更具挑战。投影矩阵（每个头的 $W_i^Q, W_i^K, W_i^V$）和最终输出投影矩阵（$W^O$）的所有参数 (parameter)都在模型训练期间进行端到端学习。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Pages: 15; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，解释了多头注意力机制的结构，包括头部输出的拼接和最终的线性投影。
- [torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) — PyTorch Development Team (2024)
  Journal: PyTorch Documentation; Publisher: PyTorch Foundation
  描述了线性变换层的功能，该层用于多头注意力模块中的输入投影（$W^Q, W^K, W^V$）和输出投影（$W^O$）。
