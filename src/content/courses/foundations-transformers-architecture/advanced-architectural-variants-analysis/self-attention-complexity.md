---
course: "foundations-transformers-architecture"
chapter: "advanced-architectural-variants-analysis"
lesson: "self-attention-complexity"
sourceId: 2036
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/self-attention-complexity"
title: "自注意力机制的计算复杂度"
description: "分析自注意力机制与序列长度相关的二次方复杂度。"
order: 1
plots: ["plots/2036-0.json"]
sourceHash: "eff60cf532bfb25edcb7e79db998c739bc069c2328fb52701d3d0821b9192e0a"
sourceCorrections: []
---

如前所述，标准的自注意力 (self-attention)机制 (attention mechanism)虽然强大，但会带来显著的计算负担，尤其当输入序列变长时。了解这种计算成本，有助于把握本章后续会介绍的更高效架构的设计初衷。

缩放点积注意力公式中的主要运算构成了Transformer中自注意力机制的基本形式。


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里，$Q$ (查询)、$K$ (键) 和 $V$ (值) 是从输入序列嵌入 (embedding)中得到的矩阵。设 $N$ 为序列长度，$d_k$ 为键和查询的维度，$d_v$ 为值的维度。

主要计算步骤如下：

1. **查询-键相似度计算：** 计算矩阵乘积 $QK^T$。

   - $Q$ 的维度为 $(N \times d_k)$。
   - $K^T$ 的维度为 $(d_k \times N)$。
   - 得到的注意力得分矩阵 $S = QK^T$ 的维度为 $(N \times N)$。
   - 此矩阵乘法的计算成本大约为 $O(N \cdot d_k \cdot N) = O(N^2 d_k)$ 浮点运算 (FLOPs)。
2. **缩放与Softmax：** 将得分按 $1/\sqrt{d_k}$ 进行缩放，并对每行应用softmax函数。

   - 缩放涉及 $N^2$ 次元素级乘法。
   - Softmax涉及对每个 $N$ 行进行指数运算和归一化 (normalization)，每行所需的运算量与行长 $N$ 成正比。总成本大约为 $O(N^2)$。
   - 与矩阵乘法相比，对于较大的 $N$ 和 $d_k$，此步骤的计算量通常较小。
3. **值聚合：** 将softmax输出 (即注意力权重 (weight)矩阵 $A$，维度为 $(N \times N)$) 与值矩阵 $V$ 相乘。

   - $A$ 的维度为 $(N \times N)$。
   - $V$ 的维度为 $(N \times d_v)$。
   - 得到的输出矩阵 $O = AV$ 的维度为 $(N \times d_v)$。
   - 此矩阵乘法的计算成本大约为 $O(N \cdot N \cdot d_v) = O(N^2 d_v)$ 浮点运算 (FLOPs)。

### 主要影响因素和总体复杂度

综合这些步骤，总计算复杂度主要取决于两次大型矩阵乘法：$O(N^2 d_k + N^2 d_v)$。

在许多标准Transformer配置中，$d_k$ 和 $d_v$ 的维度与整体模型嵌入 (embedding)维度 $d_{model}$ 成比例 (通常 $d_k = d_v = d_{model} / h$，其中 $h$ 是注意力头的数量)。因此，复杂度通常概括为：


$$
O(N^2 \cdot d_{model})
$$


这种对序列长度 $N$ 的二次方依赖是重要瓶颈。尽管这些运算在序列维度上高度并行化 (与循环模型不同)，但总运算次数随 $N$ 的增长而迅速增加。

### 内存复杂度

除了计算之外，还有显著的内存需求。中间注意力得分矩阵 $QK^T$ (softmax之前或之后) 的维度为 $(N \times N)$。存储该矩阵需要：


$$
O(N^2)
$$


的内存。对于长序列 (例如 $N > 4096$)，存储一个 $(N \times N)$ 的浮点数矩阵可能会超出典型加速器 (如GPU) 的内存容量，甚至在考虑激活值、梯度和模型参数 (parameter)所需的内存之前。



![计算成本增长：N 与 N^2 对比](plots/2036-0.json)



> 该图表显示了计算成本的迅速分化。随着序列长度 $N$ 的增加，标准自注意力 (self-attention)机制 (attention mechanism)的 $O(N^2)$ 成本迅速超过线性 $O(N)$ 增长，使其对于非常长的序列变得不切实际。Y轴使用对数刻度以适应较大的数值范围。

### 对长序列的影响

这种二次方计算和内存复杂度严重限制了标准Transformer应用于涉及极长序列的任务，例如：

- 处理被视为图像块序列的整个高分辨率图像。
- 分析完整的长篇文档或书籍。
- 对长篇时间序列数据建模 (例如，音频、传感器读数)。
- 处理基因组序列。

即使配备强大的硬件，序列长度超过几千个token后，在训练和推理 (inference)期间，从运行时和内存角度来看都会变得困难。此限制直接促使了本章后续部分讨论的其他注意力机制 (attention mechanism)和架构变体的发展，这些旨在减少这种二次方依赖。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和缩放点积注意力机制，详细阐述了其组成部分和初步的复杂度考量。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2023)
  Publisher: Cambridge University Press
  提供了深度学习概念的全面易懂的解释，包括注意力机制及其计算特性的详细分析。
