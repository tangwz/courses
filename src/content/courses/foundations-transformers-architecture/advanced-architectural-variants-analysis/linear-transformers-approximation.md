---
course: "foundations-transformers-architecture"
chapter: "advanced-architectural-variants-analysis"
lesson: "linear-transformers-approximation"
sourceId: 2042
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/linear-transformers-approximation"
title: "近似注意力机制：线性Transformer"
description: "探讨了诸如Linformer和Performer等以线性复杂度近似注意力机制的变体。"
order: 3
plots: []
sourceHash: "936f56dff9abbb445ec68b01d344c81173b1fb8e2c11c3ce89daf24a21119933"
sourceCorrections: []
---

标准自注意力 (self-attention)机制 (attention mechanism)的主要计算瓶颈是其相对于输入序列长度 $N$ 的二次复杂度。通过计算注意力得分矩阵 $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ 涉及计算 $N \times N$ 矩阵 $QK^T$。这需要查询（query）向量 (vector)和键（key）向量之间进行 $N^2$ 次点积（每个向量维度为 $d_k$），接着进行缩放、softmax归一化 (normalization)，最后乘以值（value）矩阵 $V$。整体复杂度通常由 $QK^T$ 的计算主导，导致注意力矩阵本身的计算时间复杂度为 $O(N^2 d_k)$，空间复杂度为 $O(N^2)$。这种随规模增长的特性使得将标准Transformer应用于超长序列在计算上变得难以承受（例如，被视为补丁序列的高分辨率图像、长文档或基因组数据）。

为克服此局限，一项重要的研究方向是开发 *线性Transformer*，这些变体以线性 $O(N)$ 的时间复杂度与内存复杂度来近似自注意力机制。核心思想是避免显式计算和存储完整的 $N \times N$ 注意力矩阵。

### Softmax的难点

如果我们考虑不带softmax归一化 (normalization)的注意力计算，$O_{unnorm} = (QK^T)V$，矩阵乘法的结合律允许我们将计算重排为 $O_{unnorm} = Q(K^TV)$。在此，$K^T$ 是 $d_k \times N$ 维矩阵，$V$ 是 $N \times d_v$ 维矩阵。计算 $K^TV$ 需要 $O(N d_k d_v)$ 次运算。接着，将结果乘以 $Q$ ($N \times d_k$) 又需要 $O(N d_k d_v)$ 次运算。如果嵌入 (embedding)维度 $d = d_k = d_v$ 被视为固定值或远小于 $N$，则整体复杂度在 $N$ 上呈线性关系。

然而，按行进行的softmax函数，定义为：
$\text{softmax}(Z)_i = \frac{\exp(Z_i)}{\sum_j \exp(Z_j)}$
其中 $Z_i$ 代表输入矩阵的第 $i$ 行（在此处，是 $\frac{QK^T}{\sqrt{d_k}}$ 的第 $i$ 行），阻碍了这种简单的重排。归一化项 $\sum_j \exp(\dots)$ 将一行中的所有元素耦合在一起，似乎需要为给定查询计算所有 $N$ 个点积，然后才能进行归一化和与值相乘的操作。

### 线性化策略

线性Transformer的变体采用不同的策略，以近似注意力机制 (attention mechanism)并绕过softmax瓶颈，有效地实现了类似于 $Q(K^TV)$ 的重排：

1. **核函数近似：** 该方法近似点积相似度，然后进行softmax中固有的指数函数计算，通常将 $\exp(q_i^T k_j / \sqrt{d_k})$ 视为核函数。如果这个核函数可以通过特征映射 $\phi(\cdot)$ 的内积来近似，即 $K(q_i, k_j) \approx \phi(q_i)^T \phi(k_j)$，那么注意力求和 $\sum_j K(q_i, k_j) v_j$ 可以重写为：
   $\sum_j \left( \phi(q_i)^T \phi(k_j) \right) v_j = \phi(q_i)^T \left( \sum_j \phi(k_j) v_j^T \right)$
   （注意：此处需要仔细处理向量 (vector)/矩阵的维度，这仅演示了重排方式）。
   项 $\sum_j \phi(k_j) v_j^T$ 计算基于键和值的聚合表示。这个聚合可以计算一次（在 $N$ 上呈线性），然后乘以每个查询的特征映射 $\phi(q_i)$（同样，在 $N$ 上呈线性）。难点在于找到合适的特征映射 $\phi$，使其能够准确近似指数核函数，同时允许这种分解。**Performer** 等模型使用基于随机特征的技术（特别是通过正交随机特征实现快速注意力，即FAVOR+）来构建这些映射，并对近似质量提供理论保证。这些方法的目标是达到 $O(N)$ 复杂度。我们将在“基于核函数的注意力近似（Performer）”一节中更详细地考察基于核函数的方法。
2. **低秩投影：** 该策略基于注意力矩阵 $A = \text{softmax}(QK^T / \sqrt{d_k})$ 通常是低秩的，或可以被低秩矩阵很好地近似的观察或假设。**Linformer** 模型通过引入可学习的投影矩阵 $E$ ($k \times N$) 和 $F$ ($k \times N$) 来体现这一点，其中 $k$ 是一个固定维度，远小于 $N$ ($k \ll N$)。它计算投影后的键矩阵 $K' = EK$ 和值矩阵 $V' = FV$。这些投影矩阵的维度分别为 $k \times d_k$ 和 $k \times d_v$。注意力机制随后使用这些投影矩阵进行操作，例如，通过基于 $Q(K')^T$ ($N \times k$) 计算注意力权重 (weight)，并将其应用于 $V'$ ($k \times d_v$)。复杂度从 $O(N^2)$ 降低到 $O(Nk)$，如果 $k$ 是常数或随 $N$ 亚线性增长，则实现了线性扩展。这种方法将在“低秩投影方法（Linformer）”一节中详细说明。

### 影响与权衡

线性注意力机制 (attention mechanism)在计算效率方面提供了显著优势，尤其对于长序列。

| 特性 | 标准注意力 | 线性注意力（近似） |
| --- | --- | --- |
| 时间复杂度 | $O(N^2 d)$ | $O(N d r)$ （其中 $r$ 取决于方法，例如特征维度或投影大小 $k$） |
| 内存（注意力矩阵） | $O(N^2)$ | $O(Nr)$ 或更少 |
| 计算 | 显式 $N \times N$ 矩阵 | 避免显式 $N \times N$ 矩阵 |
| 精确性 | 精确 | 近似 |
| 适用性 | 短到中等长度序列 | 长序列 |

主要的权衡在于，这些方法计算的是完整自注意力 (self-attention)机制的*近似*。虽然实证结果常表明这些近似表现出色，有时甚至在特定任务上胜过标准注意力（可能归因于隐式正则化 (regularization)效应），但它们可能无法捕捉与原始公式完全相同的关系信息。近似质量及其对下游任务性能的影响必须通过实证评估。

通过将计算负担从二次降低到线性，这些方法显著扩展了Transformer架构的适用范围，到以前受序列长度限制阻碍的领域，使模型能够更有效地处理整个文档、高分辨率图像或长时间序列数据。

## 参考资料

- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2009.14794](https://doi.org/10.48550/arXiv.2009.14794)
  介绍了Performer模型，利用正交随机特征（FAVOR+）对注意力进行核近似，实现了线性复杂度。
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) — Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2006.04768](https://doi.org/10.48550/arXiv.2006.04768)
  提出了Linformer，一种通过键和值矩阵的低秩投影实现线性复杂度的Transformer变体。
- [Long-Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.04006) — Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.04006](https://doi.org/10.48550/arXiv.2011.04006)
  本文介绍了一个标准化基准，用于评估各种Transformer架构（包括线性注意力变体）在长距离依赖任务上的效率和性能。
