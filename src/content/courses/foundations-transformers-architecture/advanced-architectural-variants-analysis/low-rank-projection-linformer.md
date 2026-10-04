---
course: "foundations-transformers-architecture"
chapter: "advanced-architectural-variants-analysis"
lesson: "low-rank-projection-linformer"
sourceId: 2045
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/low-rank-projection-linformer"
title: "低秩投影方法（Linformer）"
description: "Linformer 如何通过对键和值进行低秩投影来降低复杂度。"
order: 5
plots: []
sourceHash: "2c05932c9beefdffbf837c4994bfc2ad9784a97bed43e18f2876cf4ae52e5ca6"
sourceCorrections: []
---

标准自注意力 (self-attention)机制 (attention mechanism)的二次复杂度，即 $O(N^2 d)$（其中 $N$ 是序列长度，$d$ 是模型维度），在处理长序列时是一个主要瓶颈。尽管功能强大，但随着 $N$ 的增加，计算每对标记 (token)之间的注意力分数会变得计算成本过高。

有几种方法旨在减轻这种计算负担。一个重要方向是利用低秩投影来近似注意力机制，Linformer 模型就是其中的一个例子。

### 低秩假设

Linformer 的核心思路是基于这样一个假设：尽管自注意力 (self-attention)机制 (attention mechanism)有能力模拟整个序列中复杂的关联，但它通常可以通过一个低秩矩阵进行近似。本质上，$N \times N$ 的注意力矩阵 $P = Softmax(\frac{QK^T}{\sqrt{d_k}})$ 可能在结构上存在冗余。这意味着从输入上下文 (context)（由值 $V$ 表示）到输出上下文（加权和 $PV$）的映射不一定需要任意 $N \times N$ 矩阵的完整表达能力。如果这个假设成立，我们就可以通过使用压缩表示来大幅提高效率。

### Linformer：键和值的投影

Linformer（线性 Transformer）提出了一种巧妙的方法来实现线性复杂度，即引入学习到的投影矩阵 $E_i$ 和 $F_i$。Linformer 不计算完整的 $N \times N$ 注意力矩阵，而是在注意力计算*之前*沿序列长度维度投影键 ($K$) 和值 ($V$) 矩阵。

设输入序列长度为 $N$，头维度为 $d_k$（用于键/查询）或 $d_v$（用于值）。原始矩阵为：

- 查询 $Q \in \mathbb{R}^{N \times d_k}$
- 键 $K \in \mathbb{R}^{N \times d_k}$
- 值 $V \in \mathbb{R}^{N \times d_v}$

Linformer 引入了两个投影矩阵 $E \in \mathbb{R}^{k \times N}$ 和 $F \in \mathbb{R}^{k \times N}$，其中 $k$ 是一个投影维度，远小于 $N$ ($k \ll N$)。这些矩阵用于创建投影键矩阵和投影值矩阵：


$$
K_{proj} = E K \quad (\text{维度 } k \times d_k)
$$


$$
V_{proj} = F V \quad (\text{维度 } k \times d_v)
$$


请注意，投影如何将序列维度从 $N$ 降低到 $k$。主要步骤是注意力分数现在在原始查询矩阵 $Q$ 和*投影*键矩阵 $K_{proj}$ 之间计算：


$$
P_{proj} = Softmax\left(\frac{Q K_{proj}^T}{\sqrt{d_k}}\right) \quad (\text{维度 } N \times k)
$$


最终输出通过将这个投影注意力分数矩阵 $P_{proj}$ 乘以*投影*值矩阵 $V_{proj}$ 获得：


$$
Attention_{Linformer}(Q, K, V) = P_{proj} V_{proj} \quad (\text{维度 } N \times d_v)
$$


### 复杂度分析

让我们分析计算复杂度。原始注意力计算主要由 $Q K^T$ 矩阵乘法决定，这需要 $O(N^2 d_k)$ 时间，以及随后与 $V$ 的乘法，这需要 $O(N^2 d_v)$ 时间。

在 Linformer 中：

1. 投影键 K：$E K$ 需要 $O(Nk d_k)$ 时间。
2. 投影值 V：$F V$ 需要 $O(Nk d_v)$ 时间。
3. 计算 $Q K_{proj}^T$：这涉及将一个 $N \times d_k$ 矩阵乘以一个 $d_k \times k$ 矩阵，结果需要 $O(Nk d_k)$ 时间。
4. 计算 $P_{proj} V_{proj}$：这涉及将一个 $N \times k$ 矩阵乘以一个 $k \times d_v$ 矩阵，结果需要 $O(Nk d_v)$ 时间。

由于 $k$ 被选择为使得 $k \ll N$，Linformer 注意力机制 (attention mechanism)的总体复杂度变为 $O(Nk)$，这相对于序列长度 $N$ 是线性的。这是对标准 $O(N^2)$ 复杂度的大幅改进。

> 标准自注意力 (self-attention)与 Linformer 投影注意力的计算流程比较。Linformer 引入了投影矩阵 (E, F)，以在计算注意力分数之前降低键和值沿着序列长度轴的维度。

### 实现考量

- **投影矩阵：** 投影矩阵 $E$ 和 $F$ 通常在训练期间学习。它们可以在不同的注意力头甚至层之间共享，以进一步减少参数 (parameter)数量。一个常见实现是使用简单的线性层作用于转置的键和值矩阵 ($K^T$, $V^T$)，以高效地执行投影。
- **k 的选择：** 投影维度 $k$ 是一个超参数 (hyperparameter)。通常使用 128、256 或 512 等值，这些值远小于典型序列长度（数千或数万），Linformer 在这些情况下变得有优势。这个选择影响计算效率和模型准确性之间的权衡。较小的 $k$ 运算更快，但可能导致更大的近似误差。
- **理论保证：** Linformer 论文提供了理论分析，表明在某些假设下，自注意力 (self-attention)矩阵确实是低秩的，并且可以通过这种投影方法很好地近似。

### 优点和缺点

**优点：**

- **线性时间复杂度：** 将注意力时间复杂度从 $O(N^2)$ 降低到 $O(Nk)$。
- **线性空间复杂度：** 减少了存储注意力矩阵所需的内存占用。
- **可扩展性：** 能够处理比标准 Transformer 长得多的序列。
- **修改简单：** 通过增加投影层，实现起来相对简单。

**缺点：**

- **近似误差：** 作为一种近似方法，它可能无法捕获完整注意力矩阵的所有细节，可能导致在某些任务上性能下降，特别是对于 $N^2$ 计算量可控的较短序列。
- **超参数 (parameter) (hyperparameter)调优：** 需要调整投影维度 $k$。
- **低秩假设：** 其有效性依赖于底层假设，即注意力矩阵可以很好地由低秩结构近似，但这对于所有数据或任务可能并非都同样适用。

Linformer 代表着构建更高效 Transformer 模型的一个重要步骤。通过质疑完整二次注意力计算的必要性，并借助低秩近似的思路，它提供了一种实用的方法来扩展 Transformer 到更长的序列，为涉及大量文档、高分辨率图像或长篇音频的应用带来了可能性。

## 参考资料

- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) — Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 33; Pages: 17316-17326; DOI: [10.48550/arXiv.2006.04768](https://doi.org/10.48550/arXiv.2006.04768)
  介绍了Linformer模型，该模型通过将键和值投影到较低维度，实现了自注意力机制的线性复杂度。
- [Nyströmformer: A Nyström-Algorithm-based Efficient Transformer](https://proceedings.mlr.press/v139/xiong21a/xiong21a.pdf) — Yuqi Xiong, Ye Li, Bo Li, Zhanpeng Zeng, Eytan Hu, Lizhen Nie, Chao Zhang, Mohan Teng, Xiao Zhang, Hao Ma (2021)
  Journal: International Conference on Machine Learning (ICML); Volume: 139; Pages: 11417-11428; DOI: [10.48550/arXiv.2102.03906](https://doi.org/10.48550/arXiv.2102.03906)
  提出了一种基于Nyström方法的高效Transformer，用于注意力矩阵的低秩近似。
- [A Survey of Efficient Transformers](http://arxiv.org/abs/2009.06732) — Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler (2022)
  Journal: ACM Computing Surveys; Publisher: ACM; Volume: 35; Pages: 37718-37746; DOI: [10.48550/arXiv.2009.06732](https://doi.org/10.48550/arXiv.2009.06732)
  提供了各种高效Transformer架构的概述和分类，包括Linformer及其他线性注意力模型。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入了Transformer架构和自注意力机制的奠基性论文，指出了Linformer旨在解决的二次复杂度瓶颈。
