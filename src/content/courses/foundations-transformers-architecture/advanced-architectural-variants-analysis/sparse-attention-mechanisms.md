---
course: "foundations-transformers-architecture"
chapter: "advanced-architectural-variants-analysis"
lesson: "sparse-attention-mechanisms"
sourceId: 2039
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-6-advanced-architectural-variants-analysis/sparse-attention-mechanisms"
title: "稀疏注意力机制"
description: "介绍降低注意力复杂性的方法（例如，固定、步进、局部注意力模式）。"
order: 2
plots: []
sourceHash: "8004c478c5783281218b62b79a3247b8a4d2e91607e2b11cf144fdcc5f31e9db"
sourceCorrections: []
---

标准Transformer的计算瓶颈在于自注意力 (self-attention)机制 (attention mechanism)。计算注意力权重 (weight)需要将序列中的每个词元 (token)（查询）与所有其他词元（键）进行比较。这会导致计算和内存复杂度达到$O(N^2 d)$，其中$N$是序列长度，$d$是模型维度。虽然当$d$被视为常数时，复杂度常被简化为$O(N^2)$，但这种平方级增长使得处理非常长的序列（数千或数万个词元）在计算上难以承受。

稀疏注意力机制旨在缓解这一瓶颈，通过减少需要计算的查询-键对的数量。稀疏注意力不是让每个词元关注*所有*其他词元，而是将注意力限制在精心选择的词元*子集*上，从而形成一个稀疏注意力矩阵。核心设想是，完全注意力通常是多余的，对于给定词元，大部分相关信息可以通过关注更小、策略性选择的其他词元集来获取。

### 为何采用稀疏性？

经验分析常表明，训练有素的Transformer中学习到的注意力矩阵确实是稀疏的。许多注意力权重 (weight)接近零，这表明一个词元 (token)只强烈关注有限数量的其他词元。稀疏注意力方法试图借助这一观察结果，通过预定义或学习模式来将计算集中在潜在的重要关系上，从而大幅减少$N^2$次比较。

### 常见的稀疏模式

已提出几种结构化的稀疏模式，这些模式通常结合不同类型的注意力，以平衡局部上下文 (context)与更宽广的远程交互。

#### 1. 滑动窗口（局部）注意力

最直观的模式是局部或滑动窗口注意力。在此，每个词元 (token)只关注一个固定大小为$k$的邻近词元窗口（例如，左侧$k/2$个词元和右侧$k/2$个词元）。

> 局部注意力模式，其中词元4关注自身及其窗口大小内的邻近词元（此处k=4，包括左侧2个，右侧2个，以及自身）。计算被限制在这些邻近词元上。

这会将每个词元的复杂度从$O(N)$降至$O(k)$，从而使总复杂度变为$O(N \cdot k)$。由于$k$是一个通常远小于$N$的固定超参数 (parameter) (hyperparameter)，因此这在$N$上是有效的线性复杂度。然而，明显的局限性是信息无法在单个注意力层内传播超出窗口大小$k$。获取远程依赖关系需要堆叠许多这样的层。

#### 2. 扩张（步进）注意力

为了减轻局部注意力感受野受限的问题而不牺牲效率，可以使用扩张或步进注意力。类似于扩张卷积，一个词元关注具有递增间隔或步长的邻近词元。例如，一个词元可能关注位置$i \pm 1$、$i \pm 2$、$i \pm 4$、$i \pm 8$等。

> 扩张（步进）注意力模式，其中词元5关注自身、紧邻词元（间隔1）、间隔2的词元以及间隔4的词元。这允许以每词元固定的计算量覆盖更宽广的感受野。

这使得感受野可以随层数呈指数级增长，使得捕获更长距离的依赖关系比纯局部注意力更高效，同时保持$O(N \cdot k_{dilated})$的复杂度，其中$k_{dilated}$是关注位置的数量（通常与$N$呈对数关系）。

#### 3. 全局 + 局部注意力

许多成功的稀疏注意力模型，如Longformer和BigBird，将局部注意力与少量预先选择的`全局`词元结合起来。这些全局词元可以关注序列中的所有其他词元，并且所有其他词元也可以关注它们。

- **局部注意力：** 大多数词元使用滑动窗口注意力来获取局部上下文。
- **全局注意力：** 某些词元（例如，像`[CLS]`这样的特殊词元，或被识别为特别重要的词元）具有完全注意力能力。这确保了信息可以全局汇聚并分发回局部上下文。

这种混合方法旨在兼顾两者的优点：通过局部注意力为大多数词元提供线性扩展，以及通过全局词元获取重要的远程依赖的能力。其复杂度通常由局部注意力部分主导，保持接近$O(N \cdot k)$，前提是全局词元的数量相对于$N$较小。

### 实现考虑与权衡

实现稀疏注意力通常需要比标准注意力的密集矩阵乘法更复杂的索引操作。不是计算完整的$N \times N$注意力矩阵，而是必须收集或计算与允许的稀疏连接相对应的特定索引。库和框架越来越多地为常见的稀疏模式（例如，块稀疏操作）提供优化实现。

**优点：**

- **复杂度降低：** 计算成本从$O(N^2)$显著降至通常接近$O(N)$，从而能够处理更长的序列。
- **内存减少：** 内存占用更低，因为不再实例化完整的$N \times N$矩阵。

**缺点：**

- **潜在信息损失：** 如果预定义的稀疏模式遗漏了与任务相关的某些重要远程依赖，性能可能会比完全注意力下降（假设有足够的计算资源用于完全注意力）。
- **模式设计：** 稀疏模式（窗口大小、扩张率、全局词元 (token)）的选择成为一组需要调整的重要超参数 (parameter) (hyperparameter)。
- **实现复杂性：** 比标准密集注意力更难以正确高效地实现。

稀疏注意力代表着重要一步，使Transformer能够应用于涉及超长文档、高分辨率图像（被视为补丁序列）或扩展时间序列的任务。这是修改核心架构以克服固有扩展限制的一个典型例子，我们将在接下来的章节中通过近似技术继续讨论这一主题。

## 参考资料

- [Sparse Transformers](https://arxiv.org/abs/1904.10509) — Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever (2019)
  Journal: arXiv preprint arXiv:1904.10509; DOI: [10.48550/arXiv.1904.10509](https://doi.org/10.48550/arXiv.1904.10509)
  介绍了使用固定和步进模式的稀疏注意力方法，以降低二次复杂度。
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Iz Beltagy, Matthew E. Peters, and Arman Cohan (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2004.05150](https://doi.org/10.48550/arXiv.2004.05150)
  介绍了Longformer模型，结合了滑动窗口和全局注意力机制，用于处理长文档。
- [Big Bird: Transformers for Longer Sequences](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf) — Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 17283-17297; DOI: [10.48550/arXiv.2007.14062](https://doi.org/10.48550/arXiv.2007.14062)
  详细介绍了Big Bird模型，这是一种使用局部、全局和随机注意力机制的稀疏高效Transformer变体。
