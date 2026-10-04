---
course: "mixture-of-experts"
chapter: "moe-foundations-sparse-models"
lesson: "mathematical-formulation-moe"
sourceId: 4684
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-1-moe-foundations-sparse-models/mathematical-formulation-moe"
title: "基本MoE层的数学表述"
description: "详细阐述基本MoE层前向传播的数学公式。"
order: 4
plots: []
sourceHash: "718c5b66393dd0c6d2503c19ef97037d1690d212a5c080d484ae97e3bb595eeb"
sourceCorrections: []
---

专家混合（MoE）层是一种稀疏的神经网络 (neural network)架构，与密集架构形成对比。基本MoE层所涉及的计算被正式阐述。这种数学描述准确阐明了输入数据如何流经系统、路由决策如何做出以及专家输出如何组合。理解此过程对于后续考察更复杂的架构变体和训练优化很重要。

考虑一个集成在更大网络（例如Transformer块）中的MoE层。该层为序列中的每个标记 (token)接收一个输入表示。令 $x \in \mathbb{R}^d$ 表示单个标记的输入嵌入 (embedding)，其中 $d$ 是模型的隐藏维度。MoE层由 $N$ 个专家网络（表示为 $E_1, E_2, \dots, E_N$）和一个门控网络 $G$ 组成。

### 门控网络的工作方式

门控网络的主要功能是决定哪个或哪些专家应处理输入标记 (token) $x$。一个常见实现是使用简单的线性变换，之后是softmax函数。

1. **线性变换：** 输入标记 $x$ 使用一个可学习的权重 (weight)矩阵 $W_g \in \mathbb{R}^{d \times N}$ 进行投影。这会产生logit值 $h(x) \in \mathbb{R}^N$，每个专家对应一个：
   $h(x) = x W_g$
2. **Softmax激活：** Logit值随后通过一个softmax函数，生成归一化 (normalization)分数或概率，表明对每个专家的倾向性：
   $G(x) = \text{softmax}(h(x))$
   结果向量 (vector) $G(x) = [G(x)_1, G(x)_2, \dots, G(x)_N]$ 包含非负值且总和为1，其中 $G(x)_i$ 表示门控网络将标记 $x$ 分配给专家 $E_i$ 的分数。

> 数据流经门控网络，将输入标记表示 $x$ 转换为专家分配分数 $G(x)$。

### 专家网络

每个专家 $E_i$ 通常是一个神经网络 (neural network)模块，常见的是一个标准前馈网络（FFN），类似于Transformer块中的FFN。重要的是，每个专家 $E_i$ 都拥有自己独特的参数 (parameter)集。对于一个输入标记 (token) $x$，第 $i$ 个专家的输出表示为 $E_i(x)$。尽管结构上相似，但参数差异使得每个专家能够专门处理不同类型的输入模式或执行不同的子任务。

### 稀疏路由与输出计算

虽然门控网络为每个专家生成一个分数 $G(x)_i$，但稀疏MoE模型通过仅为每个标记 (token)激活这些专家中的一部分来运用条件计算。这通常通过**Top-k路由**实现，即只选择门控分数最高的 $k$ 个专家（通常 $k=1$ 或 $k=2$）。

1. **专家选择：** 根据 $G(x)$ 中的分数，确定Top-k个专家的索引 $I$：
   $I = \text{TopKIndices}(G(x), k)$
   例如，如果 $k=2$ 且专家 $j$ 和 $l$ 具有最高分数，则 $I = \{j, l\}$。
2. **门控分数重新归一化 (normalization)（可选但常见）：** 选定专家（$i \in I$）的门控分数通常被保留并可能重新归一化。为此处简化，我们将用于加权专家 $i$ 输出的分数表示为 $g_i = G(x)_i$。一些实现可能只对Top-k分数重新应用softmax。
3. **专家计算：** 输入标记 $x$ 仅由选定的专家处理：
   $O_i(x) = E_i(x) \quad \text{for } i \in I$
4. **加权输出组合：** MoE层对标记 $x$ 的最终输出 $y$ 计算为选定专家输出的加权和，使用它们对应的门控分数作为权重 (weight)：
   $y = \sum_{i \in I} g_i O_i(x) = \sum_{i \in I} G(x)_i E_i(x)$

这种表述确保对于任何给定的输入标记，仅有专家参数 (parameter)的一小部分（$k/N$）被使用，相比于激活所有 $N$ 个专家或使用单个大型专家，这在前向传播过程中显著节省了计算资源。

> 单个标记 $x$ 在采用 $k=2$ 路由的MoE层中的前向传播。门控网络生成分数，Top-k选择专家（此处显示E1和EN），输入仅由选定的专家处理，它们的输出使用门控分数进行组合。

### 批量处理的考量

实际中，MoE层对标记 (token)批次进行操作。对于一个包含 $T$ 个标记的批次，输入通常表示为一个矩阵 $X \in \mathbb{R}^{T \times d}$。门控网络同时计算所有标记的分数 $G(X) \in \mathbb{R}^{T \times N}$。重要的一点是，Top-k选择对批次中的每个标记（行）独立执行。这意味着同一批次中的不同标记可以被路由到不同的专家组。

尽管高效，但这种每个标记的独立路由带来了挑战，特别是在**负载均衡**方面：确保专家在整个批次中获得大致相等的计算量。如果门控网络持续将大多数标记路由到一小部分专家，其他专家则利用不足，削弱了专业化的优势，并可能导致训练不稳定。我们将在第3章讨论这些训练动态和相关的优化方法。

这个数学框架提供了必要的构建模块，用来理解基本MoE层如何执行条件计算。门控网络、稀疏路由机制和专家网络之间的关联构成了MoE方法的主要组成部分。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇奠基性论文介绍了稀疏门控专家混合层，详细阐述了门控网络、Top-k路由和加权输出组合，构成了现代MoE架构的基础。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: Journal of Machine Learning Research; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  本文扩展了大型Transformer模型中MoE层的实际实现和扩展性，讨论了架构细节、路由机制以及专家负载均衡等训练挑战。
- [Learning with Experts](https://doi.org/10.1162/neco.1991.3.1.79) — Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, Geoffrey E. Hinton (1991)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 3; Pages: 79-87; DOI: [10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79)
  这项开创性工作介绍了专家混合学习的概念，为将多个专业网络与门控网络结合以处理输入奠定了许多理论基础。
