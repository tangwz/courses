---
course: "mixture-of-experts"
chapter: "moe-foundations-sparse-models"
lesson: "sparse-moe-paradigm"
sourceId: 4682
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-1-moe-foundations-sparse-models/sparse-moe-paradigm"
title: "稀疏专家混合模型（MoE）方法"
description: "介绍稀疏专家混合模型的根本结构。"
order: 2
plots: []
sourceHash: "b9f9d10b5855e183d6a3fbecda0625cdbbd470a5e801b31de95544b4f8874cbe"
sourceCorrections: []
---

稀疏专家混合模型（MoE）方法提供了一种有组织的条件计算实现方式，使神经网络 (neural network)能够大幅增加参数 (parameter)量，同时不按比例提高每个输入的计算成本。为达成这一目标，MoE架构提供了一种具体的机制，通过只激活网络相关部分来达成这种稀疏性，在大规模Transformer模型中表现尤其明显。

其根本上，MoE层取代了标准组件，例如Transformer中的前馈网络（FFN）模块，由两大部分构成：

1. **专家网络集合**：它们是多个独立的子网络，表示为 $E_1, E_2, ..., E_N$。每个专家通常具有相同的架构（例如，标准FFN），但拥有各自独立的学习参数集。专家总数$N$可以从几个到数百或数千不等。
2. **门控网络（路由）**：这是一个较小、可训练的网络，$G$，它检查每个输入令牌$x$并决定由哪个或哪些专家来处理。路由器的作用举足轻重：它根据令牌的特征动态地进行路由，将它们导向那些推测专门处理此类输入的专家。

稀疏性源于路由器的选择机制。路由器不是将每个令牌发送给所有专家（这会带来高昂的计算成本，类似于一个集成模型），而是通常采用一种稀疏选择策略。最常用的方式是**top-k路由**，对于每个输入令牌，路由器会计算出与$N$个专家中每个专家的亲和性分数。然后它只选择分数最高的$k$个专家来处理该令牌，$k$是一个小的整数，通常是1或2，并且远小于$N$（$k \ll N$）。

> 稀疏MoE层的视图。输入令牌由门控网络处理，门控网络选择并加权少量专家子集（本例中为专家1和专家k，由实心绿线表示）。非活跃专家（虚线灰线）对此令牌未启用。活跃专家的输出组合起来生成最终输出。

对于特定的令牌$x$，只有选定的top-k专家执行计算（即所选集合中$i$的$E_i(x)$）。其余的$N-k$个专家对此令牌不活跃，其处理不增加计算负载（FLOPs）。这使得模型的总参数量（路由器和所有$N$个专家的参数之和）可以非常大，同时每个令牌的计算成本得到控制，仅随$k$和单个专家的大小而变化，而不随专家总数$N$变化。

MoE层对令牌$x$的最终输出$y$通常是激活专家输出的加权组合。权重 (weight)也由门控网络的分数决定。回顾一下一般公式：

$y = \sum_{i=1}^{N} G(x)_i E_i(x)$

在稀疏top-k情境下，$G(x)_i$仅对选定的$k$个专家非零，从而使总和变得稀疏：

$y = \sum_{i \in \text{TopK}(G(x))} G(x)_i E_i(x)$

此处，$\text{TopK}(G(x))$代表路由器$G$为输入$x$选择的$k$个专家的索引，而$G(x)_i$是分配给专家$i$输出的学习权重。

这种稀疏MoE方式提供了一个强大的机制，将模型大小（参数量）与计算成本（每令牌FLOPs）解耦。它使得构建具有万亿参数的模型成为可能，同时在训练和推理 (inference)期间保持可控的计算预算。然而，实现其潜在优势需要应对这种架构固有的特定难题，例如确保专家资源的平衡使用以及路由器训练的稳定动态，这些在后续章节中是核心议题。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇基础性论文介绍了稀疏门控的专家混合层，展示了如何利用条件计算扩展神经网络容量。
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2006.16668](https://doi.org/10.48550/arXiv.2006.16668)
  这项工作介绍了GShard，一个用于将专家混合模型扩展到千亿参数的系统，解决了条件计算和自动分片方面的挑战。
