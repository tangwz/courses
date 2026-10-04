---
course: "mixture-of-experts"
chapter: "moe-foundations-sparse-models"
lesson: "dense-vs-sparse-activation"
sourceId: 4683
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-1-moe-foundations-sparse-models/dense-vs-sparse-activation"
title: "对比密集激活与稀疏激活"
description: "比较密集模型与稀疏MoE模型的计算和参数效率。"
order: 3
plots: ["plots/4683-0.json"]
sourceHash: "a741587459666f951227737ba7fd0e1870bd9c82129e68f9ab6a9a7905454b41"
sourceCorrections: []
---

传统密集前馈网络或Transformer模块让每个输入令牌与该层内的*所有*参数 (parameter)进行交互。当输入张量$x$通过一个密集层时，计算涉及将$x$乘以该层的完整权重 (weight)矩阵$W$。后续层重复此过程。这种方法保证了网络的各个部分都参与处理每一条信息。

以Transformer中的标准前馈网络（FFN）模块为例：
$\text{前馈网络}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$
其中，$W_1$、$b_1$、$W_2$和$b_2$代表权重和偏置 (bias)。对于每个输入令牌$x$，所有这些参数都参与计算。计算成本（通常以浮点运算数（FLOPs）衡量）与这些权重矩阵的大小直接相关。如果我们增加FFN的隐藏维度或中间大小，参数数量和每个令牌的FLOPs都会大幅增加。

稀疏激活，在专家混合模型中使用，从根本上改变了这种动态。不再激活整个层，而是只使用参数的选定子集，即“专家”，来处理任何给定的输入令牌。回顾前面介绍的基本MoE公式：
$y = \sum_{i=1}^{N} G(x)_i E_i(x)$
通常，门控网络$G(x)$会从一个更大的$N$个专家池（例如$N=64$）中选择少量$k$个专家（例如$k=2$）。这意味着对于单个令牌，计算成本主要由选定的$k$个专家和门控网络的大小决定，而*不是*专家总数$N$。

让我们详细比较：

### 计算成本（FLOPs）

- **密集模型：** 每个令牌的FLOPs与密集层参数 (parameter)的大小成比例。通过增加层宽度来扩展模型容量会直接增加每个令牌的计算成本。对于输入/输出维度为$d_{model}$、中间维度为$d_{ff}$的FFN，其FLOPs大约是$2 \times d_{model} \times d_{ff}$。
- **稀疏MoE模型：** 每个令牌的FLOPs主要由选定的$k$个专家内部的计算决定。如果每个专家$E_i$的计算成本与一个较小的密集FFN大致相同，那么每个令牌的总计算成本大约是单个专家成本的$k$倍，再加上门控网络$G(x)$的少量开销。值得注意的是，增加专家总数$N$（例如从16到64）并*不会*增加每个令牌的FLOPs，前提是$k$保持不变。这种解耦使得MoE模型能够在相同的每个令牌计算预算下，比密集模型更大幅度地增加其参数数量和潜在的表示能力。



![FLOPs与总参数量对比](plots/4683-0.json)



> 该对比图显示了每个令牌的FLOPs如何随模型总参数量变化。密集模型呈现耦合增长，而稀疏MoE模型允许参数扩展，同时每个令牌的FLOPs几乎保持不变（假设k固定）。注意MoE的轻微增长是由于门控开销，并假设专家也扩展参数。坐标轴代表相对比例。

### 参数 (parameter)数量

- **密集模型：** 参数数量与计算成本紧密相关。增加一个通常会使另一个按比例增加。一个拥有千亿（100B）参数的密集模型，每次前向传播都需要处理千亿参数的计算资源。
- **稀疏MoE模型：** 参数数量与每个令牌的FLOPs解耦。一个MoE模型可能拥有万亿（1 Trillion）总参数，分布在$N$个专家中，但如果$k=2$，单个令牌的计算量可能类似于一个规模小得多的密集模型（大约是单个专家大小的两倍）。这是MoE模型在类似计算限制下，超越密集架构的可行范围，扩展模型容量的主要吸引力。

### 内存使用

- **密集模型：** 推理 (inference)时，通常需要将模型的全部参数 (parameter)加载到加速器内存（例如GPU显存 (VRAM)）中。内存需求与参数数量直接相关。
- **稀疏MoE模型：** 尽管每个令牌在计算上只激活$k$个专家，但在实际实现中，尤其是在训练期间，通常需要分布式系统中所有$N$个专家参数都可访问。这是因为批次中不同的令牌（可能在不同设备上处理）可能需要不同的专家。因此，MoE模型的*总*参数内存占用可能大得多于具有相同每令牌FLOPs的密集模型。这需要复杂的分布式训练策略（例如在第4章讨论的专家并行）来有效管理内存。训练期间的激活内存也需要仔细管理，尽管其模式与密集模型不同。

### 流程

激活模式的差异可以简单地可视化：

> 数据流对比。在密集层中，输入会激活所有参数 (parameter)。在稀疏MoE层中，门控网络将输入路由到专家的一个子集（例如专家1和专家3），只激活它们的参数。未激活的专家（例如专家2、专家4）以灰色显示。

### 影响与权衡

稀疏激活策略为构建拥有海量参数 (parameter)的模型提供了一条富有前景的道路，同时能保持每个令牌的计算成本可控。这使得模型可能具有更大的容量和专业化能力。然而，这种优势也伴随着固有的复杂性：

1. **训练挑战：** 确保所有专家都被有效利用（负载均衡）并学习不同的功能需要专门的技术，例如辅助损失函数 (loss function)（第3章）。
2. **通信开销：** 在分布式环境中，将令牌路由到位于不同设备上的正确专家会引入可观的通信成本（All-to-All通信），需要进行优化（第4章）。
3. **推理 (inference)复杂性：** 在推理过程中高效处理动态路由和批处理需要仔细的实现和针对硬件的优化（第5章）。

了解密集激活和稀疏激活之间的这种基本对比非常重要。它驱动了为专家混合模型专门开发的架构选择、训练方法和扩展策略，我们将在后续章节中讨论这些。虽然密集模型在实现和训练动态方面提供了简便性，但稀疏模型提供了一条在可行计算预算内大幅扩大模型规模的途径，尽管系统复杂性有所增加。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008
  介绍了Transformer架构，为与稀疏激活模型对比的密集前馈网络奠定了基础。
- [Sparsely-Gated Mixture-of-Experts Layers](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  提出了深度学习中专家混合层（MoE）的现代公式，详细说明了稀疏激活及其在不增加每token计算量的情况下扩展参数数量的优点。
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://doi.org/10.48550/arXiv.2006.16668) — Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen (2020)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2006.16668](https://doi.org/10.48550/arXiv.2006.16668)
  探讨了大规模专家混合模型训练的实践方面，包括分布式训练策略、内存管理和通信开销。
