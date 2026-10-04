---
course: "mixture-of-experts-advanced-implementation"
chapter: "foundations-of-moe-models"
lesson: "sparsely-gated-moe-architecture"
sourceId: 7044
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-1-foundations-of-moe-models/sparsely-gated-moe-architecture"
title: "稀疏门控专家混合架构概述"
description: "稀疏门控专家混合架构、其组成部分及其在稀疏建模中的作用的技术综述。"
order: 1
plots: []
sourceHash: "707c6af3640297c0884b05f03b422d5d4ac5a1ee79b2fe135dcbe74c2b8eca6d"
sourceCorrections: []
---

标准稠密神经网络 (neural network)基于完全激活的原理运行。对于每个输入，模型中的每个参数 (parameter)都会参与计算。这种方法非常成功，但随着模型参数量达到数千亿，它会遇到一个显著瓶颈。计算成本（以浮点运算次数FLOPs衡量）与参数数量成正比增长，使得训练和推理 (inference)成本极高。

稀疏门控专家混合（MoE）架构通过采用*条件计算*，提供了一条更高效的扩展路径。其核心思想简单而有效：对于每个输入，我们不再激活整个网络，而是仅激活一小部分相关内容。MoE模型通过维持大量“专家”子网络，并使用一个轻量级“门控网络”来动态选择哪些专家应处理每个输入标记 (token)，从而实现此目标。这使得模型容量（总参数）可以大幅增加，同时保持单次前向传播的计算成本相对恒定。

稀疏门控MoE层的架构由两个主要组成部分构成。

### 专家网络

专家是MoE层的主要工作单元。每个专家本身就是一个神经网络 (neural network)，通常是一个简单的全连接前馈网络（FFN），它学习专注于处理特定类型的信息。例如，在一个大型语言模型中，一个专家可能擅长处理Python代码中的语法，而另一个则可能专注于法语习语。

由于一个输入标记 (token)只被路由到少数几个专家（通常只有一个或两个），模型可以包含大量专家。一个模型可能有8个、64个甚至更多的专家，但对于任何给定标记，只有一小部分被使用。这是这种架构高效的原因。总参数 (parameter)数量是所有专家参数（加上小型的门控网络）的总和，但单次前向传播的活跃参数数量保持较小。

### 门控网络（路由器）

门控网络，也称为路由器，充当交通控制器。它的作用是检查传入的标记 (token)，并决定哪些可用专家最适合处理它。门控网络通常是一个小的线性层，然后是一个`softmax`函数。它接收标记的输入表示 $x$，并生成覆盖所有 $N$ 个专家的概率分布。


$$
\text{对数几率} = W_g \cdot x
$$


$$
\text{门控值} = \text{softmax}(\text{对数几率})
$$


这里，$W_g$ 是门控网络的可学习权重 (weight)矩阵。生成的`门控值`向量 (vector)包含决定选择哪些专家的分数。为了强制稀疏性，对此向量应用一个`top-k`函数。例如，如果 $k=2$，则只选择分数最高的两个专家。所有其他专家的分数都被设置为零，从而有效地禁用了它们对于当前标记的作用。

### 信息流动

单个标记 (token)的完整前向传播涉及门控网络和专家之间的协调流程。标记被发送到门控网络，以确定激活哪些专家。同时，标记被传递给选定的专家进行处理。最后，活跃专家的输出以加权和的形式结合，使用门控分数作为权重 (weight)。

> 标记通过MoE层的流动过程。输入标记被发送到门控网络，门控网络输出所有专家的分数。根据这些分数（例如，top-k），标记被路由到一小部分活跃专家（此处为专家2）。活跃专家的输出被聚合以生成最终输出。不活跃的专家不执行任何计算。

这种稀疏激活是该架构的决定性特征。在Transformer模型中，MoE层通常用于替换自注意力 (self-attention)机制 (attention mechanism)之后出现的稠密前馈网络（FFN）模块。通过这样做，我们可以大幅扩展模型FFN部分的参数 (parameter)数量，而不会导致计算需求相应激增，从而为更强大、更高效的大规模模型提供可能。在接下来的章节中，我们将查看此流程的数学细节、它带来的挑战以及用于解决这些挑战的技术。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇基础性论文介绍了稀疏门控专家混合（MoE）层，详细阐述了其架构，并展示了如何在保持计算效率的同时训练参数量庞大的模型。
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2006.16668](https://doi.org/10.48550/arXiv.2006.16668)
  本文探讨了条件计算（特别是MoE层）在将深度学习模型扩展到数千亿参数方面的应用，并介绍了高效训练和分片的技术。
