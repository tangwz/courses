---
course: "mixture-of-experts-advanced-implementation"
chapter: "training-large-scale-moes"
lesson: "combining-parallelism-strategies"
sourceId: 7060
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-3-training-large-scale-moes/combining-parallelism-strategies"
title: "结合模型并行、数据并行与专家并行"
description: "结合数据并行、模型并行与专家并行以高效训练超大型专家混合模型的策略。"
order: 2
plots: []
sourceHash: "3a64ea832cc6b2ef3813bc9ee0d94ef355b864923582c1481dbc140f4716ce41"
sourceCorrections: []
---

尽管专家并行是扩展专家混合模型参数 (parameter)数量的强大手段，但它很少单独运行。为了训练最先进的稀疏模型，这些模型可以拥有数万亿参数，但仍需在海量数据集上进行训练，因此你需要同时协调多种并行策略。每种策略都解决不同的扩展瓶颈，它们的组合使得训练如此规模的模型成为可能，否则将无法实现。

并行主要有三个维度：

1. **数据并行 (DP)：** 最简单的形式。模型在每个设备上复制，全局数据批次在它们之间分割。它提高了吞吐量 (throughput)，但要求整个模型能放在单个设备上。
2. **张量并行 (TP)：** 一种模型并行形式，其中单个层（如大型权重 (weight)矩阵）在设备间分片。这使得模型的密集组件能够超出单个GPU的内存容量。
3. **专家并行 (EP)：** MoE特有的策略，将不同的专家网络放在不同的设备上。这扩展了专家数量，从而扩展了总参数量。

成功训练大规模MoE模型需要有机结合模型并行、数据并行和专家并行等多种策略，以形成一致且高效的训练配置。

### 二维组合：数据并行与专家并行 (DP+EP)

MoE模型最常见的混合策略是结合数据并行和专家并行。这种方法适用于模型中密集组件（如注意力块和嵌入 (embedding)层）可以放在单个GPU上，但专家总数过大的情况。

在这种配置中，你通常会有一组设备，例如单个服务器节点内的8个GPU。并行方式如下：

- **数据并行：** 模型的非专家层在所有8个GPU上复制。全局批次均匀分割，每个GPU通过这些复制的层处理其自己的数据切片。
- **专家并行：** 每个MoE层中的专家在8个GPU上分片。如果一个MoE层有64个专家，每个GPU将承载 $64 / 8 = 8$ 个专家。

前向传播涉及一个重要的通信步骤。在一个token通过本地密集层后，门控网络决定它应该路由到哪个专家。如果该专家位于另一个GPU上，token的隐藏状态必须通过设备互连发送。这是通过`all-to-all`集合通信实现的，其中每个GPU将其token的子集发送给其他所有GPU，并接收返回的token。在远程专家处理token后，另一个`all-to-all`操作将其发送回其原始设备以继续前向传播。

> 一个使用4个GPU的二维并行配置。模型的密集层为数据并行而复制，而64个专家在GPU间分片以实现专家并行。虚线表示将令牌路由到其指定专家所需的`all-to-all`通信。

这种DP+EP组合通过增加专家数量有效地扩展了模型的参数 (parameter)量，同时通过更大的全局批次大小扩展了训练吞吐量 (throughput)。主要的瓶颈变为`all-to-all`通信，这会饱和设备间的互连带宽。

### 三维组合：数据并行、专家并行与张量并行 (DP+EP+TP)

对于最大的模型，即使是密集组件也变得过大，单个GPU无法承载。这就需要引入第三个维度：张量并行。将这三种策略结合起来，使得在大型多节点GPU集群上训练超大规模模型成为可能。

这种排列的层级结构可以被设想为设备的三维网格：

- **张量并行组：** 一小组GPU（例如，2、4或8个）协同工作以形成一个“虚拟设备”。它们共同存储和计算模型大型密集层的一个实例。该组内的通信频繁，通常对每个张量并行层都需要`all-reduce`或`all-gather`操作。
- **专家并行组：** 专家在这些张量并行组之间分片。例如，在一个由四个各含4个GPU的TP组组成的16个GPU配置中，你可以将专家1-16分配给第一个TP组，专家17-32分配给第二个，以此类推。现在，路由一个token意味着将其从源TP组发送到目标TP组。
- **数据并行组：** 承载张量并行和专家并行模型的整个多节点配置可以复制以处理不同的数据批次。如果你有32个GPU，你可能有两个数据并行副本，每个都是如上所述的16个GPU配置。这需要在数据并行副本之间进行最终的`all-reduce`以平均梯度。

这种三维方法在层级结构的每个级别上都产生不同的通信模式：张量并行的组内通信、专家并行的组间`all-to-all`，以及数据并行的全局`all-reduce`。

> 一个跨8个GPU的三维并行策略。GPU首先被组织成2个GPU的张量并行（TP）组。专家并行将64个专家分片到这些TP组中。最后，整个4个GPU的配置会为数据并行而复制。每种并行类型都包含不同的通信模式。

### 框架与实际实现

手动实现这些组合策略是一项巨大的工程任务。幸运的是，像DeepSpeed、Megatron-LM和JAX这样的分布式训练框架处理了大部分的复杂性。这些库提供了高级API，用于定义并行策略，通常通过指定设备网格中每个维度的大小来完成。

例如，使用JAX与`pjit`，你可能会定义一个三维设备网格：
`mesh = Mesh(devices, ('data', 'expert', 'model'))`

接着，你将使用注解来告知编译器如何沿着这个网格划分模型的权重 (weight)和中间激活。框架的编译器负责将这些注解转换为正确的底层集合通信操作（`all-reduce`、`all-to-all`、`all-gather`）。

选择合适的组合和配置取决于你具体的模型架构和硬件环境。拥有大量专家但密集层相对较小的模型将从更大的专家并行维度中获益最多。反之，拥有巨大密集层的模型可能会优先选择更大的张量并行维度。平衡这些维度以最大化硬件利用率并最小化通信瓶颈，是优化大规模MoE训练的重要组成部分。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; Publisher: arXiv; DOI: [https://doi.org/10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控专家混合层，为大型神经网络的专家并行奠定基础。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2021)
  Journal: arXiv preprint arXiv:2101.03961; DOI: [https://doi.org/10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  详细说明了万亿参数专家混合模型的训练策略，包括数据并行和专家并行组合，以实现超大规模。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2020)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [https://doi.org/10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  阐述了Megatron-LM框架中用于训练大型语言模型的模型并行（张量并行）和数据并行实现。
