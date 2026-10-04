---
course: "how-to-build-a-large-language-model"
chapter: "introduction-large-scale-language-modeling"
lesson: "computational-challenges-overview"
sourceId: 5950
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-1-introduction-large-scale-language-modeling/computational-challenges-overview"
title: "计算方面的挑战概览"
description: "概述LLM开发期间面临的主要计算和内存限制。"
order: 4
plots: []
sourceHash: "e4a941e4903d33e23eb0abffe01173121c9bdbbced44a9fbc5232e5f002a95df"
sourceCorrections: []
---

使用数十亿甚至数万亿参数 (parameter)的模型和以TB为单位的数据集进行构建，会带来艰巨的计算难题。这些模型的规模与其能力直接相关，但这种规模在计算资源方面会带来高昂的成本。了解这些难题对理解LLM开发所需的工程工作非常重要。主要的限制集中在两个基本资源上：内存和计算能力。

### 内存限制：模型及其状态的存储

训练大型神经网络 (neural network)需要将多个组件存储在硬件加速器（通常是GPU或TPU）的内存中。对于LLM，这些组件的庞大大小常常超出单个设备的内存容量。我们来分析一下主要的内存占用者：

1. **模型参数 (parameter)：** 这些是模型在训练过程中学习到的权重 (weight)和偏置 (bias)（$W$）。对于大型模型，这通常是最显而易见的内存需求。如果一个模型有$N$个参数并使用32位浮点精度（FP32），它仅存储权重就需要$N \times 4$字节。例如，一个千亿参数的模型仅参数就需要大约$100 \times 10^9 \times 4 = 400$ GB。即使使用16位精度（FP16或BF16），这仍然达到200 GB，远超典型单个加速器（可能从16GB到80GB）的内存。
2. **梯度：** 在反向传播 (backpropagation)过程中，我们计算损失函数 (loss function)相对于每个参数的梯度（$\nabla W$）。这些梯度通常具有与参数本身相同的维度和数据类型。因此，存储梯度需要额外的$N \times 4$字节（对于FP32）或$N \times 2$字节（对于FP16/BF16）。
3. **优化器状态：** 现代优化器如Adam或AdamW为每个参数维护状态信息，以调整学习率。例如，Adam为每个参数存储一阶矩（动量）和二阶矩（方差）的估算值。如果这些矩以FP32存储，它们各自需要额外的$N \times 4$字节。这意味着优化器状态可以轻易地使仅参数和梯度所需的内存翻倍或增至三倍（参数$1 \times N$，梯度$1 \times N$，Adam状态$2 \times N$ = FP32中与参数相关的总内存为$4 \times N$）。
4. **激活值：** 也许是内存占用量最大的，尤其是在批处理大小和序列长度较大的情况下，是前向传播过程中产生的中间激活值。为一层计算的每个激活值通常都需要存储，直到在反向传播过程中用于计算梯度。激活值（$A$）所需的内存大致与批处理大小（$b$）、序列长度（$s$）、隐藏维度（$h$）和层数（$L$）成比例。特别是对于Transformer，注意力机制 (attention mechanism)的中间结果（例如随序列长度呈平方关系增长的注意力分数矩阵，$O(s^2)$）可能特别占用内存。这种内存占用量可能显著超过参数所需的内存，尤其是在训练期间。通常采用激活值检查点（或梯度检查点）等技术，通过在反向传播时重新计算激活值而非全部存储，从而以计算换内存。

这些组件的组合决定了每个加速器所需的总内存。

> LLM训练期间单个加速器内部典型内存占用者的细分说明。激活值和优化器状态所需的内存量常常远超模型参数本身。

由于所需的总内存常常超出单个设备所能提供的，因此将模型及其相关状态分散到多个加速器上的分布式训练策略变得很有必要。

### 计算需求：浮点运算预算

训练LLM是计算密集型任务，需要庞大的浮点运算（FLOPs）。

- **训练所需的浮点运算：** 大部分计算发生在Transformer的前馈网络和自注意力 (self-attention)机制 (attention mechanism)中的矩阵乘法。经验研究，常被称为缩放定律（例如，Kaplan et al., 2020; Hoffmann et al., 2022），提供了计算成本的估算。一个常见的近似值是，在包含$D$个token的数据集上训练一个$N$参数 (parameter)的模型，大约需要$C \approx 6 \times N \times D$ FLOPs。对于在数万亿个token上训练的千亿参数模型，这会导致计算预算以ZettaFLOPs ($10^{21}$ FLOPs) 范围或更高来衡量。完成此类计算需要大型加速器集群运行数周或数月。
- **推理 (inference)所需的浮点运算：** 尽管生成单个token比训练快得多，但自回归 (autoregressive)生成（一个接一个地生成token）仍然需要可观的计算量。对于一个$N$参数的模型，生成一个token大约需要$2 \times N$ FLOPs（忽略位置编码 (positional encoding)、层归一化 (normalization)等）。逐个token地生成长序列可能会对延迟敏感。此外，自注意力机制对于长度为$s$且隐藏维度为$h$的序列，每层具有$O(s^2 \cdot h)$的计算复杂度。虽然像键值（KV）缓存这样的技术可以减少推理时过去token的冗余计算，但初始提示的处理仍然涉及相对于提示长度的二次方成本。

### 分布式环境中的通信开销

随着内存和计算需求使得训练必须跨多个加速器（通常是数百或数千个）进行，这些设备之间的通信成为一个决定性因素。

- **数据并行：** 需要跨设备同步梯度或模型参数 (parameter)，通常使用AllReduce操作。
- **模型并行（张量/流水线）：** 需要在负责模型不同部分或计算阶段的设备之间传输激活值和梯度。

互连（例如GPU之间的NVLink或节点之间的InfiniBand/以太网）的效率显著影响整体训练吞吐量 (throughput)。缓慢的通信可能导致加速器在等待数据时空闲，造成限制扩展的瓶颈。

这些计算难题、内存容量限制、庞大的浮点运算需求以及通信开销，促使对专用硬件、软件框架（如支持分布式的PyTorch、DeepSpeed、Megatron-LM）和并行化策略的需求，我们将在本课程中详细研究这些内容。克服这些障碍是构建大型语言模型工程实践的核心部分。

## 参考资料

- [Scaling laws for neural language models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  这项基础研究提出了经验性扩展定律，关联了模型大小、数据集大小和训练计算量与模型性能。
- [Training compute-optimal large language models](https://arxiv.org/abs/2203.15556) — Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (2022)
  Journal: arXiv preprint arXiv:2203.15556; DOI: [10.48550/arXiv.2203.15556](https://doi.org/10.48550/arXiv.2203.15556)
  这篇论文改进了扩展定律，展示了如何在模型大小和数据大小之间优化计算分配，以训练大型语言模型。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，该架构是大多数大型语言模型的基础，解释了注意力机制的计算和内存特性。
- [ZeRO: Memory Optimizations Toward Training Gigantic Neural Networks for Model Parallelism](https://arxiv.org/abs/2005.09055) — Samyam Rajbhandari, Cong Guo, Erik P. Johnson, Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: Association for Computing Machinery (ACM); Pages: 430-442; DOI: [10.1145/3416790.3429130](https://doi.org/10.1145/3416790.3429130)
  介绍了ZeRO，一项显著降低大型模型训练内存消耗的内存优化技术，通过在数据并行设备上划分优化器状态。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  描述了Megatron-LM，一个早期用于训练超大型语言模型的系统，它利用张量并行和流水线并行等多种并行技术来处理计算和内存限制。
