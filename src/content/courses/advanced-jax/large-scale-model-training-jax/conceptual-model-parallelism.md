---
course: "advanced-jax"
chapter: "large-scale-model-training-jax"
lesson: "conceptual-model-parallelism"
sourceId: 2944
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-6-large-scale-model-training-jax/conceptual-model-parallelism"
title: "模型并行策略"
description: "探讨在 JAX 环境下将大型模型拆分到多个设备的方法（张量并行、流水线并行）。"
order: 8
plots: []
sourceHash: "5c0801946c4e740022a00ada2d8be17af537b149cdc7ab0f5072da255a448a72"
sourceCorrections: []
---

尽管使用 `pmap` 的数据并行，结合梯度累积和检查点等技术，能够扩展到更大的数据集和有效的批处理大小，但当模型的参数 (parameter)和中间激活变得过大，以至于无法放入单个加速器设备的内存中时，一个显著的限制便会显现。即使使用混合精度，顶尖模型自身的庞大尺寸也可能超出现有最大 GPU 或 TPU 的容量。

当模型的单个副本无法在一台设备上运行时，您需要将*模型本身*划分到多个设备上。这是**模型并行**背后的主要思想。与数据并行不同，数据并行是在不同的数据片上运行*相同*的模型，而模型并行涉及拆分模型的组件（层甚至层的一部分）并将它们分配给不同的加速器。数据在流经划分后的模型时在这些设备之间流动。

实现模型并行有两种主要策略：

### 张量并行（层内并行）

张量并行侧重于将*单一层内*（或操作）的计算拆分到多个设备上。这对于具有巨大权重 (weight)矩阵的层（例如 Transformers 中常见的大型线性层）尤其有效。

考虑一个大型矩阵乘法，它是神经网络 (neural network)中的一个基本操作：$Y = XW$。如果权重矩阵 $W$ 对于单个设备来说过大，我们可以将其划分到多个设备上。例如，$W$ 可以按列拆分到两台设备上，$W = [W_1, W_2]$。输入 $X$ 被发送到这两台设备。每台设备计算一部分输出：设备 1 上计算 $Y_1 = XW_1$，设备 2 上计算 $Y_2 = XW_2$。最终结果 $Y$ 通过连接部分结果得到：$Y = [Y_1, Y_2]$。

> 张量并行应用于矩阵乘法 $Y = XW$ 的简化视图。权重矩阵 $W$ 被拆分为 $W_1$ 和 $W_2$，计算 $XW_1$ 和 $XW_2$ 在不同的设备上进行。输入 $X$ 被复制，部分结果 $Y_1$ 和 $Y_2$ 被收集。

或者，$W$ 可以按行拆分，这需要通信（例如，一次 all-reduce 操作）来求和部分结果。更复杂的操作，例如注意力机制 (attention mechanism)，也可以通过这种方式并行化。

**难点：**

- **通信开销：** 张量并行通常需要在单一层的前向和反向传播 (backpropagation)中进行频繁通信（例如，all-gather、reduce-scatter、all-reduce）。如果管理不当，这种高带宽通信可能会成为瓶颈，尤其是在较慢的互联网络上。
- **实现复杂性：** 正确实现张量并行需要仔细处理张量分片、通信原语以及跨分片的梯度传播。

在 JAX 中，张量并行通常涉及在分配给并行层的设备子集上使用 `pmap`，并常与应用于 `pmap` 中定义的特定 `axis_name` 维度的 `jax.lax.psum` 或 `jax.lax.all_gather` 等集合通信原语结合使用。基于 JAX 构建的库可能会提供常见张量并行模式的抽象。

### 流水线并行（层间并行）

流水线并行采用不同的方法：它在层*之间*划分模型，将顺序的阶段或层块分配给不同的设备。一个阶段（在一台设备上）的输出激活成为下一个阶段（在另一台设备上）的输入。

设想一个具有四层（L1, L2, L3, L4）的模型，分布在四台设备（D1, D2, D3, D4）上。

- D1 运行 L1。
- D2 运行 L2。
- D3 运行 L3。
- D4 运行 L4。

一个简单的实现会顺序处理一个批次：D1 计算 L1，将激活发送到 D2；D2 计算 L2，发送到 D3；D3 计算 L3，发送到 D4；D4 计算 L4。在此过程中，任何时候只有一台设备处于活动状态，导致大量空闲时间，通常称为“流水线气泡”。

> 简单的流水线并行：层被分配给顺序设备。激活在设备之间传递。

为了提高效率并减少气泡，使用了**微批处理**。输入批次被拆分为更小的微批次。一旦设备 1 完成处理第一个微批次，它就会将激活发送到设备 2，并立即开始处理第二个微批次。这允许多台设备同时处理不同的微批次，从而填充流水线。

> 带有微批处理（MB1、MB2 等）的流水线并行。设备同时处理不同的微批次，减少空闲时间（由“-”表示）。初始和最终的气泡仍然存在，但会被许多微批次分摊。

**难点：**

- **流水线气泡：** 即使使用微批处理，在处理完整批次的开始和结束时，一些空闲时间仍然无法避免。
- **负载均衡：** 确保每个阶段花费大致相同的时间对于效率很重要。不均衡的阶段会导致瓶颈。
- **复杂的调度：** 管理微批次、激活和梯度（在反向传播 (backpropagation)期间）的流动需要仔细的调度逻辑。

在 JAX 中，流水线并行通常涉及使用 `jax.device_put` 或后端特定的放置机制，手动将计算的不同部分放置到特定设备上。阶段之间的通信通常使用点对点传输，这可能会由库抽象或根据设置（单主机与多主机）需要更低级别的原语。管理微批处理和调度通常由用户或更高级别的框架完成。

### 混合方法和 JAX 考量

实践中，复杂的大规模训练设置通常结合这些策略：

- **数据并行 + 流水线并行：** 将流水线复制到多组设备上，每个副本处理不同的数据片。
- **数据并行 + 张量并行：** 在每个层内使用张量并行，并使用数据并行在设备之间复制这个张量并行模型。
- **完全分片数据并行 (FSDP)：** 一种更一体化的方法（常见于 PyTorch FSDP 等库中，JAX 生态系统中也正在出现类似的功能），它在数据并行工作器之间分片参数 (parameter)、梯度和优化器状态，仅在计算需要时执行 all-gather 操作。这可以看作是一种结合，其中参数被分片（如模型并行），但应用于数据并行环境中。

直接在 JAX 中仅使用 `pmap`、`lax.p*` 集合操作和设备放置等原语实现这些高级模型并行策略是可行的，但要求很高。它需要透彻理解设备拓扑、通信模式和精心编排。因此，实践者通常依赖 JAX 生态系统中更高级别的库（例如 Flax 的扩展或为大型模型构建的专用库），这些库为常见的张量和流水线并行模式提供抽象，使其更易于应用。

尽管本节提供了一个概述，但实际实现高效模型并行通常涉及使用这些专用库，或投入大量工程精力来构建针对特定模型架构和硬件设置的定制解决方案。然而，理解这些策略对于在仅数据并行不足时选择正确的方法和工具是必要的。

## 参考资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  介绍专门用于训练Transformer语言模型的张量并行和流水线并行技术。
- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://dl.acm.org/doi/10.1145/3433701.3433707) — Samyam Rajbhandari, Cong Guo, Erwan Hallou, Anshul Gupta, Olatunji Ruwase, Jeff Dean (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; Pages: 1-16; DOI: [10.1145/3433701.3433707](https://doi.org/10.1145/3433701.3433707)
  介绍了ZeRO优化器，这是分片参数、梯度和优化器状态的基础工作，对FSDP有重要影响。
- [Distributed arrays and automatic parallelization in JAX](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) — JAX Authors (2024)
  JAX官方文档，解释了使用`pmap`和集体通信原语进行分布式计算和自动并行化的基本机制。
