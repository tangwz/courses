---
course: "fine-tuning-adapting-large-language-models"
chapter: "optimization-deployment-considerations"
lesson: "distributed-training-strategies"
sourceId: 3085
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/distributed-training-strategies"
title: "使用分布式策略加速训练"
description: "概述数据并行、张量并行和流水线并行（例如DeepSpeed、FSDP）。"
order: 2
plots: []
sourceHash: "d3f94a08e01553c326451c409ce8b6fbef2fed591fba750f5ca290d6e4e237ec"
sourceCorrections: []
---

在大型语言模型的微调 (fine-tuning)过程中，有效的内存管理是必不可少的，这通常通过梯度累积和混合精度训练等策略来解决。然而，即使采用这些内存管理方法，计算所需的实际时间仍然是一个很大的挑战，尤其是在处理大型数据集或非常大的模型时。训练一个拥有数十亿参数 (parameter)的模型，即使进行了内存优化，在单个加速器上仍可能需要数天或数周。为大幅缩短此训练时间，可以采用分布式训练策略，将计算工作量分配到多个处理单元，通常是GPU上。

分布式训练过程涉及将工作（无论是数据、模型本身还是计算的不同阶段）划分并分配到设备集群中。目的是并行执行更多计算，从而减少完成微调周期所需的总时间。然而，这也带来了协调这些设备并管理它们之间通信的复杂性。下面我们分析一下用于大型语言模型的主要策略。

### 数据并行

数据并行是可能最直观和常见的分布式策略。其核心思想很简单：在每个可用设备（GPU）上复制整个模型，然后将每个全局训练数据批次分割成更小的微批次。每个设备独立处理其分配的微批次，并计算其本地模型副本的梯度。

重要的一步是同步。在优化器更新模型权重 (weight)之前，所有设备上计算的梯度必须进行聚合（通常是平均）。然后，这个聚合后的梯度用于更新*所有*设备上的模型参数 (parameter)，确保每个副本保持同步并从整个批次中学习。

> 数据并行的一种简化视图。模型 (M) 被复制，数据 (D) 被分割，梯度 (G) 在本地计算，然后聚合，最后更新所有模型副本。

**优点：**

- 使用PyTorch的`DistributedDataParallel` (DDP) 等库实现相对直接。
- 如果通信开销管理得当，可以大幅加快训练速度。

**缺点：**

- 整个模型、梯度和优化器状态必须适应*每个*独立GPU的内存（尽管稍后讨论的ZeRO等方法可以缓解这个问题）。这限制了纯数据并行能够处理的最大模型尺寸。
- 通信瓶颈：随着设备数量的增加，聚合所有设备梯度的`AllReduce`操作可能成为限制，尤其是在没有高速互连（如NVLink）的情况下。

对于模型能适应单个GPU的大多数常见微调 (fine-tuning)场景，PyTorch的`DistributedDataParallel` (DDP) 因其更好的性能和对Python全局解释器锁（GIL）的处理，被推荐优于旧的`DataParallel`方法。

### 张量并行

当模型的层过大，甚至无法适应单个高内存GPU时，单独的数据并行就不够了。张量并行通过在模型层*内部*进行划分来解决这个问题。它不是复制整个模型，而是将层内的各个权重 (weight)矩阵（张量）（如Transformer块中的大型`nn.Linear`层或注意力机制 (attention mechanism)）分割到多个设备上。

考虑一个大型矩阵乘法 $Y = XA$。如果矩阵 $A$ 在两个GPU上按列分割（$A = [A_1, A_2]$），那么通过让GPU 1计算 $Y_1 = XA_1$ 并且GPU 2计算 $Y_2 = XA_2$，就可以计算 $Y = [XA_1, XA_2]$。这需要将输入 $X$ 传输给两个GPU。另外，按行分割 $A$ 则需要在每个GPU上执行部分矩阵乘法，然后对结果求和。

> 张量并行在层内分割计算。这里，一个线性层的权重矩阵（A）按列分割（[A1, A2]）到两个GPU上。输入 (X) 对每个部分进行处理，需要通信来组合结果或根据操作同步中间步骤。

**优点：**

- 使训练的模型能够超越单个设备的内存容量。
- 与数据并行相比，减少了每个GPU所需的权重和激活内存。

**缺点：**

- 需要修改模型实现本身，以正确处理分割的计算和必要的通信（例如，使用Megatron-LM等库）。
- 在分割层的前向和反向传播 (backpropagation)过程中引入了大量通信开销，对GPU之间的高带宽互连要求很高。
- 比数据并行更复杂，难以实现和调试。

### 流水线并行

流水线并行采用了一种不同的模型划分方式。它不是分割单个层（张量并行）或复制整个模型（数据并行），而是垂直划分模型，将连续的层序列（阶段）分配给不同的设备。

数据按顺序流经这些阶段：GPU $i$ 上一个阶段的输出激活成为GPU $i+1$ 上下一个阶段的输入。为了提高设备利用率并减少设备空闲等待的“气泡”，输入批次通常被分割成更小的*微批次*。这些微批次按顺序送入流水线，允许多个设备在不同的微批次上并行计算。

> 流水线并行将顺序层（阶段）分配给不同的GPU。微批次流经各个阶段，支持并发处理，但需要仔细调度以最小化空闲时间（“气泡”）。

**优点：**

- 通过将层内存需求分配到不同设备上，使训练超大型模型成为可能。
- 如果激活管理得当，每个设备的内存效率可能高于张量并行。
- 通信通常是相邻阶段间的点对点通信，这比数据并行中的AllReduce更易于管理。

**缺点：**

- 容易出现流水线气泡：流水线开始或结束的设备可能在等待微批次传播时处于空闲状态。
- 有效的微批次调度（例如GPipe、PipeDream）对于最大化利用率是必要的。
- 需要仔细的负载均衡：各阶段的计算成本必须大致相等以避免瓶颈。
- 在反向传播 (backpropagation)过程中，可能会在管理跨阶段的激活和梯度方面引入复杂性。

### 策略结合与高级框架

在实际应用中，特别是在训练先进的大型语言模型时，这些策略经常被结合使用。一种常见的配置是 **3D并行**：

1. **数据并行：** 在多组设备上复制计算。
2. **流水线并行：** 在一个副本内的设备上分割模型层。
3. **张量并行：** 在一个流水线阶段内的设备上分割单个大型层。

像 **DeepSpeed** 和 **PyTorch Fully Sharded Data Parallelism (FSDP)** 这样的框架提供了精密的实现，它们集成了这些思想并增加了进一步的优化。

- **DeepSpeed：** 提供数据并行、张量并行和流水线并行的实现。其突出的贡献是 \*\*ZeRO（零冗余优化器）\*\*系列方法。ZeRO通过不仅分割数据，还分割优化器状态、梯度以及可选地模型参数 (parameter)本身到数据并行等级，从而增强了数据并行。这大大减少了每个GPU的内存占用，允许在数据并行设置中使用更大的模型或批次大小。虽然它主要是一种内存优化，但ZeRO支持扩展，间接加速了从开始训练到获得结果的时间。
- **PyTorch FSDP：** PyTorch用于大规模训练的本机解决方案，作为一种高级形式的数据并行。与ZeRO阶段3类似，FSDP将模型参数、梯度和优化器状态分片到数据并行工作器中。每个工作器只在当前执行的层中具体化完整的参数。它与PyTorch生态系统良好集成，并且通常比自定义张量或流水线并行需要更少的代码修改。

### 实现考量

选择和实现分布式策略涉及权衡：

- **模型大小与设备内存：** 模型是否对单个GPU来说太大了？如果是，则需要张量并行或流水线并行（或FSDP/ZeRO）。如果不是，数据并行则更简单。
- **硬件互连：** 张量并行要求参与张量分割的GPU之间具有极高的带宽（例如NVLink）。流水线并行和数据并行也对互连速度敏感，但可能不如张量并行那样关键。
- **实现复杂性：** 数据并行（尤其是DDP）是最简单的。FSDP提供了一种平衡。完整的流水线并行和张量并行通常需要大量的代码重构或依赖于专门的库（DeepSpeed、Megatron-LM）。
- **通信开销：** 所有策略都会引入通信。最佳选择取决于平衡节省的计算时间与增加的通信时间。通常需要仔细的性能分析。
- **可扩展性：** 随着更多设备的加入，该策略的表现如何？数据并行中的同步成本和流水线气泡可能会限制可扩展性。

通过分布式策略加速微调 (fine-tuning)对于高效处理大型模型来说是必不可少的。了解数据并行、张量并行和流水线并行，以及DeepSpeed和PyTorch FSDP等现代框架的功能，可以根据您的模型大小、硬件资源和性能目标选择并实现最适合的方法。这通常需要通过实验来找到针对您特定微调任务的最佳配置。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02057) — Samyam Rajbhandari, Cong Guo, Jeff Rasley, Shaden Smith, Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 69-82; DOI: [10.1109/SC41405.2020.00070](https://doi.org/10.1109/SC41405.2020.00070)
  介绍了ZeRO（零冗余优化器）技术系列，一种分布式训练的内存优化方法，通过在数据并行设备间分片模型状态，从而支持训练超大规模模型。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  介绍了大型语言模型的模型并行工作，详细阐述了张量并行和流水线并行策略，以及它们在高效训练数十亿参数模型中的组合应用。
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) — Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen (2019)
  Journal: NeurIPS 2019; DOI: [10.48550/arXiv.1811.06965](https://doi.org/10.48550/arXiv.1811.06965)
  介绍了GPipe，一种高效的流水线并行策略，通过将层分片到不同设备并利用微批次调度来最大化硬件利用率，从而训练超大型神经网络。
- [\`torch.distributed.fsdp\` - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/fsdp.html) — PyTorch Development Team (2022)
  PyTorch 完全分片数据并行（FSDP）的官方文档，说明了其在大型模型训练中通过分片模型参数、梯度和优化器状态来节省内存的使用方法。
