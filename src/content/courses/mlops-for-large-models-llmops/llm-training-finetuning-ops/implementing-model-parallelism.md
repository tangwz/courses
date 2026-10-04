---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "implementing-model-parallelism"
sourceId: 3142
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/implementing-model-parallelism"
title: "实现模型并行策略"
description: "通过张量并行和流水线模型并行，将大型模型放入内存中。"
order: 3
plots: []
sourceHash: "2ba48f2d9402390f618122cb07ae7f6e324f7d3fe7c4345737ff734f6b28a4b6"
sourceCorrections: []
---

当大型语言模型的参数 (parameter)和中间激活量变得太大，以至于单个加速器（如GPU或TPU）的内存无法容纳时，仅靠数据并行是不够的。虽然数据并行在多个设备上复制模型，并在每个设备上处理不同的数据批次，但整个模型仍然必须存在于*每个*设备上。我们需要将模型本身拆分到不同设备上的策略。这就是模型并行要解决的问题。

模型并行将模型的层或张量划分到多个加速器上。与复制模型不同，模型的不同部分驻留在不同的设备上，这使得集群的整体内存能够容纳比单个设备所能处理的模型大数倍的模型。实现模型并行有两种主要策略：流水线并行和张量并行。

### 流水线并行

流水线并行涉及将模型的层按顺序划分到多个设备上。可以将其想象成一条装配线：每个设备（或设备组）形成一个“阶段”，负责执行模型层的特定子集。

假设一个具有12层的模型分布在3个GPU上：

- GPU 0 计算第1-4层。
- GPU 1 使用GPU 0的输出计算第5-8层。
- GPU 2 使用GPU 1的输出计算第9-12层。

输入数据按顺序流经这些阶段。GPU 0处理第一个批次，将其输出（激活）发送到GPU 1，GPU 1处理后再将其输出发送到GPU 2，依此类推。对于反向传播 (backpropagation)，梯度则沿相反方向流动。

> 流水线并行的一个简化视图，显示了跨GPU的顺序处理。

朴素流水线并行的一个重大挑战是“流水线气泡”。当GPU 1等待GPU 0的第一个批次的输出时，以及GPU 2等待GPU 1时，GPU 1和GPU 2最初是空闲的。类似地，当GPU 0处理完其批次的最后部分后，它也会变得空闲，而后续阶段则在完成计算。这种空闲时间表示计算资源的浪费。

为了缓解这个问题，会采用**微批处理**等技术（由GPipe推广并由PipeDream等系统改进）。主要数据批次被分成更小的微批次。一旦GPU 0完成处理第一个微批次，它就会将激活值发送到GPU 1，并立即开始处理第二个微批次。这使得多个微批次可以在流水线阶段同时“运行”，从而显著减少空闲时间并提升硬件利用率。

> 时间线显示微批次（MB）流经一个3阶段流水线。请注意，当GPU 1处理微批次1时，GPU 0如何开始处理微批次2，与一次处理整个批次相比，这减少了空闲时间。反向传播（Bwd）紧随前向传播（Fwd）之后。

流水线并行可以有效减少每个GPU的内存占用，因为每个GPU只容纳模型层的一部分。但是，它会在阶段之间引入通信延迟，并且需要仔细的负载均衡以确保各个阶段的计算成本大致相等。

### 张量并行

流水线并行是在设备*之间*拆分层，而张量并行则是在一个层*内部*（特别是其大型权重 (weight)矩阵）将计算拆分到多个设备上。这对于Transformer模型尤其相关，因为多头注意力 (multi-head attention)层和前馈网络（FFN）层包含大型矩阵乘法。

考虑Transformer层中的大型矩阵乘法 $Y = XA$。如果矩阵 $A$（代表模型权重）对单个GPU的内存来说太大了，我们可以将其按列拆分到两个GPU上： $A = [A_1, A_2]$。计算变为：


$$
Y = X [A_1, A_2] = [XA_1, XA_2]
$$


GPU 0计算 $Y_1 = XA_1$，使用其部分的权重 $A_1$，GPU 1计算 $Y_2 = XA_2$，使用 $A_2$。输入 $X$ 通常会被广播或提供给两个GPU。然后，如果后续操作需要，可以收集结果 $Y_1$ 和 $Y_2$。

或者，矩阵 $A$ 可以按行拆分。考虑 $Y = XA$。将A按行拆分 $A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$。那么：


$$
Y = X \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}
$$


这在操作可能是 $Y = \text{GeLU}(XA)B$ 的前馈层中很常见。这里，$A$ 可以按列拆分为 $[A_1, A_2]$，而 $B$ 可以按行拆分为 $\begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$。
GPU 0计算 $X A_1$，GPU 1计算 $X A_2$。然后，在应用GeLU之前可能需要一次all-gather操作，或者计算可以这样进行：
GPU 0计算 $Y_1 = \text{GeLU}(XA_1)B_1$
GPU 1计算 $Y_2 = \text{GeLU}(XA_2)B_2$
最后，一次all-reduce操作对结果求和： $Y = Y_1 + Y_2$。

> 张量并行将矩阵乘法 $XA$ 拆分到两个GPU上的流程。需要通信（收集/归约）来组合部分结果。

张量并行要求层内计算*内部*进行大量通信，通常使用集体通信操作，例如`all-gather`或`reduce-scatter`。这要求参与的GPU之间具有高带宽互连（例如NVLink）。像NVIDIA的Megatron-LM这样的框架专门设计用于高效地实现Transformer模型的张量并行。

与张量并行有时一同讨论的另一个方面是**序列并行**。使用张量并行时，激活值通常需要在设备之间进行收集，这对于长序列来说可能是内存密集型的。序列并行提供了沿序列长度维度拆分激活值的策略，将此内存负担分配给张量并行设备。这进一步有助于扩展到更长的上下文 (context)长度。

### 组合并行策略

在实践中，训练大型模型很少只依赖单一并行技术。通常通过组合多种策略来达到最佳性能：

- **数据并行：** 跨多个节点或GPU组使用，以同时处理更多数据。
- **流水线并行：** 在节点内或跨节点使用，以划分模型的众多层，减少每个GPU的内存负载。
- **张量并行：** 在流水线阶段内部（节点内跨GPU）使用，用于处理经过流水线划分后仍对单个GPU内存来说太大的独立层。

> 一种混合并行策略，结合了跨节点的数据并行、节点内跨阶段的流水线并行，以及阶段内的张量并行（TP）。

### 实现考量

手动实现这些模型并行策略既复杂又容易出错。它需要仔细处理分布式设备之间的数据移动、通信同步和梯度计算。

- **框架很重要：** 像DeepSpeed（它提供灵活的流水线和张量并行实现，以及ZeRO等内存优化）、Megatron-LM（为Transformer模型率先实现了高效张量并行）、Colossal-AI和PyTorch的完全分片数据并行（FSDP）（可以配置模型分片，类似于ZeRO Stage 3）等库抽象了大部分复杂性。它们提供了定义并行策略的API，通常会自动处理通信和同步。
- **通信开销：** 张量并行严重依赖节点内带宽（如NVLink），而如果流水线阶段跨越节点，流水线并行的性能对节点间带宽很敏感。策略的选择与硬件拓扑结构紧密关联。
- **调试：** 调试分布式训练任务，特别是涉及模型并行的任务，比单设备训练困难得多。问题可能源于通信死锁、错误的数据分散或收集，或设备之间的数值差异。日志记录和专用调试工具是必需的。
- **负载均衡：** 在流水线并行中，确保每个阶段具有相似的计算工作量对于最小化气泡很重要。不均匀的阶段会导致瓶颈。自动化或基于配置文件引导的层分配会有帮助。

掌握模型并行涉及理解这些权衡并利用合适的框架来有效分配计算和内存。这是在LLM训练中扩展模型规模的基本操作要求。

## 参考资料

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://neurips.cc/paper/2019/hash/0c9aee5d14e3e48e027d11f8e6c02f0b-Abstract.html) — Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, Zhifeng Chen (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 32
  介绍流水线并行和微批处理技术，以提高大型模型训练中的硬件利用率。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  介绍了用于训练超大型Transformer模型的有效张量并行策略。
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://ieeexplore.ieee.org/document/9355153) — Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 1-16; DOI: [10.1109/SC41405.2020.00024](https://doi.org/10.1109/SC41405.2020.00024)
  阐述了ZeRO内存优化策略，对于模型扩展至关重要，常与DeepSpeed中的模型并行结合使用。
- [PyTorch FSDP: Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html) — PyTorch Documentation (2022)
  Publisher: PyTorch Foundation
  PyTorch官方文档，详细介绍了在PyTorch中实现和使用完全分片数据并行（FSDP）进行大型模型训练的方法。
