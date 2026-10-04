---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-graph-level-optimizations"
lesson: "aggressive-operator-fusion"
sourceId: 4473
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations/aggressive-operator-fusion"
title: "激进的算子融合技术"
description: "审视超越简单垂直/水平融合的高级融合策略，并考量内存和计算成本。"
order: 2
plots: []
sourceHash: "af2221fa6d53d20cffadb5e25f55ba48c362d2f331a585d2a2afd2e9f9b37db6"
sourceCorrections: []
---

算子融合是一种图优化的基本方法，主要目的是减少核函数启动和中间内存传输的开销。简单的融合策略通常结合垂直相邻的逐元素操作（如 `Add` 后的 `ReLU`），有时也结合写入相同输出缓冲区的水平并行操作。然而，为了实现最佳性能，特别是在计算密度高但内存带宽通常有限的现代加速器上，需要更精细的融合技术。这些技术超越了简单的模式，依赖于复杂的分析和成本建模。

激进的融合致力于合并操作，即使其模式不是简单的线性链或纯粹的逐元素操作。其核心思想是最大化每次核函数启动完成的工作量，并最小化主内存和计算单元之间的数据移动。

### 逐元素操作的融合

显著的性能提升来自于将计算密集型操作（如卷积或矩阵乘法）与随后的逐元素操作进行融合。考虑一个常见模式：`Conv -> BiasAdd -> ReLU`。

> 将计算密集型操作（Conv2D）与随后的逐元素操作（BiasAdd, ReLU）融合到一个核函数中。中间张量 `feature_map` 和 `biased_map` 从全局内存中消除。

在未融合的情况下，`Conv2D` 的结果（`feature_map`）被写入内存，然后由 `BiasAdd` 读取。其结果（`biased_map`）被写入内存，然后由 `ReLU` 读取。每一步都涉及大量的内存I/O和单独的核函数调用。将这些融合到一个核函数（`融合: Conv2D_BiasAdd_ReLU`）中，允许中间结果可能保留在寄存器或缓存中（如GPU共享内存或CPU L1/L2缓存），大幅减少内存带宽消耗和延迟。逐元素操作（BiasAdd, ReLU）的计算成本通常可忽略不计，与其融合时节省的内存访问时间相比。这种将计算密集型生产者与逐元素消费者结合的融合类型，非常有效。

### 多输出和复杂拓扑

激进的融合不限于线性链。它可以处理一个操作的输出被多个后续操作消费的情况。

> 将节点A与多个消费者（B，C）融合。中间结果计算一次后，在同一个核函数内立即被融合逻辑用于B和C，避免将其写入内存。

这里，`操作A` 产生一个中间结果，该结果被 `操作B` 和 `操作C` 使用。如果 `B` 和 `C` 是合适的候选（例如，逐元素操作），它们可以与 `A` 融合。融合后的核函数会计算 `A` 的结果，然后立即使用该结果计算 `B` 和 `C`，可能将中间数据保留在寄存器或片上缓存中。这避免了将中间张量写回内存，然后再读取两次（B一次，C一次）。这种模式有效地结合了垂直融合（A与B，A与C），并隐式处理了常见生产者情况。

### 成本建模：融合的仲裁者

决定是否激进地融合操作需要一个**成本模型**。盲目融合所有操作可能有害。例如，融合过多计算密集型操作可能创建一个需要比可用寄存器更多寄存器的核函数（导致寄存器溢出到局部或全局内存，这会很慢），或者如果组合的核函数代码变得过大和复杂，可能导致指令缓存未命中。

一个实用的成本模型根据几个因素估算潜在融合的性能影响：

1. **核函数启动开销：** 每次核函数启动都有一个与运行时系统调度工作相关的固定成本（CPU调度、GPU命令缓冲区提交、上下文 (context)切换）。融合直接减少了启动次数。这对于许多小型操作尤其重要。
2. **内存流量：** 估算因融合而消除的中间张量从主内存读取和写入的数据量。令 $T_{intermediate}$ 为操作Op1产生并由Op2消费的张量。融合节省了 $T_{intermediate}$ 的一次写入和一次读取。近似节省的成本可以建模为 $2 \times \text{大小}(T_{intermediate}) / \text{有效内存带宽}$。这通常是最重要的节省。
3. **计算成本：** 估算单个操作和潜在融合核函数的执行时间。通常，与内存节省相比，融合的逐元素操作的计算成本微不足道。然而，融合复杂操作需要仔细考虑组合的计算工作负载。
4. **局部性和缓存效应：** 融合操作为中间数据表现出更好的时间局部性，提升了缓存（CPU上的L1/L2，GPU上的共享内存/L1/L2）的利用率。虽然难以精确建模，但这种好处通过减少内存流量和潜在的融合核函数内更快的执行被隐式体现。
5. **资源约束：** 估算潜在融合核函数的资源需求，例如寄存器使用量、共享内存使用量（在GPU上）以及潜在的指令缓存占用。如果估算的需求超出目标硬件的限制，则通常不允许融合。

编译器使用此模型，通常针对特定硬件目标进行参数 (parameter)化，以评估潜在的融合候选（操作对或组）。仅当融合配置的估算成本（例如，执行时间）预计低于未融合配置的成本时才执行融合。

$\text{预测成本}(\text{融合操作}) < \sum_{i \in \text{操作}} \text{预测成本}(\text{操作}_i) + \sum_{j \in \text{中间结果}} \text{预测成本}(\text{内存I/O}_j) + N_{\text{节省}} \times \text{预测成本}(\text{启动})$

不同的硬件目标需要不同的成本模型和启发式方法。内存带宽受限的GPU可能会更积极地倾向于融合，以减少片外内存访问，而拥有大容量、快速缓存的CPU可能会以不同方式优先考虑融合，也许更侧重于指令流水线利用率。

### 带重计算的融合

在某些情况下，将一个操作与其一个消费者融合可能是有益的，即使另一个消费者也需要原始输出。编译器可能选择为第二个消费者在单独的融合核函数中*重新计算*中间值，而不是强制将中间结果写入内存。如果中间操作的计算成本与主内存往返的成本相比很低，这种做法是可行的。这种策略对于计算开销小但产生大张量的操作特别适用，例如广播或某些逐元素转换。

### 与布局转换的联动

激进的融合通常与数据布局转换（在`memory-aware-layout-transformations`中讨论）密切关联。有时，只有当所涉及张量的布局匹配（例如，都期望NCHW）或作为融合过程的一部分进行转换时，融合才可能或有益。例如，融合的卷积-ReLU核函数可能为了在特定目标上的效率而在内部以NHWC格式操作，即使周围的图使用NCHW，也会在核函数的序言和尾声中纳入布局更改。反之，融合操作可以简化图并消除中间张量，从而*促成*对之前被阻碍的后续操作进行有益的布局转换。

### 挑战与考量

实施有效且激进的融合策略会带来几个工程挑战：

- **编译器复杂性：** 需要复杂的图遍历算法、模式匹配能力以及准确的、硬件感知的成本建模。对于复杂图，最佳融合决策的搜索空间会变得非常庞大。
- **代码生成：** 为复杂融合核函数生成高性能代码并非易事。它通常需要高级代码生成技术，能够高效地将融合逻辑映射到目标硬件的执行模型（例如，管理CPU/GPU目标的并行性、向量 (vector)化、寄存器分配和内存层次结构，或映射到加速器上的专用指令）。这通常与张量级优化中使用的技术重叠。
- **调试：** 在激进融合的代码中查明性能退化或正确性问题可能很困难，因为原始高级操作与最终生成的机器代码之间的直接对应关系因融合转换而变得模糊。良好的编译器诊断和可视化工具变得重要。

尽管存在这些复杂性，激进的算子融合是现代机器学习 (machine learning)编译器和运行时中不可或缺的技术（如TensorFlow XLA, PyTorch JIT (通过NNC/Fuser), Apache TVM, ONNX Runtime, TensorRT）。其性能提升主要由内存带宽需求和核函数启动开销的大幅减少所驱动，对于在当代硬件上实现有竞争力的推理 (inference)和训练速度通常非常重要。它有效地将计算图重构为更大、更适合硬件的执行单元。

## 参考资料

- [TVM: An Automatic End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy (2018)
  Journal: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Publisher: USENIX Association; Pages: 578–594; DOI: [10.1145/3294344.3259969](https://doi.org/10.1145/3294344.3259969)
  介绍TVM深度学习编译器框架中图级别优化，包括操作符融合技术和成本模型。
- [AOT compilation with XLA](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG6XFf4ogn1zNwFJ2_k2Q2I6vKRpnaPBUOYrFLjeaeXLhpaN-GFFG_RnBtJmN3-8RQrzjPdOrjP863j33TQse3rm7qPJbGScPNBRKHdedwcHatBCrONaEP-On_imYWI) — TensorFlow Authors (2024)
  官方文档，说明XLA的提前编译（AOT），强调其如何通过激进的操作符融合等图优化技术提高性能。
- [Graph-Level Optimization for Deep Learning Compilers: A Survey](https://dl.acm.org/doi/10.1145/3342371) — Tan Wang, Peng Li, Jinchen Wu, Junhao Wen, Zhaoyang Zhang, Yu Zhang (2019)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 52; Pages: Article No. 63; DOI: [10.1145/3342371](https://doi.org/10.1145/3342371)
  深度学习编译器中图级别优化的全面，重点介绍各种操作符融合策略及其基本原理。
- [TensorFlow: A System for Large-Scale Machine Learning](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi) — Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng (2016)
  Journal: Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI '16); Publisher: USENIX Association; Pages: 265–283; DOI: [10.1145/2996901.2996906](https://doi.org/10.1145/2996901.2996906)
  介绍TensorFlow系统架构的基础性论文，为了解XLA等后续图优化组件及其融合能力提供了必要背景。
