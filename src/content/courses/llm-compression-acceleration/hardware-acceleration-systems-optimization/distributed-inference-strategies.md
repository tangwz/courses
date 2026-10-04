---
course: "llm-compression-acceleration"
chapter: "hardware-acceleration-systems-optimization"
lesson: "distributed-inference-strategies"
sourceId: 4413
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-6-hardware-acceleration-systems-optimization/distributed-inference-strategies"
title: "分布式推理策略"
description: "实现张量并行、流水线并行和序列并行，用于分布式LLM服务。"
order: 5
plots: []
sourceHash: "25c7cdd38dbaea476c91aeed06db834651a2115592067ab3b9f32e3083274eef"
sourceCorrections: []
---

即使是经过优化的模型，其内存容量和计算吞吐量 (throughput)也可能超出单个加速器设备（如 GPU）的能力。在部署真正的大型语言模型或追求极高的推理 (inference)吞吐量时，将工作负载分布到多个设备上是必需的。分布式推理策略可以并行化计算，使得过大的模型可以在一个设备上运行，或者通过多个加速器的组合资源来加快处理速度。

有几种既定的方法可以将推理工作负载进行划分：

### 张量并行 (TP)

张量并行（通常指层内模型并行）侧重于将单个层（或更准确地说，层内的大型矩阵乘法）的执行拆分到多个设备上。与在一个设备上计算整个操作 $Y = XA$ 不同，权重 (weight)矩阵 $A$ 在 $N$ 个设备上按列（$A = [A_1, A_2, ..., A_N]$）或按行（$A$ 水平拆分）进行划分。

- **列并行：** 如果 $A$ 按列拆分，每个设备 $i$ 计算 $Y_i = X A_i$。然后将结果 $Y_i$ 连接起来（$Y = [Y_1, Y_2, ..., Y_N]$）。这通常需要将输入 $X$ 广播或等效地发送到所有设备，但在局部计算之后不需要通信。
- **行并行：** 如果 $A$ 按行拆分，输入 $X$ 也按行拆分（$X = [X_1^T, X_2^T, ..., X_N^T]^T$）。每个设备 $i$ 使用其部分 $A$ 计算部分结果。这些部分结果必须通过 All-Reduce 通信操作在所有设备上求和，以产生最终输出 $Y$。

在 Transformer 模型中，张量并行通常应用于前馈网络 (FFN) 层中的权重矩阵以及注意力机制 (attention mechanism)的查询、键、值和输出投影矩阵。

> 张量并行：输入 `X` 在设备 0（使用权重分片 `W0`）和设备 1（使用 `W1`）上同时处理。部分结果通过 All-Reduce 或连接操作组合。

**优点：**

- 减少每个设备对权重和激活的内存使用，因为激活大小通常随批次大小和序列长度而变化。
- 如果通信速度快，可以降低单个层计算的延迟。

**缺点：**

- 需要设备之间的高带宽互连（例如，NVLink、NVSwitch），因为通信（特别是 All-Reduce）在每个层的计算中频繁发生。
- 通信开销可能会限制在单个节点内的设备扩展数量。

### 流水线并行 (PP)

流水线并行采用不同的方式，它将模型按层划分到多个设备上。每个设备（或阶段）包含模型层的一个子集。例如，在 4 个设备的设置中，设备 0 可能包含第 1-10 层，设备 1 包含第 11-20 层，以此类推。

输入数据按顺序流经这些阶段。设备 0 处理输入并将其输出（中间激活）传递给设备 1，设备 1 进一步处理并将其传递给设备 2，持续进行直到最后一个设备产生输出。

简单的实现会造成设备利用率低（“流水线气泡”），因为大多数设备在等待上一阶段的数据时处于空闲状态。为了减轻这种情况，输入批次通常被分割成更小的*微批次*。一旦设备 0 完成处理第一个微批次，它就会将结果发送到设备 1，并立即开始处理第二个微批次。这使得多个微批次可以在流水线阶段中同时进行，从而提高硬件利用率。

> 流水线并行：微批次（微批次1、微批次2等）按顺序流经各个阶段（设备 0、1、... N）。设备 `i` 在收到来自设备 `i-1` 的激活后开始处理微批次 `k`。

**优点：**

- 减少每个设备对权重 (weight)的内存占用，因为每个设备只包含一小部分层。与存储整个模型的激活相比，也减少了激活内存，尽管可能比张量并行更多，具体取决于划分方式。
- 通信通常是相邻阶段之间的点对点通信，可能比张量并行的 All-Reduce 操作所需的 All-to-All 带宽更少。这可能更适合通过多个节点进行扩展，即使互连速度较慢。
- 有效提高推理 (inference)吞吐量 (throughput)。

**缺点：**

- 由于顺序处理和流水线填充/排空时间，即使有微批次，也会增加推理延迟。延迟的下限由所有阶段的计算总和加上通信延迟决定。
- 负载均衡可能有挑战；需要仔细划分层以确保每个阶段花费大致相同的时间。
- 管理流水线调度（例如 GPipe、PipeDream）增加了复杂性。

### 序列并行 (SP)

张量并行将层内的张量拆分，流水线并行将层拆分到设备上，而序列并行则侧重于沿着序列长度维度划分输入序列本身。这项技术对注意力机制 (attention mechanism)特别有效，因为注意力机制的计算复杂度随序列长度呈二次方增长（$O(L^2)$）。

在标准的张量并行中，*整个*序列的激活通常需要在*所有*张量并行设备上，这对于非常长的序列来说可能成为内存瓶颈。序列并行允许拆分 LayerNorm、dropout 和注意力计算的特定部分等操作，使得每个设备只处理序列长度的一个片段。

例如，在注意力计算 $注意力(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ 中，操作通常可以重新设计，使得通信（例如，部分和或部分 softmax 计算）沿着序列维度发生，从而允许设备并行处理不同的序列块，而无需整个序列的完整中间激活张量。这通常需要修改标准层实现。

**优点：**

- 直接解决与长序列相关的内存瓶颈，特别是注意力层内的激活。
- 可以与张量并行和流水线并行结合。
- 与处理非常长序列的张量并行相比，减少了某些操作所需的通信量。

**缺点：**

- 主要有利于那些在序列维度上独立或可以通过特定通信模式（如注意力）并行化的操作。对于在每个 token 位置独立应用的前馈网络等操作，适用性较低。
- 可能需要自定义核函数实现或特定框架支持（如 DeepSpeed 提供）。
- 增加了分布策略的另一个复杂维度。

### 混合方法

在实践中，部署最大的模型通常涉及组合这些策略。常见的配置是在节点内使用张量并行（借助 NVLink 等快速节点内互连），并在节点间使用流水线并行（互连带宽通常较低）。

例如，一个模型可以被拆分为 8 个流水线阶段（流水线并行度 = 8）。在每个阶段内，层可以使用张量并行（张量并行度 = 8）进一步并行化到 8 个 GPU 上。这导致总共使用了 64 个 GPU。如果上下文 (context)长度非常大，还可以在此基础上添加序列并行。

策略的选择在很大程度上取决于具体的模型架构、硬件配置（设备数量、节点内和节点间带宽）以及主要的优化目标（延迟或吞吐量 (throughput)）。

### 通信开销

任何分布式策略有效性的一个重要因素是通信开销。

- **张量并行：** 通常依赖于每个层内带宽密集的 All-Reduce 操作。性能对互连速度高度敏感。
- **流水线并行：** 依赖于阶段之间激活的点对点通信。数据量取决于激活大小和微批次大小。顺序发送会增加延迟。
- **序列并行：** 通信模式取决于沿着序列轴并行化的具体操作。

最小化数据传输量并将通信与计算重叠是 PyTorch Fully Sharded Data Parallel (FSDP)、DeepSpeed、Megatron-LM 等分布式训练和推理 (inference)框架以及 NVIDIA Triton 等具有多 GPU 后端的推理服务器或 vLLM 中实现的重要优化技术。

了解这些分布策略对于将LLM推理扩展到超出单个设备限制的范围、支持部署大型模型以及实现应用程序所需的吞吐量 (throughput)非常重要。虽然框架抽象了部分实现细节，但了解其基本原理有助于选择合适的策略并调试性能瓶颈。

## 参考资料

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://ieeexplore.ieee.org/document/9355153) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2020)
  Journal: SC '20: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: IEEE; Pages: 1-13; DOI: [10.1109/SC41405.2020.00072](https://doi.org/10.1109/SC41405.2020.00072)
  介绍一种利用张量并行训练大型语言模型的方法。
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) — Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen (2019)
  Journal: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); Pages: 10835-10846; DOI: [10.5555/3454287.3455110](https://doi.org/10.5555/3454287.3455110)
  提出一种利用微批次提高硬件利用率的流水线并行方法。
- [DeepSpeed: Extreme-scale distributed training for DL models](https://arxiv.org/abs/2008.01484) — Jamie Watson, Oisin Mac Aodha, Daniyar Turmukhambetov, Gabriel J. Brostow, Michael Firman (2020)
  Journal: Accepted as an oral presentation at ECCV 2020; Pages: 3432-3440; DOI: [10.48550/arXiv.2008.01484](https://doi.org/10.48550/arXiv.2008.01484)
  描述一个用于在多设备上高效扩展深度学习模型的开源库。
