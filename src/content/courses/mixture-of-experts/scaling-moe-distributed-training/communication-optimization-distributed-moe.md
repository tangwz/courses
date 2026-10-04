---
course: "mixture-of-experts"
chapter: "scaling-moe-distributed-training"
lesson: "communication-optimization-distributed-moe"
sourceId: 4703
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-4-scaling-moe-distributed-training/communication-optimization-distributed-moe"
title: "通信优化方法（例如，重叠）"
description: "减少大规模MoE训练中通信开销的方法。"
order: 6
plots: []
sourceHash: "8e851f19d9bf4bf633b5e278a973fca90d8bb0c75dea20ee321f17f67b802964"
sourceCorrections: []
---

使用专家并行将专家分布到多个设备上会引入一个主要的通信步骤：All-to-All 交换。每个设备需要发送给远程专家的令牌，并接收分配给其本地专家的令牌。该操作，表示为根据门控决策 $g(x)$ 将令牌表示 $x$ 路由到可能位于不同设备上的正确专家 $E_j$，在分布式MoE训练中，尤其当设备数量增加时，很容易成为主要的性能瓶颈。


$$
x_{\text{在设备 } i \text{ 上}} \xrightarrow{\text{All-to-All}} E_{j (\text{在设备 } k \text{ 上})}
$$


大规模MoE训练的效率取决于减少这种All-to-All通信的开销。简单地等待通信完成后再进行下一步操作，会对计算单元造成大量空闲时间。幸运的是，有一些方法可以隐藏或减少这种通信延迟，主要是通过将其与有效计算重叠来进行。

### 计算-通信重叠的原理

重叠的主要想法很直接：在网络传输（All-to-All）进行时，执行独立的计算任务。现代硬件和分布式框架通常支持异步操作，这意味着我们可以启动一个通信任务，然后立即进行不依赖该通信结果的计算。如果存在足够的独立计算，等待网络的时间可以被有效地隐藏起来。

以下是Transformer模块中包含MoE层在前向传播时的典型顺序：

1. 输入令牌经过自注意力 (self-attention)机制 (attention mechanism)。
2. 输入令牌经过第一个前馈网络（FFN）层（MoE层的输入投影部分）。
3. 门控网络计算每个令牌的专家分配。
4. **All-to-All 通信：** 令牌被分发到其指定的专家设备。
5. 专家网络处理其分配的令牌。
6. **All-to-All 通信：** 处理后的令牌被发送回其原始设备。
7. 令牌被组合（根据门控分数进行加权求和）。
8. 输出令牌经过第二个FFN层（输出投影）。
9. 输出令牌进入下一层或操作（例如，LayerNorm，残差连接）。

两个All-to-All步骤（4和6）是进行重叠的主要选择。同样，在反向传播 (backpropagation)期间，梯度需要被路由回来，这又是一个可能重叠的All-to-All通信阶段。

### 实现重叠的策略

实现有效的重叠需要细致的调度，并借助通信库（如MPI或PyTorch的`torch.distributed`）和硬件功能（如CUDA流）提供的异步操作。

#### 异步通信原语

与其使用阻塞式通信调用（它们会停止执行直到传输完成），不如使用它们的非阻塞式对应项。例如，在PyTorch的分布式包中：

- 与其使用 `dist.all_to_all()`，您可以使用 `dist.all_to_all_single()`，在某些配置下它可能同步性较低，或者使用更底层的非阻塞原语，例如 `dist.isend()` 和 `dist.irecv()`，并结合同步机制 (`wait()`)。

典型的模式如下所示：

1. 启动非阻塞All-to-All通信（例如，`handle = dist.isend(...)`）。
2. 执行不依赖于正在传输数据的那部分计算。这可能包括：
   - 如果使用流水线并行，模型后续层的计算。
   - 网络中并行层或分支的计算。
   - 反向传播 (backpropagation)时，MoE层通信步骤*之前*处理的层的梯度计算。
   - 甚至可以是专家计算的一部分，如果通信和计算能够细致地交错进行（例如，首先处理本地可用数据）。
3. 在执行依赖于已传输数据的代码之前，等待通信完成（`handle.wait()`）。

#### 借助CUDA流

在GPU加速系统中，CUDA流提供了一种管理并发的机制。排队在不同流上的操作可能并行执行。通信库通常使用独立的CUDA流进行数据传输（将数据复制到/从CPU内存，网络传输）和计算核。通过细致管理依赖关系并使用多个流，框架可以安排计算核与通信库管理的数据传输并发运行。这需要对框架的执行模型有详细了解，或者在实现自定义核时直接操作流。

#### 重叠的可视化

重叠的益处在可视化时变得清晰。没有重叠时，计算等待通信。有了重叠，总时间会减少。

> 时间线比较了顺序执行（顶部），其中计算等待通信，与重叠执行（底部），其中独立计算（计算 B）与All-to-All传输同时发生，从而减少了总执行时间。

### 优化All-to-All集体操作本身

All-to-All操作本身的重叠性能有时可以得到提高：

- **算法选择：** NCCL（适用于NVIDIA GPU）等库为All-to-All等集体操作实现了多种算法。它们的性能可能在很大程度上取决于网络拓扑（例如，NVLink密度、节点互连）和消息大小。框架通常会尝试自动调整此项，但在特定的集群设置中，可能需要手动选择。
- **分层集体操作：** 在非常大的集群中，分阶段执行All-to-All（例如，首先在节点内部使用NVLink等快速互连，然后通过网络在节点之间）可能比在所有设备上进行单一、扁平的All-to-All更高效。像DeepSpeed这样的框架可能会实现这样的策略。
- **通信数据类型：** 对通信缓冲区使用低精度数据类型（如FP16或BF16）可以减少传输的数据量，直接加速All-to-All操作，前提是模型的数值稳定性得到保持。

### 框架实现

手动实现这些优化复杂且容易出错。幸运的是，专门的框架承担了大部分这些负担。

- **DeepSpeed：** 提供集成了通信优化的MoE实现，包括重叠启发式算法和可能的分层集体操作。其ZeRO优化器阶段也能影响梯度通信的方式。
- **Tutel：** 一个微软的库，专门侧重于优化MoE层，提供高度优化的All-to-All核和重叠调度。
- **Megatron-LM：** 包含张量并行和流水线并行，其MoE实现通常包括为其并行策略定制的通信优化。

不过，理解通信优化的基本原理，尤其是重叠，对于有效配置这些框架以及诊断大规模MoE训练任务中的性能瓶颈非常重要。能够可视化计算和通信执行时间线的分析工具（如NVIDIA Nsight系统或PyTorch Profiler的分布式跟踪功能）对于识别空闲时间和获得更好重叠的机会是不可或缺的。

## 参考资料

- [Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) — PyTorch Authors (2017)
  Publisher: PyTorch Documentation
  PyTorch分布式包的官方文档，详细说明了异步通信原语和分布式训练的最佳实践，对实现计算-通信重叠有参考价值。
- [NVIDIA Collective Communications Library (NCCL) Documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA
  NVIDIA集体通信库的官方指南，该库提供诸如All-to-All之类的高性能集体操作，对于分布式训练中高效的GPU间通信至关重要。
