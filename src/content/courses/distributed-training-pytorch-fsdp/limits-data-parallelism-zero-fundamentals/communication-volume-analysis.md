---
course: "distributed-training-pytorch-fsdp"
chapter: "limits-data-parallelism-zero-fundamentals"
lesson: "communication-volume-analysis"
sourceId: 7886
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-1-limits-data-parallelism-zero-fundamentals/communication-volume-analysis"
title: "通信量分析"
description: "计算并优化分片设置中参数同步所需的通信开销"
order: 3
plots: ["plots/7886-0.json"]
sourceHash: "594ea1cba3949028ea62eeea807ed8725c571c41e211fc9084f2f81f676e6907"
sourceCorrections: []
---

通过分片扩展内存容量带来一个必然的权衡：网络使用率增加。ZeRO 阶段通过在设备间划分状态有效解决了内存瓶颈，但它们从根本上改变了训练循环的通信模式。在标准分布式数据并行 (DDP) 设置中，通信仅限于反向传播 (backpropagation)阶段，梯度在此同步。相反，完全分片数据并行 (FSDP) 在前向传播和反向传播阶段都需要主动通信，以按需实现参数 (parameter)。

明确数据传输量对于设计集群拓扑结构以及调试性能退步很重要。如果网络互联无法维持所需的吞吐量 (throughput)，GPU 计算单元将停滞，闲置等待参数分片到达。

### 分片中的通信原语

为了分析带宽需求，我们必须首先识别出 ZeRO 使用的特定集合通信原语。与主要依赖 `AllReduce` 的 DDP 不同，FSDP 使用 `AllGather` 和 `ReduceScatter`。

1. **AllGather**：用于从分片中重建完整的模型参数 (parameter)。在特定层执行其前向或反向计算之前，每个 GPU 必须获取通信组中所有其他 GPU 上托管的参数分片。
2. **ReduceScatter**：用于同步和分片梯度。反向传播 (backpropagation)为层计算梯度后，这些梯度在所有工作器之间平均，并立即划分，以便每个 GPU 只存储与其参数分片对应的梯度分片。

#### 数学成本模型

设 $\Psi$ 表示模型参数的总数。我们假设进行混合精度训练，其中参数和梯度以 16 位格式（FP16 或 BF16）存储，意味着每个元素占用 2 字节。

在一个有 $N$ 个 GPU 的集群中，一个标准的 DDP 实现每步对梯度执行一次 `AllReduce` 操作。使用基于环或基于树的算法，`AllReduce` 的通信量是 $2\Psi$。这实际上意味着每个参数元素每训练步遍历网络两次（一次用于归约，一次用于广播），与 $N$ 无关（对于大的 $N$）。


$$
V_{DDP} \approx 2\Psi
$$


FSDP 改变了这个等式。由于参数被分片，它们在本地不可用。训练步骤遵循以下顺序：

1. **前向传播**：触发 `AllGather` 来收集参数。每个秩下载 $\frac{N-1}{N} \Psi$ 数据。
2. **反向传播（预计算）**：参数，通常在前向传播后为节省内存而被丢弃，必须再次 `AllGather` 以计算相对于输入的梯度。
3. **反向传播（后计算）**：计算出的梯度使用 `ReduceScatter` 进行同步和分片。

FSDP（特别是 ZeRO-3）每步的总数据传输量是这三项操作的总和。对于大的 $N$，`AllGather` 和 `ReduceScatter` 的成本都接近 $\Psi$。


$$
V_{FSDP} \approx \Psi_{\text{前向\_收集}} + \Psi_{\text{反向\_收集}} + \Psi_{\text{梯度\_分散}} = 3\Psi
$$


这项分析显示出一个重要的架构含义：**FSDP 需要大约 1.5 倍于 DDP 的通信带宽** ($3\Psi$ vs $2\Psi$)。

> 数据流显示了 FSDP 中完整参数的瞬时特性。与参数持续存在的 DDP 不同，FSDP 需要网络传输来为前向和反向传播实现权重 (weight)，随后对梯度进行分散操作。

### 带宽饱和和节点扩展

公式 $V_{FSDP} \approx 3\Psi$ 是一个假设 $N$ 值很大的近似值。基于环的集合操作的精确通信成本，`AllReduce` 为 $2 \frac{N-1}{N} \Psi$，`AllGather/ReduceScatter` 为 $\frac{N-1}{N} \Psi$。

随着集群大小 $N$ 的增加，项 $\frac{N-1}{N}$ 迅速接近 1。这表明增加更多 GPU 并不能减少每个 GPU 的通信量。相反，它使每个 GPU 的吞吐量 (throughput)要求保持不变，同时增加了集群的总聚合带宽。

这种恒定的每个设备带宽压力使 FSDP 对“落后者”或慢节点很敏感。如果集群中的单个链路协商到较低的速度（例如，从 InfiniBand HDR 回退到 EDR，或 TCP 重传问题），整个组的 `AllGather` 集合操作将以最慢链路的速度运行。

下图比较了每训练步所需的理论最小数据传输量，针对不同的模型大小。请注意随着模型大小增长，DDP 和 FSDP 之间的差异。



![每步通信量：DDP vs FSDP (ZeRO-3)](plots/7886-0.json)



> 16 位混合精度训练的网络流量比较。随着模型规模扩展到 700 亿参数 (parameter)，FSDP 需要在网络互联中传输 420GB 的数据，对于每个优化器步骤，这仅用于参数和梯度同步。

### 粒度和延迟隐藏

虽然带宽 (GB/s) 决定了大型张量的传输时间，但延迟（微秒）决定了小型张量的性能。FSDP 的一个朴素实现可能会尝试单独分片每个线性层。对于包含独立的键、查询和值投影层的 Transformer 块，这将触发三个单独的小 `AllGather` 操作。

启动集合通信核的开销通常超过小型负载的传输时间。为了减轻这种情况，PyTorch FSDP 将参数 (parameter)聚合成“扁平参数”。这种机制将模块内（如 Transformer 解码器层）的多个小型张量扁平化为一个连续的内存块。

这种聚合有两个作用：

1. **向量 (vector)化**：它允许 `AllGather` 对更大的数据块进行操作，更好地使用互联带宽。
2. **通信-计算重叠**：通过定义一个分片“单元”（通常是一个 Transformer 块），FSDP 可以在 GPU 忙于计算块 $N$ 的同时预取块 $N+1$ 的参数。

如果 $3\Psi$ 的通信时间超过了前向/反向传播 (backpropagation)的计算时间，训练就会受到通信限制。这个比率是调整大型集群时的主要效率指标。高效扩展需要网络互联提供足够的带宽，以使：


$$
T_{\text{通信}} \leq T_{\text{计算}}
$$


当满足此条件时，并且重叠策略配置正确，$3\Psi$ 数据传输的成本可以被有效隐藏在矩阵乘法的算术强度后面。
