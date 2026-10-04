---
course: "distributed-training-pytorch-fsdp"
chapter: "multi-node-scaling-nccl-tuning"
lesson: "nccl-collective-communication-primitives"
sourceId: 7899
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-4-multi-node-scaling-nccl-tuning/nccl-collective-communication-primitives"
title: "NCCL 集合通信原语"
description: "深入剖析 NCCL 后端中的 AllGather 和 ReduceScatter 操作。"
order: 2
plots: ["plots/7899-0.json"]
sourceHash: "00050e7a7f580c2218e671d6cfade5c021eeecbc115a481ae622e4aa20bf30a6"
sourceCorrections: []
---

分布式训练策略在其通信模式上存在基本差异。分布式数据并行 (DDP) 主要使用 `AllReduce` 来平均副本间的梯度，而全分片数据并行 (FSDP) 则分解此操作以优化内存使用。FSDP 操作几乎完全依赖于两个 NCCL (NVIDIA 集合通信库) 原语：`AllGather` 和 `ReduceScatter`。了解这些原语的机制和成本模型，对于诊断多节点集群中的瓶颈是必需的，因为在这种集群中，网络带宽而非 GPU 计算能力，通常决定着训练吞吐量 (throughput)。

### FSDP 通信循环

在分片环境中，没有单个 GPU 存储完整的模型权重 (weight)。在特定层（或一组层）进行前向或反向传播 (backpropagation)之前，必须将分布式分片组装成完整参数 (parameter)。计算完成后，必须同步结果并丢弃临时完整参数以释放内存。

这会形成一种独特的“手风琴”式内存模式，在训练步骤中由以下循环驱动：

1. **前向传播 (AllGather)：** 从所有级别收集参数分片以具体化完整权重。
2. **反向传播输入 (AllGather)：** 重新具体化完整权重（如果在前向传播后已丢弃），以计算相对于输入的梯度。
3. **反向传播输出 (ReduceScatter)：** 跨级别同步梯度；同时求和并散布结果，使每个级别最终只拥有与其特定参数分片相对应的梯度。

下图示意了这些操作期间模型块的状态转换。

> FSDP 执行期间单个 GPU 上的数据状态转换。系统通过集合通信，在低内存分片状态和高内存具体化状态之间振荡。

### AllGather：参数 (parameter)具体化

`AllGather` 是一种带宽密集型操作，负责重建完整的模型权重 (weight)。给定 $N$ 个 GPU，其中每个 GPU $i$ 存储一个参数分片 $P_i$，目标是让每个 GPU 都拥有集合 $\{P_0, P_1, \dots, P_{N-1}\}$。

在 NCCL 使用的标准基于环的实现中，通信成本很高。对于模型大小为 $M$ (字节) 的情况，每个级别都必须接收它不拥有的数据。每个级别接收到的数据量是：

$V_{rx} = M \times \frac{N-1}{N}$

随着 $N$ 增加，传输量接近总模型大小 $M$。这种扩展特性很重要；增加更多节点并不会减少前向传播中每个 GPU 的 `AllGather` 通信量。它只是更精细地分片了源数据。

在环形拓扑结构上，以总线带宽 $B$ 完成 `AllGather` 操作的理论时间是：

$T_{AllGather} = \frac{M(N-1)}{B \cdot N}$

在多节点集群中，$B$ 有效地成为节点间互连（例如 InfiniBand HDR/NDR 或 RoCEv2）的瓶颈带宽，这通常比节点内 NVLink 慢一个数量级。

#### 分层 AllGather

当运行混合分片数据并行 (HSDP) 时，通信模式会改变。如果后端支持，PyTorch 可以使用分层集合操作，或者明确的两步操作：

1. **节点内：** 节点内的级别聚合分片。
2. **节点间：** 每个节点的代表交换聚合数据。

这使用了 NVLink 的高带宽进行大部分聚合，从而减轻了较慢的 TCP/IP 或 InfiniBand 网络的压力。

### ReduceScatter：梯度同步

`ReduceScatter` 扮演着 `AllGather` 的功能逆操作，并结合了归约操作。在反向传播 (backpropagation)期间，每个 GPU 都计算完整参数 (parameter)的梯度（因为它刚刚具体化了这些参数）。然而，每个 GPU 只负责更新其优化器状态的特定分片。

因此，系统必须：

1. **归约 (求和)：** 从所有 $N$ 个级别聚合梯度以计算全局梯度。
2. **分散：** 分散这些求和后的梯度，使级别 $i$ 只接收与参数分片 $P_i$ 对应的梯度。

NCCL 将此实现为融合操作。标准的 `AllReduce` (DDP 中使用) 会导致每个级别都持有全局梯度的完整副本。`ReduceScatter` 避免了这种冗余，确保梯度的内存占用与 $1/N$ 成比例。

`ReduceScatter` 的通信量成本与 `AllGather` 相同：

$T_{ReduceScatter} = \frac{M(N-1)}{B \cdot N}$

由于反向传播涉及计算和此通信，FSDP 实现积极地将 `ReduceScatter` 与前一层的反向计算重叠。

### NCCL 中的环形算法与树形算法

尽管我们通常使用环形算法来建模成本，但 NCCL 会根据拓扑结构和消息大小动态选择算法。

1. **环形：** 对于大消息的吞吐量 (throughput)来说是最佳的。数据在逻辑环中流动 ($0 \to 1 \to \dots \to N \to 0$)。这是 FSDP 大型参数 (parameter)张量的默认选项。延迟随 $N$ 线性扩展。
2. **树形（双二叉树）：** 对于小消息的延迟来说是最佳的，或在扩展到数千个 GPU 时。它将延迟扩展从 $O(N)$ 减少到 $O(\log N)$。

在大型语言模型的 FSDP 训练中，张量足够大，以至于环形（或分段环形）算法占主导地位。然而，在极其大的集群（例如 >256 个 GPU）上，环形的线性延迟会阻碍有效的通信重叠。

下图演示了在假设模型大小和互连带宽固定不变的情况下，随着节点数量增加，理论效率如何下降。



![通信效率与节点数量的关系 (固定模型大小)](plots/7899-0.json)



> 随着 $N$ 增加，项 $\frac{N-1}{N}$ 迅速接近 1。这表明在 GPU 数量达到一定程度后，带宽需求就会饱和。增加更多节点并不会减少每 GB 模型数据的通信时间；它只是分散了计算和内存容量。

### 张量融合和分桶

PyTorch 和 NCCL 不会逐个传输张量。那样做会产生巨大的内核启动开销，并且无法使 PCIe 或 NVLink 总线饱和。相反，FSDP 采用分桶策略。

张量被展平并连接到一个连续的缓冲区（一个“桶”）中，直至达到配置的大小限制（例如 25MB 或 50MB）。集合原语在此桶上执行。

- **延迟-带宽权衡：**
  - **小桶：** 握手开销更高，受延迟影响大。不利于带宽利用。
  - **大桶：** 带宽利用率高，但引入“启动延迟”。直到计算流填充整个桶后，通信才能开始。

在 `ShardingStrategy` 中配置 `bucket_cap_mb` 是一个优化方向。对于通过以太网或 InfiniBand 进行的多节点训练，较大的桶（例如 100MB+）通常通过分摊协议开销来获得更好的吞吐量 (throughput)，而在仅使用低延迟 NVLink 的设置中，可能更倾向于使用较小的桶以收紧重叠窗口。

### 网络瓶颈和点对点传输

在多节点集群中，`AllGather` 和 `ReduceScatter` 必须穿越节点间互连结构。如果集群使用 GPU Direct RDMA (GDR)，NCCL 可以直接从 GPU 内存将数据传输到网络接口卡 (NIC)，绕过 CPU。

没有 GDR 时，数据遵循的路径是：`GPU -> CPU RAM -> NIC -> Network -> NIC -> CPU RAM -> GPU`。这会引入显著延迟并消耗 CPU 内存带宽。

当使用 PyTorch Profiler 分析 FSDP 时，NCCL 流中明显的“等待”状态通常表明集合操作被阻塞，等待来自环中最慢链路的数据。在异构集群中，一个网卡协商降级（例如，以 100Gbps 而非 400Gbps 运行）的节点，会将整个集群的 `AllGather` 速度限制到那个最低速度，无论其他 GPU 的计算速度有多快。
