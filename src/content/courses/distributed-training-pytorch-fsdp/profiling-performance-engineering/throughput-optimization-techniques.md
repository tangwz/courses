---
course: "distributed-training-pytorch-fsdp"
chapter: "profiling-performance-engineering"
lesson: "throughput-optimization-techniques"
sourceId: 7910
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-6-profiling-performance-engineering/throughput-optimization-techniques"
title: "吞吐量优化技术"
description: "计算并最大化模型浮点运算利用率 (MFU) 和吞吐量的策略。"
order: 4
plots: ["plots/7910-0.json"]
sourceHash: "46d27a8c0da7d5185daa44bb837cc3cc8edde61ed0896f3393f6a6704b8baab4"
sourceCorrections: []
---

优化分布式训练的根本在于增加单位时间内的计算密度，同时减少数据传输的延迟。一旦功能正确性得到确认，重点便转向最大化模型浮点运算利用率 (MFU)。此指标衡量您的训练循环如何高效地发挥底层硬件的理论峰值性能。在高性能计算环境中，大型语言模型 (LLM) 达到 50% 到 60% 的 MFU 被认为是很好的成绩，而未优化的 FSDP 配置通常低于 30%。

### 衡量模型浮点运算利用率

以“每秒样本数”或“每秒 token 数”衡量的原始吞吐量 (throughput)有助于比较不同运行，但它没有考虑硬件能力或模型架构变化。MFU 提供一个标准化的效率得分。要计算此值，我们首先估算单个训练步骤所需的浮点运算量。

对于基于 Transformer 的模型，每个 token 的浮点运算 (FLOPs) 数量大致与参数 (parameter)数量成比例。在不使用激活检查点的情况下，前向和后向传播的标准近似值是：

$C_{\text{步}} \approx 6 \cdot P \cdot D_{\text{批次}}$

其中 $P$ 表示可训练参数的数量，$D_{\text{批次}}$ 是全局批次中的 token 总数（序列长度 $\times$ 批次大小）。因子 6 来自前向传播 ($2P$) 和后向传播 ($4P$)。

然而，当使用 FSDP 训练大模型时，激活检查点（也称为梯度检查点）几乎总是启用以节省内存。此技术要求在反向传播 (backpropagation)阶段重新计算前向传播。因此，计算成本增加：

$C_{\text{检查点}} \approx 8 \cdot P \cdot D_{\text{批次}}$

为了确定 MFU，我们将每秒实际达到的 FLOPs 除以 GPU 的理论峰值吞吐量（例如，使用 BF16 Tensor Cores 的 NVIDIA A100 为 312 TFLOPS）。

$\text{MFU} = \frac{C_{\text{检查点}} / \text{步长耗时 (s)}}{\text{设备峰值FLOPs} \times \text{GPU数量}}$

低 MFU 表明 GPU 执行单元正在停滞，可能由于内存带宽限制（HBM瓶颈）、通信开销（网络瓶颈）或内核启动延迟（延迟瓶颈）。

> 系统性提升训练吞吐量的优化流程。

### 最大化算术强度

FSDP 训练中 MFU 较低最常见的原因是微批次大小过小。GPU 在处理大型连续数据块时表现最佳。如果每个 GPU 的本地批次大小过小，系统就会受限于内存带宽；计算核心等待 HBM 数据的时间将多于执行矩阵乘法的时间。

解决此问题的方法是，增加每个 GPU 的微批次大小，直到接近内存不足 (OOM) 限制。这会增加算术强度，即执行的 FLOPs 与访问的字节数之比。如果全局批次大小由收敛超参数 (parameter) (hyperparameter)固定，则使用梯度累积来保持全局批次大小，同时最大化硬件上的微批次大小。

例如，如果您的目标全局批次大小为 1024，并且您有 64 个 GPU：

1. **情况 A：** 每个 GPU 微批次 16 ($16 \times 64 = 1024$)。梯度累积步数 = 1。
2. **情况 B：** 每个 GPU 微批次 4 ($4 \times 64 = 256$)。梯度累积步数 = 4。

情况 A 在吞吐量 (throughput)方面严格更优，因为它启动更少、更大的内核，从而减少开销并更好地饱和 Tensor Cores。情况 B 仅在情况 A 导致 OOM 时使用。

### IO 感知内核和 FlashAttention

在 Transformer 架构中，自注意力 (self-attention)机制 (attention mechanism)的开销与序列长度成平方关系。标准实现将 $N \times N$ 注意力矩阵读写到 HBM，这会造成严重的内存瓶颈。

对于高性能训练，集成 FlashAttention（v2 或更高版本）是必须的。它将注意力操作融合到一个单独的内核中，将注意力矩阵保留在 GPU 的快速 SRAM（L1/共享内存）中，避免了对 HBM 的往返访问。这不仅加快了计算速度，还减少了内存占用，从而允许更大的批次大小。在 PyTorch FSDP 中，确保您的模型封装了 `F.scaled_dot_product_attention`（在可用时会分派到 FlashAttention）是一项高优先级的优化。

### 优化通信频率

FSDP 通过在计算前聚合分片参数 (parameter) (AllGather) 和在计算后同步梯度 (ReduceScatter) 来引入通信开销。这些消息的频率和大小会影响吞吐量 (throughput)。

FSDP 配置中的 `limit_all_gathers` 设置控制 GPU 是在层的前向传播后立即释放聚合的分片，还是将其保留用于后向传播。

- **True (默认值)：** 通过立即释放权重 (weight)来节省内存。在反向传播 (backpropagation)期间需要重新聚合权重（更多通信）。
- **False：** 保留聚合的权重直到反向传播完成。减少通信量但增加峰值内存使用。

如果分析显示由于 NCCL AllGather 操作而存在明显空白，并且 VRAM 未完全饱和，为特定层或整个模型禁用 `limit_all_gathers` 可以带来大幅的加速。

此外，调整 `bucket_cap_mb` 参数控制通过网络发送的数据块大小。小桶会增加 NCCL 调用的次数（延迟开销），而过大的桶可能会妨碍计算和通信之间的有效重叠。对于现代集群，25MB 到 100MB 之间的值通常是最佳的。

> 累积优化对 A100 GPU 上 70 亿参数模型训练吞吐量 (TFLOPS) 的影响。



![优化层带来的吞吐量提升](plots/7910-0.json)



### 数据加载和 CPU 瓶颈

尽管通常关注 GPU，CPU 可能会悄悄地限制训练速度。如果 `DataLoader` 无法足够快地向 GPU 提供数据，GPU 执行时间线将在步骤之间显示空白，其中没有内核在运行。这在 PyTorch Profiler 中通常表现为“DataLoaderNext”占用大量时间。

为缓解此问题：

1. **固定内存：** 始终在 DataLoader 中设置 `pin_memory=True` 以使用页锁定内存，从而加快从主机到设备的传输。
2. **工作进程数：** 通常将 `num_workers` 设置为 CPU 核心数除以每个节点的 GPU 数量。
3. **预取：** 确保数据加载器预取批次，以便当前步完成时，下一个批次能立即就绪。

通过系统地处理算术强度、内核效率、通信开销和数据调度，您可以将一个可用的分布式配置转变为一个高性能训练引擎，能够在合理的时间范围内处理数 TB 的数据。
