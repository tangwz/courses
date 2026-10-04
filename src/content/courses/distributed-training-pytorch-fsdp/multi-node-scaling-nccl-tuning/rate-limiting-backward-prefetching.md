---
course: "distributed-training-pytorch-fsdp"
chapter: "multi-node-scaling-nccl-tuning"
lesson: "rate-limiting-backward-prefetching"
sourceId: 7900
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-4-multi-node-scaling-nccl-tuning/rate-limiting-backward-prefetching"
title: "速率限制与反向预取"
description: "使用反向预取策略优化通信与计算重叠。"
order: 3
plots: ["plots/7900-0.json"]
sourceHash: "d3726b5bd91bf90938fbeb3d27d263ffdb2d5b8d15cb90ea2bafb667770b9a3b"
sourceCorrections: []
---

优化网络通信与GPU计算的配合，决定了多节点集群中训练吞吐量 (throughput)的上限。FSDP（完全分片数据并行）的一种简单实现方式将通信和计算视为串行依赖。GPU 需要等待 `AllGather` 操作完成，实例化当前层的完整参数 (parameter)后，才能开始该层的前向或反向传播 (backpropagation)。这种串行执行会带来大量的空闲时间，即通信延迟的“显现”。

为减少这种情况，我们采用反向预取技术。该技术在GPU忙于计算当前层梯度时，调度后续层的通信。然而，积极的预取会争用GPU内存和PCI-e带宽。因此，我们必须在重叠与速率限制之间取得平衡，以避免内存分配器抖动和内存不足（OOM）错误。

### 反向预取的工作原理

在反向传播 (backpropagation)中，梯度从输出层传播回输入层。若不进行预取，FSDP 对于每个层块会执行以下步骤：

1. 触发 `AllGather` 以重构完整参数 (parameter)。
2. 等待同步。
3. 计算相对于输入和权重 (weight)的梯度。
4. 触发 `ReduceScatter` 以同步和分片梯度。
5. 丢弃完整参数以释放内存。

这种“停-等”协议在步骤1、2和4期间使CUDA核心处于空闲状态。反向预取通过检查执行图来改变这种工作流程。考虑到反向传播是线性进行的（例如，第10层 $\rightarrow$ 第9层 $\rightarrow$ 第8层），FSDP 可以在第10层计算开始时，立即在辅助 NCCL 流上为第9层发出 `AllGather` 命令。

下图说明了串行执行与带有预取的流水线执行之间的差异。

> 串行执行与流水线执行的比较。在流水线方法中，第 N-1 层的通信与第 N 层的计算同时进行，从而隐藏了延迟。

### PyTorch 预取策略

PyTorch FSDP 通过 `backward_prefetch` 参数 (parameter)提供此功能，该参数接受 `BackwardPrefetch` 枚举中的值。理解这两种主要策略的区别对调整内存使用情况非常重要。

#### BACKWARD\_POST

这是默认且保守的策略。FSDP 在当前层的梯度计算完成后，立即为*前一层*（反向序列中的下一层）发出 `AllGather`。尽管这允许一些重叠，但它没有最大化并发的潜力，因为通信请求在周期中发出相对较晚。

#### BACKWARD\_PRE

这是积极的策略。FSDP 在当前层的梯度计算开始*之前*，为前一层发出 `AllGather`。这确保了网络传输与计算核的整个持续时间并行运行。

从数学角度看，如果 $T_{计算}$ 是计算时间，$T_{通信}$ 是通信时间，那么有效步长 $T_{步长}$ 将从：

$T_{步长} = \sum_{i=1}^{L} (T_{计算}^{(i)} + T_{通信}^{(i)})$

转变为重叠状态，此时：

$T_{步长} \approx \sum_{i=1}^{L} \max(T_{计算}^{(i)}, T_{通信}^{(i)})$

`BACKWARD_PRE` 策略通常会带来更高的模型浮点运算利用率（MFU）。然而，它会增加峰值内存消耗。由于下一层的参数在当前层的参数仍在内存中时被获取，GPU 必须同时保存两组完整参数。

### 使用 limit\_all\_gathers 进行速率限制

尽管 `BACKWARD_PRE` 最大化了重叠，但盲目预取可能导致资源争用。如果 GPU 计算速度快于网络传输数据的速度，FSDP 可能会让多个待处理的 `AllGather` 操作排队。这会用尚未消耗的大张量淹没 GPU 内存分配器，导致内存碎片化或 OOM 错误。

为控制此情况，FSDP 提供了 `limit_all_gathers` 配置（通常默认启用或可通过 `limit_all_gathers=True` 配置）。这充当指令流的信号量。如果内存中已存在特定数量的非分片参数 (parameter)集，它会限制 CPU 线程调度新的 `AllGather` 集体操作。

当 `limit_all_gathers` 启用时，系统会强制执行一个严格的实例化层窗口。对于一个标准的 Transformer 块，这通常意味着只有当前层和立即预取的层驻留在 GPU 内存中。如果预取器在第一层释放之前尝试获取第三层，速率限制器会阻塞调度，直到内存被释放。

下图展示了在调整预取深度和速率限制时，内存开销和吞吐量 (throughput)之间的权衡。



![预取策略对内存和吞吐量的影响](plots/7900-0.json)



> 内存与吞吐量的分析。`BACKWARD_PRE` 提供了最高的吞吐量，但明显增加了内存压力。添加速率限制（`BACKWARD_PRE + Limit`）保留了大部分吞吐量增益，同时将内存使用控制在安全范围内。

### 实现模式

启用这些功能需要在模型封装阶段传递特定的参数 (parameter)。这通常在您的主训练循环设置中完成，即构建 `FullyShardedDataParallel` 实例的地方。

您必须导入 `BackwardPrefetch` 枚举并将其应用于您的策略。

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    ShardingStrategy
)

# 带有安全限制的积极重叠配置
model = FSDP(
    base_model,
    # 使用积极预取来隐藏通信延迟
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,

    # 强制执行速率限制以防止 GPU 内存不足
    limit_all_gathers=True,

    # 标准分片策略（ZeRO-3）
    sharding_strategy=ShardingStrategy.FULL_SHARD,

    # 确保为 NCCL 正确设置设备网格
    device_id=torch.cuda.current_device()
)
```

### 网络带宽依赖性

反向预取的有效性与节点之间可用的网络带宽紧密相关。

在具有高延迟互连的环境中（例如没有 RDMA 的标准以太网），预取是必不可少的。传输数据所需的时间（$T_{通信}$）相对于计算时间（$T_{计算}$）较长。若无预取，GPU 将在大部分反向传播 (backpropagation)过程中处于空闲状态。在这种情况下，`BACKWARD_PRE` 可以带来 20% 到 40% 的吞吐量 (throughput)提升。

相反，在具有大带宽的集群中（例如带有 NVLink 和 InfiniBand 的 NVIDIA DGX 系统），$T_{通信}$ 非常小。`AllGather` 操作可能会几乎立即完成。在这种情况下，积极预取会产生递减的回报，并且可能只是降低内存可用性而不会增加吞吐量。

对您的特定硬件配置进行性能分析时，请观察 PyTorch Profiler 中的“NCCL 等待”内核时间。如果这些等待时间在反向传播期间明显，启用 `BACKWARD_PRE` 是优化的主要手段。如果等待时间可忽略不计，则优先考虑内存节省，通过使用 `BACKWARD_POST` 或禁用预取来允许更大的批处理大小。
