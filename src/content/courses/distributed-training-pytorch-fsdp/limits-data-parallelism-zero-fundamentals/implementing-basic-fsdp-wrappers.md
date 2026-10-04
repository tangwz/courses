---
course: "distributed-training-pytorch-fsdp"
chapter: "limits-data-parallelism-zero-fundamentals"
lesson: "implementing-basic-fsdp-wrappers"
sourceId: 7887
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-1-limits-data-parallelism-zero-fundamentals/implementing-basic-fsdp-wrappers"
title: "实现基础FSDP包装器"
description: "关于使用FSDP基础配置包装标准PyTorch模型的实践说明。"
order: 4
plots: []
sourceHash: "50330f3339a4c4acda23933e0c9edde82a71ab90d631091452d3a0cb6e9a1ef7"
sourceCorrections: []
---

FullyShardedDataParallel (FSDP) 与 DistributedDataParallel (DDP) 相比，为分布式训练提供了一种不同做法，不只简单的类替换。虽然它们的API表面看起来相似，但参数 (parameter)管理的核心机制有显著区别。在DDP中，模型包装器主要在桶级别处理梯度同步（all-reduce）。在FSDP中，包装器拥有参数，并在进程组中实际拆分张量。

本节说明FSDP的编程实现，着重于`ShardingStrategy`配置和`MixedPrecision`策略，它们是在大规模训练中保持稳定所必需的。

### FSDP构造器接口

分片操作的入口是`FullyShardedDataParallel`类。不同于DDP接受一个已在目标GPU上的模型，FSDP通常包装位于CPU上的模型，以避免初始化时立即出现内存不足（OOM）错误。

构造器签名提供了对理论分析中提到的ZeRO阶段的控制。最重要的参数 (parameter)是`sharding_strategy`，它决定了模型状态如何划分。

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)

# 基础包装结构
model = FSDP(
    module,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mixed_precision_policy,
    device_id=torch.cuda.current_device()
)
```

### ZeRO阶段到分片策略的映射

PyTorch将ZeRO优化的数学定义映射到`ShardingStrategy`枚举。选择正确的策略在于内存限制和通信带宽之间的平衡。

1. **`FULL_SHARD` (ZeRO 阶段 3)**: 这是默认行为。参数 (parameter)、梯度和优化器状态都被分片。参数只在计算特定层的正向和反向传播 (backpropagation)期间在GPU上具体化（聚集），然后立即释放。这提供了最大的内存节省（$1/N$ 缩放），但由于频繁的`AllGather`操作而产生高通信开销。
2. **`SHARD_GRAD_OP` (ZeRO 阶段 2)**: 梯度和优化器状态被分片，但参数保持复制状态。此策略避免了正向传播期间聚集权重 (weight)的通信开销，但需要足够的显存 (VRAM)来保存完整的模型参数（$\Psi$）。
3. **`NO_SHARD` (DDP 等效)**: 此模式在FSDP API中复制DDP的行为。它有助于调试，或者在缩减到少量GPU时使用，因为那时分片开销超过了其优势。

以下图表说明了这些策略在双GPU设置中的内存分配区别。

> 标准复制和完全分片之间的内存分配比较。在`FULL_SHARD`中，更新的存储和计算都已分布，设备占用量随大小线性减少。

### 混合精度配置

不同于`torch.cuda.amp`通常需要外部缩放器和autocast上下文 (context)管理器，FSDP直接将混合精度整合到分片逻辑中。这种整合是必要的，原因在于FSDP在跨等级通信张量之前必须知道精度格式。如果通信采用FP32而计算采用BF16，则会浪费带宽。

`MixedPrecision`配置类控制三种特定数据类型：

1. **`param_dtype`**: 正向和反向计算期间用于模型参数 (parameter)的类型。
2. **`reduce_dtype`**: 用于梯度归约（通信）的类型。
3. **`buffer_dtype`**: 缓冲区（例如，批量归一化 (normalization)统计数据）的类型。

对于Ampere或Hopper架构上的现代大型语言模型（LLM）训练，`bfloat16`是标准。它保留了FP32的指数范围，避免了FP16常见的下溢问题，通常无需损失缩放。

```python
import torch
from torch.distributed.fsdp import MixedPrecision

# 定义BFloat16训练策略
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,  # 将通信量减少50%
    buffer_dtype=torch.bfloat16,
)
```

当`param_dtype`设置为`bfloat16`时，FSDP会保留FP32主权重 (weight)（如果存在），但在正向传播之前将其转换为BFloat16。这与混合精度训练中的“主权重”思想一致，确保优化器步骤中的小权重更新不会因精度截断而丢失。

### 扁平参数 (parameter)与正向传播

当你用FSDP包装一个模块时，原始参数（例如`model.layer1.weight`）会被一个通用`FlatParameter`替换。这是一个单一的1D张量，将多个原始参数的存储视为一个整体。

这种扁平化提高了内存访问模式和通信效率。FSDP没有为单个权重 (weight)矩阵启动数百个小型NCCL内核，而是将它们聚合为更大的块。

然而，这带来了一个限制：包装后你不能直接访问`model.layer1.weight`，因为该属性实际上不再以原始形式存在于设备上。访问它需要使用上下文 (context)管理器`FSDP.summon_full_params(model)`，我们将在检查点部分进行说明。

### 完整实现示例

以下示例展示了一种初始化模式。它建立进程组，定义模型，并使用`FULL_SHARD`和BFloat16精度将其包装。

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 使用NCCL后端初始化GPU的进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_fsdp_step(rank, world_size):
    setup(rank, world_size)

    # 1. 定义模型 (标准PyTorch)
    # 在实际场景中，这很可能是一个Transformer
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024)
    )

    # 2. 定义策略
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # 3. 用FSDP包装
    # 注意: 我们通过device_id参数隐式地移动到设备。
    # 对于大型模型，首先加载到'meta'设备或CPU。
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=bf16_policy,
        device_id=torch.cuda.current_device(),
    )

    # 4. 优化器初始化
    # 重要: 优化器必须在包装后初始化
    # 因为FSDP会修改参数结构。
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3)

    # 5. 训练循环
    # 在正确的设备上生成模拟数据
    inputs = torch.randn(64, 1024).to(rank).bfloat16()

    optimizer.zero_grad()
    output = fsdp_model(inputs)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    print(f"Rank {rank} step complete. Loss: {loss.item()}")

    cleanup()

# 这个函数将在脚本中通过torch.multiprocessing启动
# mp.spawn(train_fsdp_step, args=(world_size,), nprocs=world_size)
```

### 设备初始化与`device_id`

在上面的示例中，`device_id`参数 (parameter)非常要紧。如果省略，FSDP可能会尝试在CPU或默认GPU上初始化分片，可能导致无声的性能下降或放置错误。

当模型能适应单个GPU（如示例所示）时，包装一个已具体化的模型是可以接受的。然而，对于接近TB级别的模型，在分片前将完整模型具体化到CPU内存是不可能的。在这些情况下，我们采用`meta`设备进行延迟初始化，允许FSDP分片参数，而无需在任何单个设备上分配完整的模型状态。这种高级初始化模式是下一章的内容。
