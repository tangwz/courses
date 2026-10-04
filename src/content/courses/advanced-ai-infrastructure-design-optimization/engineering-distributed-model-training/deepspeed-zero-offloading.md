---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "engineering-distributed-model-training"
lesson: "deepspeed-zero-offloading"
sourceId: 6982
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-2-engineering-distributed-model-training/deepspeed-zero-offloading"
title: "使用 Microsoft DeepSpeed 实现 ZeRO 和卸载"
description: "使用 DeepSpeed 的零冗余优化器（ZeRO）高效训练庞大模型。"
order: 4
plots: []
sourceHash: "f574f17fc7347b3d18a715e1196bc0ff7264979b8d4e742c70678398a6a77cf9"
sourceCorrections: []
---

尽管 PyTorch 的 `DistributedDataParallel` (DDP) 中实现的标准数据并行对于提升吞吐量 (throughput)很有效，但它引入了显著的内存瓶颈。数据并行组中的每个 GPU 都维护着模型参数 (parameter)、梯度和优化器状态的完整副本。对于大型模型，特别是那些使用 Adam 等优化器（会存储动量和方差）的模型，仅优化器状态所需的内存就可能超过模型参数大小的两倍或更多。这种复制在 GPU 计算能力达到饱和之前很久就成了限制因素。

Microsoft 的 DeepSpeed 库通过其零冗余优化器 (ZeRO) 直接解决了这种内存效率低下问题。ZeRO 不会复制整个训练状态，而是将其划分到可用数据并行设备上。这使得您可以训练比单个 GPU 所能容纳的模型大许多倍的模型，而无需依赖于纯粹的模型并行或流水线并行的复杂性。

### 内存冗余问题

为了充分理解 ZeRO 的作用，让我们细分标准数据并行训练期间单个 GPU 上的内存消耗。总内存 $M$ 可以建模为三个主要部分的和：


$$
M = M_P + M_G + M_O
$$


各部分为：

- $M_P$ 是模型参数 (parameter)的内存。
- $M_G$ 是梯度的内存，其大小通常与 $M_P$ 相同。
- $M_O$ 是优化器状态的内存。对于 Adam 这样的优化器，这包括动量和方差，通常导致 $M_O \ge 2 \times M_P$。

在包含 $K$ 个 GPU 的标准 DDP 配置中，每个 GPU 都持有这三个部分的完整副本，导致大量的冗余。

> 标准数据并行中的内存布局，每个 GPU 都持有模型参数、梯度和优化器状态的完整副本。

### ZeRO：训练状态划分

ZeRO 通过在 GPU 之间分片模型和优化器状态来系统地消除这种冗余。它分三个渐进阶段实现，让您选择最适合需求的优化级别。

#### ZeRO 阶段 1 (ZeRO-1)

第一阶段侧重于 Adam 等优化器最主要的内存冗余来源：优化器状态。ZeRO-1 在数据并行进程之间划分优化器状态。每个 GPU 现在仅负责更新其分配的参数 (parameter)分区。在优化器步骤中，`all-gather` 操作会收集所有更新的参数分片，以确保每个 GPU 都有一个完整且最新的模型，用于下一个前向传播。

- **划分：** 优化器状态 ($M_O$)。
- **内存节省：** 显著，尤其适用于 Adam/AdamW。

#### ZeRO 阶段 2 (ZeRO-2)

ZeRO-2 在阶段 1 的基础上，进一步划分梯度。在反向传播 (backpropagation)期间，不再是每个 GPU 持有完整梯度集然后通过 `all-reduce` 操作对其进行平均，而是使用 `reduce-scatter` 操作。此操作同时计算平均值并将分片结果分散到对应的 GPU。这避免了同时存储完整梯度和参数所导致的瞬时内存峰值。

- **划分：** 优化器状态 ($M_O$) + 梯度 ($M_G$)。
- **内存节省：** 大量。消除梯度冗余。

#### ZeRO 阶段 3 (ZeRO-3)

这是最先进的阶段，使得训练真正庞大的模型成为可能。ZeRO-3 划分所有内容：优化器状态、梯度和模型参数本身。每个 GPU 仅持有一个整个模型的切片。

在前向和反向传播期间，ZeRO-3 仅在计算需要时才在每个 GPU 上动态重构模型的完整层。一个层的参数在执行之前会从所有参与的 GPU 收集，并在之后立即回收内存。这意味着在任何给定时间，内存峰值使用量与单个层的大小成比例，而不是与整个模型成比例。

- **划分：** 优化器状态 ($M_O$) + 梯度 ($M_G$) + 参数 ($M_P$)。
- **内存节省：** 最大。使得支持万亿参数的模型成为可能。

> ZeRO-3 的内存布局：参数 (P)、梯度 (G) 和优化器状态 (O) 全部划分到可用 GPU 上。

### ZeRO-Offload：将内存扩展到 CPU 和 NVMe

即使使用 ZeRO-3，一个庞大模型的总状态大小也可能超过集群中所有可用 GPU 内存的总和。ZeRO-Offload 通过将训练状态的某些部分移动到更多、但速度较慢的内存层级来扩展划分层级。

- **CPU 卸载：** 您可以将 DeepSpeed 配置为将划分的优化器状态，甚至参数 (parameter)，卸载到主机 CPU 的内存中。在训练步骤中，所需数据会从 CPU 移动到 GPU 进行计算，然后移回。这对于优化器步骤尤其有效，该步骤通常计算量小但内存需求大。
- **NVMe 卸载：** 对于最极端的情况，DeepSpeed 可以将参数和优化器分区卸载到快速 NVMe 固态硬盘。这提供了可用内存的极大扩展，使得支持万亿参数模型成为可能，但代价是与 CPU 内存相比，I/O 延迟显著更高。

这种卸载能力让大型模型训练变得普及，使得在 GPU 显存 (VRAM)有限但拥有大量系统内存或快速存储的系统上也能实现训练。

### DeepSpeed 的实际实现

将 DeepSpeed 集成到 PyTorch 训练脚本中非常直接。主要改变包括初始化 DeepSpeed 引擎以及修改训练循环以使用 DeepSpeed 的方法。

首先，您创建一个 `ds_config.json` 文件。此文件是所有 DeepSpeed 功能的控制面板。

**ZeRO-2 带 CPU 卸载的 `ds_config.json` 示例：**

```json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "steps_per_print": 100,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 0.0001,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "fp16": {
    "enabled": true
  }
}
```

接下来，您修改您的训练脚本。重点是 `deepspeed.initialize` 函数，它会封装您的模型和优化器。

```python
import torch
import deepspeed

# 假设模型和优化器已定义
# model = MyTransformerModel()
# optimizer = torch.optim.AdamW(model.parameters())

# DeepSpeed 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config='ds_config.json'
)

# 训练循环
for step, batch in enumerate(data_loader):
    # 将批次移动到正确的设备
    batch = to_device(batch, model_engine.local_rank)

    # 前向传播
    loss = model_engine(batch)

    # 反向传播 - 使用模型引擎
    model_engine.backward(loss)

    # 优化器步骤 - 使用模型引擎
    model_engine.step()
```

注意训练循环中的变化：

1. `deepspeed.initialize` 返回一个 `model_engine`，它会替换您原始的模型以进行主要操作。
2. `loss.backward()` 调用被 `model_engine.backward(loss)` 替换。
3. `optimizer.step()` 调用被 `model_engine.step()` 替换。

DeepSpeed 根据您提供的配置处理所有底层分片、通信和卸载的复杂性。这种简洁的 API 使您只需更改 JSON 配置，即可尝试不同的 ZeRO 阶段和卸载策略，而无需更改您的核心训练逻辑。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://doi.org/10.1145/3418856.3418915) — Samyam Rajbhandari, Cong Guo, Jeff Rasley, Shaden Smith, Yuxiong He (2020)
  Journal: SC20: International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; DOI: [10.1145/3418856.3418915](https://doi.org/10.1145/3418856.3418915)
  介绍了零冗余优化器 (ZeRO) 及其三个阶段，用于高效扩展深度学习模型训练的内存。
- [DeepSpeed Official Documentation](https://www.deepspeed.ai/) — Microsoft DeepSpeed Team (2024)
  提供了 DeepSpeed 库的全面指南和 API 参考，包括 ZeRO 配置和卸载功能。
- [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — PyTorch Core Team (2024)
  PyTorch 标准分布式数据并行 (DDP) 的官方文档，有助于理解基线内存复制问题。
