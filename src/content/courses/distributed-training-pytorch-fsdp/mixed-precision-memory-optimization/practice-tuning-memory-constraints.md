---
course: "distributed-training-pytorch-fsdp"
chapter: "mixed-precision-memory-optimization"
lesson: "practice-tuning-memory-constraints"
sourceId: 7897
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/practice-tuning-memory-constraints"
title: "实践：调整内存限制"
description: "通过精度缩放和卸载，在受限硬件上最大化批次大小的实战操作。"
order: 5
plots: ["plots/7897-0.json"]
sourceHash: "171f30f9804a3965e1682c241808322619778c0d1cc78abf3f46c1954965081c"
sourceCorrections: []
---

用数十亿参数 (parameter)的模型在标准 GPU 硬件上训练，常常让人觉得像是在解一个棋子比棋盘大的谜题。当模型参数量扩展到 7B 或 13B 时，优化器状态、梯度和参数的总占用空间会迅速占满即使是 A100-80GB 显卡的 HBM。为解决此问题，内存优化方法被系统地应用，以便在有限的硬件预算下适应大型 Transformer 模型，同时目标是最大化有效批次大小。

### 大型模型的内存计算

在应用修补前，我们需量化 (quantization)瓶颈所在。以训练一个 7B 参数 (parameter)模型为例。在标准 FP32 训练中，仅模型状态就需要大量内存：

- **参数：** $7 \times 10^9 \times 4 \text{ 字节} \approx 28 \text{ GB}$
- **梯度：** $7 \times 10^9 \times 4 \text{ 字节} \approx 28 \text{ GB}$
- **优化器状态 (Adam)：** $7 \times 10^9 \times 8 \text{ 字节} \approx 56 \text{ GB}$

仅模型状态就总计 112 GB，这超出了一张 GPU 的容量。FSDP (ZeRO-3) 通过将这些状态分片到 $N$ 个 GPU 来解决此问题。在一个 8-GPU 集群上，每个 GPU 的占用空间下降到 $\approx 14 \text{ GB}$。然而，这个计算没有计入**激活值**，即在前向传播期间为梯度计算而生成的瞬态数据。激活值内存与序列长度和批次大小呈线性关系，即使模型状态完美分片，也经常导致内存不足（OOM）错误。

我们将执行一个调整流程来回收内存，优先采用保持训练吞吐量 (throughput)的技术，然后才使用那些会带来通信开销的方法。



![显存占用缩减策略](plots/7897-0.json)



> 随着优化的应用，内存节省的进展。请注意，CPU 卸载会大幅减少优化器状态在 GPU 上的驻留时间，但会引入延迟。

### 步骤 1：精度缩减

首要的应对方法是从 FP32 切换到混合精度 (BFloat16)。这会将参数 (parameter)和梯度所需的内存减少一半，并大幅降低激活值的占用空间。在 FSDP 中，我们通过 `MixedPrecision` 策略来配置。请注意，为了优化器步骤中的数值稳定性，我们通常将主权重 (weight)保留在 FP32 中，但前向和反向传播 (backpropagation)发生在 BF16 中。

```python
from torch.distributed.fsdp import MixedPrecision
import torch

# 定义混合精度策略
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    # BF16 中的梯度通信减少了总线带宽使用
    reduce_dtype=torch.bfloat16,  
    # 缓冲区精度影响 LayerNorm 等
    buffer_dtype=torch.bfloat16   
)

# 在 FSDP 包装期间应用
model = FSDP(
    model,
    mixed_precision=bf16_policy,
    # ... 其他配置
)
```

通过设置 `reduce_dtype=torch.bfloat16`，我们也将 `AllReduce` 通信量减少了一半，从而提高了带宽受限互连上的吞吐量 (throughput)。如果您的损失曲线显示出不稳定性，可以考虑将 `buffer_dtype` 保持为 `torch.float32`，以在归一化 (normalization)层中保留更高的精度。

### 步骤 2：用计算换内存

如果混合精度不足以适应您期望的批次大小，下一个合乎逻辑的步骤是**激活检查点（AC）**。大型 Transformer 模型由重复的相同层组成。AC 在前向传播期间丢弃这些层的中间激活值，并在反向传播 (backpropagation)期间重新计算它们。

这有效地将激活值的内存复杂程度从 $O(N)$（其中 $N$ 是层数）减少到大约 $O(\sqrt{N})$ 或每层恒定，具体取决于粒度。代价是计算开销增加 20-25%，但这通常是一个值得的权衡，能够实现 $2\times$ 或 $4\times$ 更大的批次大小。

在 PyTorch FSDP 中，我们使用 `apply_activation_checkpointing` 来应用 AC。**在**将模型包装到 FSDP 中**之前**应用此功能非常关键，以确保包装器钩子正确注册。

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# 定义一个检查函数来识别 Transformer 块
# 假设标准的 Transformer 架构类名
check_fn = lambda submodule: isinstance(submodule, TransformerBlock)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=check_fn,
)

# 在应用检查点后用 FSDP 包装
model = FSDP(model, ...)
```

这里的核心实现细节是 `check_fn`。您必须瞄准重复的解码器/编码器块（例如 `GPT2Block`、`LlamaDecoderLayer`）。对较小的层（如单个线性层）进行检查点会增加开销，而不会带来显著的内存节省。

### 步骤 3：突破显存 (VRAM)限制

当训练超大型模型或使用显存有限的 GPU（例如，24GB 消费级显卡）时，您可能需要将优化器状态和参数 (parameter)卸载到 CPU。这会使用大得多的系统 RAM（通常 512GB+）作为扩展存储。

这会带来严重的性能损失，因为 PCIe 带宽是瓶颈。数据必须从 CPU 传输到 GPU 进行计算，再返回 CPU 进行更新。这种方法通常仅用于模型无法在 GPU 集群上运行的情况。

```python
from torch.distributed.fsdp import CPUOffload

# 启用 CPU 卸载
offload_policy = CPUOffload(offload_params=True)

model = FSDP(
    model,
    cpu_offload=offload_policy,
    # ... 其他配置
)
```

使用 CPU 卸载时，请确保在通用数据加载器中将 `pin_memory=True` 设置为 `True`，以加速主机到设备的传输。

### 调整流程

获得最佳性能需要迭代的方法。不要盲目开启所有功能。遵循此决策逻辑，以最大化模型浮点运算利用率 (MFU)。

> 逐步启用内存优化的决策流程。目标是尽可能停留在右侧路径（增加批次大小），除非必要，否则不进入高开销的 CPU 卸载区域。

您应致力于最大化每个 GPU 的微批次大小。更大的微批次大小通常能提高 GPU 内核的利用率。一旦 GPU 显存 (VRAM)接近满载，使用**梯度累积**来达到收敛所需的全局目标批次大小。

例如，如果您的全局目标批次大小是 4M tokens，并且您在 8-GPU 集群上每个 GPU 只能容纳 4k tokens：
$\text{全局批次} = \text{微批次} \times \text{进程数} \times \text{累积步数}$
$4,000,000 = 4,000 \times 8 \times 125$

您将梯度累积步数设置为 125。这使您可以使用较大的有效批次大小进行训练，而无需增加瞬时内存需求。请注意，过多的梯度累积步数有时会导致训练变慢，原因在于累积逻辑的开销以及通信重叠机会的减少，因此在微批次大小和累积步数之间找到平衡是调整实践的一部分。
