---
course: "introduction-to-llm-fine-tuning"
chapter: "full-parameter-fine-tuning"
lesson: "managing-computational-resources"
sourceId: 7276
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-3-full-parameter-fine-tuning/managing-computational-resources"
title: "计算资源管理"
description: "在资源密集型全参数微调期间管理GPU内存和计算负载的方法。"
order: 3
plots: []
sourceHash: "a2007534e8c445463faec4a1c887ba8e60a000dc2ce9793816b8bcceef4d57e0"
sourceCorrections: []
---

全参数 (parameter)微调 (fine-tuning)对您的硬件提出了较高的要求，其中GPU内存（VRAM）是最常见的瓶颈。一个70亿参数的模型，如Llama 3 8B，在标准32位精度下加载时，仅模型权重 (weight)就需要大约28 GB的VRAM（$7B \times 4 \text{ 字节/参数}$）。这个数字甚至没有考虑训练期间梯度、优化器状态和激活所需的额外内存。如果不妥善处理，即使在高端GPU上尝试微调此类模型，也可能很快导致内存不足错误。

本节提供实用方法来处理这些计算需求，使您能够在现有硬件上微调更大的模型，否则这是不可能的。

### VRAM用量分析

训练期间，VRAM被四个主要组成部分占用。了解这种划分是优化内存使用的首要步骤。

- **模型参数 (parameter)：** 这是存储模型权重 (weight)所需的内存。大小取决于模型大小和使用的精度（例如，`float32`、`float16`或`bfloat16`）。
- **梯度：** 对于每个模型参数，必须存储相应的梯度以进行权重更新。这些通常以与模型参数相同的精度存储。
- **优化器状态：** 现代优化器如AdamW并非无状态。它们为每个参数维护额外信息，以在训练期间调整学习率。例如，AdamW存储两种状态：过去梯度的移动平均（动量）和过去梯度平方的移动平均（方差）。这实际上使参数相关的内存占用量增加三倍（参数+梯度+优化器状态）。
- **激活与缓冲区：** 在前向传播过程中，模型计算中间值（即激活），这些值在反向传播 (backpropagation)中梯度计算时需要。这些激活的内存随批次大小、序列长度和模型架构变化。此部分也包含深度学习 (deep learning)库（如CUDA）使用的各种临时缓冲区。

> VRAM在训练步骤中如何分配的简化说明。优化器状态和激活常常占用总内存的很大一部分。

### 内存优化方法

多种技术可以结合使用，以大幅减少全参数 (parameter)微调 (fine-tuning)的内存占用量。

#### 梯度累积

梯度累积是一种允许您在不增加内存使用量的情况下模拟更大批次大小的技术。您不是在每次前向/反向传播 (backpropagation)后执行权重 (weight)更新，而是累积多个小批次的梯度，然后进行一次更新。

例如，如果您的硬件只能处理批次大小为2的情况，但您希望获得批次大小为16的训练效果，您可以将批次大小设置为2，并累积8步的梯度。这8个小批次的梯度被求和，优化器仅使用这个累积的梯度更新模型权重一次。这实现了“有效”批次大小为16，同时内存中只保留批次大小为2的激活。在Hugging Face `Trainer`中，这由`gradient_accumulation_steps`参数控制。

#### 混合精度训练

默认情况下，模型使用32位浮点数（`float32`）进行训练。混合精度训练涉及对模型的大部分操作使用16位浮点数（`float16`或`bfloat16`）。这立即将模型参数、梯度和激活所需的内存减少多达一半。

- **`float16` (fp16)：** 一种广泛支持的格式，提供可观的内存节省。然而，其较小的动态范围有时会导致数值不稳定（梯度变为零或溢出）。这通常通过一种名为“动态损失缩放”的技术自动管理。
- **`bfloat16` (bf16)：** 一种在较新GPU（NVIDIA Ampere及更新版本）上受支持的格式。它具有与`float32`相同的动态范围但精度较低，使其对下溢和上溢问题更具弹性，无需损失缩放。

使用混合精度通常是减少内存消耗最有效的方法之一。您可以通过设置`fp16=True`或`bf16=True`在`Trainer`中启用它。

#### 梯度检查点

梯度检查点是一种以计算时间换取内存的方法。如前所述，前向传播计算并存储用于反向传播的激活。梯度检查点策略性地避免存储一些中间激活。在反向传播过程中，它在需要时即时重新计算它们。虽然这会使训练步骤变慢（通常慢20-30%），但它可以带来可观的内存节省，特别是对于层数较多的模型。这在`Trainer`中通过`gradient_checkpointing=True`启用。

#### 内存高效优化器

标准AdamW优化器需要为模型中的每个参数存储两个状态值。对于一个70亿参数的模型，这意味着额外的140亿个值必须保留在显存 (VRAM)中。内存高效优化器可以减轻这种负担。一个常用选择是8位Adam，通过`bitsandbytes`库提供。它将优化器状态量化 (quantization)为8位精度，将其内存占用减少四倍。另一个选择是Adafactor，它放弃动量并使用因子分解的二阶矩估计，大幅降低其内存需求，尽管有时会对最终模型性能产生轻微影响。

### Hugging Face Trainer的实际应用

Hugging Face `Trainer` API使结合这些技术变得简单。以下是如何配置`TrainingArguments`以在内存受限的GPU上微调 (fine-tuning)模型的示例。

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    
    # Batch size and gradient accumulation
    per_device_train_batch_size=1,       # 使用能适应的最大批次大小
    gradient_accumulation_steps=16,      # 有效批次大小 = 1 * 16 = 16
    
    # Mixed-precision training
    fp16=True,                           # 启用fp16（或在支持的硬件上启用bf16=True）
    
    # Memory-efficient optimizer
    optim="paged_adamw_8bit",            # 使用bitsandbytes中的量化优化器
    
    # Gradient checkpointing
    gradient_checkpointing=True,         # 以计算换取内存
    
    # Other training parameters
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=20,
    save_steps=200,
    warmup_steps=50,
)
```

在此配置中，我们从多个角度解决内存问题：较小的单设备批次大小通过梯度累积得到弥补，参数 (parameter)和激活的内存通过`fp16`减半，优化器状态通过`paged_adamw_8bit`量化 (quantization)，激活内存通过梯度检查点进一步减少。这种组合方法通常是成功执行消费级或上一代硬件上的全参数微调所必需的。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍混合精度训练（FP16）的概念，展示其在深度学习模型中减少内存、提高速度并保持准确性的优势。
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin (2016)
  Journal: arXiv; DOI: [10.48550/arXiv.1604.06174](https://doi.org/10.48550/arXiv.1604.06174)
  提出梯度检查点（或激活检查点）技术，用计算时间换取内存，从而训练比以往更深的网络。
- [\`transformers.TrainingArguments\`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库中`TrainingArguments`的官方文档，详细介绍了与内存管理相关的参数，如`fp16`、`bf16`、`gradient_accumulation_steps`、`gradient_checkpointing`和`optim`。
