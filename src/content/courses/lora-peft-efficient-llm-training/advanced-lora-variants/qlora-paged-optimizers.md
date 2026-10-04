---
course: "lora-peft-efficient-llm-training"
chapter: "advanced-lora-variants"
lesson: "qlora-paged-optimizers"
sourceId: 4151
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/qlora-paged-optimizers"
title: "分页优化器提升内存效率"
description: "说明分页优化器（例如AdamW 8位）如何在QLoRA训练期间进一步降低内存使用。"
order: 5
plots: []
sourceHash: "337089a176b8f5deb5bd55e4923dcec4be9348fda4ddcd01e4547d82571849cd"
sourceCorrections: []
---

尽管量化 (quantization)LoRA (QLoRA) 通过使用4位量化大幅减少了基础模型权重 (weight)的内存占用，但训练过程中所需的优化器状态仍可能是显著的内存瓶颈。像AdamW这样的标准优化器会为每个可训练参数 (parameter)维护多个状态（例如，动量和方差估计）。即使LoRA训练的参数远少于全量微调 (fine-tuning)，优化器仍可能根据*原始*模型大小分配内存，或者至少为LoRA参数本身需要大量内存，尤其是在规模很大的模型上。

这时分页优化器便发挥作用了，它提供了一种互补的方法，可进一步降低微调过程中GPU内存消耗，使QLoRA更加容易使用。

### 优化器状态的内存挑战

以AdamW优化器为例。对于每个被训练的参数 (parameter)，它通常存储：

1. 参数梯度。
2. 一阶矩估计（动量）。
3. 二阶矩估计（方差）。

如果这些状态以32位精度（FP32）存储，每个参数会占用12字节（4字节用于梯度 + 4字节用于动量 + 4字节用于方差）。虽然LoRA与全量模型相比显著减少了可训练参数的*数量*，但优化器状态内存仍可能很大，尤其是在微调 (fine-tuning)具有数十亿参数的模型时，即使只有一小部分通过LoRA进行调整。这种内存使用量随可训练参数数量增加而扩展，即使基础模型本身由于量化 (quantization)而适配，也可能阻止在VRAM有限的GPU上训练。

### 分页优化器的工作原理

分页优化器，以`bitsandbytes`等库中提供的8位AdamW实现为例，通过使用CPU内存作为优化器状态的溢出缓冲区来解决这一挑战。其核心思想类似于操作系统中的虚拟内存管理：

1. **量化 (quantization)（可选但常见）：** 通常，分页优化器也会对优化器状态本身进行量化（例如，到8位精度），立即减小其固有大小。
2. **CPU卸载：** 大部分优化器状态（可能已量化）驻留在固定CPU内存中。固定内存有助于CPU与GPU之间更快的数据传输。
3. **GPU分页：** 只有当前计算所需的特定优化器状态（即，涉及当前小批量梯度更新的参数 (parameter)的状态）才会在被需要之前从CPU内存传输（“页入”）到GPU显存 (VRAM)。
4. **计算：** 优化器步长计算在GPU上使用已页入的状态进行。
5. **写回：** 更新后的状态可能会被写回CPU内存（“页出”）。

这种动态的数据移动确保了在任何给定时刻，只有一小部分全部优化器状态需要存在于GPU显存中，从而大幅降低了优化器在GPU上的峰值内存需求。

> QLoRA训练期间分页优化器设置中的数据流。大部分优化器状态驻留在CPU内存中，仅在计算需要时才被页入GPU显存，从而最大限度地减少优化器带来的GPU内存开销。

### 与QLoRA的协作

分页优化器与QLoRA结合使用时尤其有效：

- **互补的内存节省：** QLoRA解决模型权重 (weight)内存问题，而分页优化器解决优化器状态内存问题。它们共同显著降低了微调 (fine-tuning)大型模型的门槛。
- **支持更大规模：** 这种组合让使用者能够微调那些否则在典型硬件上完全无法实现的模型，或者可以使用更大的批处理大小以在现有硬件上获得可能更快的收敛或更好的稳定性。

### 实现与注意事项

使用分页优化器通常涉及在设置训练循环时指定特定优化器实现。例如，使用`bitsandbytes`库时，您可能会选择AdamW的8位变体。

```python
# 使用训练器设置的示例
# 注意：实际实现取决于框架/库（例如，Hugging Face Transformers，自定义PyTorch循环）

import bitsandbytes.optim as bnb_optim

# ... 模型设置（已启用QLoRA） ...
# ... 训练参数 ...

# 不使用torch.optim.AdamW，而是使用bitsandbytes版本
optimizer = bnb_optim.AdamW8bit(
    model.parameters(),
    lr=training_args.learning_rate,
    # 其他AdamW参数...
    # bitsandbytes特定的参数可能可用，例如用于块大小的参数
)

# ... 训练循环的其余部分 ...
```

**注意事项：**

- **吞吐量 (throughput)：** 尽管内存节省是可观的，但CPU-GPU数据传输会引入一些开销。这可能导致与使用完全适合GPU显存 (VRAM)的标准优化器相比，训练吞吐量（每秒样本数）略有降低。然而，能够进行训练，或者使用更大的批处理大小，通常超过这种适度的速度降低。
- **CPU内存：** 确保您的系统有足够的CPU内存来容纳卸载的优化器状态。所需量取决于可训练参数 (parameter)的数量和优化器状态精度（例如，8位与32位）。
- **库依赖：** 实现依赖于特定库（如`bitsandbytes`）在您的硬件环境（CUDA版本等）中正确安装和配置。

总之，分页优化器代表了使大型模型微调 (fine-tuning)更高效和易于使用的又一个重要进展。通过智能管理CPU和GPU之间的优化器状态内存，它们与QLoRA等技术配合使用，支持在资源受限的硬件上进行高级微调流程。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS 2023); DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  介绍了QLoRA方法，该方法结合4位量化和分页优化器，显著减少了LLM微调所需的GPU内存。
- [\`bitsandbytes\` Library Documentation](https://github.com/bitsandbytes-foundation/bitsandbytes) — bitsandbytes-foundation and Contributors (2024)
  提供了在`bitsandbytes`库中实现8位优化器（包括AdamW8bit）的实用指南和示例，以支持内存高效的LLM训练。
