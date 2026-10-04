---
course: "lora-peft-efficient-llm-training"
chapter: "peft-optimization-deployment"
lesson: "peft-optimizers-schedulers"
sourceId: 4158
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-5-peft-optimization-deployment/peft-optimizers-schedulers"
title: "用于PEFT的优化器和学习率调度器"
description: "讨论适用于PEFT的优化器（AdamW，8比特Adam）和学习率调度策略。"
order: 2
plots: ["plots/4158-0.json", "plots/4158-1.json"]
sourceHash: "c17e36c685dfee8a87c911c168728619d9b55d2cc1353482a3f917966a803f82"
sourceCorrections: []
---

选择合适的优化器和学习率调度器是模型训练成功的核心，这对于参数 (parameter)高效微调 (fine-tuning)（PEFT）也同样适用。尽管PEFT相较于完全微调能大幅减少可训练参数的数量，但更新这些低秩矩阵或适配器层的动态变化需要仔细考量优化策略，以确保稳定收敛和理想性能。PEFT工作流程中常用的优化器和调度技术在此说明，并强调平衡性能与内存效率的选择。

### PEFT优化器的选择

优化器的作用是根据计算出的梯度更新可训练参数 (parameter)（例如，LoRA矩阵 $A$ 和 $B$、适配器权重 (weight)）。尽管存在许多优化器，但某些选择因其经验表现和内存效率上的调整而成为PEFT的通用选择。

#### AdamW：可靠的通用选择

AdamW仍然是深度学习 (deep learning)（包括PEFT）中非常流行且高效的优化器。它为每个参数单独调整学习率，引入动量（追踪过往梯度）并基于过往梯度平方进行缩放。与原始Adam优化器相比，它的独特之处在于改进了权重衰减的处理。AdamW不是将权重衰减与基于梯度的更新混合，而是在主优化步骤*之后*直接应用于权重，这通常能带来更好的泛化能力。

对于PEFT，AdamW提供了一个基准。其自适应特性有助于处理不同PEFT参数的敏感性。典型配置包括设置学习率、beta参数（$\beta_1$，$\beta_2$）、epsilon（$\epsilon$）和权重衰减。

- **学习率：** 通常需要调整。PEFT有时可能容忍比完全微调 (fine-tuning)稍高的学习率，但从微调常用的值（例如 $1e-4$、$5e-5$、$2e-5$）开始是一个合理的做法。
- **Betas：** $\beta_1$（动量衰减）通常设为0.9左右，$\beta_2$（RMSprop衰减）通常设为0.999。这些值通常开箱即用，表现良好。
- **Epsilon：** 一个小值（$1e-8$），用于数值稳定性。
- **权重衰减：** 一个正则化 (regularization)项。0.01或0.1等值是常见的起始点，但由于PEFT可训练参数较少，其影响可能不那么明显。进行调整甚至将其设为0通常是可行的。

#### 内存高效优化器：8比特Adam

PEFT的一个重要优点是减少内存需求。AdamW等标准优化器为每个可训练参数维护内部状态（动量和方差估计），通常采用32位浮点精度。即使PEFT中的可训练参数较少，这些优化器状态仍会占用大量GPU内存，特别是在与梯度累积或大批量结合使用时。

为此，内存高效的优化器变体已得到开发。PEFT（特别是QLoRA）中使用最突出的例子是**8比特Adam**。此变体量化 (quantization)优化器状态，使用8位数据类型而非32位进行存储。这大幅减少了优化器的内存占用，通常减少近4倍。

工作原理：

1. 优化器照常使用32位精度计算更新。
2. 内部状态（动量、方差）被量化为8位表示，以便在步骤之间存储。这涉及分块量化等技术，以便在最重要的部分保持精度。
3. 在下一次更新之前，8位状态被反量化回32位精度进行计算。

这种方法在大多数PEFT场景中提供了大量的内存节省，同时对收敛行为或最终模型性能的影响微乎其微。`bitsandbytes`等库提供了与PyTorch和Hugging Face的`transformers`等框架集成的实现。使用8比特Adam通常像在训练器配置期间指定不同的优化器名称或类一样简单。



![优化器内存占用](plots/4158-0.json)



> 优化器状态的相对内存使用量比较。8比特Adam相较于标准32比特AdamW，大幅减少了所需的内存。

#### 其他优化器

尽管AdamW及其8比特变体普遍使用，但也可以考虑其他优化器，如**Adafactor**。Adafactor旨在通过分解二阶矩估计矩阵来提高内存效率，从而避免存储整个矩阵。它可以在不要求明确量化库（如`bitsandbytes`）的情况下提供内存节省。然而，其相对于AdamW的性能对超参数 (hyperparameter)选择更敏感，特别是学习率调度。

### 学习率调度策略

学习率决定了参数 (parameter)更新时的步长。固定的学习率很少是理想的。学习率调度器在训练期间动态调整学习率，通常从较高值开始并随时间降低。这有助于实现更快的初始进展，并在训练收敛时进行更精细的调整。

#### PEFT常用的学习率调度

1. **恒定学习率：** 最简单的方法，在整个训练过程中使用固定的学习率。虽然易于配置，但通常导致次优结果。它对于非常短的微调 (fine-tuning)运行或特定调试场景可能是可接受的。
2. **线性衰减：** 学习率在训练过程中从初始值线性下降到最终值（通常为0）。这是一种常见且有效的策略。
3. **余弦衰减：** 学习率遵循余弦曲线，从初始值下降到最小值（例如0）。此调度在训练的开始和结束时学习率下降得更慢，而在中间更快。实践中通常表现稍好于线性衰减。
4. **热身阶段：** 强烈推荐将衰减调度（线性或余弦）与初始热身阶段结合。在热身期间，学习率从非常低的值（通常为0）开始，并在指定步数（例如总训练步数的5-10%）内线性增加到其目标初始值。这种初始的平缓阶段有助于稳定训练，防止模型快速适应时早期出现可能具有破坏性的大幅更新。



![常用学习率调度](plots/4158-1.json)



> 学习率在10步热身阶段后，接着是100总步数内的线性或余弦衰减的变化曲线。热身后的初始学习率归一化 (normalization)为1。

#### PEFT特定的调优注意事项

- **超参数 (hyperparameter)敏感性：** 理想的初始学习率、热身持续时间和衰减策略可能很大程度上取决于具体的PEFT方法（LoRA vs. 适配器）、LoRA中的秩（`r`）、数据集以及基础模型。通常需要进行实验。
- **与批量大小的关系：** 有效批量大小（全局批量大小 = 单设备批量大小 × 梯度累积步数 × 设备数量）对学习率有影响。更大的有效批量大小通常允许稍高的稳定学习率。
- **训练时长：** PEFT微调通常比完全微调需要更少的训练步数。请确保学习率调度（尤其是衰减和热身步数）适当调整以适应总计划训练步数。

### 实际操作

大多数深度学习 (deep learning)框架和库（如Hugging Face的`transformers`）提供了便捷的优化器和学习率调度器配置方式。使用`Trainer` API时，您通常在`TrainingArguments`中指定优化器类型（例如`'adamw_torch'`、`'adamw_bnb_8bit'`）、学习率、权重 (weight)衰减以及学习率调度器类型（例如`'linear'`、`'cosine'`）和热身步数。

```python
# Hugging Face Trainer参数使用示例
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # 有效批量大小 = 4 * 4 = 16
    optim="adamw_bnb_8bit",        # 使用8比特AdamW
    learning_rate=2e-4,            # 初始学习率
    lr_scheduler_type="cosine",    # 余弦衰减调度
    warmup_steps=50,               # 热身步数
    weight_decay=0.01,
    max_steps=500,                 # 总训练步数
    fp16=True,                     # 使用混合精度（PEFT常用）
    # ... 其他参数
)
```

选择和调整优化器和学习率调度是一个迭代过程。从既定默认值（AdamW或8比特Adam，带热身的余弦/线性衰减）开始，提供了一个强大的起始点。密切监控训练损失和验证指标，以指导调整，从而在您的PEFT设置中获得最佳性能。

## 参考资料

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov and Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  介绍AdamW，Adam的一种变体，它将权重衰减独立于梯度更新应用，以获得更好的泛化能力。
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235) — Noam Shazeer and Mitchell Stern (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1804.04235](https://doi.org/10.48550/arXiv.1804.04235)
  描述了Adafactor，一种通过因子化二阶矩估计来设计内存高效的优化器。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov and Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  介绍余弦退火学习率调度，常与热身阶段结合以改善训练动态。
