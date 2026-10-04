---
course: "lora-peft-efficient-llm-training"
chapter: "advanced-lora-variants"
lesson: "lora-initialization-strategies"
sourceId: 4144
sourceUrl: "https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/lora-initialization-strategies"
title: "LoRA 初始化策略"
description: "讨论LoRA矩阵（A和B）的不同初始化方法及其对训练稳定性的作用。"
order: 1
plots: ["plots/4144-0.json"]
sourceHash: "581bcfb39052aa270799fe36d067bd0dda9e74a10516bd5225bbb562901fd4ec"
sourceCorrections: []
---

对于复杂的LoRA应用，了解如何初始化可训练参数 (parameter)非常重要。设置低秩矩阵$A$和$B$初始值的方式会影响训练稳定性、收敛速度，并最终影响微调 (fine-tuning)模型的质量。常见的初始化策略及其作用将被审视。

回顾一下，对于权重 (weight)矩阵$W_0$，LoRA更新$\Delta W$计算为$\Delta W = BA$，其中$B \in \mathbb{R}^{d \times r}$且$A \in \mathbb{R}^{r \times k}$，对于权重矩阵$W_0 \in \mathbb{R}^{d \times k}$。微调期间修改后的权重矩阵是$W = W_0 + \alpha \frac{BA}{r}$，尽管为了简化初始化讨论，我们通常关注核心的$BA$项，同时记住缩放因子$\alpha/r$稍后应用。初始化的目标是适当地设置$A$和$B$的起始值。

### 默认初始化：B为零，A为高斯分布

最广泛采用且通常为默认的LoRA初始化策略是：

1. **初始化矩阵$A$** 使用标准随机初始化方法，通常是Kaiming均匀或高斯分布（$\mathcal{N}(0, \sigma^2)$），且方差较小。这能确保投影到低秩空间的初始表示具有一定的变动性。
2. **初始化矩阵$B$** 为全零。

这种方法的核心思想是确保在微调 (fine-tuning)过程的最开始（步骤$t=0$），自适应项$\Delta W = BA$精确为零。


$$
W_{t=0} = W_0 + BA = W_0 + 0 \cdot A = W_0
$$


这意味着带有LoRA层的模型初始行为与预训练 (pre-training)基础模型完全相同。微调过程随后逐步学习$B$的非零值，使得自适应$\Delta W$可以出现并根据特定任务数据调整模型的行为。

**优点：**

- **稳定性：** 从$\Delta W = 0$开始能避免对预训练模型精心学习到的表示造成任何初始干扰。这通常会带来更稳定的训练动态，尤其是在初始阶段。
- **保留预训练知识：** 确保微调过程完全从基础模型的状态开始，没有来自LoRA层的任何初始随机扰动。

**缺点：**

- **初始学习可能较慢：** 由于$B$从零开始，梯度信息需要回流以将$B$更新为非零值，然后才能发生有意义的自适应。这可能会使收敛的初始阶段略微慢于$\Delta W$从一开始就非零的方法。

这种策略在流行的库中默认实施，例如Hugging Face的PEFT (`peft`)。例如，`LoraLayer`通常将`lora_B`权重 (weight)初始化为零，并使用Kaiming均匀初始化来初始化`lora_A`权重。

### A和B的高斯初始化

另一种方法是使用随机分布初始化矩阵$A$和$B$，通常是精心选择（通常较小）方差$\sigma^2$的高斯分布（$\mathcal{N}(0, \sigma^2)$）。

在这种情况下，当$t=0$时，自适应项$\Delta W = BA$将是一个非零矩阵，尽管如果$\sigma$较小，其项的值也可能较小。


$$
W_{t=0} = W_0 + BA \neq W_0 \quad (\text{除非 } BA \text{ 碰巧为零})
$$


**原理：**

这里的想法是，从小的非零随机自适应开始，可能让模型更快地学习到所需的调整，从而可能加速收敛。初始的随机$\Delta W$提供了一个即时但可能带噪声的自适应方向。

**注意事项：**

- **方差选择（$\sigma^2$）：** 这是一个重要的超参数 (parameter) (hyperparameter)。如果$\sigma$过大，初始的随机$\Delta W$可能会显著干扰预训练 (pre-training)权重 (weight)$W_0$，导致训练不稳定或立即性能下降。如果$\sigma$过小，其作用可能微不足道，并类似于将$B$初始化为零。最佳方差通常需要调优。
- **与学习率的相互作用：** 与将$B$零初始化策略相比，非零的初始$\Delta W$可能需要更小的初始学习率以保持稳定性。

**优点：**

- **潜在的更快初始收敛：** 模型不需要“打破$B=0$的对称性”，并且如果随机初始化提供有用信号，则可能在初始步骤中更快地自适应。

**缺点：**

- **不稳定性风险：** 如果未适当缩放，初始随机$\Delta W$可能会干扰预训练模型的功能。
- **对超参数的敏感性增加：** 需要更仔细地调优初始化方差（$\sigma^2$）以及可能的学习率。



![LoRA初始化对更新幅度的作用](plots/4144-0.json)



> 此图说明了LoRA更新项$BA$的幅度可能如何演变。将$B$零初始化会使更新从零开始，而对$A$和$B$进行高斯初始化则会从一个小的非零更新开始。

### 实际考虑和建议

对于大多数实际使用场景，**建议**从\*\*默认策略（$B$零初始化，$A$随机初始化）\*\*开始。它提供了一个稳定可靠的基线，在微调 (fine-tuning)开始时保留预训练 (pre-training)模型的完整性。与随机初始化两个矩阵相比，这种方法对超参数 (parameter) (hyperparameter)选择的敏感性较低。

仅在以下情况下考虑对$A$和$B$进行随机初始化：

- 使用默认方法时，您观察到初始收敛特别缓慢。
- 您有足够的计算资源对初始化方差和可能的学习率进行仔细的超参数调优。
- 您正在为了研究目的试验不同变体。

记住，有效的初始更新幅度也受LoRA秩$r$和缩放因子$\alpha$的影响。更新是$W_0 + \alpha \frac{BA}{r}$。即使对$A$和$B$进行高斯初始化，非常小的$\alpha$或非常大的$r$也会减弱由$BA$引起的初始扰动。相反，大的$\alpha$或小的$r$会放大它。这些因素共同作用，同时调整它们是优化LoRA性能的一部分。

尽管可以设计其他初始化方案，或许借鉴矩阵分解技术或整合关于任务的先验知识，但$A$采用Kaiming/高斯初始化，$B$采用零初始化仍然是应用LoRA普遍且有效的起始点。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了原始的LoRA方法，详细阐述了其架构和默认的参数初始化策略，该策略确保预训练模型在初始阶段不发生变化。
- [LoRA (PEFT Library) Documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora) — Hugging Face (2024)
  提供了LoRA层在广泛使用的Hugging Face PEFT库中的实际实现细节和默认初始化选择。
