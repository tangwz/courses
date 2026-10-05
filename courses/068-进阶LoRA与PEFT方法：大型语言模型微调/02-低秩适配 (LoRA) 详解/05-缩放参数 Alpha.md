# 缩放参数 Alpha

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/lora-scaling-alpha)

[返回章节目录](README.md) · [返回课程目录](../README.md)

缩放参数 (parameter)，记作 $\alpha$，是低秩适应 (LoRA) 中一个重要的超参数 (hyperparameter)。此参数作为LoRA更新（通常表示为 $\Delta W = BA$）的一个标量乘数。它用于调整微调 (fine-tuning)后的权重 (weight)对原始预训练 (pre-training)权重 $W_0$ 的影响程度，与低秩矩阵 $A$ 和 $B$ 以及选定的秩 $r$ 共同作用，以控制适应的整体影响。

修正后的前向传播，包含了 $\alpha$，将输入 $x$ 的最终输出 $h$ 表示为：


$$
h = W_0 x + \Delta W x = W_0 x + \alpha (BAx)
$$


此处，$W_0$ 代表冻结的预训练权重，而 $BA$ 形成了在微调过程中学到的低秩更新。参数 $\alpha$ 直接缩放了此更新的贡献。

然而，需要注意一个常见的实现惯例，尤其是在Hugging Face的PEFT (`peft`) 等库中很常见。实际上，更新在训练期间常通过 $\frac{\alpha}{r}$ 进行动态缩放。当使用此惯例时，有效的前向传播计算如下：


$$
h = W_0 x + \frac{\alpha}{r} (BAx)
$$


这种通过 $\frac{\alpha}{r}$ 进行的缩放旨在将权重调整的量级与秩 $r$ 的选择分离。如果矩阵 $A$ 的元素使用标准分布（例如高斯分布）初始化，并且 $B$ 初始化为零（这是一种常见的做法，以确保初始状态与预训练模型匹配），那么乘积 $BA$ 的方差可能会随 $r$ 变化。除以 $r$ 有助于使这种影响正常化，从而使 $\alpha$ 能够更稳定地作用于控制整体调整的强度，并在一定程度上独立于 $r$。

### Alpha 的作用

可以将 $\alpha$ 视为控制对基础模型表示施加的微调 (fine-tuning)调整的“强度”或幅度。它微调了学到的任务特定调整 ($BA$) 与原始冻结权重 (weight) ($W_0$) 相比，对输出改变了多少。

- **$\alpha$ 值较高**：更侧重于学到的LoRA调整。这可以加速适应目标任务，但会增加大幅偏离预训练 (pre-training)期间学到的表示的风险。过大的值可能导致过拟合 (overfitting)或不稳定。
- **$\alpha$ 值较低**：减小LoRA更新的影响，确保模型的行为更接近原始预训练模型。这可能导致在微调任务上收敛速度较慢，但通过更好保留基础模型的能力，可以提升泛化能力。

实际上，设定 $\alpha$ 涉及平衡 $W_0$ 中包含的通用预训练知识和在 $\Delta W$ 中学到的特定任务调整的贡献。它是一个重要的超参数 (parameter) (hyperparameter)，通常需要根据具体任务、数据集、模型架构以及选定的秩 $r$ 进行经验性调整。

### Alpha 调优的实际考量

对于 $\alpha$ 并没有一个单一的、普遍适用的最佳值。它的选择与其他超参数 (parameter) (hyperparameter)有关联，特别是秩 $r$ 以及用于训练矩阵 $A$ 和 $B$ 的学习率。常用的方法包括：

1. **设定 $\alpha = r$**：这是一种常用的启发式方法。如果实现中使用 $\frac{\alpha}{r}$ 缩放因子，将 $\alpha = r$ 设定会有效抵消该因子，从而得到更简单的更新形式 $h = W_0 x + BAx$。尽管直接，但这表示如果您修改 $r$，调整的实际幅度可能会随之隐式变化。
2. **将 $\alpha$ 设定为固定值**：实践者通常选择一个固定的 $\alpha$ 值（例如16、32、64），而不考虑 $r$。当使用 $\frac{\alpha}{r}$ 缩放时，这会将 $\alpha$ 视为期望调整强度的更直接衡量，相对于秩 $r$ 提供的容量进行缩放。
3. **将 $\alpha$ 视为独立的超参数**：就像学习率或秩 $r$ 一样，$\alpha$ 可以使用网格搜索、随机搜索或更复杂的贝叶斯优化等方法进行系统地调整。实验可能表明，当 $\alpha$ 与 $r$ 有很大不同时，例如 $\alpha = 2r$ 或 $\alpha = r/2$，模型表现最佳。

最佳方法通常取决于经验验证。如果使用LoRA进行微调 (fine-tuning)显得过于激进（例如，验证损失迅速增加）或过于保守（例如，模型性能低于预期），调整 $\alpha$ 是一个主要的控制手段，它与调整秩 $r$ 和优化器的学习率相辅相成。

总而言之，$\alpha$ 提供了一个重要的机制来缩放LoRA调整。尽管它常与秩 $r$ 相关联的缩放因子（即 $\alpha/r$）一起实现，但其主要目的是调节应用于冻结基础模型权重 (weight)的低秩更新的强度。在使用LoRA时，仔细考量和调整 $\alpha$ 是提升模型性能的必要步骤。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍低秩适应 (LoRA) 的原始研究论文，正式定义其架构和超参数，包括缩放参数 alpha。
- [Parameter-Efficient Fine-Tuning (PEFT) Documentation: LoRA Overview](https://huggingface.co/docs/peft/en/conceptual_guides/lora_overview) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，提供 LoRA 的实际实现细节和惯例，例如 alpha/r 缩放。

---

[上一节](04-%E7%A7%A9%E9%80%89%E6%8B%A9%E7%AD%96%E7%95%A5.md) · [下一节](06-LoRA%20%E5%B1%82%E7%9A%84%E5%AE%9E%E6%96%BD.md)
