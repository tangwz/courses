---
course: "deep-learning-regularization-optimization"
chapter: "optimization-refinements-tuning"
lesson: "learning-rate-warmup"
sourceId: 5003
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-7-optimization-refinements-tuning/learning-rate-warmup"
title: "预热策略"
description: "讲解学习率预热周期的原理和作用。"
order: 6
plots: ["plots/5003-0.json"]
sourceHash: "9f0f8a6a9bf6b230e13558cd2db05793fe1f96fd60776b7abb536fb8646c63ec"
sourceCorrections: []
---

尽管在训练*期间*使用阶梯衰减或指数衰减等策略调整学习率有助于后续收敛的微调 (fine-tuning)，但训练初期有其自身特殊的问题。当网络权重 (weight)刚被初始化（通常是随机的）时，计算出的初始梯度可能很大，或者指向对稳定学习没有直接帮助的方向。从第一步开始就应用一个相对较大的初始学习率有时可能导致不稳定，使损失剧烈波动甚至发散。对于复杂的模型或使用某些自适应优化器（其内部状态需要时间稳定）时，情况尤其如此。

在深度学习 (deep learning)模型的训练初期，当权重刚刚初始化时，直接使用较高的学习率可能导致训练不稳定，造成损失剧烈震荡甚至发散。学习率预热 (learning rate warmup) 正是为了解决此类初始不稳定问题而设计。其主要思想很简单：你不是从目标初始学习率（例如 $\alpha = 0.001$）开始，而是从一个小的学习率开始，并在预设的初始训练步数或时期内逐渐提高它。一旦“预热阶段”完成，学习率达到其目标初始值，然后标准的学习率策略（如阶梯衰减、指数衰减或余弦退火）将接管剩余的训练过程。

### 为何使用预热？

预热的主要优点是**训练的稳定**。通过在初期使用小的学习率，您可以让模型参数 (parameter)温和地调整，从而避免由于初始梯度不稳定而可能产生的大幅、潜在破坏性更新。这为优化过程提供了“稳定下来”的时间，然后再进行更大的步长。

这种技术通常在以下具体情况中特别有用：

1. **大批量大小：** 使用非常大的小批量进行训练有时会导致损失函数 (loss function)更陡峭或产生优化困难。预热有助于减轻与大批量训练所需的大学习率相关的不稳定性。
2. **自适应优化器：** 像Adam这样的优化器会维护梯度矩的内部估计。在训练早期，这些估计基于很少的样本，可能存在噪声。预热期允许这些估计在学习率达到其完整值之前变得更稳定。
3. **复杂模型：** 对于深度或复杂架构（如Transformer），预热几乎已成为标准做法，因为它显著提升了初始训练阶段的可靠性和稳定性。

### 常见预热策略

最常见的方法是**线性预热**。在这种策略中，在 $N_{warmup}$ 步内，步长 $t$ 处的学习率 $\alpha_t$ 从一个小的起始值 $\alpha_{start}$（有时甚至为0）线性增加到目标初始学习率 $\alpha_{target}$：


$$
\alpha_t = \alpha_{start} + (\alpha_{target} - \alpha_{start}) \times \frac{t}{N_{warmup}} \quad \text{对于 } 0 \le t < N_{warmup}
$$


在 $N_{warmup}$ 步之后，学习率达到 $\alpha_{target}$，并且通常会开始后续的衰减策略。

另一种更简单但不太常见的变体是**恒定预热**，即在 $N_{warmup}$ 步内使用一个小的恒定学习率，之后学习率会突然跳到 $\alpha_{target}$。

预热阶段的持续时间 $N_{warmup}$ 是一个超参数 (parameter) (hyperparameter)。它可以指定为训练步数（例如，1000步）或时期数（例如，5个时期）。最佳持续时间取决于所用的数据集、模型、批量大小和优化器。



![学习率策略：线性预热 + 阶梯衰减](plots/5003-0.json)



> 一个学习率策略，结合了前10步的线性预热（从 $10^{-5}$ 到 $10^{-3}$），随后在第30步进行阶梯衰减（到 $10^{-4}$）。注意Y轴的对数刻度。

### 实施考量

许多深度学习 (deep learning)框架提供了实现预热的机制，通常通过组合调度器或使用自定义lambda函数来完成。例如，在PyTorch中，您可以使用 `LambdaLR` 或通过 `SequentialLR` 链接调度器，将线性预热与 `StepLR` 或 `ExponentialLR` 等其他调度器结合起来实现。

以下是使用 `LambdaLR` 进行线性预热的示例：

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# 假设模型和优化器已定义
# optimizer = optim.Adam(model.parameters(), lr=0.001) # 此处设置的初始学习率对LambdaLR不重要

target_lr = 0.001
warmup_steps = 1000

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        # 线性预热
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # 预热后，使用一个常数因子（或在此处实现衰减）
        # 例如，预热后的常数：
        return 1.0
        # 示例：预热后的阶梯衰减
        # decay_start_step = 10000
        # decay_factor = 0.1
        # if current_step >= decay_start_step:
        #     return decay_factor
        # else:
        #     return 1.0

# 注意：优化器的初始学习率作为base_lr的乘数
optimizer = optim.Adam(model.parameters(), lr=target_lr) # 将优化器学习率设置为目标学习率
scheduler = LambdaLR(optimizer, lr_lambda)

# 训练循环
for step in range(num_training_steps):
    # ... 执行训练步骤 ...
    optimizer.step()
    scheduler.step() # 更新学习率
    optimizer.zero_grad()
```

此代码片段定义了一个lambda函数，它在 `warmup_steps` 期间将学习率乘数从接近零线性增加到1.0。`optimizer` 使用 `target_lr` 初始化，调度器根据lambda函数在每一步的输出缩放此值。预热期结束后，这个简单示例将学习率保持在 `target_lr` 不变，但您可以轻松修改 `lr_lambda` 函数，以便在此之后实现衰减策略。像Hugging Face的 `transformers` 库通常提供包含预热阶段的预构建调度器。

总而言之，学习率预热是一种有用的技术，用于稳定深度学习训练的初始阶段，防止因大的初始学习率作用于随机初始化的权重 (weight)或不稳定的自适应优化器状态而导致的潜在发散或震荡。它常与其他学习率衰减策略一起使用，对复杂模型和大规模批量训练尤其有益。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  本文介绍了Transformer架构及其学习率调度，其中包含线性热身阶段。它对复杂深度学习模型中热身策略的广泛采用产生了深远影响。
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) — Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He (2017)
  Journal: arXiv preprint arXiv:1706.02677; DOI: [10.48550/arXiv.1706.02677](https://doi.org/10.48550/arXiv.1706.02677)
  这项研究强调了在使用大批量进行训练时学习率热身策略的有效性。它解决了初始不稳定问题并有助于实现更快的收敛，这是本节讨论的一个重要方面。
- [Learning rate scheduling - PyTorch Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate-for-warmup) — PyTorch Authors (2025)
  Publisher: PyTorch Foundation
  PyTorch学习率调度器的官方文档，提供了实现各种调度（包括本节中演示的`LambdaLR`等热身策略）的详细指南和示例。
