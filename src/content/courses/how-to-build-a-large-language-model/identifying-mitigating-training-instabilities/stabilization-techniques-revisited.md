---
course: "how-to-build-a-large-language-model"
chapter: "identifying-mitigating-training-instabilities"
lesson: "stabilization-techniques-revisited"
sourceId: 6075
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-24-identifying-mitigating-training-instabilities/stabilization-techniques-revisited"
title: "稳定方法回顾（梯度裁剪、学习率、预热）"
description: "强调梯度裁剪、学习率调度和预热的作用。"
order: 5
plots: ["plots/6075-0.json"]
sourceHash: "29a69a5a27f3673be0f123e6858b5ec6222b33fb83ed6a0d87357a0f6d30c601"
sourceCorrections: []
---

在训练大型语言模型时，损失值突然飙升或发散（出现`NaN`值）等不稳定情况遗憾的是很常见。某些技术对于有效训练以及专门预防和减轻这些问题来说都非常重要。三种重要的稳定方法——梯度裁剪、学习率调度和学习率预热——对于解决不稳定问题至关重要。当训练运行开始出现异常时，正确配置和监控这些组成部分通常是第一道防线。

### 梯度裁剪

梯度裁剪直接解决梯度爆炸问题，这是导致`NaN`损失值或突然发散的常见原因。通过对梯度的大小施加一个上限，它能避免模型权重 (weight)更新过大，而过大的更新会使训练过程不稳定，特别是在深度网络或Transformer中固有的循环结构中。

最常用的技术是对所有模型参数 (parameter)的梯度*范数*进行裁剪。如果整个模型（或有时按参数组）的梯度L2范数（欧几里得范数）超过预设阈值`max_norm`，梯度会按比例缩小以匹配此阈值。


$$
\mathbf{g} \leftarrow \frac{\text{max\_norm}}{||\mathbf{g}||_2} \mathbf{g} \quad \text{if } ||\mathbf{g}||_2 > \text{max\_norm}
$$


这里，$\mathbf{g}$ 表示所有梯度的向量 (vector)。

在PyTorch中，这通常在反向传播 (backpropagation)之后，但在优化器步进之前应用：

```python
import torch
from torch.nn.utils import clip_grad_norm_

# 假设模型、损失和优化器已定义
# ... 前向传播 ...
loss = compute_loss(outputs, targets)

# ... 反向传播 ...
loss.backward()

# 裁剪梯度
max_norm = 1.0 # 一个常见起始值，需要调整
clip_grad_norm_(model.parameters(), max_norm)

# 优化器步进
optimizer.step()
optimizer.zero_grad()
```

**选择`max_norm`**：`max_norm`的值是一个超参数 (hyperparameter)。常用值范围是0.5到10.0，1.0是一个常见的默认值。设置过低会因更新收缩过多而阻碍学习，而设置过高则无法有效对抗大幅飙升。在稳定运行时监控梯度范数（如“监控训练指标”部分所述）可以提供选择合适值的依据。如果在稳定期间观察到范数频繁超过您选择的`max_norm`，您可能裁剪得过于激进。反之，如果发生损失飙升且此时范数远高于您的`max_norm`，裁剪很可能起到帮助作用。

尽管有效，但梯度裁剪不应被视为万能药。它是一种稳定机制。如果裁剪持续活跃或在非常低的阈值下才起作用，那可能表明模型架构、初始化、学习率或数据质量存在潜在问题，也应检查这些问题。

### 学习率调度

固定学习率很少适合训练大型复杂模型。学习率调度能在训练过程中动态调整学习率，平衡早期的探索和后期的精细收敛。这种受控调整对稳定性很重要。整个训练过程中学习率过高很容易导致震荡或发散，而衰减过慢则可能在接近最小值时仍进行大步、可能不稳定的更新。

大型语言模型常用的调度方案，通常在初始预热阶段*之后*应用，包括：

1. **线性衰减**：学习率在设定的步数或整个训练持续时间内，从基础值线性降低到一个最小值（通常为零）。
2. **余弦衰减**：学习率遵循余弦曲线，从基础值降低到一个最小值。与线性衰减相比，这通常导致初始衰减较慢，然后接近结束时衰减加快。

以下是您如何使用PyTorch的调度器结合预热和余弦衰减：

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR
)

# 假设模型已定义
optimizer = AdamW(model.parameters(), lr=1e-4) # 基础学习率

# 定义预热和衰减阶段
warmup_steps = 500
total_training_steps = 10000
decay_steps = total_training_steps - warmup_steps

# 预热调度器（从0线性增加到基础学习率）
# 注意：LinearLR的因子从接近零（1/warmup_steps）开始，并变为1.0
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=1.0/warmup_steps,
    end_factor=1.0,
    total_iters=warmup_steps
)

# 余弦衰减调度器（从基础学习率到0）
decay_scheduler = CosineAnnealingLR(
    optimizer, T_max=decay_steps, eta_min=0
) # T_max 是一个半周期的步数

# 串联调度器
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, decay_scheduler],
    milestones=[warmup_steps]
)

# 在训练循环中：
# ... 训练步进 ...
optimizer.step()
scheduler.step() # 更新学习率
optimizer.zero_grad()
```

配置不当的调度方案可能导致不稳定。例如，如果衰减过慢或最终学习率（`CosineAnnealingLR`中的`eta_min`）过高，优化器可能在训练结束时超调或震荡。反之，衰减过快可能导致收敛停滞。当遇到后期不稳定时，检查学习率调度的行为是一个有益的诊断步骤。

### 学习率预热

训练的最初阶段是一个特别脆弱的时期。模型权重 (weight)通常是随机初始化的，立即应用完整的目标学习率可能导致非常大、混乱的更新，在最初几次迭代中就出现损失值飙升或`NaN`值。像AdamW这样的自适应优化器也需要一些初始步骤来建立梯度矩的可靠估计。学习率预热通过从一个非常小的学习率（通常为零或接近零）开始，并在预设的初始训练步数（`warmup_steps`）内逐步增加到目标基础学习率来解决这个问题。这给模型和优化器时间在应用较大更新前稳定下来。线性预热是最常见的策略。



![学习率调度：预热 + 余弦衰减](plots/6075-0.json)



> 一个典型的学习率调度方案，包括500步线性预热，随后在剩余步数中余弦衰减到零。

**选择`warmup_steps`**：预热步数是另一个超参数 (parameter) (hyperparameter)。它通常范围从数百到数千步，取决于总体训练时长、批量大小和数据集。常见做法是将其设置为总预期训练步数的一小部分（例如1-10%）。过少的预热步数可能无法避免初始不稳定，而过多则会减慢有效学习的开始。如果在训练初期就遇到不稳定，增加预热时长或降低初始学习率是首先要考虑的调整。

### 关联与监控

这三种技术彼此联系。例如：

- 有效的预热可能允许稍高的峰值学习率而不会立即发散。
- 激进的梯度裁剪可能使训练过程对次优的学习率调度方案更具容忍度。
- 优化器的选择（例如AdamW与SGD）也会影响对学习率的敏感度和预热的有效性。

诊断不稳定时，监控与这些技术相关的行为很重要：

- **学习率**：记录优化器在每一步实际使用的学习率，以确认调度方案按预期运行。
- **梯度范数**：跟踪梯度范数（裁剪前）以了解更新的幅度以及裁剪是否频繁活跃。梯度范数突然、持续增加通常先于损失值飙升发生。

再次审视并仔细调整梯度裁剪、学习率调度和预热阶段，是解决训练不稳定问题的必要步骤。它们提供了控制更新动态的手段，防止可能阻碍大规模语言模型训练进展的爆炸和发散情况。

## 参考资料

- [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 5998–6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构及其学习率调度，包括预热阶段，这已成为大型语言模型的标准实践。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8: Optimization for Training Deep Models
  一本全面的深度学习教科书，涵盖了基础概念，包括对梯度裁剪作为优化稳定技术的详细解释。
- [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — PyTorch Contributors (2024)
  PyTorch官方文档，提供了梯度范数裁剪的实现细节，包括使用示例和参数说明。
- [How to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — PyTorch Contributors (2025)
  PyTorch官方文档，解释了学习率调度器，包括课程内容中演示的`LinearLR`、`CosineAnnealingLR`和`SequentialLR`的示例。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf) — Ilya Loshchilov and Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR 2017); DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  介绍了余弦退火，这是一种学习率衰减策略，因其平滑衰减和跳出局部最优的潜力而被广泛采用。
