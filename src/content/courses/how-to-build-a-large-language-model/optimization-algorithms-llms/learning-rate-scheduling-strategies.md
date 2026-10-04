---
course: "how-to-build-a-large-language-model"
chapter: "optimization-algorithms-llms"
lesson: "learning-rate-scheduling-strategies"
sourceId: 6035
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-17-optimization-algorithms-llms/learning-rate-scheduling-strategies"
title: "学习率调度策略"
description: "实现预热后接线性或余弦衰减调度。"
order: 3
plots: ["plots/6035-0.json"]
sourceHash: "d66719653aa9d28452c867a3d15bb0419aa8111a41069e5a185684e14f236f42"
sourceCorrections: []
---

选择合适的学习率 $\eta$ 是模型训练成功的基础。然而，在整个训练过程中使用固定学习率通常不是最优解，特别是对于像 Transformer 这样的大型复杂模型。在训练初期，模型参数 (parameter)离最优值较远，如果学习率过高，大的梯度可能导致不稳定或发散。相反，在训练后期，当模型接近收敛时，通常需要较小的学习率进行微调 (fine-tuning)并找到一个良好的最小值。学习率调度通过在每个训练步 $t$ 动态调整学习率 $\eta_t$ 来应对此情况。

对于大型语言模型，一种常见且有效的策略是结合*预热*阶段和随后的*衰减*阶段。

### 学习率预热

在训练的初始阶段，特别是当使用 Adam 或 AdamW 等自适应优化器时，由于观察到的样本数量有限，方差估计可能不可靠。此外，随机初始化的模型会产生显著误差，因此初始梯度可能较大且带有噪声。立即应用大学习率可能导致数值不稳定或使模型发散。

为缓解这种情况，通常会采用预热期。在此阶段，学习率会在预设的步数 `warmup_steps` 内，从一个很小的值（通常为 0）逐渐增加到其目标峰值 $\eta_{peak}$。最常见的方法是线性预热：


$$
\eta_t = \eta_{peak} \times \frac{t}{\text{预热步数}} \quad \text{对于 } 0 \le t < \text{预热步数}
$$


这种逐渐增加使 AdamW 等优化器中的自适应动量稳定下来，并防止训练初期出现大的、可能破坏稳定的更新。`warmup_steps` 是一个超参数 (parameter) (hyperparameter)，通常设置为几千步，或是总训练步数的一小部分百分比（例如 1-10%），这取决于数据集大小和批次大小。

### 学习率衰减

一旦预热阶段完成且学习率达到 $\eta_{peak}$，在剩余的训练步数中逐步降低学习率通常是有益的。这能使模型稳定到一个良好的最小值。有几种常见的衰减策略：

#### 线性衰减

学习率从 `warmup_steps` 到总训练步数 $T$ 之间，从 $\eta_{peak}$ 线性降低到最小值 $\eta_{min}$（通常为 0）。


$$
\eta_t = \eta_{min} + (\eta_{peak} - \eta_{min}) \times \frac{T - t}{T - \text{预热步数}} \quad \text{对于 } \text{预热步数} \le t \le T
$$


#### 余弦衰减

这是一种训练大型模型的流行策略。学习率遵循余弦曲线从 $\eta_{peak}$ 下降到 $\eta_{min}$。它起初下降缓慢，然后加速，接着在接近 $\eta_{min}$ 时再次放慢速度。这种平滑的衰减曲线在实践中通常效果良好。


$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{peak} - \eta_{min}) \left( 1 + \cos\left( \pi \frac{t - \text{预热步数}}{T - \text{预热步数}} \right) \right) \quad \text{对于 } \text{预热步数} \le t \le T
$$


#### 其他衰减调度

其他调度，如反平方根衰减 ($\eta_t \propto 1/\sqrt{t}$) 或多项式衰减也常被使用，但线性衰减和余弦衰减（尤其是余弦衰减）是预训练 (pre-training)大型 Transformer 模型时非常普遍的选择。

### 在 PyTorch 中实现调度

PyTorch 在 `torch.optim.lr_scheduler` 中提供了灵活的学习率调度工具。你可以使用 `LambdaLR` 实现自定义调度，或使用内置调度器。像 Hugging Face 的 `transformers` 等库也提供了便捷的辅助函数。

以下是定义一个适用于 `LambdaLR` 的调度函数的方法，该函数实现线性预热后接余弦衰减：

```python
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 假设优化器已定义：
# model = YourTransformerModel(...)
# optimizer = AdamW(
#     model.parameters(),
#     lr=peak_lr,
#     betas=(0.9, 0.98),
#     eps=1e-6,
#     weight_decay=0.1
# )

# 配置
num_training_steps = 100000 # 示例总步数
num_warmup_steps = 10000   # 示例预热步数
peak_lr = 3e-4            # 目标峰值学习率
                          # (优化器初始学习率)
min_lr = 3e-5             # 目标最小学习率

def lr_lambda(current_step: int):
    # 线性预热阶段
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    # 余弦衰减阶段
    progress = (float(current_step - num_warmup_steps) /
                float(max(1, num_training_steps - num_warmup_steps)))

    # 计算余弦衰减因子，确保它不会低于
    # min_lr / peak_lr
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    # 将余弦衰减与由 min_lr 设定的下限结合
    decay_factor = cosine_decay * (1 - min_lr / peak_lr) + (min_lr / peak_lr)

    return decay_factor

# 创建调度器
lr_scheduler = LambdaLR(optimizer, lr_lambda)

# 训练循环摘录：
# for step, batch in enumerate(dataloader):
#     ... 执行前向传播、反向传播 ...
#     optimizer.step()
#     lr_scheduler.step() # 更新学习率
#     optimizer.zero_grad()
#     if step >= num_training_steps:
#         break
```

或者，使用 `transformers` 库可以简化这一点：

```python
from transformers import get_scheduler, AdamW

# 配置（同前）
num_training_steps = 100000
num_warmup_steps = 10000
peak_lr = 3e-4

# model = YourTransformerModel(...)
# optimizer = AdamW(model.parameters(),
#                   lr=peak_lr,
#                   betas=(0.9, 0.98),
#                   eps=1e-6,
#                   weight_decay=0.1)

# 获取调度器
lr_scheduler = get_scheduler(
    name="cosine", # 也可以是 "linear"（线性），"polynomial"（多项式）等。
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
    # 注意：transformers 中的余弦衰减通常默认衰减到 0。
    # 自定义 min_lr 可能需要自定义 LambdaLR 方法
    # 或修改源代码。
)

# 训练循环相同：在 optimizer.step() 后调用 lr_scheduler.step()
```

以下图表展示了一个典型的学习率调度，包含线性预热和余弦衰减：



![学习率调度：线性预热 + 余弦衰减](plots/6035-0.json)



> 学习率曲线显示，在 10,000 预热步中线性增加到峰值 3e-4，随后在剩余的 90,000 步中余弦衰减至 0。

选择合适的调度、`warmup_steps`、`peak_lr` 和 `min_lr` 通常需要一些试验，但预热与后续衰减（尤其是余弦衰减）的组合是训练大型语言模型的一个良好起点。它在早期提供了稳定性，并使得后期训练能够有效收敛。

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.5555/3295222.3295232](https://doi.org/10.5555/3295222.3295232)
  介绍了Transformer架构及其包含预热阶段的特定学习率调度，是大型语言模型的基础。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov, Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  引入了余弦退火，这是一种学习率遵循余弦曲线的方法，构成了余弦衰减调度的基础。
- [How to adjust learning rate - PyTorch Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — PyTorch Developers (2025)
  PyTorch学习率调度器的官方文档，提供了各种策略的详细用法和示例。
- [Optimizers and Learning Rate Shchedulers - Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/main/main_classes/optimizer_and_lr_scheduler) — Hugging Face Team (2024)
  Publisher: Hugging Face
  Hugging Face Transformers库中学习率调度实用工具的官方文档，广泛应用于大型语言模型训练。
