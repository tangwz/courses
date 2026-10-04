---
course: "advanced-pytorch"
chapter: "optimization-training-strategies"
lesson: "advanced-lr-scheduling"
sourceId: 2710
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-3-optimization-training-strategies/advanced-lr-scheduling"
title: "高级学习率调度"
description: "实现学习率策略，如带重启的余弦退火、线性/多项式衰减和预热。"
order: 2
plots: ["plots/2710-0.json"]
sourceHash: "99de4fe6d70fa6f5f58266046fd5cf02d96b19e230ead1a2c76ca59cb1865c69"
sourceCorrections: []
---

虽然固定学习率对于较简单的问题可能有效，但训练大型、复杂的模型通常会从优化过程中动态调整学习率中大大受益。精心选择的学习率调度可以加速收敛，帮助处理复杂的损失函数 (loss function)形态，并与使用单一、静态学习率相比，带来更好的最终模型性能。PyTorch通过`torch.optim.lr_scheduler`模块提供了一个灵活的架构，用于实现各种调度策略。

大多数学习率调度背后的核心思想是直观的：在训练初期，当参数 (parameter)距离最优值较远时，从一个相对较大的学习率开始，以取得快速进展。然后，随着训练的进行，模型接近一个可能的最小值时，逐渐降低学习率，以便进行更精细的调整，并防止越过最佳点。这种动态调整有助于在寻找最佳模型参数时平衡发现与利用。

### 常用高级调度策略

PyTorch提供了多种内置调度器，让您可以用最少的代码实现复杂的学习率调整。让我们来看看高级训练方案中一些最有效且常用的策略。

#### 余弦退火

余弦退火是一种流行的调度技术，它按照余弦曲线平滑地降低学习率。它从优化器中指定的初始学习率开始，并在定义的回合数或步数（`T_max`）内逐渐将其降低到最小值（`eta_min`）。第$t$回合的学习率$\eta_t$计算如下：


$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t \pi}{T_{max}}\right)\right)
$$


这里，$\eta_{max}$是初始学习率。

这种平滑、渐进的衰减有助于优化器在训练结束时稳定到较好的最低点，避免了基于步长的衰减方法带来的突变。

这里是如何在PyTorch中实现`CosineAnnealingLR`：

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt # 用于可视化

# 示例设置
model_params = [torch.randn(10, 5, requires_grad=True)]
optimizer = SGD(model_params, lr=0.1)

# 余弦退火：将学习率从0.1退火到0，持续100个epoch
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# 模拟训练循环以可视化学习率变化
lrs = []
for epoch in range(150): # 模拟超过T_max的epoch数
    # optimizer.step() # 通常在loss.backward()之后调用
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# # 简单绘图（课程集成请替换为Plotly）
# plt.figure()
# plt.plot(range(150), lrs)
# plt.title("CosineAnnealingLR (T_max=100)")
# plt.xlabel("回合")
# plt.ylabel("学习率")
# plt.grid(True)
# plt.show()
```

注意学习率如何沿着余弦曲线下降，直到达到`T_max`（100个回合），然后对于后续回合保持在`eta_min`（0）。

#### 带热启动的余弦退火

一个变体是`CosineAnnealingWarmRestarts`。这个调度器不是只退火一次，而是定期重新启动余弦退火周期。它在`T_0`个回合内退火学习率，然后通过将学习率重置回其初始值并开始新的退火周期来“重新启动”。后续周期的长度可以选择性地通过因子`T_mult`增加。


$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur} \pi}{T_i}\right)\right)
$$


这里，$T_i$是当前周期的长度（最初是$T_0$，每次重启后乘以$T_{mult}$），$T_{cur}$是自上次重启以来经过的回合数。

这些“热启动”可以帮助优化器摆脱可能在退火阶段陷入的次优局部最小值。

```python
import torch
from torch.optim import AdamW # 常与高级调度器配合使用
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt # 用于可视化

# 示例设置
model_params = [torch.randn(10, 5, requires_grad=True)]
optimizer = AdamW(model_params, lr=0.01) # 初始学习率

# 带热启动的余弦退火：
# 每50个回合重启一次 (T_0=50)。
# 每次重启后周期长度加倍 (T_mult=2)。
# 最小学习率为1e-5。
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)

# 模拟训练循环
lrs_restarts = []
num_epochs = 300 # T_0 + T_0*T_mult + T_0*T_mult*T_mult = 50 + 100 + 200 = 350
for epoch in range(num_epochs):
    # optimizer.step()
    lrs_restarts.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# # 简单绘图
# plt.figure()
# plt.plot(range(num_epochs), lrs_restarts)
# plt.title("CosineAnnealingWarmRestarts (T_0=50, T_mult=2)")
# plt.xlabel("回合")
# plt.ylabel("学习率")
# plt.grid(True)
# plt.show()
```

这个调度器创建了退火周期，每个周期可能比前一个更长。

#### 学习率预热

直接以较大的学习率开始训练，尤其是在使用Adam或AdamW等自适应优化器时，有时会导致早期训练不稳定或发散。初始参数 (parameter)梯度可能很大且噪声多，从而导致很大的、可能产生负面影响的更新。

缓解这种情况的常用方法是*学习率预热*。在训练的前几个回合（或批次）中，学习率从一个非常小的值（例如，接近零）逐渐增加到目标初始学习率。这使得模型在应用较大更新之前得以稳定。

预热通常不是PyTorch的`lr_scheduler`模块中的独立调度器，而是通过组合调度器或使用`LambdaLR`来实现的。一种常用方法是实现线性预热，然后是另一个衰减调度，如余弦退火。

这里是一个使用`LambdaLR`实现线性预热后接余弦退火的示例：

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math

# 示例设置
model_params = [torch.randn(10, 5, requires_grad=True)]
initial_lr = 0.01
optimizer = AdamW(model_params, lr=initial_lr)

# 参数
warmup_epochs = 10
total_epochs = 100
cosine_epochs = total_epochs - warmup_epochs

# 调度器1：线性预热
def lr_lambda_warmup(current_epoch):
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(max(1, warmup_epochs))
    else:
        # 预热结束后，让余弦调度器间接接管
        # 我们计算相对于预热阶段结束的衰减因子
        progress = float(current_epoch - warmup_epochs) / float(max(1, cosine_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay # 这个因子将乘以initial_lr

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

# 模拟训练循环
lrs_warmup_cosine = []
for epoch in range(total_epochs + 20): # 模拟稍长一点的时间
    # optimizer.step()
    lrs_warmup_cosine.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

# # 简单绘图
# plt.figure()
# plt.plot(range(total_epochs + 20), lrs_warmup_cosine)
# plt.title("线性预热 (10 回合) + 余弦退火")
# plt.xlabel("回合")
# plt.ylabel("学习率")
# plt.grid(True)
# plt.show()
```

这种方法使用单个`LambdaLR`调度器，其`lr_lambda`函数整合了预热逻辑和随后的余弦衰减逻辑。请注意，较新的PyTorch版本还提供了`SequentialLR`和`ChainedScheduler`，用于更明确地组合不同的调度器。

#### 其他调度器

虽然余弦退火和预热非常普遍，但其他调度器也存在：

- **`StepLR`**: 每`step_size`个回合将学习率衰减`gamma`因子。简单，但突然的下降有时会扰乱训练势头。
- **`MultiStepLR`**: 类似于`StepLR`，但允许指定衰减学习率的确切回合数（`milestones`）。
- **`ExponentialLR`**: 每个回合将学习率衰减`gamma`因子。
- **`PolynomialLR`**: 按照多项式函数衰减学习率，在线性和其他衰减形状之间提供灵活性。
- **`ReduceLROnPlateau`**: 当监控的指标（例如，验证损失）在指定回合数（`patience`）内停止改进时，降低学习率。这是根据性能自适应的，而非固定调度。

### 将调度器集成到训练循环中

在PyTorch中使用学习率调度器通常涉及两个步骤：

1. **初始化：** 在创建优化器之后，创建调度器实例，并将优化器传递给调度器的构造函数。
2. **步进：** 在训练循环中的适当位置调用调度器的`step()`方法。

`scheduler.step()`的放置位置非常重要：

- **基于回合的调度器：** 对于`StepLR`、`CosineAnnealingLR`、`CosineAnnealingWarmRestarts`（当按回合定义时）、`MultiStepLR`等调度器，您通常每个回合调用`scheduler.step()`一次，通常在验证循环之后或训练回合结束时。
- **基于批次的调度器：** 一些调度策略，特别是更精细的或与迭代紧密耦合的（如潜在的基于批次的预热或周期实现），可能需要在每个批次之后（即在`optimizer.step()`之后）调用`scheduler.step()`。务必查阅特定调度器的文档。对于这里提到的大多数常见基于回合的调度器，每个回合步进一次是标准做法。请注意，`ReduceLROnPlateau`需要将其指标值传递给其`step()`方法（例如，`scheduler.step(validation_loss)`）。

```python
# 典型训练循环片段（基于回合的步进）

optimizer = AdamW(model.parameters(), lr=initial_lr)
# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = # 在此处初始化您选择的调度器

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 如果需要，基于批次的调度器步进会放在这里

    # --- 回合结束 ---
    # 执行验证、日志记录等

    # 步进调度器（针对基于回合的调度器）
    scheduler.step() # 对于ReduceLROnPlateau：scheduler.step(validation_loss)

    print(f"回合 {epoch+1}/{num_epochs}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
```

### 常用调度器的可视化

可视化学习率随时间的变化有助于理解和调试您选择的调度器。



![学习率调度示例](plots/2710-0.json)



> 比较不同学习率调度在回合中的表现（y轴对数刻度）。余弦退火表现出平滑衰减，热启动引入了周期性重置，而预热+余弦则包含一个初始的爬升阶段。

### 选择与调优

最佳的学习率调度及其参数 (parameter)（初始学习率、`T_max`、`T_0`、`T_mult`、`eta_min`、预热持续时间）很大程度上取决于具体问题、数据集、模型架构和所选优化器。没有适用于所有情况的单一最佳调度器。

- **余弦退火**调度（带或不带重启，常与预热结合）是许多现代架构（如Transformer）的良好通用选择。
- **预热**对大型模型或使用AdamW等自适应优化器时尤其有益。
- \*\*`ReduceLROnPlateau`\*\*在进度直接与可衡量的验证指标相关联时可以很有效，但如果指标因学习率之外的原因频繁停滞，它可能反应缓慢。

实验很重要。可视化计划的调度、监控训练/验证损失曲线，以及使用超参数 (hyperparameter)优化工具（本章后面会讨论）对于找到适合您特定任务的最有效调度策略是不可或缺的。请记住，优化器和学习率调度之间的配合很重要，因此它们通常应该一起调整。

## 参考资料

- [\`torch.optim.lr_scheduler\` documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) — PyTorch Contributors (2025)
  PyTorch中实现学习率调度器的官方指南，涵盖各种内置策略。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov and Frank Hutter (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  提出余弦退火与热启动的学习率调整方法，一种用于训练深度神经网络的策略。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  详细说明了结合线性预热和逆平方根衰减的学习率计划，尤其适用于Transformer模型。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，提供优化理论背景，包括学习率调整方法。
