---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "implementing-adam-rmsprop"
sourceId: 4996
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/implementing-adam-rmsprop"
title: "实现 Adam 和 RMSprop"
description: "介绍如何在深度学习框架中应用 Adam 和 RMSprop 优化器。"
order: 9
plots: ["plots/4996-0.json"]
sourceHash: "c75eed9e19c2a00f089c1b12356a12f91a739b7d8eb7aff72531cdf00197b372"
sourceCorrections: [{"kind": "plain-math-control-character", "line": 101}]
---

实现 RMSprop 和 Adam 等自适应优化算法需要使用常见的深度学习 (deep learning)框架。幸运的是，PyTorch 等库提供了直接的实现，使得使用这些高效的优化器变得简单。

### 使用 RMSprop

RMSprop 根据梯度的平方移动平均值来调整每个参数 (parameter)的学习率，有助于应对复杂的损失曲面。在 PyTorch 中，你可以通过导入并实例化 `torch.optim.RMSprop` 类来使用 RMSprop。

下面是一个基本示例，展示了如何定义一个简单的神经网络 (neural network)，然后为其参数创建一个 RMSprop 优化器：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设 'model' 是你定义的神经网络实例
# 示例：model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
# model_parameters = model.parameters() # 获取模型的参数

# 定义一个用于演示的占位符模型
model = nn.Linear(10, 2) # 一个简单的线性层
model_parameters = model.parameters()

# 实例化 RMSprop 优化器
learning_rate = 1e-3
alpha_param = 0.99 # 对应 RMSprop 描述中的 rho
epsilon_param = 1e-8
weight_decay_param = 0 # L2 正则化，通常单独处理或在此处设置
momentum_param = 0 # 可选的动量项

optimizer_rmsprop = optim.RMSprop(
    model_parameters,
    lr=learning_rate,
    alpha=alpha_param,
    eps=epsilon_param,
    weight_decay=weight_decay_param,
    momentum=momentum_param,
    centered=False # 如果为 True，计算中心化 RMSProp（通过方差估计来归一化梯度）
)

# 训练循环步骤示例（简化）：
# optimizer_rmsprop.zero_grad()
# loss = calculate_loss(model(inputs), targets)
# loss.backward()
# optimizer_rmsprop.step()
```

`torch.optim.RMSprop` 的重要参数：

- `params`：一个可迭代对象（如 `model.parameters()`），包含要优化的参数。
- `lr` (float, 可选)：学习率（默认值：1e-2）。这是一个起始点，通常需要调整。`1e-3` 或 `1e-4` 等值也是常见的起始点。
- `alpha` (float, 可选)：平滑常数（默认值：0.99），对应算法描述中的 $\rho$。它控制平方梯度移动平均值的衰减率。
- `eps` (float, 可选)：添加到分母中的一个小项，用于数值稳定性（默认值：1e-8）。它防止除以零。
- `weight_decay` (float, 可选)：向损失中添加 L2 惩罚（权重 (weight)衰减）（默认值：0）。
- `momentum` (float, 可选)：对梯度更新施加动量（默认值：0）。
- `centered` (bool, 可选)：如果为 `True`，计算 RMSprop 的中心化版本，其中梯度通过其方差的估计值进行归一化 (normalization)，而不仅仅是二阶矩。有时可以使训练更稳定，但会增加计算成本（默认值：`False`）。

`alpha` 的默认值（0.99）是标准设置。学习率 `lr` 是你最常需要调整的参数。

### 使用 Adam

Adam（自适应矩估计）可以说是一种非常常用且通常高效的通用优化器。它结合了动量（使用梯度的移动平均值）和 RMSprop（使用平方梯度的移动平均值）的思路。在 PyTorch 中实现 Adam 同样直接，只需使用 `torch.optim.Adam`。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设 'model' 是你定义的神经网络实例
# model_parameters = model.parameters()

# 定义一个用于演示的占位符模型
model = nn.Linear(10, 2) # 一个简单的线性层
model_parameters = model.parameters()

# 实例化 Adam 优化器
learning_rate = 1e-3 # Adam 常见的默认起始点
beta1 = 0.9
beta2 = 0.999
epsilon_param = 1e-8
weight_decay_param = 0

optimizer_adam = optim.Adam(
    model_parameters,
    lr=learning_rate,
    betas=(beta1, beta2), # 用于计算移动平均值的系数
    eps=epsilon_param,
    weight_decay=weight_decay_param,
    amsgrad=False # 是否使用 AMSGrad 变体
)

# 训练循环步骤示例（简化）：
# optimizer_adam.zero_grad()
# loss = calculate_loss(model(inputs), targets)
# loss.backward()
# optimizer_adam.step()
```

`torch.optim.Adam` 的重要参数 (parameter)：

- `params`：包含要优化参数的可迭代对象。
- `lr` (float, 可选)：学习率（默认值：1e-3）。尽管 Adam 通常对学习率的敏感度低于 SGD，但这仍是一个需要调整的超参数 (hyperparameter)。
- `betas` (Tuple[float, float], 可选)：用于计算梯度 ($\beta$\_1) 及其平方 ($\beta$\_2) 的运行平均值的系数（默认值：(0.9, 0.999)）。这些默认值被普遍使用，并由 Adam 论文建议。$\beta$\_1 控制类似于动量的部分，而 $\beta$\_2 控制类似于 RMSprop 的自适应学习率部分。
- `eps` (float, 可选)：添加到分母中的项，用于数值稳定性（默认值：1e-8）。
- `weight_decay` (float, 可选)：L2 惩罚（默认值：0）。
- `amsgrad` (bool, 可选)：是否使用 Adam 的 AMSGrad 变体，该变体旨在某些情况下优化收敛特性（默认值：`False`）。

### 实际考量

- **默认值通常效果良好：** 对于 RMSprop 和 Adam，PyTorch 等库提供的默认超参数 (parameter) (hyperparameter)（尤其是 `alpha`、`betas`、`eps`）基于常见实践和研究结果。它们通常开箱即用，表现出色，特别是 Adam 的默认设置（`lr=0.001`、`betas=(0.9, 0.999)`、`eps=1e-8`）。在进行大量调整之前，可以从这些默认值开始。
- **学习率调整：** 即使使用自适应优化器，学习率 (`lr`) 仍然是一个需要调整的超参数。虽然默认值是良好的起始点，但你可能需要对其进行调整（例如，尝试 `3e-4`、`1e-4`、`1e-5`）。
- **Epsilon (`eps`)：** 当平方梯度的移动平均值非常小时，此参数可防止除以零。默认值（如 `1e-8`）通常足够，很少需要调整。
- **权重 (weight)衰减：** 你可以通过手动向损失函数 (loss function)添加惩罚项，或者直接在优化器中使用 `weight_decay` 参数来应用 L2 正则化 (regularization)。使用优化器的参数通常更方便。
- **监控表现：** 如前一章所述，在尝试不同的优化器或超参数时，要监控你的训练和验证损失曲线。Adam 和 RMSprop 通常在初期比 SGD 或 Momentum 收敛更快，但这并不总是能确保最终更好的泛化能力。



![优化器收敛示意图](plots/4996-0.json)



> 训练损失曲线表明，对于某些问题，RMSprop 和 Adam 等自适应优化器可能比基本 SGD 具有更快的收敛速度。实际结果很大程度上取决于具体任务和超参数。

在你的训练代码中实现 RMSprop 和 Adam 通常是一个简单的替换操作。通过了解它们的参数，你可以应用这些先进的优化技术，从而可能加速训练并提升深度学习 (deep learning)模型的表现。下一步通常涉及微调 (fine-tuning)学习率以及可能包括权重衰减在内的其他超参数，我们将在后续讨论。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了深度学习中的核心Adam优化算法。
- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8.5.2 (RMSprop)
  作为一本标准的深度学习教材，对RMSprop算法进行了全面解释。
- [torch.optim](https://pytorch.org/docs/stable/optim.html) — PyTorch Core Team (2025)
  PyTorch优化算法的官方文档，包含RMSprop和Adam的实现细节。
