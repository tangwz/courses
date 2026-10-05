# 实现SGD和动量

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/implementing-sgd-momentum)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随机梯度下降 (gradient descent) (SGD)、动量 (Momentum) 和 Nesterov 加速梯度 (NAG) 这些优化算法，可以通过 PyTorch 等常用深度学习 (deep learning)框架来实现。深度学习框架通常提供便捷的实现，让开发者能够专注于选择和配置优化器，而不是从头编写更新规则。

### 使用 torch.optim.SGD

PyTorch 的 `torch.optim` 模块包含各种优化算法的实现。我们介绍过的优化器，包括标准 SGD、动量和 NAG，都可以通过 `torch.optim.SGD` 类访问。

#### 标准 SGD 的实现

要使用标准 SGD，您需要实例化 `SGD` 优化器，传入模型的参数 (parameter)（通过 `model.parameters()` 获得）和所需的学习率 (`lr`)。

```python
import torch
import torch.optim as optim
import torch.nn as nn

# 假设 'model' 是您定义的神经网络（例如，nn.Linear(10, 2)）
# model = nn.Linear(10, 2) # 示例模型定义

# 实例化 SGD 优化器
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- 在典型训练循环中 ---
# 假设 'data' 和 'targets' 是从数据加载器中获取的批次
# 假设 'criterion' 是您的损失函数（例如，nn.CrossEntropyLoss()）

# for data, targets in dataloader: # 示例循环开始
#     # 1. 将上一步的梯度清零
#     optimizer.zero_grad()
#
#     # 2. 执行模型的前向传播
#     outputs = model(data)
#
#     # 3. 计算损失
#     loss = criterion(outputs, targets)
#
#     # 4. 执行反向传播以计算梯度
#     loss.backward()
#
#     # 5. 使用优化器更新模型参数
#     optimizer.step()
# --- 训练循环代码片段结束 ---

print(f"优化器已实例化: {optimizer}")
```

核心训练循环结构保持不变，无论在此系列中使用了哪种特定优化器：梯度清零、前向传播、计算损失、反向传播 (backpropagation)，然后是 `optimizer.step()`。`optimizer.step()` 调用是发挥作用的地方；它应用由优化器实例定义的特定更新规则。对于标准 SGD，这对应于更新 $w \leftarrow w - \eta \nabla L(w)$，其中 $w$ 表示参数，$\eta$ 是学习率，$\nabla L(w)$ 是损失的梯度。

#### 添加动量

要加入动量，只需在创建 `SGD` 优化器时指定 `momentum` 超参数 (hyperparameter)。此参数通常取值在 0.5 到 0.99 之间，其中 0.9 是一个常用起始点。

```python
import torch
import torch.optim as optim
import torch.nn as nn

# 假设 'model' 是您定义的神经网络
# model = nn.Linear(10, 2) # 示例模型定义

# 实例化带有动量的 SGD 优化器
learning_rate = 0.01
momentum_coefficient = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_coefficient)

# 训练循环结构与标准 SGD 保持一致。
# optimizer.step() 现在应用动量更新规则。

print(f"优化器已实例化: {optimizer}")
```

在幕后，当指定 `momentum` 时，`optimizer.step()` 会为每个参数维护速度向量 (vector) $v$。然后它会应用之前讨论过的动量更新规则：
$v \leftarrow \gamma v + \eta \nabla L(w)$
$w \leftarrow w - v$
这里，$\gamma$ 是您提供的 `momentum` 系数。训练循环代码本身不会改变，但 `optimizer.step()` 执行的参数更新现在得到了加速。

#### 启用 Nesterov 加速梯度 (NAG)

Nesterov 加速梯度 (NAG) 也可在 `torch.optim.SGD` 类中使用。要使用它，您必须提供一个非零的 `momentum` 值（类似于标准动量），并且额外将布尔参数 `nesterov` 设置为 `True`。

```python
import torch
import torch.optim as optim
import torch.nn as nn

# 假设 'model' 是您定义的神经网络
# model = nn.Linear(10, 2) # 示例模型定义

# 实例化带有 Nesterov 动量的 SGD 优化器
learning_rate = 0.01
momentum_coefficient = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_coefficient, nesterov=True)

# 训练循环结构保持不变。
# optimizer.step() 现在应用 NAG 更新规则。

print(f"优化器已实例化: {optimizer}")
```

将 `nesterov=True` 设置为 `True` 会修改更新步骤，使其包含 NAG 特有的“超前”梯度计算。请记住，NAG 需要将 `momentum` 参数设置为大于 0 的值；在未设置 `momentum` 的情况下将 `nesterov` 设置为 `True` 将不会激活 Nesterov 更新。

### 选择与调优

尽管有了 PyTorch 这样的库，实现过程变得简单，但选择正确的优化器变体并调整其超参数 (parameter) (hyperparameter)（如 `lr` 和 `momentum`）对于获得良好的模型性能和高效训练非常重要。

- **标准 SGD：** 收敛速度慢，并且对学习率的选择非常敏感。它可能难以在损失曲面中沿着狭长山谷（沟壑）前进，常常在山谷中来回震荡，而不是沿着山谷移动。
- **动量：** 通常比标准 SGD 收敛更快，特别是在存在沟壑的情况下，它通过在一致的梯度方向上累积速度并抑制震荡来做到这一点。它引入了 `momentum` 超参数，需要与学习率一同进行调整。
- **NAG：** 在实践中通常提供比标准动量略快的收敛速度，特别是对于某些凸优化问题，有时在深度学习 (deep learning)中也是如此。它使用与标准动量相同的超参数（`lr`、`momentum`）。

通常需要进行实验。一种常见做法是首先使用带有动量（或 NAG）的 SGD，采用典型的超参数值（例如 `lr=0.01`、`momentum=0.9`），然后根据观察到的训练动态（例如学习曲线）进行调整。可视化不同优化器或超参数设置的训练损失曲线可以提供有益的见解。



[交互图表：优化器收敛比较（示意）](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-5-foundational-optimizers/implementing-sgd-momentum#plot-deo3gw)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "优化器收敛比较（示意）",
    "xaxis": {
      "title": "训练轮次"
    },
    "yaxis": {
      "title": "训练损失",
      "type": "log",
      "range": [
        -1,
        0.5
      ]
    },
    "legend": {
      "title": "优化器"
    }
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines",
      "name": "SGD",
      "x": [
        1,
        5,
        10,
        15,
        20,
        25,
        30
      ],
      "y": [
        2.1,
        1.5,
        1.1,
        0.9,
        0.8,
        0.75,
        0.72
      ],
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "动量",
      "x": [
        1,
        5,
        10,
        15,
        20,
        25,
        30
      ],
      "y": [
        1.8,
        0.9,
        0.5,
        0.3,
        0.2,
        0.15,
        0.12
      ],
      "line": {
        "color": "#ae3ec9"
      }
    },
    {
      "type": "scatter",
      "mode": "lines",
      "name": "NAG",
      "x": [
        1,
        5,
        10,
        15,
        20,
        25,
        30
      ],
      "y": [
        1.7,
        0.8,
        0.4,
        0.25,
        0.18,
        0.13,
        0.1
      ],
      "line": {
        "color": "#f76707"
      }
    }
  ]
}
```

</details>



> 任务上 SGD、动量和 NAG 训练损失收敛的示意比较。注意动量和 NAG 通常如何导致损失下降比标准 SGD 更快。具体表现很大程度上取决于数据集、模型架构、初始化和超参数设置。

这些优化器，特别是带有动量/NAG 的 SGD，仍然广泛使用，并构成理解更复杂的自适应方法的根基。在下一章中，我们将研究 AdaGrad、RMSprop 和 Adam 等算法，它们引入了在训练期间自动调整学习率的机制。

## 参考资料

- [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch SGD优化器的官方文档，其中包含标准SGD、动量和Nesterov加速梯度的实现。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 286-302
  一本基础教材，对深度学习进行了全面的学术处理，其中有专门章节讨论随机梯度下降、动量和Nesterov加速梯度的理论和机制。
- [Some methods of speeding up the convergence of iteration processes](https://doi.org/10.1016/0041-5553%2864%2990082-5) — Boris T. Polyak (1964)
  Journal: USSR Computational Mathematics and Mathematical Physics; Publisher: Elsevier; Volume: 4; Pages: 1-17; DOI: [10.1016/0041-5553(64)90082-5](https://doi.org/10.1016/0041-5553%2864%2990082-5)
  介绍重球动量概念的原始学术论文，这是动量优化器的核心组成部分。

---

[上一节](07-%E6%B6%85%E6%96%AF%E6%8D%B7%E7%BD%97%E5%A4%AB%E5%8A%A0%E9%80%9F%E6%A2%AF%E5%BA%A6%20%28NAG%29.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%AF%94%E8%BE%83%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E3%80%81%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%92%8C%E5%8A%A8%E9%87%8F%E6%B3%95.md)
