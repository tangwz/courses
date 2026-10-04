# 优化器 (`torch.optim`)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/optimizers-torch-optim)

[返回章节目录](README.md) · [返回课程目录](../README.md)

优化器在神经网络 (neural network)训练中扮演着重要角色。在定义神经网络架构（例如使用 `torch.nn.Module`）并选定一个合适的损失函数 (loss function)来衡量模型预测与实际目标之间的差异后，优化器的作用是更新模型的参数 (parameter)（如权重 (weight)和偏置 (bias)），以最小化计算出的损失。`torch.optim` 包为深度学习 (deep learning)中常用的多种优化算法提供了实现。

回顾 Autograd 一章，调用 `loss.backward()` 会计算损失相对于所有 `requires_grad=True` 的模型参数的梯度。这些梯度指示了每个参数为减少损失所需的改变方向和大小。然而，仅仅计算梯度是不够的；我们需要一种机制来*应用*这些更新。优化器提供了这种机制。

## 优化过程

核心来说，训练神经网络 (neural network)是一个优化问题。我们希望找到一组参数 (parameter)（权重 (weight) $w$ 和偏置 (bias) $b$）来最小化损失函数 (loss function) $L$。梯度下降 (gradient descent)是实现这一目的的基础算法。基本思路是迭代地沿着梯度的反方向调整参数：

$\theta_{new} = \theta_{old} - \eta \nabla_{\theta} L$

这里，$\theta$ 代表一个参数（如权重或偏置），$\nabla_{\theta} L$ 是损失 $L$ 对 $\theta$ 的梯度，而 $\eta$ (eta) 是学习率，一个控制步长的超参数 (hyperparameter)。

PyTorch 的 `torch.optim` 包实现了这一核心思想，以及一些旨在提高收敛速度和稳定性的更精巧变体。

## 使用 `torch.optim`

要在 PyTorch 中使用优化器，首先需要导入该包：

```python
import torch.optim as optim
```

接下来，实例化一个优化器对象。创建时，你必须告诉优化器它应该管理哪些参数 (parameter)。通常，你会使用 `model.parameters()` 方法传入模型的参数。你还需要指定学习率 (`lr`) 以及其他可能与算法相关的超参数 (hyperparameter)。

```python
# 假设 'model' 是你的 nn.Module 子类的一个实例
# 示例：使用随机梯度下降 (SGD)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 示例：使用 Adam 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

`model.parameters()` 调用会返回一个迭代器，遍历你的模型中所有可学习的参数。优化器持有对这些张量的引用，并知道如何根据它们的 `.grad` 属性（该属性在 `loss.backward()` 调用期间填充）来更新它们。

## 常用优化器

尽管 `torch.optim` 提供了许多算法，但随机梯度下降 (gradient descent) (SGD) 和 Adam 是两个最常用的起始点。

### 随机梯度下降 (gradient descent) (SGD)

SGD 是一种经典的优化算法。在其 PyTorch 实现中，它可以在小批量数据（这是标准做法）而非单个样本上运行。它根据当前小批量数据计算出的梯度更新参数 (parameter)。

`optim.SGD` 优化器有几个重要参数：

- `params`: 要优化的参数的可迭代对象（例如，`model.parameters()`）。
- `lr`: 学习率 ($\eta$)。这是一个关键的超参数 (hyperparameter)。选择过小的值可能导致收敛缓慢，而过大的值可能导致不稳定或发散。
- `momentum`: 一种有助于加速 SGD 朝相关方向前进并抑制振荡的方法。它将之前更新向量 (vector)的一部分添加到当前更新向量中。典型值为 0.9。
- `weight_decay`: 在更新步骤中隐式地向损失函数 (loss function)添加 L2 正则化 (regularization)（对大权重 (weight)的惩罚）。这有助于防止过拟合 (overfitting)。

```python
# 带有动量和权重衰减的 SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

### Adam (自适应矩估计)

Adam 是一种自适应学习率优化算法，这意味着它为不同参数 (parameter)计算各自的学习率。它结合了 RMSprop（根据最近梯度平方的平均值调整学习率）和动量的思想。Adam 通常比 SGD 收敛更快，且相对有效，常常在默认设置下表现良好。

`optim.Adam` 的重要参数：

- `params`: 要优化的参数。
- `lr`: 初始学习率（Adam 在内部进行调整）。常见的起始值是 `1e-3` 或 `0.001`。
- `betas`: 一个元组 `(beta1, beta2)`，控制动量估计的指数衰减率（通常是 `(0.9, 0.999)`）。
- `eps`: 添加到分母中的一个小项，用于数值稳定性（通常是 `1e-8`）。
- `weight_decay`: 添加 L2 正则化 (regularization)。

```python
# 带有默认 betas 和指定学习率的 Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 带有自定义 betas 和权重衰减的 Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)
```

其他常用优化器，如 `RMSprop`、`Adagrad` 和 `AdamW`（改进了权重 (weight)衰减处理的 Adam）也在 `torch.optim` 中提供。选择通常取决于具体问题和实际表现。

## 将优化器整合到训练循环中

在训练循环中使用优化器涉及每次迭代的两个主要步骤，通常在计算损失和梯度之后执行：

1. **`optimizer.zero_grad()`**: 在计算当前小批量数据的梯度（通过 `loss.backward()`）之前，必须清除之前迭代累积的梯度。PyTorch 默认在每次调用 `backward()` 时累积梯度。如果你忘记将其归零，来自多个批次的梯度将会混合，导致不正确的更新。这通常在循环开始时或在调用 `backward()` 之前执行。
2. **`optimizer.step()`**: 在用 `loss.backward()` 计算梯度后，调用 `optimizer.step()` 会更新所有已在优化器中注册的参数 (parameter)。它会使用计算出的梯度（存储在 `parameter.grad` 中）和学习率来应用特定的优化算法（如 SGD 或 Adam）。

以下是整合了优化器的训练迭代的简化结构：

```python
# 假设模型、准则（损失函数）和优化器已定义
# 假设 data_loader 提供批量的输入和目标

model.train() # 将模型设置为训练模式
for inputs, targets in data_loader:
    # 1. 梯度归零
    optimizer.zero_grad()

    # 2. 前向传播：计算模型预测
    outputs = model(inputs)

    # 3. 计算损失
    loss = criterion(outputs, targets)

    # 4. 反向传播：计算梯度
    loss.backward()

    # 5. 更新权重
    optimizer.step()

    # (可选：日志记录、指标计算等)
```

下图展示了优化器在标准训练周期中的作用：

> 优化器使用 `loss.backward()` 计算出的梯度，通过 `optimizer.step()` 更新模型参数，在此之前确保使用 `optimizer.zero_grad()` 清除了之前的梯度。

## 调整学习率

有时，在训练期间调整学习率是有益的。例如，你可能希望以较大的学习率开始以加快初始进展，之后再降低学习率以更精细地调整参数 (parameter)。PyTorch 在 `torch.optim.lr_scheduler` 中为此提供了学习率调度器。这些调度器根据预设规则（例如，每隔几个 epoch 减少一次，或者当验证性能稳定时）调整与优化器相关的学习率。尽管它们功能强大，但调度器的详细用法通常在更高级的背景下介绍。

总结来说，`torch.optim` 是 PyTorch 中训练神经网络 (neural network)不可或缺的工具。通过选择合适的优化器、配置其学习率等超参数 (hyperparameter)，并将 `optimizer.zero_grad()` 和 `optimizer.step()` 正确整合到训练循环中，你便为模型提供了从数据中学习并最小化损失的机制。尝试不同的优化器和学习率是开发有效深度学习 (deep learning)模型的常规部分。

## 参考资料

- [torch.optim - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/optim.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  PyTorch `torch.optim` 包的官方文档，详细介绍了可用的优化器及其用法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍 Adam 优化算法的原始研究论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习理论基础的综合性教科书，包括优化算法。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出了一种将权重衰减与梯度更新分离的方法，从而产生了 AdamW 等算法。

---

[上一节](06-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%20%28%60torch.nn%60%20%E6%8D%9F%E5%A4%B1%29.md) · [下一节](08-%E7%BB%83%E4%B9%A0%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%BD%91%E7%BB%9C.md)
