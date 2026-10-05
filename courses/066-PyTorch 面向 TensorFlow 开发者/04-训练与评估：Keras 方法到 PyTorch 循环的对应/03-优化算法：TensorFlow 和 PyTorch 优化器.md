# 优化算法：TensorFlow 和 PyTorch 优化器

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/optimizers-pytorch-tf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当你在TensorFlow中使用Keras训练模型时，你可能在`model.compile()`步骤中指定了优化算法，从`tf.keras.optimizers.Adam`或`tf.keras.optimizers.SGD`等选项中选择。PyTorch将其优化算法组织在`torch.optim`包中。这些优化器是推动学习的引擎，它们通过根据计算出的梯度调整模型的参数 (parameter)（权重 (weight)和偏置 (bias)）来使损失函数 (loss function)最小化。

在PyTorch中，你需要实例化一个优化器，并将其与模型的参数明确关联起来。这与Keras略有不同，Keras在编译期间会将优化器更广泛地关联到模型对象。

### 实例化优化器

要使用`torch.optim`中的优化器，你首先需要创建模型的一个实例（它是一个`nn.Module`子类）。然后，将模型的参数 (parameter)以及任何算法特定的超参数 (hyperparameter)（如学习率）传递给优化器的构造函数。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设 'model' 是你的 nn.Module 子类的一个实例
# 例如：
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#
# model = SimpleNet()

# 用于优化器实例化示例的占位模型
model = nn.Linear(10, 2) # 一个带有可学习参数的简单模型

# 实例化一个SGD优化器
learning_rate_sgd = 0.01
momentum_sgd = 0.9
optimizer_sgd = optim.SGD(model.parameters(), lr=learning_rate_sgd, momentum=momentum_sgd)

# 实例化一个Adam优化器
learning_rate_adam = 0.001
optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate_adam)
```

`model.parameters()`方法返回模型中所有可学习参数的迭代器。这告诉优化器它负责更新哪些张量。这是连接你定义的模型结构与优化过程的一个重要步骤。

### 训练循环中的优化器流程

如前所述，PyTorch要求你明确编写训练循环。优化器在此循环中扮演着核心角色，通常为每批数据涉及三个不同的步骤：

1. **`optimizer.zero_grad()`**：此调用会重置优化器管理的所有模型参数 (parameter)的梯度。在每次迭代开始时（或在当前批次的`loss.backward()`调用之前）执行此操作非常重要。默认情况下，当多次调用`loss.backward()`时，PyTorch会累积梯度。如果不将其归零，来自前一批次的梯度将干扰当前批次的梯度计算，导致更新不正确。TensorFlow Keras的`model.fit()`会在后台自动处理梯度重置。
2. **`loss.backward()`**：计算当前批次的损失后，在损失张量上调用`backward()`会计算损失相对于所有`requires_grad=True`且参与损失计算的模型参数的梯度。这些梯度存储在每个参数张量的`.grad`属性中。
3. **`optimizer.step()`**：此方法更新模型参数的值。它使用存储在每个参数`.grad`属性中的梯度（由`loss.backward()`计算）并应用特定优化算法的更新规则（例如，带有动量的SGD，Adam的更新）。如果你在TensorFlow中编写过自定义训练循环，这类似于`optimizer.apply_gradients()`。

下面是一个图，说明了优化器在单个训练迭代中的作用：

> 优化器的循环：归零梯度，允许从损失计算新梯度，然后对模型参数应用更新。

### 常见优化器：比较

你在TensorFlow Keras中熟悉的大多数优化算法在`torch.optim`中都有直接对应。它们的基本数学原理通常相同，尽管默认的超参数 (parameter) (hyperparameter)值或命名约定有时可能略有不同。

#### 随机梯度下降 (gradient descent) (SGD)

SGD是一种基本优化器。在PyTorch中，它作为`torch.optim.SGD`提供。

- **PyTorch**: `optim.SGD(params, lr=<required>, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
- **TensorFlow Keras**: `tf.keras.optimizers.SGD(learning_rate=<required>, momentum=0.0, nesterov=False, ...)`

```python
# PyTorch SGD
optimizer_sgd_pytorch = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

PyTorch的`SGD`中的`weight_decay`参数通过在损失函数 (loss function)中添加一个项来实现L2正则化 (regularization)，从而有效地惩罚大权重 (weight)。

#### Adam（自适应矩估计）

Adam是一种流行的自适应学习率优化算法。

- **PyTorch**: `optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)`
- **TensorFlow Keras**: `tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, ...)`

```python
# PyTorch Adam
optimizer_adam_pytorch = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

注意，PyTorch使用元组`betas=(beta1, beta2)`，而TensorFlow Keras则将`beta_1`和`beta_2`作为单独的参数。默认的`epsilon`值也略有不同。PyTorch的`Adam`还包含一个用于L2正则化的`weight_decay`参数，这与AdamW中解耦的权重衰减不同。

#### AdamW（带有解耦权重衰减的Adam）

AdamW通过将权重衰减与基于梯度的更新解耦来改进Adam。这通常比直接将L2正则化应用于Adam能带来更好的性能和泛化能力。

- **PyTorch**: `optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)`
- **TensorFlow Keras**: `tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-7, ...)`

```python
# PyTorch AdamW
optimizer_adamw_pytorch = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

两个框架都提供了Adam的这个改进版本。默认的`weight_decay`值可能不同，因此在移植精确的超参数时请务必查阅文档。

#### RMSprop

RMSprop是另一种自适应学习率算法。

- **PyTorch**: `optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)`
- **TensorFlow Keras**: `tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False, ...)`

```python
# PyTorch RMSprop
optimizer_rmsprop_pytorch = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, momentum=0.0)
```

一个主要参数名称的区别是PyTorch中用于平滑常数的`alpha`，而TensorFlow中使用的是`rho`。

下面的表格总结了这些常见的优化器及其用法模式：

| 特性/优化器 | TensorFlow Keras (`tf.keras.optimizers`) | PyTorch (`torch.optim`) | 说明 |
| --- | --- | --- | --- |
| **实例化** | `optimizer = SGD(lr=0.01)` | `optimizer = optim.SGD(model.parameters(), lr=0.01)` | PyTorch优化器在初始化时需要`model.parameters()`。 |
| **SGD** | `SGD(...)` | `SGD(...)` | `lr`、`momentum`、`nesterov`通常保持一致。PyTorch增加了`weight_decay`。 |
| **Adam** | `Adam(beta_1, beta_2, epsilon)` | `Adam(betas=(b1,b2), eps)` | `beta`和`epsilon`的命名和一些默认值可能有所不同。PyTorch增加了`weight_decay`。 |
| **AdamW** | `AdamW(weight_decay, ...)` | `AdamW(weight_decay, ...)` | 解耦的权重衰减。推荐使用此方法而非带L2正则化的Adam。默认`weight_decay`可能不同。 |
| **RMSprop** | `RMSprop(rho, ...)` | `RMSprop(alpha, ...)` | 平滑常数：`rho`（TF）与`alpha`（PyTorch）。默认`lr`和`epsilon`可能不同。 |
| **梯度归零** | 在`model.fit()`中自动执行；在自定义循环中手动执行 | 必须明确调用`optimizer.zero_grad()` | 如果不清除，PyTorch默认会累积梯度。 |
| **权重更新** | 在`model.fit()`中自动执行；`opt.apply_gradients()` | 必须明确调用`optimizer.step()` | 根据参数的`.grad`属性更新参数。 |

### 参数 (parameter)特定选项

PyTorch优化器的一项强大能力是能够为不同组的模型参数指定不同的超参数 (hyperparameter)。这是通过向优化器传递一个字典列表来实现的，其中每个字典定义一个参数组及其特定选项。这对于微调 (fine-tuning)特别有用，在这种情况下，你可能希望预训练 (pre-training)的基础模型层具有比新添加的分类头小得多的学习率。

```python
# model.feature_extractor = nn.Sequential(...)
# model.classifier = nn.Linear(...)

optimizer_param_groups = optim.SGD([
    {'params': model.feature_extractor.parameters(), 'lr': 1e-4}, # 基础层使用较小的学习率
    {'params': model.classifier.parameters(), 'lr': 1e-2}        # 新层使用较大的学习率
], momentum=0.9)
```

如果一个超参数未在组字典中指定，它将默认为传递给优化器构造函数的值（例如，上面示例中的`momentum=0.9`适用于所有组）。

### 学习率调度

在训练期间调整学习率是提高收敛速度和最终模型性能的常见方法。在Keras中，你可能会使用`ReduceLROnPlateau`或`LearningRateScheduler`等回调。PyTorch通过其`torch.optim.lr_scheduler`模块提供了类似的机制。

调度器通过传递一个优化器实例和调度器特定的参数 (parameter)来实例化。然后，你通常在每个epoch之后（或有时在每个批次之后，取决于调度器类型和你的策略）调用`scheduler.step()`。

```python
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# 示例：使用StepLR，每隔step_size个epoch将学习率按gamma因子衰减
# optimizer = optim.SGD(model.parameters(), lr=0.1) # 假设优化器已定义
scheduler_steplr = StepLR(optimizer_sgd, step_size=30, gamma=0.1)

# 示例：使用ReduceLROnPlateau，当某个指标停止改进时降低学习率
# val_loss将在验证期间跟踪
scheduler_plateau = ReduceLROnPlateau(optimizer_adam, 'min', patience=5, factor=0.5, verbose=True)

# 在你的训练循环中，在optimizer.step()之后：
# 对于基于epoch的调度器，如StepLR：
# for epoch in range(num_epochs):
#   # ... 训练阶段 ...
#   scheduler_steplr.step()

# 对于基于指标的调度器，如ReduceLROnPlateau：
# for epoch in range(num_epochs):
#   # ... 训练阶段 ...
#   val_loss = # ... 验证阶段 ...
#   scheduler_plateau.step(val_loss)
```

有多种调度器可用，包括用于指数衰减、余弦退火和更复杂策略的调度器。

从TensorFlow在`model.compile()`中处理优化器到PyTorch明确使用`torch.optim`，需要更亲力亲为的方法。然而，这种明确性让你对训练过程有了更精细的控制，并更清楚地了解参数更新是如何发生的。通过了解如何实例化优化器、如何使用`zero_grad()`、`backward()`和`step()`管理其流程，以及如何使用参数组和学习率调度器等功能，你可以有效地训练PyTorch模型。

## 参考资料

- [torch.optim](https://pytorch.org/docs/stable/optim.html) — PyTorch Authors (2025)
  PyTorch优化算法的官方文档，包含详细的API规范和使用示例。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: 3rd International Conference for Learning Representations, San Diego, 2015; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化算法的奠基性论文。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  引入AdamW的论文，将权重衰减与梯度更新分离，改进了正则化。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍深度学习中优化算法及其理论基础的教科书。
- [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html) — PyTorch Authors (2024)
  Publisher: PyTorch
  PyTorch学习率调度器的官方文档，解释了不同的策略和用法。

---

[上一节](02-TensorFlow%20%E5%92%8C%20PyTorch%20%E4%B8%AD%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md) · [下一节](04-PyTorch%20%E4%B8%AD%E7%9A%84%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%E4%B8%8E%E6%9D%83%E9%87%8D%E6%9B%B4%E6%96%B0.md)
