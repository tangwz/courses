---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-training-loops-for-keras-devs"
lesson: "gradients-weight-updates-pytorch"
sourceId: 5761
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/gradients-weight-updates-pytorch"
title: "PyTorch 中的梯度计算与权重更新"
description: "详细了解反向传播过程：PyTorch 中梯度清零、梯度计算和优化器步骤。"
order: 4
plots: []
sourceHash: "10ebc5e41db5e8237e80e3fe403b98ffb4cb2befd0002bf5ef8706ffa5ef44f5"
sourceCorrections: []
---

在模型架构定义好、损失函数 (loss function)选定、优化器配置完成后，训练过程中的下一步就是实际计算梯度，并使用它们更新模型的权重 (weight)。对于 TensorFlow 用户来说，他们可能习惯于使用 `tf.GradientTape` 记录操作，然后 `tape.gradient()` 计算梯度，接着是 `optimizer.apply_gradients()`。PyTorch 提供了一个类似明确但略有不同的工作流程，其核心是 `loss.backward()` 方法和 `optimizer.step()` 调用。掌握这个过程是编写高效 PyTorch 训练循环的基础。

本节涉及 PyTorch 如何使用其自动微分引擎 `autograd` 计算梯度，以及优化器如何应用这些梯度来调整模型参数 (parameter)的具体内容。

## 核心步骤：概览

在 PyTorch 的每次训练迭代中，计算完损失后，与梯度和权重 (weight)更新相关的有两个主要操作：

1. **反向传播 (backpropagation) (`loss.backward()`)**：这会计算损失相对于所有 `requires_grad=True` 且是导致损失的计算图一部分的模型参数 (parameter)的梯度。这些梯度存储在每个相应参数张量的 `.grad` 属性中。
2. **优化器步进 (`optimizer.step()`)**：这会根据你选择的优化算法（例如，SGD、Adam），使用反向传播中计算出的梯度来更新模型参数的值。

然而，还有一个虽小但非常重要的准备步骤：

- **梯度清零 (`optimizer.zero_grad()`)**：在计算当前批次的新梯度之前，你必须清除任何从之前批次累积的梯度。

让我们详细了解每一个步骤。

### 清零旧梯度：`optimizer.zero_grad()`

在 PyTorch 中，梯度默认是累积的。这意味着当调用 `loss.backward()` 时，新计算的梯度会 *添加* 到每个参数 (parameter) `.grad` 属性中的现有值上。虽然这种累积在高级情况（例如为不适合内存的大批次实现梯度累积）中可能有用，但对于常规训练，你希望仅基于当前批次计算梯度。

因此，在你的训练循环中，对于每个批次，在正向传播之前或紧随上一个优化器步进之后，第一个步骤就是清零梯度。

```python
# 假设 optimizer 是 torch.optim.Optimizer 的一个实例
optimizer.zero_grad()
```

如果你忘记这一步，你的梯度将是来自当前批次和之前批次的混合，导致不正确的更新，并可能阻止你的模型有效学习。可以把它想象成在每次新计算前清除旧记录。这与 TensorFlow 的 `tf.GradientTape` 略有不同，`tf.GradientTape` 通常在每次调用 `tape.gradient()` 时重新计算梯度，除非磁带被设置为持久化。

### 反向传播 (backpropagation)：`loss.backward()`

在你的模型执行正向传播以获得预测，然后计算损失（例如，`loss = criterion(outputs, targets)`）之后，下一个重要的命令是：

```python
loss.backward()
```

这一行代码触发了 PyTorch 的 `autograd` 引擎。以下是底层发生的情况：

1. **动态计算图**：PyTorch 在你的操作执行时动态构建一个计算图。每个 `requires_grad` 属性设置为 `True`（模型参数 (parameter)默认如此）并参与操作的张量都会成为这个图的一部分。对于反向传播，`loss` 张量通常是这个图的根节点。
2. **梯度计算**：`loss.backward()` 从 `loss` 张量向后遍历此图。它应用链式法则来计算损失相对于图中每个 `requires_grad=True` 且是导致损失的操作的输入的张量的导数。
3. **存储梯度**：对于每个参数（它们是此图中的叶子节点，通常是 `torch.nn.Parameter` 实例），计算出的梯度 $dL/dp$（其中 $L$ 是损失，$p$ 是参数）会累积到其 `.grad` 属性中。例如，如果 `layer.weight` 是一个参数，在 `loss.backward()` 之后，`layer.weight.grad` 将保存损失相对于 `layer.weight` 的梯度。

如果你熟悉 TensorFlow 的 `tf.GradientTape`，那么 `loss.backward()` 类似于调用 `tape.gradient(loss, model.trainable_variables)`。主要区别在于 `loss.backward()` 直接填充参数自身的 `.grad` 属性，而不是返回一个梯度张量列表。

### 应用更新：`optimizer.step()`

一旦 `loss.backward()` 填充了模型参数 (parameter)的 `.grad` 属性，优化器就拥有了更新权重 (weight)所需的所有信息。你可以通过以下方式触发此更新：

```python
optimizer.step()
```

优化器（例如 `torch.optim.SGD` 或 `torch.optim.Adam` 的一个实例）是用模型的参数初始化的（例如，`optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`）。当调用 `optimizer.step()` 时，它会遍历它被给予的所有参数。对于每个参数 `p`，它使用 `p.grad` 及其特定的更新规则来修改 `p.data`（实际的张量值）。

例如，一个简单的 SGD 优化器会执行如下更新：
$p.\text{数据} = p.\text{数据} - \text{学习率} \times p.\text{梯度}$

这类似于 TensorFlow 中的 `optimizer.apply_gradients(zip(grads, model.trainable_variables))`，其中 `grads` 是从 `tape.gradient()` 获得的梯度。

## 综合起来：一个代码骨架

以下是这些步骤如何融入典型训练循环迭代中：

```python
# 假设：
# model: 你的 torch.nn.Module 实例
# data_loader: 提供 (输入, 目标) 批次数据
# criterion: 你的损失函数 (例如, nn.CrossEntropyLoss())
# optimizer: 你的优化器 (例如, optim.Adam(model.parameters()))

model.train() # 将模型设置为训练模式

for inputs, targets in data_loader:
    # 1. 清零上一迭代的梯度
    optimizer.zero_grad()

    # 2. 执行正向传播：获取预测
    outputs = model(inputs)

    # 3. 计算损失
    loss = criterion(outputs, targets)

    # 4. 执行反向传播：计算损失相对于模型参数的梯度
    loss.backward()

    # 5. 使用计算出的梯度更新模型参数
    optimizer.step()

    # (可选：日志记录、指标计算等)
```

这种循环结构是 PyTorch 中训练模型的标准方式，提供了学习过程每个阶段的清晰视图。

## 流程可视化

梯度计算和权重 (weight)更新的操作序列可以可视化如下：

> 单个 PyTorch 训练迭代中操作序列及其对模型参数 (parameter)的影响。每一步都为学习过程的某个部分做准备或执行该部分，从清除旧梯度到应用新更新。

## 给 TensorFlow 开发者的要点

从 TensorFlow 的 `GradientTape` 过渡到 PyTorch 的 `loss.backward()` 和 `optimizer.step()` 涉及一些视角的转变：

- **梯度累积**：记住 `optimizer.zero_grad()`。PyTorch 的默认累积行为不同于 `GradientTape` 的典型用法。
- **隐式梯度存储**：`loss.backward()` 直接将 `.grad` 属性填充到参数 (parameter)张量上。你通常不需要在将梯度张量传递给优化器之前显式处理它们，这与使用 `tape.gradient()` 的方式不同。
- **明确性**：PyTorch 的方法使得训练循环中梯度计算和更新过程的每一步都非常明确。

通过掌握这三个核心组成部分：`optimizer.zero_grad()`、`loss.backward()` 和 `optimizer.step()`，你将获得对训练过程的细粒度控制，这是 PyTorch 的突出优势之一。这种明确的控制有助于更轻松地调试和实现更复杂的训练方案。

## 参考资料

- [Autograd: automatic differentiation](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Developers (2024)
  PyTorch自动微分引擎的官方文档，解释了计算图、梯度计算和`requires_grad`属性。
- [torch.optim](https://pytorch.org/docs/stable/optim.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  涵盖PyTorch优化包的官方文档，详细介绍了根据各种算法清零梯度和更新参数的功能。
- [Dive into Deep Learning](https://d2l.ai/index.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2023)
  Publisher: Cambridge University Press
  一本内容全面的开源教材，提供使用PyTorch训练神经网络的实践示例，包括梯度计算和权重更新。
