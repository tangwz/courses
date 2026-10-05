# Autograd 引擎机制

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/autograd-mechanics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`autograd` 是驱动动态计算图进行梯度计算的引擎。该系统是 PyTorch 自动计算梯度的基础，是训练神经网络 (neural network)通过反向传播 (backpropagation)不可或缺的。

其核心是，`autograd` 执行反向模式自动微分。当您对 `requires_grad` 设为 `True` 的张量进行操作时，PyTorch 会构建一个表示操作序列的图。此图是随着计算的发生动态构建的。`autograd` 引擎随后从最终输出（通常是标量损失）开始，反向遍历此图，以计算该输出相对于参数 (parameter)（图的叶节点，通常是模型权重 (weight)和偏置 (bias)）的梯度。

## `backward()` 调用

此过程通常通过在一个张量上调用 `.backward()` 方法来启动，最常见的是在正向传播结束时计算的标量损失值。

```python
import torch

# 示例设置
w = torch.randn(5, 3, requires_grad=True)
x = torch.randn(1, 5)
# 如果 x 仅是输入数据，确保它不需要梯度
# x.requires_grad_(False) # 或在创建时不设置 requires_grad

y = x @ w  # y 通过矩阵乘法依赖于 w
z = y.mean() # z 是从 y 派生的标量

# 从 z 开始计算梯度
z.backward()

# 梯度现在填充在 w.grad 中
# d(z)/dw 梯度被计算并存储
print(w.grad.shape)
# 输出: torch.Size([5, 3])
```

当调用 `z.backward()` 时，`autograd` 从 `z` 开始反向运算。由于 `z` 是一个标量，`backward()` 隐式使用 `1.0` 作为起始梯度。这意味着 $\frac{\partial z}{\partial z} = 1$。如果您在一个非标量张量 `t` 上调用 `backward()`，则必须提供一个 `gradient` 输入参数 (parameter)，它应与 `t` 的形状相同。此输入表示某个最终标量损失 $L$ 相对于张量 `t` 的梯度，即 $\frac{\partial L}{\partial t}$。

从本质上讲，`autograd` 计算向量 (vector)-雅可比乘积 (VJP)。回想一下导数的链式法则。如果我们有一个标量损失 $L$，它是向量 $\mathbf{y}$ 的函数，$L=g(\mathbf{y})$，并且 $\mathbf{y}$ 本身是另一个向量 $\mathbf{x}$ 的函数，$\mathbf{y}=f(\mathbf{x})$，那么 $L$ 相对于 $\mathbf{x}$ 的梯度由下式给出：


$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}
$$


此处，$\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$ 是 $f$ 相对于 $\mathbf{x}$ 的雅可比矩阵，而 $\frac{\partial L}{\partial \mathbf{y}}$ 是一个行向量，表示 $L$ 相对于 $\mathbf{y}$ 的梯度。由 `y.backward(gradient=dL_dy)` 计算的 VJP 正是乘积 $\mathbf{v}^T \mathbf{J}$，其中 $\mathbf{v}$ 是上游梯度（由 `gradient=dL_dy` 表示），而 $\mathbf{J}$ 是雅可比矩阵 $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$。在一个标量损失 $z$ 上调用 `z.backward()` 对应于使用初始梯度向量 $\mathbf{v} = [1.0]$。与显式构建可能庞大的雅可比矩阵相比，这种 VJP 方法在计算上更为高效。

## 使用 `grad_fn` 遍历图

每个由至少一个 `requires_grad=True` 的张量参与操作而产生的张量，都将具有 `grad_fn` 属性。此属性是指向在正向传播期间创建该张量的函数对象（例如 `AddBackward0`、`MulBackward0`、`MmBackward0` 等）的引用。重要的是，此函数对象存储对其输入的引用，并包含其相应反向操作的实现，这是梯度计算所必需的。

我们来检查之前示例中的 `grad_fn` 属性：

```python
# 我们需要中间张量来检查它们的 grad_fn
w = torch.randn(5, 3, requires_grad=True)
x = torch.randn(1, 5)
y = x @ w
z = y.mean()

print(f"y 源自: {y.grad_fn}")
# 输出: y originated from: <MmBackward0 object at 0x...>

print(f"z 源自: {z.grad_fn}")
# 输出: z originated from: <MeanBackward0 object at 0x...>

# 用户创建的叶张量没有 grad_fn
print(f"w.grad_fn: {w.grad_fn}")
# 输出: w.grad_fn: None
print(f"x.grad_fn: {x.grad_fn}")
# 输出: x.grad_fn: None
```

`grad_fn` 对象形成一个有向无环图 (DAG)，记录计算历史。当执行 `z.backward()` 时：

1. `autograd` 引擎从目标张量 `z` 开始。
2. 它访问 `z.grad_fn`（即 `MeanBackward0`）。使用传入梯度（隐式为 `1.0`），`MeanBackward0` 计算均值操作相对于其输入 `y` 的梯度。我们称之为 $\frac{\partial z}{\partial y}$。
3. 引擎随后移动到 `y` 指示的图中下一个节点。它使用 `y.grad_fn`（`MmBackward0`）和传入梯度 $\frac{\partial z}{\partial y}$ 来计算矩阵乘法相对于其输入 `x` 和 `w` 的梯度。这涉及计算 $\frac{\partial z}{\partial y} \frac{\partial y}{\partial w}$ 和 $\frac{\partial z}{\partial y} \frac{\partial y}{\partial x}$。
4. 由于 `w` 是一个叶张量（由用户创建）并且 `requires_grad=True`，计算出的梯度 $\frac{\partial z}{\partial w}$ 将*累积*到 `w.grad` 属性中。
5. 由于 `x` 的 `requires_grad=False`，沿此路径的梯度计算停止，`x.grad` 保持为 `None`。

此递归过程持续进行，将链式法则反向应用于图，直到所有返回到需要梯度的叶张量的路径都已处理完毕。

> `z.backward()` 启动的反向传播 (backpropagation)图示。Autograd 沿着 `grad_fn` 指针（由从张量指向函数节点的箭头表示）从输出 `z` 反向回溯。计算出的相对于 `w` 的梯度累积到 `w.grad` 中。对于涉及 `requires_grad=False` 的张量（如 `x`）的路径，计算会停止。

## 梯度累积

`autograd` 行为的一个重要方面是梯度会*累积*到叶张量的 `.grad` 属性中。每次调用 `backward()` 时，为参数 (parameter)新计算的梯度都会*添加*到其 `.grad` 属性中当前存储的值上。如果 `.grad` 最初为 `None`，则会用第一次计算的梯度对其进行初始化。

这种设计选择在典型的训练循环中需要明确的管理。在计算损失并对新批次数据执行反向传播 (backpropagation)之前，您必须重置所有模型参数的梯度。否则，当前批次的梯度将与前一批次的梯度相加，导致错误的权重 (weight)更新。执行此操作的标准方法是使用优化器的 `zero_grad()` 方法：

```python
# 假设已定义 model、optimizer、criterion、dataloader

for inputs, targets in dataloader:
    # 1. 重置上一迭代的梯度
    optimizer.zero_grad()

    # 2. 执行正向传播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 3. 执行反向传播以计算梯度
    loss.backward()

    # 4. 使用计算出的梯度更新模型参数
    optimizer.step()
```

此累积行为可有意用于诸如实现梯度累积以模拟更大批次大小的场景，尤其是在 GPU 内存有限时。在这种情况下，您将在调用 `optimizer.step()` 和 `optimizer.zero_grad()` 之前，执行多次正向和反向传播同时累积梯度。

## 控制梯度计算

PyTorch 对 `autograd` 引擎的操作提供精细控制：

- **`requires_grad` 标志：** 这个基本张量属性决定了涉及该张量的操作是否应跟踪以进行梯度计算。直接创建的张量通常默认为 `requires_grad=False`。`torch.nn.Module` 中的参数 (parameter)会自动设置为 `requires_grad=True`。您可以使用 `my_tensor.requires_grad_(True)`（原地操作）手动更改此标志。
- **`torch.no_grad()` 上下文 (context)管理器：** 这是一个广泛使用的工具，用于在特定代码块内禁用梯度计算。在 `with torch.no_grad():` 块内执行的任何操作都不会被 `autograd` 跟踪，即使输入张量具有 `requires_grad=True`。这会大幅减少内存消耗并加速计算，使其非常适合模型评估、推理 (inference)或任何不需要梯度的代码部分。

  ```python
  with torch.no_grad():
      # 此处的操作将不被跟踪
      predictions = model(validation_data)
      # 内存使用量更低，计算速度更快
  ```
- **`torch.enable_grad()` 上下文管理器：** 相反，此上下文管理器在其范围内重新启用梯度跟踪。如果您需要为恰好位于较大 `torch.no_grad()` 块内的一小部分代码计算梯度，这会很有用。
- **`.detach()` 方法：** 调用 `tensor.detach()` 会创建一个新张量，它与原始张量共享底层数据存储，但已明确与计算图历史分离。新张量将具有 `requires_grad=False`。梯度不会通过此分离的张量流回原始图。当您需要使用张量的值而不影响与其历史相关的梯度计算时，这会很有用。

扎实掌握这些 `autograd` 机制对于调试训练问题（如梯度爆炸或梯度消失）、优化内存使用、理解复杂模型的行为以及有效实现自定义操作或训练循环非常有价值。尽管 `autograd` 会自动处理微分的复杂性，但了解其工作原理有助于您更熟练地使用 PyTorch。

## 参考资料

- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch developers (2025)
  Publisher: PyTorch Foundation
  PyTorch 官方文档，解释了其自动微分引擎、图构建、反向传播和核心功能。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717726) — Andreas Griewank, Andrea Walther (2008)
  Publisher: Society for Industrial and Applied Mathematics (SIAM); DOI: [10.1137/1.9780898717726](https://doi.org/10.1137/1.9780898717726)
  一本关于自动微分的权威著作，详细说明了其原理、各种模式（包括反向模式）和数学基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本标准教科书，提供了关于反向传播、计算图及其在神经网络训练中作用的基础知识。

---

[上一节](02-%E7%90%86%E8%A7%A3%E8%AE%A1%E7%AE%97%E5%9B%BE.md) · [下一节](04-%E8%87%AA%E5%AE%9A%E4%B9%89%20Autograd%20%E5%87%BD%E6%95%B0%EF%BC%9A%E5%89%8D%E5%90%91%E4%B8%8E%E5%8F%8D%E5%90%91.md)
