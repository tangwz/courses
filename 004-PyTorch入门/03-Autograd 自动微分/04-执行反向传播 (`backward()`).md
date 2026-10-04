# 执行反向传播 (`backward()`)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/performing-backpropagation-backward)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在使用 PyTorch 时，张量会被设置，并且需要梯度的张量会被标记 (token)为 `requires_grad=True`。PyTorch 会在其动态计算图中忠实地跟踪这些张量上的操作。为了计算这些梯度，`backward()` 方法是必不可少的。

`backward()` 方法是驱动 PyTorch 自动微分的引擎。当在一个张量上调用它时，通常是模型最终的标量损失值，它会启动使用链式法则在整个计算图中计算梯度。它会计算被调用的张量相对于图中所有 `requires_grad=True` 的“叶”张量的梯度（这些通常是你的模型参数 (parameter)或你需要梯度的初始输入）。

### 启动梯度计算

你几乎总是在一个标量张量上调用 `backward()`，这通常是你的损失函数 (loss function)计算的结果。例如，如果 `loss` 包含代表模型批次误差的单个数值：

```python
import torch

# 示例设置（想象这些是模型的结果）
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 执行一些操作（构建图）
y = w * x + b  # y = 3*2 + 1 = 7
loss = y * y   # loss = 7*7 = 49 (a scalar)

# 反向传播之前，梯度为 None
print(f"Gradient for x before backward: {x.grad}")
print(f"Gradient for w before backward: {w.grad}")
print(f"Gradient for b before backward: {b.grad}")

# 计算梯度
loss.backward()

# 反向传播之后，梯度被填充
print(f"Gradient for x after backward: {x.grad}") # d(loss)/dx = d(y^2)/dx = 2*y*(dy/dx) = 2*y*w = 2*7*3 = 42
print(f"Gradient for w after backward: {w.grad}") # d(loss)/dw = d(y^2)/dw = 2*y*(dy/dw) = 2*y*x = 2*7*2 = 28
print(f"Gradient for b after backward: {b.grad}") # d(loss)/db = d(y^2)/db = 2*y*(dy/db) = 2*y*1 = 2*7*1 = 14
```

输出：

```text
Gradient for x before backward: None
Gradient for w before backward: None
Gradient for b before backward: None
Gradient for x after backward: 42.0
Gradient for w after backward: 28.0
Gradient for b after backward: 14.0
```

如你所见，调用 `loss.backward()` 计算了梯度 $\frac{\partial \text{loss}}{\partial x}$、$\frac{\partial \text{loss}}{\partial w}$ 和 $\frac{\partial \text{loss}}{\partial b}$ 并将它们存储在 `x`、`w` 和 `b` 张量各自的 `.grad` 属性中。

### 为什么在标量上调用 `.backward()`？

Autograd 被设计用于计算雅可比向量 (vector)积 (JVP)。当你在一个标量张量 $L$ 上调用 `backward()` 时，它隐式地等同于以 $1.0$ 的起始梯度调用 `backward()`。这使得 PyTorch 可以使用链式法则，从标量损失向后传播，高效地计算所有参数 (parameter) $\mathbf{p}$ 的梯度 $\frac{\partial L}{\partial \mathbf{p}}$。

如果你尝试在一个**非标量张量**（即包含多个元素的张量）上调用 `.backward()`，PyTorch 无法隐式知道如何根据最终（未显式）的标量损失来为该张量中每个元素的梯度加权。你将收到一个运行时错误，要求提供 `gradient` 参数：

```python
# 继续前面的例子，但使用非标量 y
x_vector = torch.tensor([2.0, 4.0], requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y_non_scalar = w * x_vector + b # y_non_scalar 现在是非标量张量，包含两个元素：[7.0, 13.0]

try:
    y_non_scalar.backward() # 这将导致错误
except RuntimeError as e:
    print(f"Error calling backward() on non-scalar: {e}")

# 要使其工作，需要提供一个与 y_non_scalar 形状匹配的梯度张量
# 这代表了某个最终损失相对于 y_non_scalar 的梯度。
# 为演示目的，我们使用 torch.ones_like(y_non_scalar)
grad_tensor = torch.ones_like(y_non_scalar)
y_non_scalar.backward(gradient=grad_tensor)
print(f"Gradient for x_vector after y_non_scalar.backward(gradient=...): {x_vector.grad}")
print(f"Gradient for w after y_non_scalar.backward(gradient=...): {w.grad}")
```

输出：

```text
Error calling backward() on non-scalar: grad can be implicitly created only for scalar outputs
Gradient for x_vector after y_non_scalar.backward(gradient=...): tensor([3., 3.])
Gradient for w after y_non_scalar.backward(gradient=...): 6.0
```

在大多数标准的训练循环中，你将计算一个单一的标量损失值，代表一个批次或样本的误差，并直接在该标量上调用 `loss.backward()`，无需提供 `gradient` 参数。

### 计算图和梯度流

`loss.backward()` 触发对创建 `loss` 的操作图进行反向遍历。

> 一个简化的计算图，显示了输入 `x`、`w`、`b`、中间结果 `y` 和最终的标量 `loss`。虚线红色箭头说明了在 `loss.backward()` 过程中计算 `x`、`w` 和 `b` 梯度所经过的路径。

默认情况下，PyTorch 在 `backward()` 被调用后会清除计算图的中间缓冲区，以节省内存。这意味着如果你需要在图的*相同*部分多次调用 `backward()`（这种情况较少，通常用于高级技术或调试），你需要在第一次 `backward()` 调用时传入 `retain_graph=True`。然而，对于标准训练，你会构建一个图，计算损失，调用 `backward()`，更新权重 (weight)，然后为下一个批次重复这个过程，这会构建一个新的图。

理解 `backward()` 对于在 PyTorch 中训练模型非常重要。它是将模型输出和损失函数 (loss function)与需要调整的参数 (parameter)联系起来的机制。在接下来的章节中，我们将了解优化器如何访问和使用这些计算出的梯度。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6: Deep Feedforward Networks, Chapter 8: Optimization for Training Deep Models
  提供了反向传播、计算图和链式法则的理论解释，这些是`torch.autograd.backward()`的数学基础。
- [CS231n: Deep Learning for Computer Vision - Lecture Notes on Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — Andrej Karpathy, Justin Johnson, and Fei-Fei Li (2017)
  提供了关于反向传播在神经网络中如何工作的理解，补充了PyTorch的实现细节。
- [Automatic Differentiation with \`torch.autograd\` (PyTorch Tutorial)](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  一份侧重代码的`autograd`使用指南，通过不同场景演示`backward()`方法并解释其与计算图的交互。

---

[上一节](03-%E5%BC%A0%E9%87%8F%E4%B8%8E%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%20%28%60requires_grad%60%29.md) · [下一节](05-%E8%AE%BF%E9%97%AE%E6%A2%AF%E5%BA%A6%EF%BC%88%60.grad%60%EF%BC%89.md)
