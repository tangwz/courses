# PyTorch 计算图

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/pytorch-computation-graphs)

[返回章节目录](README.md) · [返回课程目录](../README.md)

PyTorch 运用**计算图**作为其通过 Autograd 自动计算梯度的底层机制。每次你执行涉及需要梯度的张量运算时（你很快会了解如何指定这一点），PyTorch 都会动态构建一个图，表示计算序列。

可以将此图视为一个有向无环图 (DAG)，其中：

- **节点**代表张量或对其执行的运算。
- **边**代表数据（张量）的流动以及运算和张量之间的函数依赖关系。

考虑一个简单的计算：

```python
import torch

# Tensors that require gradients
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Operations
y = w * x  # Intermediate result 'y'
z = y + b  # Final result 'z'

print(f"Result z: {z}")
```

输出：

```
Result z: 7.0
```

当这些代码行执行时，PyTorch 会在后台构建一个图。它看起来像这样：

> 这是 $z = (w * x) + b$ 的计算图表示。蓝色框代表输入张量，黄色椭圆代表运算，绿色框代表输出/中间张量。边表示数据流和运算间的依赖关系。由运算产生的张量上的 `grad_fn` 属性指向创建它们的函数。

### 动态特性

PyTorch 计算图的一个重要特点是它们的**动态**特性。与那些要求你在运行计算前定义整个图结构的框架不同，PyTorch 会在你的 Python 代码执行时动态构建图。

- **灵活性：** 这允许标准的 Python 控制流语句（如 `if` 条件或 `for` 循环）直接影响每次迭代中的图结构。如果你的模型架构需要在正向传播期间根据输入数据进行更改，PyTorch 会很自然地处理这种情况。
- **调试：** 动态图通常更容易使用标准 Python 工具进行调试，因为图的构建与你熟悉的程序执行同时进行。

### 正向传播和反向传播 (backpropagation)

1. **正向传播：** 当你执行张量运算（如 `y = w * x`）时，你正在进行正向传播。PyTorch 会记录所涉及的运算和张量，从而构建图。由 Autograd 追踪的运算产生的张量将具有 `grad_fn` 属性（如示例中的 `y` 和 `z`）。此属性引用了创建该张量的函数，并保存了对其输入的引用，从而形成了图中的反向链接。用户创建的张量（如 `x`、`w`、`b`）通常具有 `grad_fn=None`。
2. **反向传播：** 当你稍后在标量张量（通常是最终的损失值）上调用 `.backward()` 时，Autograd 会从该节点向后遍历此图。它使用微积分的链式法则，在每一步都由 `grad_fn` 指引，以计算标量输出相对于最初标记 (token)为 `requires_grad=True` 的张量（通常是模型参数 (parameter)或输入）的梯度。

### 叶张量和梯度

在 Autograd 中：

- **叶张量：** 这些是位于图“开始”处的张量。通常，它们是用户直接创建的张量（例如，使用 `torch.tensor()`、`torch.randn()`）且 `requires_grad=True`。模型参数 (parameter)（`nn.Parameter`，我们稍后会看到）也是叶张量。
- **非叶张量（中间张量）：** 这些是在图内运算后产生的张量（如上文的 `y` 和 `z`）。它们关联着 `grad_fn`。

默认情况下，在 `.backward()` 调用期间计算的梯度仅保留并累积在具有 `requires_grad=True` 的**叶张量**的 `.grad` 属性中。中间张量的梯度会进行计算，但使用后通常会被丢弃以节省内存，除非另有明确请求（例如，使用 `.retain_grad()`）。

理解这种图结构对于掌握 Autograd 的工作方式非常重要。它将你在模型中定义的正向计算与优化所需的梯度计算直接关联起来。接下来，我们将研究如何使用 `requires_grad` 来明确控制梯度跟踪。

## 参考资料

- [Automatic differentiation with torch.autograd](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Foundation (2017)
  PyTorch自动微分引擎及其底层计算图的主要官方文档。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  一本基础教材，在通用背景下提供了自动微分和计算图的理论背景。
- [CS231n: Deep Learning for Computer Vision. Lecture Notes: Backpropagation, Neural Networks, and Computation Graphs](https://cs231n.github.io/optimization-2/) — Stanford University (2024)
  来自顶尖大学课程的教育笔记，解释了反向传播和计算图的通用概念。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Foundation (2017)
  Publisher: PyTorch Foundation
  PyTorch官方教程，提供了Autograd机制和基于图的操作的实践性和详细探索。

---

[上一节](01-%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86%E7%9A%84%E5%8E%9F%E7%90%86.md) · [下一节](03-%E5%BC%A0%E9%87%8F%E4%B8%8E%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%20%28%60requires_grad%60%29.md)
