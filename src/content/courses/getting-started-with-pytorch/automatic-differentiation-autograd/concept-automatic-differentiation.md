---
course: "getting-started-with-pytorch"
chapter: "automatic-differentiation-autograd"
lesson: "concept-automatic-differentiation"
sourceId: 2128
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-3-automatic-differentiation-autograd/concept-automatic-differentiation"
title: "自动微分的原理"
description: "介绍自动微分及其在神经网络训练中的作用。"
order: 1
plots: []
sourceHash: "f6b4eb8a0ba53e7824a51ed3c18b0eb5b25bc8681f59de811bcd7df3153b27d4"
sourceCorrections: []
---

训练神经网络 (neural network)包括迭代地调整模型参数 (parameter)（权重 (weight)和偏置 (bias)）以最小化损失函数 (loss function)。这种调整依赖于知晓每个参数的微小变化如何影响最终的损失值。从数学上讲，这种敏感性由损失函数对每个参数的梯度来体现。对于损失 $L$ 和参数 $w$，我们需要计算 $\frac{\partial L}{\partial w}$。

对于非常简单的模型，手动使用微积分规则计算这些梯度是可行的，但对于当今常见的深度多层网络来说，这很快就会变得异常复杂且容易出错。想象一下为拥有数百万参数的模型推导导数！这时，\*\*自动微分（AD）\*\*就派上用场了。

AD 是一系列以数值方式评估由计算机程序定义的函数导数的方法。与*符号微分*（它操作数学表达式，常导致复杂且低效的公式）或*数值微分*（它使用有限差分近似导数，可能存在截断和舍入误差）不同，AD 通过在构成整体计算的基本运算（加法、乘法、三角函数等）层面系统地应用微积分的链式法则，高效地计算出精确的梯度。

### 链式法则：AD 的核心

其核心是，AD 依赖于链式法则。如果有一系列函数，例如 $y = f(x)$ 和 $z = g(y)$，链式法则告诉我们如何找到复合函数 $z = g(f(x))$ 对 $x$ 的导数：


$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$


AD 将复杂的计算分解为一系列基本运算。然后，它计算每个小步骤的局部导数，并使用链式法则将它们组合起来以得到整体梯度。

考虑一个简单例子：$L = (w \cdot x + b)^2$。令 $y = w \cdot x + b$。则 $L = y^2$。
要找到 $\frac{\partial L}{\partial w}$，链式法则给出：


$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}
$$


我们知道 $\frac{\partial L}{\partial y} = 2y$ 和 $\frac{\partial y}{\partial w} = x$。将 $y$ 代回，我们得到：


$$
\frac{\partial L}{\partial w} = (2(w \cdot x + b)) \cdot x
$$


AD 自动完成这个过程，即使运算链非常长。

### 正向模式与反向模式

在 AD 中应用链式法则主要有两种方式：

1. **正向模式（正向累积）：** 通过从输入到输出遍历计算步骤来计算导数。它计算*一个*输入的改变如何影响*所有*中间变量和最终输出。当输入数量相对于输出数量较少时，它比较高效。
2. **反向模式（反向累积）：** 通过从最终输出到输入反向遍历计算步骤来计算导数。它计算*最终输出*的改变如何受到*所有*中间变量和输入的影响。当输出数量相对于输入数量较少时，这种模式明显更高效，这正是深度学习 (deep learning)中的情况，因为我们通常只有一个标量损失值和数百万个参数 (parameter)（损失函数 (loss function)的输入）。

PyTorch 的 Autograd 系统使用**反向模式自动微分**。

### Autograd 如何使用 AD

当您对 `requires_grad` 属性设置为 `True` 的 PyTorch 张量执行操作时，PyTorch 会在后台构建一个有向无环图（DAG）。这个图通常被称为计算图，它记录了操作序列（节点）和涉及的张量（边）。

让我们直观地看一下 $L = (a \cdot x + b)^2$ 的简单计算图，假设 $a$、$x$ 和 $b$ 是输入张量（或之前计算的结果），并且我们想得到 $\frac{\partial L}{\partial a}$、$\frac{\partial L}{\partial x}$ 和 $\frac{\partial L}{\partial b}$。

> $L = (a \cdot x + b)^2$ 计算图的表示。实线表示正向传播，构建图。虚线表示反向传播 (backpropagation)期间梯度的流动，应用链式法则。

当您在最终输出张量（通常是标量损失 $L$）上调用 `.backward()` 时，Autograd 会从该输出开始并反向遍历图。在每个步骤（节点），它根据后续节点的梯度和当前节点执行运算的局部导数来计算梯度，有效地应用了链式法则。然后，对每个需要梯度的张量（如模型参数 (parameter)）计算出的梯度会累积到它们的 `.grad` 属性中。

这种机制使 PyTorch 能够自动计算由张量运算序列定义的任意复杂模型的梯度，让您摆脱手动推导的繁琐且容易出错的任务。接下来的章节将演示如何实际使用 Autograd 的功能：定义需要梯度的张量、隐式构建计算图、触发反向传播、访问梯度以及控制梯度计算。

## 参考资料

- [Evaluating Derivatives: Principles and Techniques of Automatic Differentiation](https://doi.org/10.1137/1.9780898717752) — Andreas Griewank, Andrea Walther (2008)
  Publisher: SIAM; Pages: 459; DOI: [10.1137/1.9780898717752](https://doi.org/10.1137/1.9780898717752)
  自动微分的标准且全面的参考文献，详细介绍了其数学原理以及前向和反向模式的算法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  一本内容全面的教科书，其中第八章解释了反向传播作为神经网络中反向模式自动微分的应用。
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Contributors (2024)
  PyTorch 官方文档，解释了其 Autograd 引擎的内部工作原理，包括计算图和用于梯度计算的反向传播。
