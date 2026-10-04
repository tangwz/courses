---
course: "getting-started-with-jax"
chapter: "automatic-differentiation-grad"
lesson: "how-autodiff-works"
sourceId: 1723
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-3-automatic-differentiation-grad/how-autodiff-works"
title: "自动微分的工作方式：反向模式"
description: "`jax.grad` 使用的反向模式自动微分概述。"
order: 3
plots: []
sourceHash: "48ca007ffc544dec7194f66a9dd54006468b625c3957e59801dc4f84fb108932"
sourceCorrections: []
---

自动微分（AD）是 `jax.grad` 背后的技术。它是一套用于数值计算计算机程序所定义函数导数的方法。与符号微分（操作数学表达式）或数值微分（使用有限差分，可能产生近似误差）不同，AD 通过在基本运算层面系统应用微积分链式法则，高效地计算出精确导数。

AD 主要有两种模式：正向模式和反向模式。`jax.grad` 主要使用**反向模式自动微分**，在机器学习 (machine learning)社区中常被称为**反向传播 (backpropagation)**。让我们来了解它是如何运作的。

## 计算图

反向模式 AD 的核心是将任何计算看作一系列基本运算（如加法、乘法、sin、cos 等）应用于某些输入值的过程。这个序列可以表示为**计算图**，图中的节点代表输入变量或运算，有向边代表数据流。

考虑一个简单函数：$f(x) = \sin(x^2)$。我们可以将其分解为中间步骤：

1. $a = x^2$ (平方运算)
2. $y = \sin(a)$ (正弦运算)

计算图如下所示：$x \rightarrow \text{平方} \rightarrow a \rightarrow \sin \rightarrow y$。

> $f(x) = \sin(x^2)$ 的一个简单计算图。数据从输入 $x$ 经过中间运算 $a$ 流向最终输出 $y$。

## 正向传播

当你正常执行函数，例如 `f(2.0)` 时，你将进行一次图的**正向传播**：

1. 输入：$x = 2.0$
2. 计算 $a = x^2 = 2.0^2 = 4.0$
3. 计算 $y = \sin(a) = \sin(4.0) \approx -0.757$

在此正向传播过程中，JAX 等 AD 系统不仅计算最终值 $y$，通常还会存储中间值（如 $a=4.0$）以及图的结构。这些是下一阶段所需的。

## 反向传播 (backpropagation)：应用链式法则

目标是计算最终输出 $y$ 相对于初始输入 $x$ 的梯度，即 $dy/dx$。反向模式通过从输出开始，将导数反向传播通过图来实现这一目标。

1. **初始化：** 输出相对于自身的导数显然是1。我们将这种敏感度或伴随表示为 $\bar{y} = dy/dy = 1$。这是从末端进入图的初始“梯度信号”。
2. **反向步骤（sin 节点）：** 我们从 $y$ 反向移动到 $a$。我们需要找到 $a$ 的变化如何影响 $y$，这就是局部导数 $dy/da$。然后我们使用链式法则找到到达 $a$ 的梯度信号：

   
   $$
   \bar{a} = \frac{dy}{da} = \frac{dy}{dy} \frac{dy}{da} = \bar{y} \frac{dy}{da}
   $$
   

   这个 $\bar{a}$ 表示最终输出 $y$ 对中间变量 $a$ 变化的敏感度。因为 $y = \sin(a)$，所以局部导数 $dy/da = \cos(a)$。因此，$\bar{a} = 1 \times \cos(a)$。使用从正向传播中存储的值 $a=4.0$，$\bar{a} = \cos(4.0) \approx -0.654$。
3. **反向步骤（平方节点）：** 我们从 $a$ 继续反向到 $x$。我们需要局部导数 $da/dx$。我们再次使用链式法则，将传入的梯度信号 $\bar{a}$（$y$ 对 $a$ 的敏感度）乘以局部导数 $da/dx$：

   
   $$
   \bar{x} = \frac{dy}{dx} = \frac{dy}{da} \frac{da}{dx} = \bar{a} \frac{da}{dx}
   $$
   

   因为 $a = x^2$，所以局部导数 $da/dx = 2x$。因此，$\bar{x} = \bar{a} \times (2x)$。使用 $\bar{a} \approx -0.654$ 和输入值 $x=2.0$（也可能被存储或重新计算），$\bar{x} \approx -0.654 \times (2 \times 2.0) = -0.654 \times 4.0 \approx -2.616$。这个 $\bar{x}$ 就是我们所需的梯度 $dy/dx$。

反向传播本质上是计算最终输出对每个中间变量和输入的敏感程度，从输出开始反向工作，并在每一步应用链式法则。

> 正向传播（计算值）和反向传播（使用链式法则计算梯度）的流程。反向传播需要正向传播期间计算得到的值（x, a），并反向传播敏感度。

## 机器学习 (machine learning)中的效率

为什么反向模式是 `jax.grad` 的默认模式，并在机器学习中普遍使用？考虑一个典型的神经网络 (neural network)损失函数 (loss function)：$L = f(W_1, W_2, ..., W_n, x, y)$，其中 $W_i$ 是许多权重 (weight)矩阵/向量 (vector)（参数 (parameter)），$x$ 是输入数据，$y$ 是目标。这个函数可能接收数百万个输入（参数和数据），但只产生一个标量输出（损失 $L$）。

反向模式的计算成本相对而言对*输入*的数量不敏感。它需要一次通过计算图的正向传播（用于计算中间值）和一次反向传播 (backpropagation)（用于计算梯度）。总成本通常是评估原始函数成本的一个小常数倍（例如，2-4倍）。这使得它在同时计算 $\nabla L$ 相对于*所有*参数的梯度时非常高效，这正是梯度下降 (gradient descent)等基于梯度的优化算法所需要的。

正向模式则一次计算一个输入值的导数。其成本与你需要计算梯度的输入数量呈线性关系。虽然在某些情况下有用，但对于训练拥有数百万参数的大型模型来说，它的计算成本过高。

## JAX 的作用

当你将 `jax.grad` 应用于你的 Python 函数时，JAX 会在底层执行以下步骤：

1. **追踪：** 它会使用特殊的抽象“追踪器”对象而不是具体的数值来执行你的函数一次。这个过程会记录所执行的基本运算序列，有效地构建了计算图的内部表示（称为“jaxpr”）。
2. **正向传播：** 为了计算梯度，JAX 使用追踪到的图结构。它执行图中定义的运算，以计算梯度计算所需的中间值。（如果使用 `jax.value_and_grad`，此传播也会计算最终函数输出）。
3. **反向传播 (backpropagation)：** JAX 自动解释追踪到的图，以生成并执行反向传播的代码。它系统地将链式法则反向应用于图的运算，使用正向传播中存储的中间值，以计算相对于指定输入的梯度。

你作为用户，使用熟悉的 Python 和 NumPy 类似语法定义正向计算。JAX 负责根据该定义，通过反向模式 AD 推导并执行高效的梯度计算。这种关注点分离使你能够专注于模型逻辑，同时依赖 JAX 进行高性能的微分。

## 参考资料

- [Automatic Differentiation of Algorithms: From Ideology to Tools](https://doi.org/10.1007/978-3-540-68494-1) — Andreas Griewank and Andrea Walther (2008)
  Publisher: Springer; DOI: [10.1007/978-3-540-68494-1](https://doi.org/10.1007/978-3-540-68494-1)
  自动微分的综合处理，详细介绍了正向和反向模式的数学基础和实用算法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第6章提供了反向传播的易懂解释，阐明了它与神经网络反向模式自动微分的关系。
- [Automatic Differentiation in JAX](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) — JAX Core Developers (Current)
  官方文档，演示了使用 `jax.grad` 进行自动微分的原理和用法，包括反向模式的具体细节。
- [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) — David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams (1986)
  Journal: Nature; Publisher: Springer Nature; Volume: 323; Pages: 533-536; DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0)
  推广反向传播算法的开创性论文，该算法是用于训练神经网络的反向模式自动微分的特例。
