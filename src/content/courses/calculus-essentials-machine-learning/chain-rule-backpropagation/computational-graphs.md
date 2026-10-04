---
course: "calculus-essentials-machine-learning"
chapter: "chain-rule-backpropagation"
lesson: "computational-graphs"
sourceId: 1422
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-5-chain-rule-backpropagation/computational-graphs"
title: "计算图"
description: "使用计算图可视化计算的前向和反向传播过程。"
order: 6
plots: []
sourceHash: "5794d88df2cf84ab20c8dc0420ff70015234c4af0a4dbf514e0cca67dee8f2d7"
sourceCorrections: []
---

正如我们所见，神经网络 (neural network)本质上是大型的嵌套函数。计算损失函数 (loss function)相对于这个嵌套结构中可能存在的数百万个参数 (parameter)的梯度，需要一种系统的方法。手动重复应用链式法则很快就会变得难以应对且容易出错。此时，计算图提供了一种强大且实用的工具。

**计算图**是一种将数学表达式或一系列操作表示为有向图的方法。图中的每个节点代表一个变量（输入数据、参数、中间值）或一个操作（如加法、乘法、激活函数 (activation function)）。有向边表示数据流以及这些节点之间的依赖关系；从节点 A 到节点 B 的边意味着节点 B 的计算依赖于节点 A 的输出。

### 可视化计算

让我们看一个简单例子。假设我们有函数 $L = (x \cdot w + b - y)^2$。这可以表示一个非常简单的线性模型 $z = x \cdot w + b$ 的平方误差损失，其中 $y$ 是目标值。我们可以将其分解为基本操作：

1. $p = x \cdot w$ (乘法)
2. $a = p + b$ (加法)
3. $d = a - y$ (减法)
4. $L = d^2$ (平方)

我们可以将这个序列表示为计算图：

> 一个计算图，分解了计算 $L = (x \cdot w + b - y)^2$。椭圆形代表变量或计算值，矩形代表操作。边表示计算的方向。

### 前向传播

**前向传播**包括从输入到最终输出遍历图来计算表达式的值。你从输入（如 $x$, $y$）和参数 (parameter)（如 $w$, $b$）的值开始，然后根据其父节点的值计算每个操作节点的值。

对于我们的例子，如果 $x=2, w=3, b=1, y=5$:

1. $p = x \cdot w = 2 \cdot 3 = 6$
2. $a = p + b = 6 + 1 = 7$
3. $d = a - y = 7 - 5 = 2$
4. $L = d^2 = 2^2 = 4$

图结构清晰地定义了操作的顺序。

### 反向传播 (backpropagation)：图上的反向传播

计算图的真正作用在**反向传播**过程中变得显而易见，这也是反向传播的实现方式。目标是计算最终输出（我们的损失 $L$）相对于每个输入和参数 (parameter)的梯度（即 $\frac{\partial L}{\partial x}, \frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}, \frac{\partial L}{\partial y}$）。

我们从最终输出节点（$L$）开始，向后遍历图。梯度计算依赖于在每个节点局部应用的链式法则。

1. **从终点开始：** 输出相对于自身的梯度总是1： $\frac{\partial L}{\partial L} = 1$。这是我们“输入”到反向传播中的初始梯度值。
2. **向后传播梯度：** 对于任何产生输出 $z$ 的节点 $N$，如果我们知道最终损失 $L$ 相对于 $z$ 的梯度（我们称之为 $\frac{\partial L}{\partial z}$），我们可以使用链式法则计算 $L$ 相对于该节点 $N$ 的任何输入 $u$ 的梯度：
   $\frac{\partial L}{\partial u} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial u}$
   这里，$\frac{\partial z}{\partial u}$ 是节点 $N$ 处操作相对于其特定输入 $u$ 的**局部梯度**。
3. **在分叉处求和梯度：** 如果一个变量（比如更复杂图中的 $p$）输入到多个后续操作中，其总梯度是每个路径反向传播回来的梯度之和。

让我们为我们的示例图追踪此过程：

- **节点 `sq` ($L=d^2$):**

  - 输入梯度： $\frac{\partial L}{\partial L} = 1$。
  - 局部梯度： $\frac{\partial L}{\partial d} = \frac{\partial (d^2)}{\partial d} = 2d$。根据我们的值，$2d = 2(2) = 4$。
  - 输出梯度（到节点 `d`）： $\frac{\partial L}{\partial d} = \frac{\partial L}{\partial L} \cdot \frac{\partial L}{\partial d} = 1 \cdot (2d) = 2d = 4$。
- **节点 `sub` ($d=a-y$):**

  - 输入梯度： $\frac{\partial L}{\partial d} = 4$。
  - 局部梯度： $\frac{\partial d}{\partial a} = 1$， $\frac{\partial d}{\partial y} = -1$。
  - 输出梯度：
    - 到节点 `a`： $\frac{\partial L}{\partial a} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial a} = 4 \cdot 1 = 4$。
    - 到节点 `y`： $\frac{\partial L}{\partial y} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial y} = 4 \cdot (-1) = -4$。
- **节点 `add` ($a=p+b$):**

  - 输入梯度： $\frac{\partial L}{\partial a} = 4$。
  - 局部梯度： $\frac{\partial a}{\partial p} = 1$， $\frac{\partial a}{\partial b} = 1$。
  - 输出梯度：
    - 到节点 `p`： $\frac{\partial L}{\partial p} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial p} = 4 \cdot 1 = 4$。
    - 到节点 `b`： $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial b} = 4 \cdot 1 = 4$。
- **节点 `mul` ($p=x \cdot w$):**

  - 输入梯度： $\frac{\partial L}{\partial p} = 4$。
  - 局部梯度： $\frac{\partial p}{\partial x} = w$， $\frac{\partial p}{\partial w} = x$。根据我们的值，$\frac{\partial p}{\partial x} = 3$， $\frac{\partial p}{\partial w} = 2$。
  - 输出梯度：
    - 到节点 `x`： $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial x} = 4 \cdot w = 4 \cdot 3 = 12$。
    - 到节点 `w`： $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial w} = 4 \cdot x = 4 \cdot 2 = 8$。

我们现在已经计算出所有需要的梯度：$\frac{\partial L}{\partial x} = 12$，$\frac{\partial L}{\partial w} = 8$，$\frac{\partial L}{\partial b} = 4$，以及 $\frac{\partial L}{\partial y} = -4$。图结构使我们能够系统地应用链式法则，而不会迷失在嵌套的函数结构中。

### 与深度学习 (deep learning)框架的相关性

现代深度学习库，如 TensorFlow 和 PyTorch，都高度依赖这种方法。它们根据你在模型代码中定义的操作，自动构建计算图（要么在执行前静态构建，要么在执行时动态构建）。这种图表示使得它们能够：

1. **高效地进行前向传播计算。**
2. **自动计算所有参数 (parameter)的梯度**，通过反向传播 (backpropagation)（通常称为“自动微分”或自动求导）。这使开发者无需手动推导和实现梯度计算，这对于复杂架构来说非常重要。
3. **优化计算**，例如，通过识别可并行化的操作或简化图的某些部分。

理解计算图有助于了解这些强大框架背后的机制，并阐明链式法则如何实现非常深的神经网络 (neural network)的训练。它将潜在复杂的梯度计算过程转变为图的结构化遍历。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  为深度学习中的计算图和反向传播提供了全面的理论基础。
- [PyTorch Autograd: Automatic Differentiation](https://pytorch.org/docs/stable/autograd.html) — PyTorch Development Team (2025)
  Publisher: PyTorch Foundation
  阐述了PyTorch中如何利用动态计算图实现自动微分。
- [CS231n: Deep Learning for Computer Vision, Lecture Notes on Backpropagation](https://cs231n.github.io/optimization-2/) — Andrej Karpathy (2023)
  提供了关于神经网络中反向传播和计算图的清晰说明及可视化。
