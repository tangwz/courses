---
course: "calculus-essentials-machine-learning"
chapter: "chain-rule-backpropagation"
lesson: "chain-rule-multivariable"
sourceId: 1411
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-5-chain-rule-backpropagation/chain-rule-multivariable"
title: "多变量函数的链式法则"
description: "将链式法则扩展到涉及多个变量和中间依赖的函数。"
order: 2
plots: []
sourceHash: "4a5c3449ae9cd916dd3a03afc589866f759290bb25bfb527a8c7b39b549de9ab"
sourceCorrections: []
---

计算单变量函数的导数是一个基础概念。当处理复合函数时，比如 $f(g(x))$，单变量链式法则（$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}$）可以帮助求出总体的变化率。然而，机器学习 (machine learning)模型，特别是神经网络 (neural network)，通常涉及多个输入和输出的函数，这会产生更复杂的依赖关系。一种扩展链式法则以处理这些多变量情况的方法是必需的。

设想一个函数 $z$ 依赖于两个中间变量 $x$ 和 $y$。另外，假设 $x$ 和 $y$ 都依赖于一个单一的基础变量 $t$。因此，我们有 $z = f(x, y)$、$x = g(t)$ 和 $y = h(t)$ 这样的关系。那么当 $t$ 变化时，$z$ 如何变化？

由于 $t$ 通过*两条*路径（一条经过 $x$，一条经过 $y$）影响 $z$，我们需要考虑沿这两条路径传播的变化。由于 $t$ 的微小变化而引起的 $z$ 的变化，是经由 $x$ 和 $y$ 传播的变化之和。

这就引出了这种情况下多变量的链式法则：


$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}
$$


我们来分解一下：

- $\frac{\partial z}{\partial x}$: 这是 $z$ 对 $x$ 的偏导数。它说明了在*保持 $y$ 不变*的情况下，$z$ 对 $x$ 变化的敏感程度。我们使用偏导数符号（$\partial$），因为 $z$ 是多于一个变量（$x$ 和 $y$）的函数。
- $\frac{dx}{dt}$: 这是 $x$ 对 $t$ 的常导数。它说明了当 $t$ 变化时，$x$ 变化的快慢。我们使用常导数符号（$d$），因为 $x$ 是只有一个变量（$t$）的函数。
- $\frac{\partial z}{\partial x} \frac{dx}{dt}$: 这个乘积表示由 $t$ 的变化*通过* $x$ *引起的* $z$ 的变化率。
- 类似地，$\frac{\partial z}{\partial y} \frac{dy}{dt}$ 表示由 $t$ 的变化*通过* $y$ *引起的* $z$ 的变化率。
- 这个和结合了来自两条路径的贡献。

我们可以使用简单的图来可视化这些依赖关系：

> 一个依赖关系图，说明了 $t$ 的变化如何通过中间变量 $x$ 和 $y$ 传播，从而影响最终输出 $z$。标签表示每条路径上的相关导数。

### 多变量链式法则的推广

现在，我们考虑一个在机器学习 (machine learning)中更普遍的情况。假设我们有一个最终输出变量 $z$，它依赖于几个中间变量 $u_1, u_2, ..., u_m$。这些中间变量又分别依赖于几个输入变量 $x_1, x_2, ..., x_n$。

因此，$z = f(u_1, u_2, ..., u_m)$，并且对于每个从 1 到 $m$ 的 $i$，$u_i = g_i(x_1, x_2, ..., x_n)$。

我们通常想知道最终输出 $z$ 相对于某个特定输入变量（例如 $x_j$）如何变化。为了求偏导数 $\frac{\partial z}{\partial x_j}$，我们需要考虑 $x_j$ 影响 $z$ 的所有路径。输入 $x_j$ 可以影响任何中间变量 $u_1, u_2, ..., u_m$，而这些变量又反过来可以影响 $z$。

多变量链式法则的通用形式将所有这些中间路径上的影响相加：


$$
\frac{\partial z}{\partial x_j} = \sum_{i=1}^{m} \frac{\partial z}{\partial u_i} \frac{\partial u_i}{\partial x_j}
$$


这个公式说明，$z$ 相对于 $x_j$ 的总变化率是所有中间变量 $u_i$ 的和，即：
（$z$ 相对于 $u_i$ 的变化率）乘以（$u_i$ 相对于 $x_j$ 的变化率）。

### 与神经网络 (neural network)的关系

这种推广的链式法则正是神经网络中反向传播 (backpropagation)的数学基础。可以把 $z$ 看作网络的损失函数 (loss function)（例如，均方误差）。输入 $x_j$ 可以是网络中某个特定层的权重 (weight)或偏置 (bias)。中间变量 $u_i$ 表示后续层中神经元的激活值或输出。

为了使用梯度下降 (gradient descent)训练网络，我们需要计算损失 $z$ 相对于每个权重和偏置 $x_j$ 的梯度。网络结构形成了一个深层的依赖链：损失依赖于最终层的输出，最终层的输出又依赖于前一层的输出和权重，依此类推，一直追溯到特定的权重 $x_j$。

多变量链式法则提供了计算这些所需梯度 $\frac{\partial z}{\partial x_j}$ 的方法。反向传播本质上是一种算法，它能有效地递归应用此链式法则，逐层从输出层开始向输入层反向计算每一步所需的偏导数。因此，理解这个法则对于理解神经网络如何学习是十分重要的。我们将在本章后面看到它在反向传播中具体是如何应用的。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6, Section 6.5
  解释了链式法则在神经网络反向传播算法中的应用。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/10.1007/978-0-387-45528-0) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 5, Section 5.3.3; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  对机器学习提供了严谨的数学处理，包括链式法则在神经网络训练中的作用。
- [Chain Rule: Several Variables (from 18.02SC Multivariable Calculus)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHDgWSLgrzIulixBkSdzHDRjP48dMeczvYqw-EPpEPnasNGt9_GoHQg-h1uvOOiKMzD5vlh5bacmBAGCwwJ1z-rAoTQLp676Dz3B9R2YWicJJvuj-AI0XcocnyhRcsRkhOcX66PyMK8Gv3EbpoPgrSWHOJ6H9RuUpXmiFIjHKS36bjaU7TOMwHDCv5M5igAtR2p17yl3s2aA_b2snCRf3DFQtWph4ivvFeyhFvIJL7csX3qVCrXvdE=) — MIT OpenCourseWare (2010)
  Publisher: MIT OpenCourseWare
  提供了来自大学课程的多变量链式法则的在线说明。
