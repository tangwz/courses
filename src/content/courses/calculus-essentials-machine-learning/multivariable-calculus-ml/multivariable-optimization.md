---
course: "calculus-essentials-machine-learning"
chapter: "multivariable-calculus-ml"
lesson: "multivariable-optimization"
sourceId: 1380
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-3-multivariable-calculus-ml/multivariable-optimization"
title: "多变量优化原理"
description: "将优化原理（寻找最小值/最大值）推广到多变量函数，并使用梯度和海森矩阵。"
order: 6
plots: ["plots/1380-0.json"]
sourceHash: "ff7ae37163d9ad9aa7b50b08283c415f7323f3322df044bd5762e5edf2d3ca59"
sourceCorrections: []
---

单变量微积分通常通过识别导数 $f'(x) = 0$ 的临界点来寻找函数 $f(x)$ 的潜在最小值和最大值。在这些点，切线是水平的，表明函数暂时平坦。随后，二阶导数 $f''(x)$ 可以用于对这些点进行分类：$f''(x) > 0$ 表示局部最小值（向上凹），而 $f''(x) < 0$ 表示局部最大值（向下凹）。

我们可以将此思想推广到多变量函数，例如 $f(x, y)$，或更一般地表示为 $f(\mathbf{x})$，其中 $\mathbf{x}$ 是一个变量向量 (vector) $(x_1, x_2, ..., x_n)$。

### 借助梯度寻找临界点

对于多变量函数，导数为零的对应物是**梯度向量 (vector)为零向量**。函数 $f(\mathbf{x})$ 的**临界点**（或驻点）$\mathbf{x}_0$ 是梯度为零的点：


$$
\nabla f(\mathbf{x}_0) = \mathbf{0}
$$


请记住，梯度 $\nabla f$ 是一个包含所有偏导数的向量：$\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n} \right)$。因此，条件 $\nabla f(\mathbf{x}_0) = \mathbf{0}$ 意味着在 $\mathbf{x}_0$ 处，*所有*偏导数必须同时为零：


$$
\frac{\partial f}{\partial x_1}(\mathbf{x}_0) = 0, \quad \frac{\partial f}{\partial x_2}(\mathbf{x}_0) = 0, \quad ..., \quad \frac{\partial f}{\partial x_n}(\mathbf{x}_0) = 0
$$


这直观地意味着什么？梯度指向最陡峭上升的方向。如果梯度是零向量，则表示从该点没有上升（或下降）的方向。函数在所有平行于输入轴的方向上局部“平坦”。就像单变量情况一样，这些临界点可能是局部最小值、局部最大值或其他类型的点。

### 临界点分类：海森矩阵的作用

在单变量情况下，仅有零导数不足够；我们需要二阶导数判别法。对于多变量函数，包含所有二阶偏导数的**海森矩阵** $\mathbf{H}$ 扮演着二阶导数的角色。


$$
\mathbf{H} = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}
$$


在临界点 $\mathbf{x}_0$（其中 $\nabla f(\mathbf{x}_0) = \mathbf{0}$）评估海森矩阵有助于我们了解函数在该点附近的曲率。多变量函数的**二阶导数判别法**使用海森矩阵 $\mathbf{H}(\mathbf{x}_0)$ 的性质：

1. **局部最小值：** 如果 $\mathbf{H}(\mathbf{x}_0)$ 是**正定**的，则 $f$ 在 $\mathbf{x}_0$ 处有局部最小值。直观地，这意味着函数在从 $\mathbf{x}_0$ 辐射的所有方向上“向上凹”。 （如果对于所有非零向量 (vector) $\mathbf{v}$，都有 $\mathbf{v}^T \mathbf{H} \mathbf{v} > 0$，则矩阵是正定的。这通常对应于所有特征值均为正）。
2. **局部最大值：** 如果 $\mathbf{H}(\mathbf{x}_0)$ 是**负定**的，则 $f$ 在 $\mathbf{x}_0$ 处有局部最大值。这意味着函数在所有方向上“向下凹”。 （负定通常对应于所有特征值均为负）。
3. **鞍点：** 如果 $\mathbf{H}(\mathbf{x}_0)$ 是**不定**的（意味着它既不正定也不负定，通常同时具有正负特征值），则 $f$ 在 $\mathbf{x}_0$ 处有**鞍点**。这是函数在某些方向上向上弯曲而在其他方向上向下弯曲的点，像马鞍一样。请看下面的图示。
4. **不确定：** 如果海森矩阵判别法没有结论（例如，如果它是正半定或负半定，意味着某些特征值可能为零），则需要其他方法。

对于机器学习 (machine learning)中的大多数优化问题，我们主要关注于寻找局部（或理想情况下全局）最小值。鞍点对某些优化算法构成一个难题，因为梯度为零，可能中止进展，即使它不是最小值。



![鞍点示例：z = x² - y²](plots/1380-0.json)



> 在函数 $f(x, y) = x^2 - y^2$ 的原点 (0,0) 处，梯度 $\nabla f = (2x, -2y)$ 为 $(0, 0)$。海森矩阵是 $\begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$，它是不定的。这表明是一个鞍点：函数沿x轴增加但沿y轴减小。

### 与机器学习 (machine learning)的关联性

在机器学习中，我们想要优化的函数 $f$ 通常是**成本函数**或**损失函数 (loss function)**，它衡量模型表现有多差。变量 $\mathbf{x}$ 是模型的参数 (parameter)（权重 (weight)和偏置 (bias)）。我们的目标几乎总是*最小化*此成本函数。

因此，寻找成本函数关于参数的梯度为零的点是一个基本步骤。虽然使用海森矩阵的二阶导数判别法提供了一种分类这些点的方法，但计算和分析海森矩阵可能计算成本高昂，特别是对于具有数百万参数的模型（如深度神经网络 (neural network)）。

此外，对于复杂的机器学习模型，通过解析地求解偏导数方程组来寻找 $\nabla f = \mathbf{0}$ 的点通常是难以处理的。这促使使用迭代优化算法，例如**梯度下降 (gradient descent)**（我们将在下一章中介绍），它们使用梯度 $\nabla f$ 来遍历成本曲面并逐步移向最小值，而无需计算海森矩阵或显式地求解 $\nabla f = \mathbf{0}$。然而，理解梯度、临界点和曲率的原理对于理解这些算法如何以及为何有效，以及它们可能遇到哪些难题（例如鞍点）仍然非常重要。

## 参考资料

- [Mathematics for Machine Learning](https://mml-book.com/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679901](https://doi.org/10.1017/9781108679901)
  本书弥合了基础数学与机器学习之间的鸿沟，提供了关于向量微积分、线性代数和优化的专门章节，与理解机器学习中的多元优化直接相关。可在线免费获取。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; DOI: [10.7551/mitpress/9780262035613.001.0001](https://doi.org/10.7551/mitpress/9780262035613.001.0001)
  一本全面的深度学习教材，详细讨论了优化算法、鞍点带来的挑战以及Hessian矩阵在高维参数空间中的计算影响。可在线免费获取。
