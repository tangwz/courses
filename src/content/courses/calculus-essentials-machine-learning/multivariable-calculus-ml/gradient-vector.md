---
course: "calculus-essentials-machine-learning"
chapter: "multivariable-calculus-ml"
lesson: "gradient-vector"
sourceId: 1372
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-3-multivariable-calculus-ml/gradient-vector"
title: "梯度向量"
description: "定义梯度向量，它指向函数最陡峭上升的方向。"
order: 3
plots: ["plots/1372-0.json"]
sourceHash: "f5267231eb9441e0d2fcb2a10be9f08f1e7ace65f82bc5b289587e01c02cf072"
sourceCorrections: []
---

偏导数衡量多变量函数沿着标准坐标轴方向（如 $x$ 方向或 $y$ 方向）的变化率。但如果我们想知道函数从给定点增加 *最快* 的方向是什么？这正是梯度向量 (vector)告诉我们的。

想象一下你站在由函数 $f(x, y)$ 表示的丘陵地带上。如果你沿着正东方向 ($\frac{\partial f}{\partial x}$) 或正北方向 ($\frac{\partial f}{\partial y}$) 走，偏导数会告诉你山坡有多陡峭。梯度将这些信息组合成一个单一的向量，它直接指向上坡，即最陡峭坡度的方向。

### 定义梯度

对于一个依赖于 $n$ 个变量的标量函数 $f(x_1, x_2, \dots, x_n)$，其**梯度**是一个包含所有一阶偏导数的向量 (vector)。它通常表示为 $\nabla f$（读作“nabla f”或“del f”）。


$$
\nabla f(x_1, x_2, \dots, x_n) = \left\langle \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right\rangle
$$


梯度向量的每个分量都是函数对一个变量的偏导数。因此，梯度是一个向量场；它将一个向量（代表最陡峭上升的方向和大小）分配给函数定义域中的每个点。

### 计算例子

我们来看函数 $f(x, y) = x^2 + y^3$。这个函数有两个输入，$x$ 和 $y$。

1. **求偏导数：**

   - $\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(x^2 + y^3) = 2x$ (将 $y$ 视为常数)
   - $\frac{\partial f}{\partial y} = \frac{\partial}{\partial y}(x^2 + y^3) = 3y^2$ (将 $x$ 视为常数)
2. **构成梯度向量 (vector)：**

   
   $$
   \nabla f(x, y) = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle = \langle 2x, 3y^2 \rangle
   $$
   

在特定点，例如 $(x, y) = (1, 2)$，梯度是：


$$
\nabla f(1, 2) = \langle 2(1), 3(2^2) \rangle = \langle 2, 12 \rangle
$$


这个向量 $\langle 2, 12 \rangle$ 表示从点 $(1, 2)$ 开始，函数 $f(x, y) = x^2 + y^3$ 增加最快的方向。

### 几何解释：方向和大小

梯度 $\nabla f$ 在任何给定点都有两个重要性质：

1. **方向：** 向量 (vector) $\nabla f$ 指向函数在该点 **最陡峭上升** 的方向。如果你想尽可能快地增加函数值，你应该沿着梯度指示的方向移动。
2. **大小：** 梯度向量的模长（或长度），记作 $||\nabla f||$，代表该最陡峭方向上的 **增加率**。模长越大，表示函数增加得越快。
   
   $$
   ||\nabla f|| = \sqrt{\left(\frac{\partial f}{\partial x_1}\right)^2 + \left(\frac{\partial f}{\partial x_2}\right)^2 + \dots + \left(\frac{\partial f}{\partial x_n}\right)^2}
   $$
   

反之，与梯度相反的方向，$-\nabla f$，指向 **最陡峭下降** 的方向。这对于机器学习 (machine learning)中的优化很重要。如果函数 $f$ 代表我们希望最小化的成本或误差，沿着 $-\nabla f$ 的方向移动可以使我们在每一步中最有效地降低成本。

### 可视化梯度

等高线图有助于可视化多变量函数。等高线连接函数值相同的点。梯度向量 (vector)在任何点总是垂直（正交）于通过该点的等高线。

我们来可视化一个简单函数的梯度，$f(x, y) = x^2 + y^2$。梯度是 $\nabla f = \langle 2x, 2y \rangle$。等高线是以原点为中心的圆。



![函数 f(x, y) = x^2 + y^2 的梯度向量](plots/1372-0.json)



> 函数 $f(x, y) = x^2 + y^2$ 的等高线图。灰色圆圈是 $f$ 值恒定的等高线。蓝色箭头表示在选定点处的梯度向量 $\nabla f = \langle 2x, 2y \rangle$。请注意梯度向量如何向外指向，垂直于等高线，表示从原点 (0,0) 的最小值向外最陡峭增加的方向。

### 梯度在机器学习 (machine learning)中的作用

在机器学习中，我们通常定义一个**成本函数**（或损失函数 (loss function)）来衡量模型表现的优劣。这个成本函数通常依赖于许多参数 (parameter)（权重 (weight)和偏差）。训练模型涉及找到使该成本函数最小化的参数值。

由于负梯度 $-\nabla f$ 指向最陡峭下降的方向，它提供了调整模型参数以降低成本的最有效方向。梯度下降 (gradient descent)等算法直接基于此原理：通过沿着成本函数梯度的反方向迈出小步，迭代更新参数。

因此，理解梯度向量 (vector)对于理解许多机器学习模型的训练和优化方式很重要。它是指引我们穿过复杂、高维参数，走向更低成本点的指南针。我们将在下一章学习基于梯度的优化算法。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  对于理解梯度在梯度下降等机器学习优化算法中的应用至关重要，为模型训练提供了数学基础。
