---
course: "calculus-fundamentals-machine-learning"
chapter: "partial-derivatives-multiple-inputs"
lesson: "partial-derivatives-concept"
sourceId: 2288
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-4-partial-derivatives-multiple-inputs/partial-derivatives-concept"
title: "偏导数：其思想"
description: "解释偏导数的思想：对一个变量求导，同时保持其他变量不变。"
order: 2
plots: ["plots/2288-0.json"]
sourceHash: "d2b58f13c07ef3e82fa7d675fc3f577a91b14aadc92e0263ebfcb4187cc26ad6"
sourceCorrections: []
---

具有多个输入的函数广泛应用于模拟复杂现象。例如，预测房价时会考虑多种因素。价格不仅取决于房屋的面积 ($x_1$)；它还取决于卧室数量 ($x_2$)、房屋的年代 ($x_3$)，以及到市中心的距离 ($x_4$)。此情境下的预测函数可以表示为 $Price = f(x_1, x_2, x_3, x_4)$。

当我们有一个像 $f(x, y)$ 这样的函数时，我们通常想知道：如果只调整 $x$ 而保持 $y$ 不变，输出会如何变化？或者，如果只调整 $y$ 而保持 $x$ 不变，它会如何变化？这正是偏导数帮助我们理解的内容。

想象你正站在一个山坡上。你的海拔高度（输出值，我们称之为 $z$）取决于你在两个方向上的位置：你向东的距离（我们称之为 $x$）以及你向北的距离（我们称之为 $y$）。因此，$z = f(x, y)$。

如果你想知道山坡在 *确切的东向* 有多陡峭，你只会考虑你向东或向西移动时海拔的变化，而不向北或向南移动。你实际上是将南北方向的位置 ($y$) 在那一刻视为常数，并只沿着东西方向 ($x$) 测量坡度。

这就是 **偏导数** 的主要思想。

偏导数衡量一个多变量函数相对于某个特定变量的变化率，同时保持所有其他变量不变。它告诉我们函数 *沿着与某个输入轴平行的方向* 的斜率。

我们不使用像 $\frac{df}{dx}$ 这样的 $d$ 符号（用于单变量函数），而是使用一个卷曲的符号 $\partial$，称为“del”或简称为“偏导数符号”。

- 函数 $f$ 对 $x$ 的偏导数写作 $\frac{\partial f}{\partial x}$，有时也写作 $f_x$。它说明了当只有 $x$ 变化时，$f$ 如何变化。
- 函数 $f$ 对 $y$ 的偏导数写作 $\frac{\partial f}{\partial y}$，有时也写作 $f_y$。它说明了当只有 $y$ 变化时，$f$ 如何变化。

考虑函数 $f(x, y) = x^2 + y^2$。这个函数描述了三维空间中的一个碗状曲面。



![Interactive chart](plots/2288-0.json)



> 函数 $f(x, y) = x^2 + y^2$ 的图。曲面的高度表示函数的输出。

如果我们想在一个特定点，比如 $(1, 1)$ 处找到 $\frac{\partial f}{\partial x}$，我们是在问：如果我们站在曲面上 $x=1$ 和 $y=1$ 的点，并且我们只在 $x$ 方向移动（保持 $y=1$ 不变），那么曲面的斜率是多少？

为了思考如何计算 $\frac{\partial f}{\partial x}$，我们将变量 $y$ 视为一个常数。同样地，当我们想计算 $\frac{\partial f}{\partial y}$ 时，我们将 $x$ 视为一个常数。实际的微分规则（例如你之前学过的幂法则）仍然适用，但你只将它们应用于你正在对其进行微分的变量，而其他变量则被视为任何其他数字。我们将在下一节介绍这些计算方法。

这里的主要思想是 *隔离*：偏导数隔离了改变单个输入变量对函数输出的影响。这对于理解复杂模型如何响应不同特征或参数 (parameter)的变化非常重要，并且它是梯度的组成部分，我们很快就会看到。

## 参考资料

- [Multivariable Calculus (18.02SC)](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/) — MIT OpenCourseWare (2010)
  Publisher: Massachusetts Institute of Technology
  麻省理工学院提供的综合视频讲座和配套笔记，提供了偏导数的基础解释。
