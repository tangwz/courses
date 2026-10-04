---
course: "calculus-fundamentals-machine-learning"
chapter: "calculus-action-optimization"
lesson: "linear-regression-cost-function"
sourceId: 2307
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization/linear-regression-cost-function"
title: "为线性回归定义成本函数"
description: "为线性模型定义一个简单的成本函数（例如，均方误差）。"
order: 3
plots: ["plots/2307-0.json"]
sourceHash: "93b676712ab19d22ee0963220602c8f973f353b8b9e512bf15fa6e31435c6597"
sourceCorrections: []
---

简单线性模型，通常表示为 $y = mx + b$，在许多分析中是基础。当处理数据点时，例如（学习时长，考试分数）这样的数据对，目标通常是找到一条直线，它能最好地描述这些数据对之间的关系。在数学上，定义“最佳”意味着什么对于这个过程是必要的。

设想一下，你有一些数据点绘制在图表上。你可以通过这些点画出许多不同的直线。有些线会非常接近这些点，而另一些则会偏离较远。我们需要一种方法来衡量由特定斜率 $m$ 和截距 $b$ 定义的特定直线与我们数据的*拟合程度*。这个衡量标准就是我们所说的**成本函数**（有时也称为*损失函数 (loss function)*或*目标函数*）。

主要思想是量化 (quantization)我们的直线对每个数据点产生的“误差”或“错误”。假设我们有 $n$ 个数据点：$(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$。对于任何给定的输入 $x_i$，我们当前的直线 $y = mx + b$ 会预测一个输出值。我们将这个预测值称为 $\hat{y}_i$（读作“y-hat”）：


$$
\hat{y}_i = mx_i + b
$$


该数据点的实际输出值是 $y_i$。实际值与预测值之间的差 $y_i - \hat{y}_i$ 告诉我们预测针对该特定点的偏离程度。这个差值通常被称为*残差*。

如果实际值 $y_i$ 大于预测值 $\hat{y}_i$，残差为正。如果预测值大于实际值，残差为负。如果我们只是将所有数据点的这些残差加起来，正误差和负误差可能会相互抵消，即使直线拟合效果很差，也可能导致总误差显得误导性地小。

为了避免这种抵消并确保更大的误差对我们衡量总的“不佳程度”贡献更多，我们通常对每个残差进行平方：$(y_i - \hat{y}_i)^2$。平方运算使所有误差都变为正值，并对较大的偏差施加比小偏差更显著的惩罚（例如，误差为2变为4，而误差为3变为9）。

现在我们有了每个单独点的误差衡量，我们需要一个数字来代表我们的直线在*所有* $n$ 个数据点上的*总体*误差。一种非常常见的方法是计算这些平方误差的平均值。这给了我们\*\*均方误差（MSE）\*\*成本函数。

在数学上，我们将均方误差成本函数（我们称之为 $J$）定义为我们模型参数 (parameter) $m$ 和 $b$ 的函数：


$$
J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$


让我们来分解一下：

- $J(m, b)$：这表明成本 $J$ 取决于所选的斜率 $m$ 和截距 $b$ 的值。我们的目标是找到使 $J$ 尽可能小的 $m$ 和 $b$ 值。
- $n$：我们拥有的数据点的总数。
- $\sum_{i=1}^{n}$：这是求和符号。它表示“从 $i=1$ 到 $i=n$，将以下表达式对每个数据点进行累加”。
- $(x_i, y_i)$：第 $i$ 个数据点的输入 ($x_i$) 和实际输出 ($y_i$)。
- $\hat{y}_i$：我们的模型 ($mx_i + b$) 对输入 $x_i$ 预测的输出。
- $(y_i - \hat{y}_i)^2$：第 $i$ 个数据点的平方误差。
- $\frac{1}{n}$：将平方误差的总和除以 $n$ 得到*平均*平方误差。

我们可以将 $\hat{y}_i$ 的定义直接代入成本函数公式中：


$$
J(m, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + b))^2
$$


这个公式给我们一个数字 $J$，它告诉我们直线 $y = mx + b$ 与数据平均拟合的糟糕程度。$J$ 值越小，意味着直线的预测平均来看更接近实际数据点，表明拟合效果更好。$J$ 值越大则意味着拟合效果越差。



![模型误差可视化（均方误差思想）](plots/2307-0.json)



> 此图显示了样本数据点（蓝色圆点）、一个可能的线性模型（红色直线），以及实际数据点与模型预测之间的垂直误差（残差，灰色虚线）。MSE 成本函数计算这些虚线长度的平方平均值。

我们的优化目标现在已明确：找到 $m$ 和 $b$ 的具体值，使 $J(m, b)$ 的值尽可能低。最小化此成本函数意味着找到一条直线，使其与数据点的平均平方垂直距离最小。

为什么选择均方误差？除了对较大误差施加更多惩罚外，结果表明 MSE 具有有利的数学性质。具体来说，它是一个平滑、连续的函数，而且重要的是，我们可以轻松地计算它对 $m$ 和 $b$ 的导数。正如我们在前几章中看到的，这些导数（梯度）正是我们引导梯度下降 (gradient descent)算法趋向最小成本所需。

在下一节中，我们将使用这个成本函数 $J(m, b)$，并应用我们所学的微积分知识来计算其梯度。这将是推动我们优化过程的引擎。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/s/ISLR_full_print.pdf) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Publisher: Springer
  一本标准教材，对统计学习提供了应用介绍，涵盖了线性回归和均方误差等成本函数的概念。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  本教材将数学主题（包括微积分和优化）与它们在机器学习算法中的应用直接联系起来，例如定义和最小化成本函数。
- [CS229 Lecture Notes, Linear Regression](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf) — Andrew Ng, Tengyu Ma (2022)
  Publisher: Stanford University
  广受好评的机器学习课程官方讲义，清晰地解释了线性回归、均方误差成本函数及其在优化中的作用。
