---
course: "calculus-fundamentals-machine-learning"
chapter: "calculus-action-optimization"
lesson: "simple-linear-regression-model"
sourceId: 2305
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization/simple-linear-regression-model"
title: "示例：简单线性回归模型"
description: "介绍一个基本的线性模型 (y = mx + b) 作为示例。"
order: 2
plots: ["plots/2305-0.json"]
sourceHash: "1c7fbd32b901e6b0d5941327f142d5cd5386e83338a99d54d47571df2a5110df"
sourceCorrections: []
---

好的，我们来运用微积分工具。我们已经讨论了如何使用导数求最小值和最大值，以及梯度下降 (gradient descent)的思路。现在，我们将这些思路应用于一个基本的机器学习 (machine learning)模型：**简单线性回归**。

简单线性回归的目标是找到一条最佳直线，来描述单个输入特征与目标输出之间的关系。想象一下，你有一些数据点绘制在图表上。线性回归试图在这些点中画一条尽可能拟合的直线。

### 模型方程

直线的方程可能对你来说不陌生：


$$
y = mx + b
$$


让我们在机器学习 (machine learning)的背景下分析这些组成部分：

- **$x$**: 这是我们的输入特征。它是我们已知的信息，也称作自变量。例如，$x$ 可以是房屋的平方英尺。
- **$y$**: 这是我们想要预测的目标变量，也称作因变量。例如，$y$ 可以是房屋的售价。
- **$m$**: 这是直线的**斜率**。它决定了当 $x$ 增加一个单位时，$y$ 变化多少。在我们的房屋示例中，它表示每增加一平方英尺，价格会增加多少。
- **$b$**: 这是直线的**y截距**。它是当 $x$ 为零时 $y$ 的值。在某些情况下，这可能代表一个基准值（例如，不考虑大小的房屋基本价格，尽管这种解释并非总是有意义，具体取决于上下文 (context)）。

在机器学习术语中，$m$ 和 $b$ 是我们模型的**参数 (parameter)**（有时也称作**权重 (weight)**或**系数**）。我们训练时的目标是找到 $m$ 和 $b$ 的最优值，使我们的直线最能拟合数据。

### 目标的可视化

考虑一个表示学习时间 ($x$) 与考试分数 ($y$) 之间关系的小数据集。



![简单数据与潜在直线](plots/2305-0.json)



> 蓝色点表示我们的数据。绿色虚线似乎很好地捕捉了趋势，而红色虚线显然拟合得很差。线性回归旨在找到能够产生最佳拟合（如绿线或更好）的直线参数 (parameter)（$m$ 和 $b$）。

寻找“最佳拟合”线的过程是一个监督学习 (supervised learning)任务。我们得到示例对 $(x, y)$（学习时间，考试分数），并且我们希望机器学习 (machine learning)算法学习它们之间的关系，这种关系由参数 $m$ 和 $b$ 表示。一旦学习完成，我们就可以使用模型 $y = mx + b$ 来预测新的学习时间下的考试分数。

核心思路是我们可以衡量特定直线（由 $m$ 和 $b$ 的特定值定义）对数据的拟合*程度*。如果我们可以衡量这种“拟合优度”（或者反过来，衡量“拟合差度”或误差），那么我们就可以使用微积分，特别是梯度下降 (gradient descent)，系统地调整 $m$ 和 $b$，使直线越来越好地拟合数据。

这种“拟合度量”就是我们所说的**成本函数**，我们将在下一节中定义它。请记住，对于我们的优化算法，$m$ 和 $b$ 是我们将要调整的变量，以使此成本最小化。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2013)
  Publisher: Springer; Pages: 59-120; DOI: [10.1007/978-1-4614-7137-0](https://doi.org/10.1007/978-1-4614-7137-0)
  一本基础教材，清晰介绍简单线性回归的原理和解释，是学习机器学习概念的优秀入门资料。
- [Mathematics for Machine Learning](https://mml-book.com/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; Pages: 119-158
  为机器学习算法提供数学严谨性，详细解释导数、梯度和优化技术，这些对于理解线性回归训练至关重要。
- [Machine Learning](https://www.coursera.org/learn/machine-learning) — Andrew Ng (2012)
  Journal: Coursera; Publisher: DeepLearning.AI and Stanford Online
  广受好评的在线课程，清晰展示了如何使用梯度下降法构建和优化简单线性回归模型，并提供了实用示例和直观解释。
