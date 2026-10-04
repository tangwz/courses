---
course: "introduction-to-machine-learning"
chapter: "supervised-learning-regression"
lesson: "cost-functions-measuring-error"
sourceId: 1482
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-3-supervised-learning-regression/cost-functions-measuring-error"
title: "成本函数：衡量误差"
description: "了解成本函数（如均方误差）如何量化模型预测误差。"
order: 4
plots: ["plots/1482-0.json"]
sourceHash: "bd9b84a987b9191fba1125cfe0a2be2dce81770ef2c98a394ab7eb42f5a5a345"
sourceCorrections: []
---

我们知道线性回归试图在数据点中找到一条最符合的直线。但到底是什么让一条直线“最符合”呢？我们如何量化 (quantization)某条直线与数据的拟合程度？我们需要一种方法来衡量给定直线相关的误差，或者说“成本”。这种衡量方法帮助我们比较不同的可能直线，并告诉学习算法如何调整直线以改进其拟合。

可以这样考虑：对于我们绘制的穿过数据的任何一条直线，有些点会靠近直线，而另一些点可能会离得更远。实际数据点与我们的直线预测点之间的距离代表了该特定预测的误差。

### 计算单个点的误差

机器学习 (machine learning)中，为找到表现最优的模型，需要量化 (quantization)模型预测误差的方法。特别是线性回归模型，目标是拟合一条最能代表数据的直线。那么，如何衡量一条直线与数据点的契合程度？如何判断哪条直线表现最优？这要求定义一种方式来量化给定直线的误差，或称为“成本”。这种度量标准有助于比较不同的潜在直线，并指导学习算法调整直线以改进其拟合效果。

误差 = 实际值 - 预测值
误差 = $y_i - \hat{y}_i$

这种差值通常被称为 *残差*。正残差表示预测过低，负残差表示预测过高。残差为零表示该数据点的预测是完美的。



![残差：单个点的误差](plots/1482-0.json)



> 垂直虚线显示了一个数据点的误差（残差）：实际值（蓝点）与直线预测值（灰线上的点）之间的差。

### 聚合误差：成本函数

我们需要一个单一的数字来概括训练集中 *所有* 数据点的 *总* 误差。简单地将单个误差 ($y_i - \hat{y}_i$) 相加并没有太大作用，因为正误差和负误差可能会相互抵消，即使直线拟合效果很差，也会给我们一个误导性的很小的总误差。

一种常见的方法是：

1. **将每个单独误差平方：** $(y_i - \hat{y}_i)^2$。这解决了抵消问题（平方总是非负的），并且对更大的误差给予比小误差更重的惩罚。误差为 2 变为 4，而误差为 10 变为 100。
2. **计算这些平方误差的平均值：** 将所有平方误差相加，然后除以数据点的数量。

这就得到了 **均方误差 (MSE)**，这是回归问题中非常常用的成本函数。

MSE 的公式是：

$J(\theta_0, \theta_1) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$

让我们分解说明：

- $N$: 训练集中的数据点总数。
- $i=1$: 我们从第一个数据点开始求和...
- $... N$: ...直到最后一个数据点。
- $\sum$: 求和符号，表示我们将每个点的结果相加。
- $y_i$: 第 $i$ 个数据点的实际目标值。
- $\hat{y}_i$: 我们的线性模型对第 $i$ 个数据点预测的值。请记住，对于简单线性回归，$\hat{y}_i = \theta_0 + \theta_1 x_i$（其中 $\theta_0$ 是截距，$\theta_1$ 是斜率，这些是我们的模型需要学习的参数 (parameter)）。
- $(\hat{y}_i - y_i)^2$: 第 $i$ 个数据点的平方误差。（注意：有时你会看到 $(y_i - \hat{y}_i)^2$，这在平方后结果相同）。
- $\frac{1}{N}$: 我们将平方误差的总和除以 $N$ 以获得 *平均值*。
- $J(\theta_0, \theta_1)$: 这表明 MSE 是模型参数 ($\theta_0$ 和 $\theta_1$) 的一个 *函数*。斜率和截距的不同值将导致不同的直线、不同的预测值 ($\hat{y}_i$)，因此也会产生不同的 MSE 值。

*有时，特别是在统计学环境或其他课程中，你可能会看到公式中使用 $1/(2N)$ 而不是 $1/N$。因子 2 是为了在之后计算导数（特别是梯度下降 (gradient descent)）时便于数学运算而添加的，但这不会改变最小误差的位置。为了方便理解这个思想，$1/N$ 代表平均值通常更清楚。*

### 目标：最小化成本

“均方误差为我们提供了一个单一的正值，它表示由特定参数 (parameter) $\theta_0$ 和 $\theta_1$ 定义的直线与整体数据的拟合程度。完美拟合的 MSE 将为 0（尽管这在数据中很少发生）。拟合效果差的直线将具有较大的 MSE。”

因此，我们学习算法的目标是 **找到使 MSE 尽可能低的 $\theta_0$ 和 $\theta_1$ 值**。

最小化这个成本函数意味着找到一条直线，它在预测训练数据中的目标值时，平均而言会产生最小的平方误差。在下一节关于梯度下降 (gradient descent)的内容中，我们将看到算法 *如何* 系统地调整 $\theta_0$ 和 $\theta_1$ 以降低这个成本函数的值，从而有效地找到最符合的直线。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Publisher: Springer; Pages: Chapter 3; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  广泛使用的统计学习入门教材，清晰地解释了线性回归、残差以及最小化平方误差的概念。
- [CS229 Lecture Notes: Linear Regression](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnOs4qRdqTwKcLXe_8MmoeHqzsVTKPvbrGVGH1KU_uIi9BchTM0ddcmvyPu1s1uPmZ3xPvMFngftK9AseyPbqRQ7coXVIYXem4PVkP1A-YdeqhKs9WxlZmevKih2iKOZs8Zg==) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  一门基础机器学习课程的讲义，提供了线性回归模型和均方误差成本函数的详细且易懂的解释。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 3; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本权威教材，对线性模型和损失函数进行了更高级、更全面的统计处理，包括均方误差的理论基础。
