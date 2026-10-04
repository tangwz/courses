---
course: "probability-statistics-essentials-ml"
chapter: "introduction-regression-analysis"
lesson: "least-squares-estimation"
sourceId: 1439
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-6-introduction-regression-analysis/least-squares-estimation"
title: "最小二乘估计法"
description: "学习最小二乘法如何用于估计回归系数。"
order: 2
plots: ["plots/1439-0.json"]
sourceHash: "851b9bf3b00502217cb0d1278f4f3a29b991ee232a73e0b7242be33a066a5839"
sourceCorrections: []
---

简单线性回归旨在找到最能描述自变量 $x$ 与目标变量 $y$ 之间关系的直线 $y = \beta_0 + \beta_1 x$。但“最好”在数学上意味着什么？需要一种准确的方法来量化 (quantization)一条直线与观测数据点 $(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)$ 的拟合程度。

设想在数据的散点图中绘制一条可能的直线。对于每个数据点 $(x_i, y_i)$，该直线预测一个值 $\hat{y}_i = \beta_0 + \beta_1 x_i$。实际观测值 $y_i$ 与预测值 $\hat{y}_i$ 之间的差值被称为该点的**残差**或误差：

$e_i = y_i - \hat{y}_i = y_i - (\beta_0 + \beta_1 x_i)$

残差 $e_i$ 代表实际数据点与直线之间的垂直距离。一条好的直线应总体上具有较小的残差。

简单地将残差求和 ($\sum e_i$) 并不是衡量拟合度的好方法。为什么？因为正残差（在线上方的点）和负残差（在线下方的点）可能会相互抵消。一条直线的残差和可能为零，但它仍然可能与数据拟合得非常差。

为了解决这个抵消问题并确保所有误差都对我们的不拟合度量有正向贡献，我们在求和之前对每个残差进行平方。这便得到了**残差平方和 (SSR)**，也称为残差平方和 (RSS) 或误差平方和 (SSE)：

$SSR = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2$

**最小二乘法**的原则是将“最佳”拟合线定义为使残差平方和 (SSR) *最小化*的直线。我们希望找到截距 ($\beta_0$) 和斜率 ($\beta_1$) 的特定值，使数据点到直线的垂直距离平方和尽可能小。

我们如何找到这些通常表示为 $\hat{\beta}_0$ 和 $\hat{\beta}_1$ 的最佳值呢？这涉及使用微积分。我们将 SSR 视为 $\beta_0$ 和 $\beta_1$ 的函数。我们分别对 SSR 求关于 $\beta_0$ 和 $\beta_1$ 的偏导数，将两个导数都设为零，然后求解所得到的两个线性方程组（通常称为“正规方程”）。虽然我们在此不进行完整的推导，但其解提供了估计系数的公式：

估计斜率 ($\hat{\beta}_1$) 的计算方法如下：

$\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$

此处，$\bar{x}$ 是自变量值的均值，$\bar{y}$ 是目标变量值的均值。请注意，分子与 $x$ 和 $y$ 之间的样本协方差成比例，分母与 $x$ 的样本方差成比例。直观上，斜率反映了 $y$ 相对于 $x$ 的变化趋势（协方差）的变化程度（方差）。

一旦我们有了估计斜率 $\hat{\beta}_1$，估计截距 ($\hat{\beta}_0$) 的计算方法很简单：

$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$

此公式确保回归线通过点 $(\bar{x}, \bar{y})$，即数据的重心。

我们来看看这个的图示。想象数据点的散点图。最小二乘法找到的是唯一一条使以残差为边长的正方形面积之和最小的直线。



![最小二乘回归线和残差](plots/1439-0.json)



> 数据点（蓝色），计算出的最小二乘回归线（红色），以及残差（垂直灰色线）。该方法使以这些残差形成的灰色正方形的面积之和最小。

对于大型数据集，手动计算这些和与公式可能很乏味。幸运的是，统计软件和像 Python 这样的编程语言中的库能高效处理这些计算。Scikit-learn 和 Statsmodels 等库提供了实现最小二乘法（以及更高级技术）的函数，可以直接从您的数据中估计回归系数，让您专注于解释结果和评估模型。我们将在本章后面了解如何使用这些库。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Publisher: Springer
  简要介绍了线性回归和最小二乘法，连接了统计理论与机器学习的实践应用。
- [\`statsmodels.regression.linear_model.OLS\`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) — Statsmodels Developers (2024)
  Statsmodels Python库中普通最小二乘法（OLS）实现的官方文档，用于实际应用。
- [CS229 Lecture Notes: Linear Regression](https://github.com/mxc19912008/Andrew-Ng-Machine-Learning-Notes/blob/master/cs229-notes1.pdf) — Andrew Ng (2009)
  Publisher: Stanford University
  斯坦福大学机器学习课程的讲义，涵盖了最小二乘法作为监督学习的基础技术。
