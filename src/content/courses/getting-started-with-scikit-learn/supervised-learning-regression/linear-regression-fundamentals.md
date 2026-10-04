---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-regression"
lesson: "linear-regression-fundamentals"
sourceId: 1841
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/linear-regression-fundamentals"
title: "线性回归基本原理"
description: "简要概述简单线性回归和多元线性回归背后的原理。"
order: 2
plots: ["plots/1841-0.json"]
sourceHash: "fdc235c4bb97567f1bc4a784fb2b5d21ee479e5df987852ee9fe20377285595d"
sourceCorrections: []
---

线性回归是监督学习 (supervised learning)中回归问题的一种基本算法。它的主要目的是通过将线性方程拟合到观测数据，来建立一个或多个解释变量（特征）与连续目标变量之间的关系模型。可以将其理解为在数据点中找出“最拟合”的直线（或在更高维度中的超平面）。

### 简单线性回归：用一个特征建立模型

*简单线性回归*是一种预测目标变量 $y$ 的方法，它基于单个特征 $x$。这种关系由直线方程表示：


$$
y = \beta_0 + \beta_1 x + \epsilon
$$


让我们来解释这个方程：

- $y$: 我们想要预测的目标变量（例如，房价）。
- $x$: 我们用于预测的输入特征（例如，房屋面积，以平方英尺计）。
- $\beta_0$: *截距*（或偏置 (bias)）。这是当 $x$ 为零时 $y$ 的预测值。几何上，它是回归线与 y 轴的交点。
- $\beta_1$: 特征 $x$ 的*系数*（或斜率）。它表示特征 $x$ 增加一个单位时，目标变量 $y$ 的变化量。正的 $\beta_1$ 表示 $y$ 倾向于随 $x$ 的增加而增加，而负的 $\beta_1$ 则表示反向关系。
- $\epsilon$: *误差项*（或残差）。它表示 $y$ 的实际观测值与模型预测值（$\beta_0 + \beta_1 x$）之间的差异。它衡量了 $y$ 中未由与 $x$ 的线性关系解释的变化量。

训练简单线性回归模型的目标是找到截距（$\beta_0$）和斜率（$\beta_1$）的最优值，从而最小化所有数据点的总误差（$\epsilon$）。



![简单线性回归示例](plots/1841-0.json)



> 一张散点图，显示了数据点以及通过简单线性回归找到的最拟合直线。

### 多元线性回归：用多个特征建立模型

在大多数情况下，目标变量依赖于不止一个特征。*多元线性回归*通过考虑多个输入特征（$x_1, x_2, ..., x_n$）来预测目标 $y$，从而扩展了简单的情况。

方程变为：


$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
$$


这里：

- $y$: 目标变量。
- $x_1, x_2, \dots, x_n$: 输入特征（例如，房屋面积、卧室数量、房龄）。
- $\beta_0$: 截距，表示当所有特征（$x_1, \dots, x_n$）都为零时 $y$ 的预测值。
- $\beta_1, \beta_2, \dots, \beta_n$: 每个对应特征的系数。每个 $\beta_i$ 表示特征 $x_i$ 增加一个单位时 $y$ 的预期变化，*假设所有其他特征保持不变*。这种“保持不变”的属性对于解释结果很重要。
- $\epsilon$: 误差项，衡量了未由特征的线性组合解释的变化量。

模型在 n 维特征空间中找到最拟合的超平面。目标保持不变：找到最小化预测误差的系数（$\beta_0, \beta_1, \dots, \beta_n$）。

### “线性”假设

需要认识到，线性回归其本质上假设特征与目标变量之间存在*线性关系*。模型试图使用直线或超平面捕获其基础趋势。如果真实关系是非线性的（例如，急剧弯曲），简单的线性模型可能无法提供准确的预测（它可能*欠拟合 (underfitting)*数据）。我们将在后面讨论处理非线性的方法，但对于许多问题，线性回归提供了一个不错的起点和一个高度可解释的模型。

### 如何找到“最拟合”

算法如何确定系数（$\beta$s）的最优值？最常用的方法是*普通最小二乘法 (OLS)*。OLS 通过寻找一条线（或超平面），使得实际目标值 ($y$) 与模型预测值 ($\hat{y}$) 之间*平方差的和*最小。将差值平方可以确保正负误差不会相互抵消，并对较大的误差施以更重的惩罚。尽管数学细节涉及微积分或线性代数，但目标很简单：最小化总平方误差。

### 可解释性

线性回归的一个重要优点是它的可解释性。所学到的系数（$\beta_1, \dots, \beta_n$）直接量化 (quantization)了每个特征与目标之间的关系，假设其他特征保持不变。例如，如果“房屋面积”的 $\beta_1$ 为 150，这表明，平均而言，每增加一平方英尺的房屋面积，房价会增加 150 美元，同时保持卧室数量等其他因素不变。这使得理解模型的行为并解释其预测变得更容易。

既然我们已经掌握了线性回归背后的原理，后续章节将演示如何使用 Scikit-learn 实现、训练和评估这些模型。

## 参考资料

- [An Introduction to Statistical Learning: with Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor (2023)
  Publisher: Springer
  优秀的入门教材，涵盖线性回归、普通最小二乘法、模型解释以及相关的统计学习概念和Python应用。
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) — Scikit-learn Developers (2024)
  Scikit-learn官方文档，介绍了`LinearRegression`估计器的参数、属性和实际使用方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本标准参考书，对线性模型、普通最小二乘法及其统计理论提供了更高级和全面的阐述。
- [CS229 Lecture Notes: Supervised Learning: Linear Regression](http://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (2012)
  Publisher: Stanford University
  提供了清晰简洁的线性回归学术介绍，包括普通最小二乘解的推导。
