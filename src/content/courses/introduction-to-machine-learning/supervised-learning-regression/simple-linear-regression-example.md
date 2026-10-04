---
course: "introduction-to-machine-learning"
chapter: "supervised-learning-regression"
lesson: "simple-linear-regression-example"
sourceId: 1486
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-3-supervised-learning-regression/simple-linear-regression-example"
title: "简单线性回归示例"
description: "逐步讲解简单线性回归的实践示例。"
order: 6
plots: ["plots/1486-0.json", "plots/1486-1.json"]
sourceHash: "49284e1507118ea3157404551bb1ba9f8118410e61de9785df7216e94796fb31"
sourceCorrections: []
---

让我们通过一个直接的例子来巩固对线性回归的理解。请记住，回归的目标是预测一个连续的数值。在线性回归中，我们寻求找到一条最佳直线，用以描述输入变量（特征）与输出变量（目标）之间的关系。

假设我们想了解学生学习时长与期末考试分数之间的关系。这是一个典型的回归问题：给出学习时长（输入），我们希望预测考试分数（输出）。

### 场景：预测考试分数

假设我们收集了五名学生的数据：

| 学习时长 (x) | 考试分数 (y) |
| --- | --- |
| 1 | 40 |
| 2 | 55 |
| 3 | 65 |
| 4 | 70 |
| 5 | 80 |

我们的输入特征 $x$ 是“学习时长”，输出目标 $y$ 是“考试分数”。我们希望找到它们之间的一个线性关系。

### 数据可视化

在构建模型之前，最好先将数据可视化。散点图很适合这样做，它将每位学生的数据点表示为根据其学习时长和考试分数绘制的点。



![考试分数 vs. 学习时长](plots/1486-0.json)



> 散点图显示了五名学生的学习时长与考试分数之间的关系。

从图中看，这些点大致呈现上升趋势：学习时长越长，通常考试分数越高。用一条直线来近似这种关系似乎是合理的。

### 模型：一条直线

简单线性回归使用直线方程对这种关系进行建模：

$y = mx + b$

或者，使用机器学习 (machine learning)中常用的符号：

$\hat{y} = \theta_1 x + \theta_0$

这里：

- $\hat{y}$（读作“y-hat”）是*预测的*考试分数。
- $x$ 是输入特征，“学习时长”。
- $\theta_1$（或 $m$）是直线的*斜率*。它表示当学习时长 $x$ 增加一个单位时，预测分数 $\hat{y}$ 的变化量。
- $\theta_0$（或 $b$）是直线的*y轴截距*。它是当学习时长 $x$ 为零时，预测分数 $\hat{y}$。

### 找到最佳拟合直线

我们如何找到 $\theta_0$ 和 $\theta_1$ 的具体数值，从而得到“最佳”直线？正如我们之前讨论的，“最佳”意味着最小化我们直线的预测分数（$\hat{y}$）与数据集中实际分数（$y$）之间的差异。这种差异通过*成本函数*（如均方误差）来衡量。

*梯度下降 (gradient descent)*算法通常用于迭代调整 $\theta_0$ 和 $\theta_1$，逐步降低成本函数，直到找到使误差最小化并使直线最紧密拟合数据的数值。

假设在应用此过程后，我们找到的最佳拟合直线大约是：

$\hat{y} = 10x + 30$

因此，我们的模型参数 (parameter)是 $\theta_1 = 10$ 和 $\theta_0 = 30$。

让我们在散点图上将这条直线可视化：



![考试分数 vs. 学习时长与回归线](plots/1486-1.json)



> 散点图显示了计算出的线性回归线以及数据的趋势。

这条直线并没有完美地穿过每个点（这在真实数据中很少见），但它捕捉了整体趋势。

### 进行预测

既然我们有了模型（$\hat{y} = 10x + 30$），我们就可以使用它来预测学生基于其学习时长的考试分数。

例如，我们会预测学习了2.5小时的学生能得多少分？我们将 $x = 2.5$ 代入方程：

$\hat{y} = 10 \times (2.5) + 30$
$\hat{y} = 25 + 30$
$\hat{y} = 55$

我们的模型预测，学习2.5小时的学生考试分数为55分。

### 解释模型

我们的模型参数 (parameter)也提供了帮助我们理解的信息：

- **斜率 ($\theta_1 = 10$):** 学生每多学习一小时，我们的模型预测其分数将增加约10分。
  "\* **截距 ($\theta_0 = 30$):** 模型预测，如果学生学习0小时，分数将是30分。考虑截距在实际情况中是否有意义是很重要的。预测0小时学习的分数可能合理，也可能超出了我们数据的范围（我们没有接近0小时学习的学生数据）。"

### 示例总结

在此示例中，我们：

1. 从一个将学习时长与考试分数联系起来的简单数据集开始。
2. 使用散点图可视化了数据。
3. 理解了简单线性回归旨在将一条直线（$\hat{y} = \theta_1 x + \theta_0$）拟合到这些数据上。
4. 确定了最小化预测误差的最佳拟合直线参数 (parameter)（$\theta_1 = 10$，$\theta_0 = 30$）（使用梯度下降 (gradient descent)等方法，之前已讨论）。
5. 使用了得到的模型方程（$\hat{y} = 10x + 30$）对新输入进行预测。
6. 在问题情境中解释了斜率和截距的含义。

"这个简单的示例阐明了线性回归背后的核心思想：对数据中的线性关系进行建模以进行预测。尽管问题通常涉及更多特征和复杂性，但基本原理保持不变。"

## 参考资料

- [An Introduction to Statistical Learning with Applications in R](https://doi.org/10.1007/978-1-4614-7138-7) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Publisher: Springer; Pages: 440; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  全面介绍了线性回归，涵盖其公式、解释和基本统计原理。详见第3章。
- [CS229 Lecture Notes: Linear Regression and Gradient Descent](https://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (2012)
  解释了线性回归的理论基础，包括成本函数和用于参数优化的梯度下降算法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  在现代机器学习框架内，为线性回归、成本函数和梯度下降提供了实用而清晰的介绍。参考第2版，早期章节。
