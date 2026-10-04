---
course: "introduction-to-machine-learning"
chapter: "supervised-learning-classification"
lesson: "decision-boundaries"
sourceId: 1495
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-4-supervised-learning-classification/decision-boundaries"
title: "决策边界"
description: "直观呈现并理解分类模型如何创建决策边界。"
order: 4
plots: ["plots/1495-0.json"]
sourceHash: "ad0da230476b7f6327680d360d8b83a6463f19a549c3eae68f47fdc4811f4d02"
sourceCorrections: []
---

让我们直观地了解分类算法，比如我们刚讨论的逻辑回归，是如何实际分隔数据中的不同组或类别的。设想您有一个数据点的散点图，其中每个点都属于特定类别（例如“垃圾邮件”或“非垃圾邮件”、“猫”或“狗”）。模型如何决定一个新点属于哪个类别呢？它通过一种被称为**决策边界**的方式来做到这一点。

可以把决策边界看作算法学习到的一条看不见的线或一个看不见的曲面。这个边界将数据所在的空间划分为不同区域，每个区域都对应一个预测类别。如果一个数据点落在边界的一侧，模型将其归为一个类别；如果落在另一侧，则将其归为另一个类别。

### 逻辑回归中的决策边界

分类算法旨在区分数据中的不同组或类别。想象一下，数据点散布在图表中，每个点都属于特定类别（例如“垃圾邮件”或“非垃圾邮件”，“猫”或“狗”）。模型如何决定一个新点属于哪个类别呢？它通过一个称为**决策边界**的机制来完成。

- 如果 $p \ge 0.5$，预测为类别1。
- 如果 $p < 0.5$，预测为类别0。

决策边界正是模型不确定的时候，意味着概率恰好是0.5。Sigmoid函数何时输出0.5？当其输入正好是0时。

请记住，逻辑回归中Sigmoid函数的输入通常是特征的线性组合，例如对于两个特征（$x_1$, $x_2$）来说，$z = w_0 + w_1 x_1 + w_2 x_2$。因此，决策边界由以下方程定义：

$w_0 + w_1 x_1 + w_2 x_2 = 0$

对于具有两个特征的数据，这个方程代表一条直线。这条线将二维平面分成两个区域：一个是模型预测为类别1的区域（$z > 0$，因此 $p > 0.5$），另一个是预测为类别0的区域（$z < 0$，因此 $p < 0.5$）。

### 直观呈现线性决策边界

让我们具体化这一点。设想我们有属于两个类别（红色和蓝色）的数据点，它们根据两个特征（特征1和特征2）绘制。一个在此数据上训练的逻辑回归模型可能会找到一个线性决策边界，如下所示。



![线性决策边界示例](plots/1495-0.json)



> 一个简单的散点图，显示了两类数据点（红色和蓝色），它们由逻辑回归等模型学习到的线性决策边界（灰色线）分隔。通常在线的上方和右侧的点将被分类为蓝色（类别1），而在线的下方和左侧的点将被分类为红色（类别0）。

图上绘制的任何新数据点都将根据其落在灰色线的哪一侧进行分类。这种可视化方式有助于理解模型如何根据输入特征做出判断。

### 线性边界

需要提及的是，决策边界并非总是直线。虽然基本的逻辑回归通常产生线性边界，但分类问题常常需要更复杂的形状才能有效地分隔类别。

设想类别以更复杂的方式混合，或许一个类别集中在中间，另一个形成环绕。一条直线在分隔这些方面不会很好。其他算法，包括我们接下来会介绍的K近邻（KNN）算法，或者对逻辑回归的修改（例如使用多项式特征），可以生成非线性决策边界（曲线、圆形甚至更不规则的形状）。

### 决策边界为何重要

理解决策边界有助于您：

1. **可视化模型行为：** 它为您提供了一种直观的图形感受，了解模型如何分离数据。
2. **理解模型复杂性：** 简单的边界（如直线）意味着模型更简单，而非常弯曲的边界表明模型更复杂。
3. **识别潜在问题：** 过于复杂的边界如果完美地分离了所有训练点，可能表示过拟合 (overfitting)（模型对训练数据学习得过于细致，包括噪声，可能在新数据上表现不佳）。未能很好地分离类别的边界可能表示欠拟合 (underfitting)（模型过于简单）。

当我们查看不同的分类算法时，请注意它们倾向于创建哪种决策边界。这会让你对它们的优点和缺点有所了解，针对不同类型的数据分布。接下来，我们将考察K近邻算法，它采用一种截然不同的分类方法，并产生一种不同类型的决策边界。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  提供了分类算法的全面统计处理，包括对逻辑回归等各种模型的决策边界的详细解释。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.
  提供了各种分类算法（包括逻辑回归）决策边界的实际示例和清晰解释，适合初学者。
- [CS229 Lecture Notes: Supervised Learning](https://cs229.stanford.edu/main_notes.pdf) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  从基础机器学习视角涵盖了逻辑回归和决策边界概念，备受推崇。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-40087-2](https://doi.org/10.1007/978-0-387-40087-2)
  提供了分类、决策理论以及不同模型决策边界构建的全面理论处理。
