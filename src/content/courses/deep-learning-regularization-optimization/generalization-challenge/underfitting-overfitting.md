---
course: "deep-learning-regularization-optimization"
chapter: "generalization-challenge"
lesson: "underfitting-overfitting"
sourceId: 4870
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-1-generalization-challenge/underfitting-overfitting"
title: "理解欠拟合与过拟合"
description: "阐述模型训练中欠拟合和过拟合的含义。"
order: 2
plots: ["plots/4870-0.json"]
sourceHash: "16de1184168ccdeb167120a2c1a8bcf49a3edd8fda432f2e5daabd3b546cf952"
sourceCorrections: []
---

如前所述，训练深度学习 (deep learning)模型的主要目标不仅是在其训练数据上获得高准确率，而是在新的、未见过的数据上表现良好。这种能力称为泛化。当模型未能有效泛化时，通常属于两种情况之一：欠拟合 (underfitting)或过拟合 (overfitting)。让我们更仔细地审视这些。

### 欠拟合 (underfitting)：模型过于简单

想象一下，你试图用一条直线穿过一组明显呈曲线分布的点。这条线不够复杂，无法反映数据的潜在模式。这便是**欠拟合**的实质。

欠拟合模型未能学习到训练数据中重要的模式。它通常过于简单，可能因为容量不足（层数或神经元过少）或训练时间不够长。

**欠拟合的迹象：**

- **训练误差高：** 模型即使在其训练过的数据上表现也很差。
- **验证/测试误差高：** 因此，模型在新的数据上表现也很差。验证误差可能与训练误差非常接近，但两者都高得无法接受。

当模型欠拟合时，这表明它没有学习到输入特征与目标输出之间的相关关系。它具有高*偏差*，表明它对数据结构的设定过于简单或不正确。增加模型复杂度、添加更多相关特征或延长训练时间，可能有助于缓解欠拟合。

### 过拟合 (overfitting)：模型学得过多

现在，考虑相反的情况。想象一下，你画了一条高度复杂、弯曲的线，它精确地穿过训练集中的每一个点，包括任何随机噪声或异常值。尽管这条线完美地描述了训练数据，但它不太可能代表真实潜在趋势，并且在预测新点时很可能会表现不佳。这便是**过拟合**。

过拟合模型学训练数据学得*太*好了。它不仅学到了潜在的模式，还学到了训练集特有的噪声和随机波动。它实际上是记忆了训练样本，而不是学习管理数据的一般规律。

**过拟合的迹象：**

- **训练误差低：** 模型在训练数据上表现优异。
- **验证/测试误差高：** 模型在新的、未见过的数据上的表现明显差于训练数据。训练误差与验证误差之间存在显著差距。

过拟合通常发生于模型容量过大（相对于训练数据量过于复杂）或训练时间过长时。模型开始拟合噪声，导致泛化能力差。它具有高*方差*，表明其预测对所见到的具体训练数据高度敏感。正则化 (regularization)、获取更多数据或使用早停等方法是应对过拟合的常用手段。

### 直观理解差异

训练误差和验证误差在训练周期内的关系提供了一个有用的诊断工具。我们可以直观了解欠拟合 (underfitting)、过拟合 (overfitting)以及良好拟合模型的典型模式。



![训练误差对比验证误差](plots/4870-0.json)



> 训练期间误差曲线的比较。欠拟合时，训练（蓝色虚线）和验证（蓝色实线）误差都较高。过拟合时，训练误差（红色虚线）下降，但验证误差（红色实线）在某个点后上升。良好拟合则显示两种误差均下降并趋于收敛（绿色线条）。

在模型复杂度和数据中的模式之间找到恰当的平衡点是基本要求。过于简单的模型（欠拟合）学不到足够信息，而过于复杂的模型（过拟合）会学到不正确的东西（噪声）。后续章节中讨论的方法，即正则化 (regularization)和优化手段，旨在帮助把握这种平衡，并构建在新的数据上泛化良好的模型。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖深度学习模型中泛化、欠拟合和过拟合的基本概念。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  对偏差-方差权衡、模型复杂度、欠拟合和过拟合进行严谨的统计学阐述。
