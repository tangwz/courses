---
course: "introduction-to-machine-learning"
chapter: "unsupervised-learning-clustering"
lesson: "choosing-number-of-clusters-k"
sourceId: 1510
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/choosing-number-of-clusters-k"
title: "选择聚类数量 (K)"
description: "讨论了选择最佳聚类数量（K）的难点和简单方法。"
order: 4
plots: ["plots/1510-0.json", "plots/1510-1.json"]
sourceHash: "541c923f1c890d65b43c04261bbba1042f480e57a35a1d29c55d41618d9bbe18"
sourceCorrections: []
---

在使用K-Means算法时，您可能会注意到一个比较突出的要求：您需要*在*运行算法*之前*，准确地告知它要找出多少个聚类，即$K$值。这与监督学习 (supervised learning)不同，在监督学习中，类别数量通常由数据中的标签决定。而在无监督学习 (unsupervised learning)中，由于我们没有标签，因此确定最佳分组数量（$K$）是这项工作的挑战之一。

选择合适的$K$为何重要？如果$K$值过小，算法可能会将不同的数据组合并在一起。如果$K$值过大，它可能会将自然形成的组拆分成更小、意义不大的碎片。找到合适的$K$有助于确保算法发现的聚类能有效反映数据中固有的结构。

那么，如何选择一个好的$K$值呢？没有一个单一的完美方法，这通常需要一些尝试和判断。以下是几种适合初学者的常见方法：

### 使用背景知识

有时，您可能已经对数据或正在尝试解决的问题有一些了解。

- **例子：** 如果您正在对零售业务的客户数据进行聚类，您可能有一个业务目标，就是将客户分为特定数量的组，例如“高消费客户”、“偶尔购物者”和“不活跃用户”。在这种情况下，先前的知识表明$K=3$可能是一个合理的起始点。

这并非总是可行，特别是在处理新数据集时，但请务必考虑现有知识能否指导您的选择。

### 可视化检查（适用于低维度数据）

如果您的数据只有两到三个特征（维度），您通常可以绘制图表并进行目视检查，以了解存在多少个自然分组。



![二维数据的可视化检查](plots/1510-0.json)



> 这是一个具有两个特征的数据点的简单散点图。从视觉上看，这些点似乎形成了大约三个不同的组。

“虽然有用，但这种方法有局限性。大多数数据集的特征数量远远超过能轻松可视化的程度。此外，视觉判断可能带有主观性。”

### 肘部法则

这是估算$K$值的最常见的定量方法之一。核心思想是针对不同范围的$K$值（例如，从$K=1$到$K=10$）多次运行K-Means算法。对于每个$K$值，您会计算一个分数，以衡量聚类的表现如何。

一个标准分数是**簇内平方和 (WCSS)**，有时也称为*惯性*。

- **WCSS衡量什么：** 它计算每个数据点与其所属簇的中心点（质心）之间的平方距离之和。
- **解读：** 更小的WCSS值意味着数据点平均而言更接近其所属簇的中心。这表明聚类更紧密、更紧凑。

随着您增加聚类数量（$K$），WCSS通常会减小，因为数据点会更接近更多的质心。如果$K$等于数据点数量，WCSS将变为零，但这并不是有用的聚类。

肘部法则涉及绘制每个$K$值的WCSS图。然后您需要寻找图上的一个“肘部”点。这是WCSS下降速度开始明显放缓的点，形成一个看起来像手臂肘部的角度。



![选择K的肘部法则](plots/1510-1.json)



> 显示不同聚类数量（$K$）的WCSS值的图表。“肘部”出现在$K=3$附近，此时增加更多聚类在减少WCSS方面的收益递减。

肘部点对应的$K$值通常被认为是聚类数量的良好候选值。它代表一种平衡：增加更多聚类几乎不会显著减少簇内方差。

**重要提示：** 肘部并非总是清晰明确。有时图表可能会显示平滑曲线，这会使选择变得更模糊。在这种情况下，您可能需要将此方法与其他技术结合使用，或者更多依赖背景知识或分析的具体目标。

### 总结

为K-Means选择合适的聚类数量（$K$）是无监督学习 (supervised learning) (unsupervised learning)中的重要一步。尽管像可视化检查（对于低维度数据）和肘部法则这样的方法提供了有价值的指导，但通常没有一个单一的“正确”答案。最佳$K$值取决于您的数据以及您希望通过聚类实现什么。尝试这些方法建议的几个不同$K$值，并评估所得聚类的质量和可解释性，是一种常见做法。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; Pages: Chapter 10
  提供K-Means聚类算法的入门介绍，并讨论选择聚类数量的实际挑战，包括肘部法则等方法。
- [Machine Learning (Coursera Course)](https://www.coursera.org/learn/machine-learning) — Andrew Ng, Aarti Bagul, Geoff Ladwig, Eddy Shyu (2012)
  Publisher: Coursera
  这门高评价的入门课程清晰地解释了K-Means算法，并提供了选择聚类数量的实用指导，包括肘部法则。
- [\`sklearn.cluster.KMeans\` - scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) — scikit-learn developers (2023)
  提供K-Means实现的技术文档，包括`n_clusters`参数及其选择的实际考量。
