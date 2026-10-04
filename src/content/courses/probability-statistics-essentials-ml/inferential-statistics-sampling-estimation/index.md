---
course: "probability-statistics-essentials-ml"
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-4-inferential-statistics-sampling-estimation"
sourceId: 457
chapter: "inferential-statistics-sampling-estimation"
title: "推断统计：抽样与估计"
order: 4
description: "了解抽样方法、中心极限定理，以及构建总体参数的点估计和区间估计。"
hasQuiz: true
---

到目前为止，我们一直专注于使用描述性统计来总结手头的数据。现在，我们将注意力转向仅根据一小部分数据，即*样本*，来对更大的总体做出有依据的推断。这很必要，因为获取或分析整个总体（例如，所有潜在客户，所有传感器读数）通常是不可能或成本过高的。

本章介绍*推断统计*的基本原理。你将学习如何：

*   区分总体和样本，并理解抽样的必要性。
*   识别不同的抽样方法。
*   理解*中心极限定理* (CLT) 及其在统计推断中的重要作用，特别是关于样本均值的分布。
*   计算总体参数的*点估计*，例如使用样本均值 $\bar{x}$ 来估计总体均值 $\mu$。
*   构建和解释*置信区间*，它根据样本数据为总体参数提供一个合理的值范围。

我们将把这些内容应用于实践，使用 Python 模拟抽样分布并计算估计值，从而连接理论与应用。
