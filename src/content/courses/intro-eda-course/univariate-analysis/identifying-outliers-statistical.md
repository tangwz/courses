---
course: "intro-eda-course"
chapter: "univariate-analysis"
lesson: "identifying-outliers-statistical"
sourceId: 1162
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis/identifying-outliers-statistical"
title: "使用统计方法识别异常值"
description: "介绍用于识别潜在异常值的统计方法（如Z分数法、IQR规则）。"
order: 7
plots: []
sourceHash: "63bf0a37b41ad571865cca6ded0380f987ada6101239b9410e687461ac1254e5"
sourceCorrections: []
---

虽然直方图和箱线图等可视化图表能直观地展现数据分布和潜在的极端值，但统计方法提供了量化 (quantization)方式来标记 (token)潜在的异常值。异常值通常定义为在来自总体随机样本中，与其它数值距离异常远的观测点。识别这些点是EDA（探索性数据分析）的重要组成部分，原因有几点：它们可能表示数据录入错误、测量问题，或者数据中真实存在的、不寻常的现象。然而，它们也可能显著影响统计汇总结果（如均值和标准差），并影响某些机器学习 (machine learning)模型的性能。

识别数值数据中潜在异常值的两种常用统计技术是Z分数法和四分位数间距（IQR）法。

### Z分数法

Z分数衡量一个数据点与分布均值相距多少个标准差。假设数据大致服从正态（钟形）分布，Z分数超出某个阈值的数据点通常被认为是潜在异常值。

计算数据点 $x$ 的Z分数的公式是：

$Z = \frac{x - \mu}{\sigma}$

其中：

- $x$ 是单个数据点。
- $\mu$ 是数据集的均值。
- $\sigma$ 是数据集的标准差。

一个常用的经验是，将绝对Z分数大于3的数据点标记 (token)为潜在异常值。这个阈值基于正态分布的性质，大约99.7%的数据落在均值3个标准差之内。因此，超出此范围的值在统计上是罕见的。

**注意事项：**

- Z分数法对均值 ($\mu$) 和标准差 ($\sigma$) 很敏感。由于这两个度量本身都可能受到极端异常值的影响，因此此方法有时可能无法识别异常值，或在偏斜分布中错误识别点。
- 它在数据大致呈正态分布时效果最好。在使用Z分数法之前，请务必检查分布形状（例如，使用直方图）。

在Pandas中，可以使用 `(s - s.mean()) / s.std()` 为序列 `s` 计算Z分数。

### 四分位数间距（IQR）法

四分位数间距（IQR）法提供了一种更有效的方法来检测异常值，特别适用于非正态分布的数据集，或者当极端值可能不适当地影响均值和标准差时。此方法侧重于数据的第一四分位数（Q1，第25百分位数）与第三四分位数（Q3，第75百分位数）之间的范围。IQR代表数据中间50%的分布范围。

步骤如下：

1. 计算第一四分位数（Q1）和第三四分位数（Q3）。
2. 计算四分位数间距：$IQR = Q3 - Q1$。
3. 定义异常值检测的上下限：
   - 下限 = $Q1 - 1.5 \times IQR$
   - 上限 = $Q3 + 1.5 \times IQR$
4. 任何低于下限或高于上限的数据点都被标记 (token)为潜在异常值。

**为何是1.5？** 乘数1.5是一个标准惯例，源于John Tukey在探索性数据分析方面的工作。它为识别明显偏离数据中心主体的值提供了一个合理的平衡。有时你可能会看到使用乘数3来识别“极端”异常值。标准箱线图中的“须”通常延伸到1.5倍IQR范围内最远的数据点。

**注意事项：**

- IQR法对极端值不像Z分数法那么敏感，因为它依赖于Q1和Q3，这些是基于百分位数的值，不像均值和标准差那样容易受异常值影响。
- 它不假设数据具有任何特定分布。

在Pandas中，可以使用 `.quantile()` 方法计算Q1和Q3（例如，`s.quantile(0.25)` 和 `s.quantile(0.75)`），计算IQR，然后应用这些边界来筛选数据。

### 理解潜在异常值

需要记住的是，这些统计方法仅是*标记 (token)潜在异常值*。它们不自动意味着该数据点是错误的或应该被移除。一旦被识别，潜在异常值需要进一步检查：

- **检查错误：** 是数据录入时的拼写错误吗？测量设备故障？
- **领域知识：** 这个值在问题背景下合理吗？它是否代表一个罕见但有效的事件？
- **影响评估：** 这个点对汇总统计数据或潜在模型结果有多大影响？

如何处理异常值（例如，修正、移除、转换数据或使用其他统计方法）的决定，在很大程度上取决于具体背景、异常值的来源以及您的分析目标。在EDA过程中，主要目的通常只是识别并理解这些不寻常的点。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  本书提供了数据科学中统计方法的实践指南，涵盖了数据清理和准备背景下的Z-score和IQR等异常值检测技术。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James and Daniela Witten and Trevor Hastie and Robert Tibshirani (2021)
  Publisher: Springer
  这本教材涵盖了统计学习的基本概念，包括数据预处理和处理数据异常的重要性，这对于理解异常值的影响和处理相关。
