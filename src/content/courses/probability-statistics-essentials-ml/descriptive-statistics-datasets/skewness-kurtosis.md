---
course: "probability-statistics-essentials-ml"
chapter: "descriptive-statistics-datasets"
lesson: "skewness-kurtosis"
sourceId: 1395
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-3-descriptive-statistics-datasets/skewness-kurtosis"
title: "理解偏度和峰度"
description: "解释偏度和峰度，以便认识数据分布的形态。"
order: 3
plots: ["plots/1395-0.json", "plots/1395-1.json"]
sourceHash: "a9121fad5e555b772fe9f7836b9b02f3fc4be4cdbdbc5cdea20c594b9d2c90df"
sourceCorrections: []
---

均值、中位数、众数、方差和标准差等统计量可以告诉我们数据的中心位置和离散程度，但它们无法展现数据的全貌。两个数据集可能拥有相同的均值和标准差，但其形态可能大相径庭。此时，描述形态的统计量，特别是偏度和峰度，就显得很重要。它们分别帮助我们了解数据分布的不对称性和“尾部特征”。

### 偏度：衡量不对称性

偏度量化 (quantization)了分布偏离完全对称的程度。一个对称的分布，例如经典的钟形曲线（正态分布），其偏度为零。它的左右两边以中心峰值为轴呈镜像对称。

- **正偏态（右偏）：** 如果分布的尾部延伸到右侧更远，则该分布呈正偏态。这表示存在一个低值聚集区，而尾部则指向高值。在正偏态分布中，均值通常大于中位数，而中位数又通常大于众数。出现这种情况是因为尾部的大数值将均值向上拉动。可以设想收入分布；大多数人收入适中，但少数高收入者会使平均收入明显升高。偏度值 > 0。
- **负偏态（左偏）：** 相反，负偏态分布的尾部延伸到左侧更远。存在一个高值聚集区，而尾部则指向低值。在这种情况下，均值通常小于中位数，而中位数又通常小于众数。尾部的低数值将均值向下拉动。可以设想退休年龄；大多数人会在某个年龄段退休，但有些人会提早许多退休，从而拉低了平均值。偏度值 < 0。
- **零偏态（对称）：** 该分布完全对称。均值、中位数和众数通常相等（或在实际中非常接近）。偏度值 ≈ 0。



![偏度示例](plots/1395-0.json)



> 比较对称（蓝色）、正偏态（橙色）和负偏态（紫色）分布。注意长尾相对于主峰的位置。

了解偏度很重要，因为高度偏斜的数据可能违反某些统计检验和机器学习 (machine learning)模型的假设（特别是那些假定数据正态性的模型，如线性回归）。有时，会对偏斜数据进行变换（如对数变换），使其在建模前更趋对称。

### 峰度：衡量尾部厚度和峰度

峰度衡量分布的“尾部特征”——即数据在尾部相对于中心部分的集中程度。它常与正态分布进行比较，正态分布被认为是**中峰态**。

标准的衡量方式是*超额峰度*，计算公式如下：
$\text{超额峰度} = \text{峰度} - 3$
正态分布的峰度为3，因此其超额峰度为0。

- **尖峰态（超额峰度 > 0）：** 超额峰度为正的分布称为尖峰态。它们拥有比正态分布更厚的尾部，且通常峰值更尖锐。这表示极端值（异常值）出现的可能性比在正态分布中更大。金融市场回报常表现出尖峰态，这意味着大的收益或损失比正态模型预测的更常见。峰度值 > 3。
- **中峰态（超额峰度 ≈ 0）：** 这描述的是尾部与正态分布相似的分布，例如正态分布本身。峰度值 ≈ 3。
- **平峰态（超额峰度 < 0）：** 超额峰度为负的分布称为平峰态。它们拥有更薄的尾部，并且倾向于比正态分布更平坦（峰值更低）。这表示极端值出现的可能性较小。连续均匀分布是平峰态分布的一个例子。峰度值 < 3。

一个常见的误解是峰度*只*衡量分布的峰值高低。尽管峰值高低常有关联，但峰度的主要决定因素是尾部的“厚度”（权重 (weight)）。一个分布可以有高耸的峰值但尾部较薄，也可以有较低的峰值但尾部较厚。峰度具体反映了极端值的影响。



![峰度示例](plots/1395-1.json)



> 比较不同峰度的分布：尖峰态（红色，厚尾）、中峰态（蓝色，正态尾部）和平峰态（绿色，薄尾）。

高峰度（尖峰态）表明可能存在显著的异常值或肥尾现象，这对风险管理和模型选择很重要。低峰度（平峰态）则可能表示数据比正态分布更集中或更均匀。

偏度和峰度一起，对数据分布提供了更详尽的描述，补充了简单的中心和离散程度信息。计算这些值是探索性数据分析（EDA）中的一个标准步骤，并有助于确定后续的分析选择。像 Pandas 这样的库使得计算这些指标变得简单，我们将在本章后面看到。

## 参考资料

- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  一本Python数据分析实用指南，展示了如何将偏度和峰度的计算与解释作为探索性数据分析（EDA）的一部分。
- [Introduction to Probability and Statistics (Lecture 4: Descriptive Statistics: Mean, Variance, Skew, Kurtosis)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/pages/lecture-notes/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  大学课程讲义，提供了偏度和峰度的概念解释及视觉示例。
- [SciPy Reference Guide: Statistical Functions (scipy.stats)](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy Developers (2024)
  SciPy官方文档，详细介绍了Python中计算偏度和峰度等描述性统计量的函数及其数学定义。
