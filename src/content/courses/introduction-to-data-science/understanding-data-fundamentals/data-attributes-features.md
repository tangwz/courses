---
course: "introduction-to-data-science"
chapter: "understanding-data-fundamentals"
lesson: "data-attributes-features"
sourceId: 2101
sourceUrl: "https://apxml.com/zh/courses/introduction-to-data-science/chapter-2-understanding-data-fundamentals/data-attributes-features"
title: "理解数据属性和特征"
description: "了解属性（列）和特征在数据集中的含义。"
order: 5
plots: []
sourceHash: "df9f8a2892aa61a5595b6e18070831e63a803d44e95f304d98871344e753963f"
sourceCorrections: []
---

当我们讨论数据集，特别是以表格形式呈现的结构化数据时，我们经常使用特定术语来指代其组成部分。您会遇到的两个非常常用的术语是**属性**和**特征**。

在数据科学的许多情境中，这些术语可以互换使用。它们都指所观察现象的**可测量性质或特点**。可以把它们看作您为数据集中每个项目收集的不同类型的信息。

### 作为列的属性和特征

如果您想象您的数据以表格形式组织，比如电子表格：

- 每**行**通常代表一个观测值、记录或实例（例如，一个客户、一笔交易、一个传感器读数）。
- 每**列**代表一个**属性**或**特征**（例如，客户年龄、交易金额、传感器温度）。

因此，属性或特征描述了为数据集中每个观测值记录的特定信息片段。

我们来看一个简单的例子。想象一个关于杂货店里不同种类水果的数据集：

| 水果名称 | 颜色 | 重量（克） | 价格（美元） |
| --- | --- | --- | --- |
| 苹果 | 红色 | 150 | 0.50 |
| 香蕉 | 黄色 | 120 | 0.25 |
| 橙子 | 橙色 | 180 | 0.60 |
| 苹果 | 绿色 | 165 | 0.55 |
| 葡萄 | 紫色 | 5 | 0.05 |

在此表中：

- 每行是一个代表特定水果项的观测值。
- '水果名称'、'颜色'、'重量（克）'和'价格（美元）'列是这些水果的**属性**或**特征**。

### 与数据类型的关联

如本章前面所述，每个属性或特征都将具有特定的数据类型。看我们的水果示例：

- '水果名称'和'颜色'包含描述性信息，使它们成为**定性**属性。
- '重量（克）'和'价格（美元）'包含数值测量，使它们成为**定量**属性。

理解每个属性的类型是基础的，因为它决定了您可以应用的分析和可视化方法。您可以计算平均重量（定量），但不能计算平均颜色（定性）。相反，您可能需要统计不同颜色的出现次数。

### 为什么这种区分重要

识别并理解数据集中的属性或特征是任何数据分析过程中的一个基础步骤。这些是您将：

- 检查以了解它们的个体特点（例如它们的范围、平均值或最常见值）。
- 比较以寻找它们之间的关联或模式（例如，水果的重量是否与其价格相关？）。
- 可能用于构建预测模型（例如，根据水果类型和重量预测价格）。

### 术语说明

虽然'属性'和'特征'经常互换使用，但您有时可能会看到'特征'在机器学习 (machine learning)的背景下使用得更具体。在那个方面，'特征'通常指经过选择或设计，专门用作预测模型输入的属性。然而，为了基础的理解，将它们视为数据表中列的可互换术语是完全可以接受的。您还会听到其他相关术语，如'变量'（统计学中常见）、'字段'（数据库中常见）或简单的'列'。

识别属性或特征是关于确定您为数据集中每个记录收集的不同信息片段。它们构成了所有进一步分析和解释的基础。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  定义变量、特征和数据类型的经典教材，对理解结构化数据和统计建模至关重要。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  实用指南，介绍数据结构和表格数据中属性/特征作为列的基本概念，常伴有数据类型讨论。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  从机器学习角度阐释特征，说明其如何作为模型输入，并澄清属性与特征之间的关系。
- [CS229 Lecture Notes: Supervised Learning](http://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (2023)
  Journal: Stanford University; Publisher: Stanford University; Pages: 1-2
  在机器学习背景下介绍特征概念，提供学术定义和框架。
