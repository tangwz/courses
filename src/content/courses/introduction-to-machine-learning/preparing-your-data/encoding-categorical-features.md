---
course: "introduction-to-machine-learning"
chapter: "preparing-your-data"
lesson: "encoding-categorical-features"
sourceId: 1527
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-6-preparing-your-data/encoding-categorical-features"
title: "编码分类特征"
description: "学习如何使用独热编码将分类（非数值）特征转换为数值格式。"
order: 4
plots: []
sourceHash: "a20c987e6ea06b3b5819e61d8776dd30ecb60446e5f17d2b25a41eb62d95f826"
sourceCorrections: []
---

机器学习 (machine learning)算法，特别是数学类算法，例如线性回归，主要处理数值数据。它们执行加法、乘法和距离计算，而这些操作对“红色”、“蓝色”、“纽约”或“巴黎”之类的文本标签没有意义。那么，当数据集包含描述性的非数值类别时，该怎么办呢？需要一种方法将这些分类特征转换为算法能够理解的数值格式，同时不丢失原始信息或引入误导性的关联。这个过程称为编码。

### 什么是分类特征？

分类特征表示可以分为不同组或类别的数据。这些类别可能没有自然的数值顺序。举例如下：

- **颜色:** '红色', '绿色', '蓝色'
- **城市:** '伦敦', '东京', '纽约'
- **产品类型:** '电子产品', '服装', '食品杂货'
- **客户等级:** '青铜', '白银', '黄金' (此项*可能*有顺序，称为序数类别，但很多只是标签，称为名义类别)

直接将这些文本标签输入大多数算法是行不通的。例如，我们无法计算“红色”和“蓝色”的平均值。我们需要一个数值表示。

### 介绍独热编码

一种常见且有效的方法，用于处理名义分类特征（即类别之间没有固有顺序的情况），是**独热编码**。

这个思路很简单：对于一个有 $k$ 个独特类别的分类特征，我们创建 $k$ 个新的二元特征（列）。每个新列对应一个原始类别。对于给定的数据点（行），与其原始类别对应的列赋值为 1，而与该原始特征相关的其他所有新列则赋值为 0。

让我们用一个例子来说明。假设我们有一个名为“颜色”的特征，它有三个可能的值：“红色”、“绿色”和“蓝色”。

**原始数据:**

| ID | 颜色 | 其他特征 |
| --- | --- | --- |
| 1 | 红色 | 10 |
| 2 | 绿色 | 15 |
| 3 | 蓝色 | 12 |
| 4 | 红色 | 18 |

对“颜色”特征应用独热编码后，它会变成这样：

**独热编码后:**

| ID | Color\_Red | Color\_Green | Color\_Blue | 其他特征 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 10 |
| 2 | 0 | 1 | 0 | 15 |
| 3 | 0 | 0 | 1 | 12 |
| 4 | 1 | 0 | 0 | 18 |

注意，“颜色”这一列被三个新列取代：“Color\_Red”、“Color\_Green”和“Color\_Blue”。每行现在都在这些列中的一个精确位置有“1”，表示原始颜色，其他位置为“0”。这种数值表示可以轻松地被机器学习 (machine learning)算法使用。

> 独热编码将单个分类特征转换为多个二元特征的示意图。

### 为何使用独热编码？

- **无隐含顺序：** 它将类别转换为算法理解的格式，而不会暗示类别之间的任何顺序或关联（例如，它不暗示“蓝色”比“红色”拥有“更大”的值）。
- **兼容性：** 大多数标准机器学习 (machine learning)算法可以直接处理这些二元特征。

### 潜在考量

独热编码的一个潜在问题是，当一个分类特征拥有非常大量的独特值时（例如，邮政编码、用户ID）。这可能导致数据集中列（特征）的数量显著增加，有时被称为“维度灾难”。虽然我们在此不会详细说明解决方案，但请注意，对于此类情况，在更进阶的情境下，可能会考虑其他编码方法或特征工程技术。

### 那如果只分配数字（0、1、2...）呢？

您可能会想：“为什么不直接为每个类别分配一个数字，比如红色=0、绿色=1、蓝色=2？”这称为**标签编码**。尽管更简单，但通常**不推荐用于名义分类特征**。赋值0、1、2暗示了顺序和大小（$2 > 1 > 0$）。算法可能会将这些数字解释为具有数学意义，而这在原始数据中并不存在（例如，假设蓝色某种程度上是绿色的“两倍”）。这可能会对模型性能产生负面影响。标签编码通常保留给*序数*特征，即类别*确实*具有有意义的顺序（如“小”、“中”、“大”）的情况。对于没有自然顺序的类别，独热编码通常是更安全的方法。

### 接下来的步骤

编码分类特征是为机器学习 (machine learning)准备数据的标准步骤。独热编码提供了一种将非数值标签转换为适合算法的格式的方法，且不强加人为的顺序。Python 中的 Scikit-learn 等库提供了方便的函数来实现这项技术，您将在后续的实际练习中遇到。拥有数值编码的特征使我们离将数据输入学习算法又近了一步。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了数据预处理的实践指导，包括使用Python Scikit-learn进行分类特征编码的详细解释和示例。
- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) — scikit-learn developers (2023)
  Scikit-learn OneHotEncoder的官方文档，详细说明其参数、方法和独热编码分类特征的示例。
- [Introduction to Machine Learning with Python: A Guide for Data Scientists](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/) — Andreas C. Müller and Sarah Guido (2016)
  Publisher: O'Reilly Media
  讨论了分类变量表示的重要性，以及在Scikit-learn生态系统中OneHotEncoder和LabelEncoder的实际实现。
