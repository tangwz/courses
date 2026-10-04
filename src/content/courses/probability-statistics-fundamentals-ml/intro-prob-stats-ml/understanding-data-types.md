---
course: "probability-statistics-fundamentals-ml"
chapter: "intro-prob-stats-ml"
lesson: "understanding-data-types"
sourceId: 2399
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-1-intro-prob-stats-ml/understanding-data-types"
title: "了解数据类型"
description: "学习机器学习中遇到的各类数据，比如分类数据和数值数据。"
order: 3
plots: []
sourceHash: "bbd3384833710e4824aa862286ade609fc851cb9e6e9a2c6636303815bbd0c76"
sourceCorrections: []
---

在我们分析数据或构建机器学习 (machine learning)模型之前，首先要了解所处理数据的*类型*。可以将数据类型想象成信息所说的不同语言。就像你需要正确的翻译来理解一门语言一样，你也需要针对每种数据类型选择合适的统计方法和可视化工具。将为数值测量设计的方法应用于分类标签，通常没有意义，并可能导致错误的结论。

在统计学和机器学习中，数据通常分为两大类：**分类数据**和**数值数据**。接下来我们对它们进行具体说明。

## 分类数据

分类数据代表用于对项目进行分组的特征或标签。它们是定性的描述。我们可以将分类数据进一步分为两种类型：

### 名义数据

名义数据由没有固有顺序或等级的类别组成。你可以对它们进行计数，但不能按逻辑从低到高排序。

- **示例：**
  - 颜色（红、蓝、绿）
  - 水果种类（苹果、香蕉、橙子）
  - 原产国（美国、加拿大、墨西哥）
  - 性别（男、女、其他）

对于名义数据，计算平均值之类的运算是没有意义的。“红色”和“蓝色”的平均值是什么？这无法计算。不过，我们可以计算每个类别的出现次数（例如，有多少红色物品，有多少蓝色物品）。

### 有序数据

有序数据代表那些*确实*具有有意义的顺序或等级的类别，但类别之间的间隔可能不相等或不可量化 (quantization)。

- **示例：**
  - 学历水平（高中、学士、硕士、博士）——有明确的顺序。
  - 客户满意度评分（很不满意、不满意、中立、满意、非常满意）——我们知道“满意”比“中立”好，但“满意”和“非常满意”之间的差异是否与“中立”和“满意”之间的差异相同？不一定。
  - T恤尺码（小号、中号、大号、特大号）——按大小排序。

尽管存在顺序，但像加法或平均值这样的算术运算通常仍然不适合有序数据，因为等级之间的差异没有精确定义。不过，我们可以确定中位数（排序后的中间值）或找到百分位数。

## 数值数据

数值数据代表可以测量或计数的数量。它们是定量的值。数值数据也有两种形式：

### 离散数据

离散数据由可计数且通常取整数值的数据组成。可能的值之间存在明显的间隔。你可以对这些项目进行计数。

- **示例：**
  - 客户投诉数量（0、1、2、3...）——你不能有2.5个投诉。
  - 停车场内的汽车数量（0、1、2...）
  - 抛掷硬币10次时正面的数量（0、1...10）

你可以对离散数据执行算术运算（加法、减法、平均）。

### 连续数据

连续数据可以在给定范围内取任何值。这些值通常是测量而非计数得到的，理论上可以无限精确，仅受限于测量仪器。

- **示例：**
  - 人的身高（例如，1.75米、1.751米、1.7512米...）
  - 物体的重量
  - 温度（例如，25.5 °C、25.54 °C）
  - 房屋价格（$350,000、$350,500.75）

算术运算对于连续数据完全有效。这是最常与均值、方差等统计测量相关联的数据类型。

这是一张图表，简要说明了数据类型的层级：

> 一张图表，说明了常见数据的主要类别和子类型。

## 数据类型为何重要

理解这些区别非常重要，原因如下：

1. **合适的分析：** 不同的统计方法适用于不同的数据类型。计算T恤尺码的平均值没有意义，但找到最常出现的尺码（众数）则有意义。计算平均身高是标准做法。
2. **正确的可视化：** 数据类型决定了最佳的可视化方式。柱状图非常适合表示分类频率，而直方图和散点图则适合数值数据。
3. **机器学习 (machine learning)算法：** 许多机器学习算法是为特定类型的输入数据设计的。有些直接处理分类数据，而另一些则需要数值输入。通常，分类数据在输入算法之前需要转换为（编码为）数值格式。了解原始数据类型有助于选择正确的编码方法。

## Python (Pandas) 中的数据类型

当你使用像 Pandas 这样的库（我们将在本章稍后简要介绍）加载数据时，它通常会尝试自动推断数据类型。不过，验证这些推断的类型始终是好的做法。

Pandas 通常使用这些常见的 `dtype` 对象：

- `object`：常用于文本或混合类型列。通常代表分类数据（包括名义数据和有序数据）。
- `int64`：代表整数值。通常对应于离散数值数据。
- `float64`：代表浮点（小数）数字。通常对应于连续数值数据。
- `bool`：代表布尔（真/假）值。这是一种分类数据类型。
- `category`：一种专为分类数据优化的特定 Pandas 类型（可以代表名义数据或有序数据）。
- `datetime64`：代表日期和时间值。

能够识别你所拥有的数据类型，是进行有意义的分析和构建有效机器学习 (machine learning)模型的第一步。在后续章节中，我们将看到我们使用的方法如何根据处理的是分类数据还是数值数据而变化。

## 参考资料

- [OpenIntro Statistics](https://www.openintro.org/stat/textbook.php?stat_book=os) — David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel (2019)
  Publisher: OpenIntro; Pages: Chapter 1, Section 1.2: Data Basics
  一本全面的统计学入门教材，全面涵盖了不同的数据类型及其基本特性。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 2: End-to-End Machine Learning Project (section on Data Preprocessing)
  一本实用指南，讨论了数据类型及其转换作为构建机器学习模型的初始步骤。
- [Working with data types (dtypes)](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes) — The Pandas development team (2023)
  Journal: Pandas Documentation
  Pandas 官方文档，解释了该库如何处理和表示不同的数据类型，包括类型推断和显式转换。
