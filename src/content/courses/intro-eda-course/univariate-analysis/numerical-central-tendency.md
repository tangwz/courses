---
course: "intro-eda-course"
chapter: "univariate-analysis"
lesson: "numerical-central-tendency"
sourceId: 1152
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis/numerical-central-tendency"
title: "分析数值变量：集中趋势"
description: "计算和解释数值列的集中趋势度量（均值、中位数、众数）。"
order: 1
plots: ["plots/1152-0.json"]
sourceHash: "d289f3a24481764920df463c934edaba5f3bb118c1d75e5f2783dafcd27db7ce"
sourceCorrections: []
---

加载并熟悉数据集结构后，对数值列进行单变量分析的一个首要步骤是了解它们的“中心”。数值倾向于聚集在哪里？集中趋势度量提供一个单一值，概括数据的典型或中心点。我们将考察三种最常用的度量：均值、中位数和众数。

### 均值：算术平均数

最常见的集中趋势度量是**均值**，通常简称为平均值。它的计算方法是，将变量中所有值求和，然后除以值的数量。

对于一组 $n$ 个值 $x_1, x_2, ..., x_n$，样本均值 ($\bar{x}$) 的计算公式如下：
$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \dots + x_n}{n}$

在 Pandas 中，使用 `.mean()` 方法计算 Series（DataFrame 中的一列）的均值很简单。

```python
import pandas as pd
import numpy as np

# 示例数据，可能表示用户年龄
data = {'age': [25, 31, 45, 22, 58, 31, 28, 35]}
df = pd.DataFrame(data)

# 计算平均年龄
mean_age = df['age'].mean()

print(f"平均年龄: {mean_age}")
# 预期输出：平均年龄：34.375
```

均值能很好地表示大致对称分布数据的中心值。然而，它有一个主要缺点：**对异常值敏感**。单个极高或极低的值能将其均值大幅拉向其方向，可能无法准确代表“典型”值。

### 中位数：中间值

**中位数**是将数据集的高一半与低一半分开的值。要找到它，您首先对数据排序，然后选择中间值。

- 如果数据点数量 ($n$) 为奇数，中位数是位于 $(n+1)/2$ 位置的值。
- 如果 $n$ 为偶数，中位数是位于 $n/2$ 和 $(n/2) + 1$ 位置的两个中间值的平均值。

Pandas 提供了 `.median()` 方法：

```python
# 使用相同的 DataFrame 计算中位年龄
median_age = df['age'].median()

print(f"中位年龄: {median_age}")

# 排序后的年龄：[22, 25, 28, 31, 31, 35, 45, 58]
# 中间的两个值是 31 和 31。它们的平均值是 (31+31)/2 = 31.0
# 预期输出：中位年龄：31.0
```

中位数相对于均值的主要优点是它**对异常值的鲁棒性**。极端值对中位数几乎没有影响，因为它只取决于排序后中间位置的值。因此，对于偏斜分布或存在已知或可疑异常值的数据集，中位数通常是更好的集中趋势度量。如果均值和中位数明显不同，这通常表示存在偏斜或异常值。



![年龄分布（示例）](plots/1152-0.json)



> 一个简单的直方图呈现了年龄数据。虚线红线表示均值 (34.375)，而点状蓝线显示中位数 (31.0)。异常值 (58) 将均值拉高至中位数之上。

### 众数：最频繁的值

**众数**是数据集中出现频率最高的值。一个数据集可以有一个众数（单峰）、两个众数（双峰）或多个众数（多峰）。如果所有值出现的频率相同，数据集也可能没有众数。

尽管众数可以为数值数据计算，但它通常对分类数据或具有有限数量唯一值的离散数值数据更有指导意义。对于连续数值数据，除非数据已经分箱，否则众数可能没有太大意义。

Pandas 使用 `.mode()` 方法计算众数。请注意，`.mode()` 总是返回一个 Pandas Series，因为可能存在多个众数（所有具有最高频率的值）。

```python
# 计算众数年龄
# 注意：.mode() 返回一个 Series
mode_age = df['age'].mode()

print(f"众数年龄:\n{mode_age}")
# 预期输出：
# 众数年龄：
# 0    31
# dtype: int64

# 多个众数的例子
data_multi_mode = {'value': [10, 20, 30, 20, 40, 10, 50]}
df_multi = pd.DataFrame(data_multi_mode)
mode_multi = df_multi['value'].mode()

print(f"\n多个众数:\n{mode_multi}")
# 预期输出：
# 多个众数：
# 0    10
# 1    20
# dtype: int64
```

在我们的 `age` 例子中，值 `31` 出现了两次，比任何其他年龄都多，因此它是众数。如果另一个年龄也出现两次，`.mode()` 将同时返回它们。

### 选择正确的度量

- 当您的数据是数值型且对称分布（例如，遵循正态分布）且没有显著异常值时，使用**均值**。
- 当您的数据是数值型且偏斜，或者怀疑存在可能扭曲均值的异常值时，使用**中位数**。
- 主要将**众数**用于分类数据，或在您需要了解最常见值时用于数值数据。它不常作为连续数值数据的唯一集中趋势度量使用。

了解这三种度量提供了数据集中位置的基本概括。比较均值和中位数通常是检查数据偏斜的快速方法，有助于指导进一步的分析和可视化选择。

## 参考资料

- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce (2020)
  Publisher: O'Reilly Media
  一本实践指南，涵盖基础统计概念，包括集中趋势度量，侧重于使用 Python 和 R 在数据科学中的应用。
- [An Introduction to Statistical Learning: With Applications in R, Python and Other Popular Languages](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  一本涵盖统计学习的综合教科书，其早期章节为描述性统计和数据探索奠定坚实基础。
