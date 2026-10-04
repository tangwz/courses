---
course: "probability-statistics-fundamentals-ml"
chapter: "descriptive-statistics"
lesson: "measures-dispersion-range"
sourceId: 2412
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/measures-dispersion-range"
title: "衡量变异性：极差"
description: "计算数据集的极差作为一种基本的变异性度量。"
order: 2
plots: []
sourceHash: "17029e7241c3aafaf13708d9251d69dfe2d2fe7a9879f6daedaaf90fa59d4c5d"
sourceCorrections: []
---

在使用均值或中位数等方法找到数据的中心之后，接下来需要考虑的问题是：数据点分散程度如何？它们是紧密聚集在中心附近，还是广泛分散？*离散度*（也称作*变异性*）度量可以帮助我们回答这个问题。最基本的离散度度量是**极差**。

### 什么是极差？

极差是一个数据集中最高值（最大值）与最低值（最小值）之间的差值。它能让你迅速了解数据的总跨度。

### 计算方法

计算极差的方法很直接：

1. 确定数据集中最大值。
2. 确定数据集中最小值。
3. 从最大值中减去最小值。

公式是：


$$
\text{极差} = \text{最大值} - \text{最小值}
$$


### 例子

假设我们有一周内记录的每日最高气温（摄氏度）：`[21, 25, 19, 28, 22, 26, 20]`。

1. **确定最大值：** 最高气温是 28°C。
2. **确定最小值：** 最低气温是 19°C。
3. **计算极差：**
   
   $$
   \text{极差} = 28 - 19 = 9
   $$
   
   因此，该周的气温极差是 9°C。这表示该周的气温从最低点到最高点变化了 9 度。

### 解释

更大的极差表示数据有更大的分散度或变异性，即数值分布在更宽的区间内。更小的极差表明数据点更紧密，表示变异性较小。

### 极差的局限性

尽管极差计算和理解起来都很方便，但它有一个重要的局限性：**它只考虑数据集中两个最极端的数值。** 这使得它对*异常值*（异常高或低的数值）非常敏感。

再次看我们的气温数据。如果有一天异常炎热，得到以下气温数据：`[21, 25, 19, 45, 22, 26, 20]`。

1. **最大值：** 45°C
2. **最小值：** 19°C
3. **极差：**
   
   $$
   \text{极差} = 45 - 19 = 26
   $$
   

现在极差是 26°C，远大于之前的 9°C。这个单一的异常值（45°C）极大地增大了极差，可能对日常气温的典型变动给出误导性的印象。数据的主体部分可能仍然聚集在一起，但极差未能反映这一点。

### 使用 Python 计算极差

在 Python 中，可以使用 NumPy 或 Pandas 等库方便地计算极差。

```python
import numpy as np

temperatures = np.array([21, 25, 19, 28, 22, 26, 20])

# 使用 NumPy 函数计算极差
data_range = np.ptp(temperatures) # ptp 代表“峰值到峰值”

# 另一种方法，手动计算
max_temp = np.max(temperatures)
min_temp = np.min(temperatures)
manual_range = max_temp - min_temp

print(f"气温数据: {temperatures}")
print(f"最高气温: {max_temp}")
print(f"最低气温: {min_temp}")
print(f"极差（使用 np.ptp）：{data_range}")
print(f"极差（手动计算）：{manual_range}")

# 带异常值的例子
temperatures_with_outlier = np.array([21, 25, 19, 45, 22, 26, 20])
range_with_outlier = np.ptp(temperatures_with_outlier)
print(f"\n带异常值气温数据: {temperatures_with_outlier}")
print(f"带异常值极差: {range_with_outlier}")
```

由于它对异常值很敏感，极差通常只作为初步检查的一个快速方法，而不是唯一的变异性度量。在接下来的章节中，我们将了解更多度量方法，如方差和标准差，它们会考虑所有数据点，受极端值的影响较小。

## 参考资料

- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media; Pages: Chapter 1
  将统计概念直接与数据科学应用联系起来，为机器学习从业者提供了描述性统计的实用解释。
- [numpy.ptp](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html) — NumPy Developers (2023)
  Publisher: NumPy Project/Community
  NumPy 官方文档中关于 'ptp' (峰谷差) 函数的说明，该函数可有效计算数组的极差，与 Python 实现相关。
- [Introduction to Probability and Statistics (18.05)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2018)
  Publisher: MIT OpenCourseWare
  麻省理工学院开放课程资料，提供了大学水平的概率论与统计学入门，包含描述性统计的基础概念。
