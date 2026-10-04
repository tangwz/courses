---
course: "intro-eda-course"
chapter: "univariate-analysis"
lesson: "categorical-frequency-counts"
sourceId: 1159
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis/categorical-frequency-counts"
title: "分析分类变量：频数统计"
description: "使用Pandas计算分类变量的频数分布和计数。"
order: 5
plots: []
sourceHash: "0d7fb64977dee0c112b48af6958e5be660479eb0cef31e20284fa045ecac98ef"
sourceCorrections: []
---

虽然数值变量适合通过均值、中位数和标准差等统计量进行分析，但分类变量则需要不同的方法。由于计算“平均”类别没有意义，我们转而关注变量中不同类别之间观测值的分布。汇总这种分布的主要方式是计算频数。

频数告诉我们数据集中每个唯一类别出现的次数。这种简单的计数对于理解分类数据非常重要。它有助于发现：

- 最常见和最不常见的类别。
- 数据在各类别的整体分布情况。
- 潜在的数据录入错误或意外的类别值。
- 某些类别是否在数据集中占主导地位。

### 使用Pandas计算频数

总结分类变量分布的主要方法是计算频数。Pandas提供了一个专门用于此目的的便捷方法：`value_counts()`。当应用于Pandas Series（例如DataFrame的单列）时，它会返回一个新的Series，该Series的索引是原始Series中的唯一值，对应的值是这些唯一值的频数（计数）。默认情况下，结果按频数降序排列，便于我们先看到最常见的类别。

假设我们有一个名为`df`的DataFrame，其中包含一个名为`customer_segment`的列，该列存储分类数据。

```python
import pandas as pd
import numpy as np

# 示例DataFrame
data = {'customer_segment': ['Standard', 'Premium', 'Standard', 'VIP', 'Premium', 'Standard', 'Standard', np.nan, 'Premium']}
df = pd.DataFrame(data)

print("示例DataFrame:")
print(df)

# 计算'customer_segment'的频数
segment_counts = df['customer_segment'].value_counts()

print("\ncustomer_segment的频数统计:")
print(segment_counts)
```

运行这段代码将产生类似于以下的输出：

```
Sample DataFrame:
  customer_segment
0         Standard
1          Premium
2         Standard
3              VIP
4          Premium
5         Standard
6         Standard
7              NaN
8          Premium

Frequency Counts for customer_segment:
Standard    4
Premium     3
VIP         1
Name: customer_segment, dtype: int64
```

注意`value_counts()`如何自动处理不同类别（'Standard'、'Premium'、'VIP'），统计它们的出现次数，并进行排序。它也默认排除缺失值（`NaN`）。

### 相对频数（比例）

尽管绝对计数很有用，但有时了解每个类别相对于非缺失观测值总数的比例或百分比会提供更多信息。这被称为相对频数。它有助于比较不同大小数据集间的分布情况。

要使用`value_counts()`获取相对频数，可以使用`normalize=True`参数 (parameter)。

```python
# 计算相对频数（比例）
segment_proportions = df['customer_segment'].value_counts(normalize=True)

print("\ncustomer_segment的相对频数（比例）:")
print(segment_proportions)

# 以百分比形式显示
segment_percentages = df['customer_segment'].value_counts(normalize=True) * 100
print("\ncustomer_segment的相对频数（百分比）:")
print(segment_percentages.round(2).astype(str) + '%') # 更好的格式化
```

输出会是这样：

```
Relative Frequencies (Proportions) for customer_segment:
Standard    0.500
Premium     0.375
VIP         0.125
Name: customer_segment, dtype: float64

Relative Frequencies (Percentages) for customer_segment:
Standard    50.0%
Premium     37.5%
VIP         12.5%
Name: customer_segment, dtype: object
```

这表示在非缺失条目中，50%是'Standard'，37.5%是'Premium'，12.5%是'VIP'。

### 在计数中包含缺失值

如前所述，`value_counts()`默认忽略缺失值（`NaN`）。在EDA中，了解分类变量存在多少缺失值通常很重要。您可以通过将`dropna`参数 (parameter)设置为`False`，将`NaN`值包含在计数中。

```python
# 计算包含NaN值的频数
segment_counts_incl_na = df['customer_segment'].value_counts(dropna=False)

print("\ncustomer_segment的频数统计（包含NaN）:")
print(segment_counts_incl_na)

# 计算包含NaN值的比例
segment_proportions_incl_na = df['customer_segment'].value_counts(normalize=True, dropna=False)

print("\ncustomer_segment的相对频数（包含NaN）:")
print(segment_proportions_incl_na)
```

输出现在包含缺失值的计数和比例：

```
Frequency Counts (including NaN) for customer_segment:
Standard    4
Premium     3
VIP         1
NaN         1
Name: customer_segment, dtype: int64

Relative Frequencies (including NaN) for customer_segment:
Standard    0.444444
Premium     0.333333
VIP         0.111111
NaN         0.111111
Name: customer_segment, dtype: float64
```

分析频数和比例提供了对每个分类变量分布的坚实定量认识。这些数值汇总构成了下一个合理步骤的依据：使用条形图等工具可视化这些分布，我们将在稍后介绍。

## 参考资料

- [pandas.Series.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) — Wes McKinney and the pandas development team (2024)
  Pandas `value_counts()` 方法的官方文档，解释了其在 Series 频率分析中的用法和参数。
- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, John H. Harland Dean, Daniela Witten, Trevor Hastie, Rob Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer
  从统计学角度深入阐述描述性分析，包括汇总和理解分类变量的方法。
- [OpenIntro Statistics](https://www.openintro.org/book/os/) — David Diez, Mine Çetinkaya-Bartoni, Christopher Barr (2022)
  Publisher: OpenIntro
  一本易于理解的开源教科书，涵盖了基本的统计概念，包括通过频率计数对分类数据进行汇总。
