---
course: "intro-eda-course"
chapter: "univariate-analysis"
lesson: "numerical-dispersion"
sourceId: 1154
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis/numerical-dispersion"
title: "数值变量分析：离散程度"
description: "计算并理解数值数据的离散程度度量（方差、标准差、极差、四分位距）。"
order: 2
plots: []
sourceHash: "f9c4d69973ffdb81a2203de44d07dbf2fa712301822457d698ae73b9f5bcb43e"
sourceCorrections: []
---

尽管均值、中位数和众数等集中趋势度量能让我们了解数值数据集中“典型”值的情况，但它们无法完全反映全貌。两个数据集可能拥有相同的均值，但在数据点分散程度方面却显得大不相同。这种分散性或变动性，正是离散程度度量所量化 (quantization)的内容。理解离散程度对于掌握变量的分布情况以及发现不一致数据或极端值等可能存在的问题非常重要。

### 极差

最简单的离散程度度量是**极差**，它就是数据集中最大值和最小值之间的差值。

`极差 = 最大值 - 最小值`

尽管易于计算和理解，但极差对异常值非常敏感。单个极端高值或低值可以大幅改变极差，可能对整体数据分散情况给出误导性的印象。

集中趋势的测量，例如平均值、中位数和众数，可以帮助我们了解数值数据集中“典型”值，但它们并不能提供数据的全貌。两个数据集可能具有相同的平均值，但在数据点分布的广度上可能存在显著差异。这种广度或变异性，正是离散度测量所量化 (quantization)的。理解离散度对于掌握变量的分布以及识别潜在问题（如不一致数据或极端值）是基础。极差是一种简单的离散度测量方式，其计算方法是找出数据中的最大值和最小值并将其相减。例如，在 Pandas 中可按以下方式计算极差：

### 方差

**方差**通过考虑每个数据点与均值偏离的程度来提供一个衡量分散性的指标。它计算每个数据点与均值之间差值的平方的平均值。对差值进行平方可以确保高于和低于均值的偏差不会相互抵消，并赋予较大的偏差更大的权重 (weight)。

对于样本数据集（这通常是您在数据分析中处理的数据），方差 ($s^2$) 的公式为：
$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}$
其中：

- $x_i$ 代表每个独立的数据点。
- $\bar{x}$ 是样本均值。
- $n$ 是样本中的数据点数量。
- 我们除以 $n-1$（贝塞尔校正），以便从样本中获得总体方差的无偏估计。

方差越大表示数据点平均而言离均值越远，意味着更大的分散性。方差越小则表示数据点更紧密地聚集在均值周围。方差的主要缺点是其单位是原始数据单位的平方（例如，美元平方、米平方），这使得解释起来不太直观。

Pandas 提供了一个方便的 `var()` 方法：

```python
# 计算样本方差
sample_variance = df['numerical_col'].var()
print(f"样本方差: {sample_variance}")
```

### 标准差

**标准差**（$s$）就是方差的平方根。它通常比方差更受欢迎，因为它将分散性度量恢复到数据的原始单位，使得解释起来更加容易。

$s = \sqrt{s^2} = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}}$

标准差代表了数据点与均值的典型或平均距离。

- 较小的标准差表示数据点趋于接近均值。
- 较大的标准差表示数据点分布在更宽的值范围内。

与方差一样，标准差在其计算中使用了均值，因此对异常值很敏感。

在 Pandas 中，使用 `std()` 方法：

```python
# 计算样本标准差
sample_std_dev = df['numerical_col'].std()
print(f"样本标准差: {sample_std_dev}")
```

您通常可以从均值和标准差的角度来思考分布。例如，对于大致服从正态分布的数据，大约 68% 的数据落在均值的一个标准差范围内（$\bar{x} \pm s$），大约 95% 落在两个标准差范围内（$\bar{x} \pm 2s$），大约 99.7% 落在三个标准差范围内（$\bar{x} \pm 3s$）。这被称为经验法则。

### 四分位距 (IQR)

**四分位距 (IQR)** 是另一个重要的离散程度度量，因其不受异常值影响的特性而备受重视。它代表了数据中间 50% 的范围。

要计算 IQR，您首先需要找到四分位数：

1. **第一四分位数 (Q1):** 25% 的数据低于此值（也称为第 25 百分位数）。
2. **第三四分位数 (Q3):** 75% 的数据低于此值（也称为第 75 百分位数）。

IQR 的计算公式为：
$\text{四分位距} = Q3 - Q1$

因为 IQR 专注于分布的中心部分，并忽略两端的极端值，所以它不受异常值影响。这使得它成为偏态分布或极端值可能扭曲方差或标准差的数据集的一种可靠的分散性度量。IQR 也在箱线图中通过箱体直观地表示。

您可以使用 Pandas 中的 `quantile()` 方法计算四分位数和 IQR：

```python
# 计算 Q1（第 25 百分位数）和 Q3（第 75 百分位数）
Q1 = df['numerical_col'].quantile(0.25)
Q3 = df['numerical_col'].quantile(0.75)

# 计算 IQR
iqr = Q3 - Q1

print(f"Q1（第 25 百分位数）: {Q1}")
print(f"Q3（第 75 百分位数）: {Q3}")
print(f"IQR: {iqr}")
```

Pandas 中的 `describe()` 方法可以方便地计算这些统计量中的几个，包括最小值、最大值、均值、标准差和四分位数（25%、50% 即中位数和 75%），从而为您提供集中趋势和离散程度的快速汇总：

```python
# 获取描述性统计信息的汇总
print(df['numerical_col'].describe())
```

理解这些离散程度度量，包括极差、方差、标准差和 IQR，可以为您的数值数据的变异性和一致性提供重要的见解，完善了集中趋势度量所呈现的全貌。

## 参考资料

- [Practical Statistics for Data Scientists: 50 Essential Concepts for Data Science](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media
  一本面向数据科学的实用统计学指南，包含各种离散度量及其在理解数据变异性方面的用途。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用 pandas 进行 Python 数据操作和分析的权威资源，包含计算和解读描述性统计量的方法。
