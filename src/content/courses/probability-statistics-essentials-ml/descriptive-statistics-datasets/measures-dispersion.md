---
course: "probability-statistics-essentials-ml"
chapter: "descriptive-statistics-datasets"
lesson: "measures-dispersion"
sourceId: 1391
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-3-descriptive-statistics-datasets/measures-dispersion"
title: "离散度量：方差、标准差、极差"
description: "使用方差、标准差、极差和四分位距量化数据分散程度。"
order: 2
plots: ["plots/1391-0.json"]
sourceHash: "b7e3b2bc622815ceab0e67ec9ffc67a889a22e42602271bcfa40daa55b0e9106"
sourceCorrections: []
---

尽管均值、中位数和众数等度量告诉我们数据的典型值或中心位置，但它们无法提供完整的信息。考虑两组考试分数：{70, 75, 80, 85, 90} 和 {60, 70, 80, 90, 100}。两组的均值和中位数都是80，但第二组的分数显然更加分散。为了全面了解数据集，我们还需要量化 (quantization)这种分散程度，通常称为离散度或变异性。本节将介绍衡量数据点围绕中心聚集的紧密或松散程度的标准方法。

### 极差

最简单的离散度量是极差。它通过从数据集中最大值中减去最小值来计算。

`极差 = 最大值 - 最小值`

对于我们的示例分数集：

- 集合 1：极差 = 90 - 70 = 20
- 集合 2：极差 = 100 - 60 = 40

极差可以快速了解数据覆盖的总跨度。然而，它的主要缺点是对极端值或异常值的敏感性。一个非常高或非常低的值可能会显著夸大极差，从而可能对整体离散度产生误导性印象。因为它只使用两个数据点，所以它忽略了其余数据的分布情况。

在 Pandas 中，您可以轻松计算极差：

```python
import pandas as pd

data = {'scores_set1': [70, 75, 80, 85, 90],
        'scores_set2': [60, 70, 80, 90, 100]}
df = pd.DataFrame(data)

range_set1 = df['scores_set1'].max() - df['scores_set1'].min()
range_set2 = df['scores_set2'].max() - df['scores_set2'].min()

print(f"Range for Set 1: {range_set1}")
print(f"Range for Set 2: {range_set2}")
# 输出：
# Range for Set 1: 20
# Range for Set 2: 40
```

### 四分位距 (IQR)

一个更可靠且受异常值影响较小的离散度量是四分位距（IQR）。顾名思义，它涉及到四分位数，它们将排序后的数据分为四个相等的部分。

- Q1（第一四分位数）：低于该值的有25%的数据（即第25百分位数）。
- Q2（第二四分位数）：中位数，低于该值的有50%的数据（即第50百分位数）。
- Q3（第三四分位数）：低于该值的有75%的数据（即第75百分位数）。

四分位距是第三四分位数和第一四分位数之间的差值：

`四分位距 = Q3 - Q1`

它代表中间50%数据的跨度。因为它舍弃最低的25%和最高的25%的值，所以它不受分布尾部极端异常值的影响。这使得它对于偏斜数据集或存在异常值的数据来说是一个特别有用的度量。箱线图（我们将在可视化部分遇到）以图形方式表示四分位距。

使用 Pandas，我们使用 `quantile` 方法计算四分位距：

```python
q1_set1 = df['scores_set1'].quantile(0.25)
q3_set1 = df['scores_set1'].quantile(0.75)
iqr_set1 = q3_set1 - q1_set1

q1_set2 = df['scores_set2'].quantile(0.25)
q3_set2 = df['scores_set2'].quantile(0.75)
iqr_set2 = q3_set2 - q1_set2

print(f"IQR for Set 1: {iqr_set1}")
print(f"IQR for Set 2: {iqr_set2}")
# 输出：
# IQR for Set 1: 10.0
# IQR for Set 2: 20.0
```

请注意，四分位距也反映了集合2比集合1更分散，与极差类似，但它关注数据的核心部分。

### 方差

虽然极差和四分位距提供了有用的概括，但它们没有包含*所有*数据点的信息来描述相对于中心的离散度。方差正好做到了这一点。它衡量每个数据点与均值的平均平方差。

为什么要平方差？如果我们简单地对差异 $(x_i - \text{均值})$ 求平均值，正负差异会相互抵消，无论实际分散程度如何，结果总和通常接近零。平方差异使得所有贡献都为正，并突出了较大的偏差。

方差有两种常用公式，取决于您是处理整个总体还是从总体中抽取的样本：

1. **总体方差 ($\sigma^2$)**：如果您的数据集代表感兴趣的整个总体。
   $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$
   这里，$N$ 是总体中数据点的总数，$x_i$ 代表每个独立的数据点，$\mu$ 是总体均值。
2. **样本方差 ($s^2$)**：如果您的数据集是样本，并且您想估计抽取样本的更大总体的方差。
   $s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$
   这里，$n$ 是样本大小，$x_i$ 代表样本中的每个数据点，$\bar{x}$ 是样本均值。

重要的区别在于分母：总体方差用 $N$，样本方差用 $n-1$。除以 $n-1$（称为贝塞尔校正）使得 $s^2$ 成为真实总体方差 $\sigma^2$ 的*无偏估计量*。在实践中，特别是在数据分析和机器学习 (machine learning)中，您几乎总是处理样本，因此样本方差公式是通常使用的公式。

方差的主要缺点是其单位。如果您的数据代表分数（点），则方差的单位是平方点，这在原始数据尺度下无法直接解释。

Pandas 默认使用 `.var()` 方法计算*样本*方差：

```python
variance_set1 = df['scores_set1'].var() # 默认 ddof=1（样本方差）
variance_set2 = df['scores_set2'].var()

print(f"Sample Variance for Set 1: {variance_set1}")
print(f"Sample Variance for Set 2: {variance_set2}")
# 输出：
# Sample Variance for Set 1: 62.5
# Sample Variance for Set 2: 250.0

# 若要计算总体方差（如果数据代表整个总体）
pop_variance_set1 = df['scores_set1'].var(ddof=0)
print(f"Population Variance for Set 1: {pop_variance_set1}")
# 输出：
# Population Variance for Set 1: 50.0
```

### 标准差

为了解决方差平方单位的可解释性问题，我们使用标准差。它仅仅是方差的平方根。

- **总体标准差 ($\sigma$)**:
  $\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$
- **样本标准差 ($s$)**:
  $s = \sqrt{s^2} = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$

标准差代表数据点与均值的典型或平均距离，以数据的*原始单位*衡量。这使得它比方差更直观。较小的标准差表示数据点趋于接近均值，而较大的标准差表示数据点分布在一个更广的范围内。

对于我们的分数集：

- 集合 1（样本标准差）：$\sqrt{62.5} \approx 7.91$ 分
- 集合 2（样本标准差）：$\sqrt{250.0} \approx 15.81$ 分

这清楚地显示，平均而言，集合2中的分数与均值（80）的偏差比集合1中的分数更大。

与方差类似，标准差使用所有数据点，但对异常值也敏感，尽管平方根减轻了与方差相比的影响。它是一个基础统计量，通常与均值结合使用，尤其是在数据近似正态（高斯）分布时。

Pandas 默认使用 `.std()` 计算样本标准差：

```python
std_dev_set1 = df['scores_set1'].std() # 默认 ddof=1（样本标准差）
std_dev_set2 = df['scores_set2'].std()

print(f"Sample Standard Deviation for Set 1: {std_dev_set1:.2f}")
print(f"Sample Standard Deviation for Set 2: {std_dev_set2:.2f}")
# 输出：
# Sample Standard Deviation for Set 1: 7.91
# Sample Standard Deviation for Set 2: 15.81
```

以下是使用箱线图的可视化比较，它清楚地显示了两组之间离散度（由箱体长度/四分位距和须线长度/极差表示）的差异，即使它们共享相同的中位数（箱体内的线）。



![分数分布比较](plots/1391-0.json)



> 箱线图比较了两个分数集。集合2显示出更大的极差和四分位距（箱体长度），表明围绕共同中位数80的离散程度更大。

总之，离散度量量化 (quantization)数据的分散程度：

- **极差**：最快但对异常值最敏感。
- **四分位距**：不受异常值影响，关注数据的中间50%。
- **方差**：与均值的平均平方偏差，使用所有数据，单位为平方。
- **标准差**：方差的平方根，单位可解释，与均值一起是最常用的离散度量。

选择合适的度量取决于数据特征（特别是是否存在异常值）和您的分析目标。了解集中趋势和离散度对于有效地总结和解释数据集是基础的。

## 参考资料

- [OpenIntro Statistics](https://www.openintro.org/book/os/) — David M Diez, Christopher D Barr, Mine Çetinkaya-Rundel (2019)
  Publisher: OpenIntro
  一本教科书，提供描述性统计的基本概念，包括集中趋势和离散度测量、总体与样本以及箱线图。
- [Practical Statistics for Data Scientists: 50 Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce (2020)
  Publisher: O'Reilly Media
  本书将方差和标准差等统计概念与数据科学和机器学习应用结合，解释它们的实际用途。
- [pandas.DataFrame.std](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.std.html) — pandas development team (2024)
  Pandas 计算标准差的官方文档，详细说明了用于样本和总体计算的 `ddof` 参数。
- [MIT OpenCourseWare 18.05 Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: Massachusetts Institute of Technology
  提供概率与统计的全面学术基础，包括描述性统计主题。
