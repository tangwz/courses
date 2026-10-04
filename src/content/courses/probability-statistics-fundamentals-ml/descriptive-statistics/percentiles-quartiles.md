---
course: "probability-statistics-fundamentals-ml"
chapter: "descriptive-statistics"
lesson: "percentiles-quartiles"
sourceId: 2413
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-2-descriptive-statistics/percentiles-quartiles"
title: "理解百分位数与四分位数"
description: "了解百分位数和四分位数如何划分数据以及它们如何帮助理解数据分布。"
order: 4
plots: ["plots/2413-0.json"]
sourceHash: "f44ef61d96c9db488071134d6d22bc558ebeccd3fc4d65e87420fc7745027581"
sourceCorrections: []
---

平均值和标准差等统计量能说明数据的中心和整体离散程度，但它们不能告诉我们某个特定数据点相对于其他数据点的位置。有时，我们想知道哪个值代表数据的底部25%，或者哪个值处于顶部10%。这时，百分位数和四分位数就成了描述数据分布中值位置的有用方法。

### 百分位数：数据点排序

假设你有一份按分数从低到高排列的班级考试成绩列表。百分位数是一种度量，它说明了总数据点中有多少百分比的值低于特定值。

例如，如果你的分数在第80百分位数，这意味着班级中80%的分数低于你的分数，而20%的分数高于你的分数。第50百分位数是50%的数据低于它的值——你可能已经知道，这便是我们之前讨论过的衡量集中趋势的*中位数*。

百分位数能为单个数据点提供背景信息，并帮助我们理解分布的形态。第10百分位数表示数据的下限，而第90百分位数表示数据的上限。

### 四分位数：将数据分为四等份

四分位数是特定的、常用的百分位数，它们将数据集分为四个相等的部分。可以将它们视为将已排序数据在25%、50%和75%处进行切割。

- **第一四分位数 (Q1)：** 这是第25百分位数。25%的数据点小于或等于Q1。
- **第二四分位数 (Q2)：** 这是第50百分位数。50%的数据点小于或等于Q2。如前所述，这与数据集的**中位数**完全相同。
- **第三四分位数 (Q3)：** 这是第75百分位数。75%的数据点小于或等于Q3。

四分位数是统计学中用于描述数据分布位置的度量，包括Q1、Q2（中位数）和Q3。它们共同对数据分布进行了简明扼要的总结，说明了大部分值的所在区间。

### 四分位距 (IQR)：衡量中间离散程度

利用四分位数，我们可以计算另一种离散程度的度量，称为**四分位距 (IQR)**。它就是第三四分位数 (Q3) 与第一四分位数 (Q1) 之间的差值：


$$
IQR = Q3 - Q1
$$


IQR表示数据中间50%的范围。它为什么有用？与总范围（最大值 - 最小值）不同，IQR不受极端异常值的影响。如果你有一个非常高或非常低的值，它不会显著改变Q1或Q3，这使得IQR成为衡量偏斜数据集或含有异常值的数据离散程度的更可靠方法。

### 使用Python计算百分位数和四分位数

手动排序数据和查找百分位数是繁琐的。NumPy和Pandas等Python库使得这项工作变得简单。

**使用NumPy：**
`numpy.percentile()`函数是常用的。你提供数据以及你想要的百分位数（值在0到100之间）。

```python
import numpy as np

# 样本数据（例如，响应时间，单位毫秒）
response_times = np.array([120, 150, 155, 160, 175, 180, 190, 210, 230, 250, 300, 500])

# 计算第25百分位数 (Q1)
q1 = np.percentile(response_times, 25)

# 计算第50百分位数（中位数，Q2）
median = np.percentile(response_times, 50) # 或 np.median(response_times)

# 计算第75百分位数 (Q3)
q3 = np.percentile(response_times, 75)

# 计算IQR
iqr = q3 - q1

print(f"数据: {response_times}")
print(f"Q1 (第25百分位数): {q1}")
print(f"中位数 (第50百分位数): {median}")
print(f"Q3 (第75百分位数): {q3}")
print(f"IQR (Q3 - Q1): {iqr}")
```

**使用Pandas：**
如果你的数据在Pandas Series或DataFrame中，可以使用`.quantile()`方法。请注意，它预期分位数的值在0到1之间。`.describe()`方法也方便地包含了Q1 (25%)、中位数 (50%) 和Q3 (75%)。

```python
import pandas as pd

# Sample data in a Pandas Series
response_times_pd = pd.Series([120, 150, 155, 160, 175, 180, 190, 210, 230, 250, 300, 500])

# Calculate specific quantiles
q1_pd = response_times_pd.quantile(0.25)
median_pd = response_times_pd.quantile(0.50) # Or response_times_pd.median()
q3_pd = response_times_pd.quantile(0.75)
iqr_pd = q3_pd - q1_pd

print(f"\n使用Pandas：")
print(f"Q1 (第25百分位数): {q1_pd}")
print(f"中位数 (第50百分位数): {median_pd}")
print(f"Q3 (第75百分位数): {q3_pd}")
print(f"IQR (Q3 - Q1): {iqr_pd}")

# Using .describe() for a summary including quartiles
print("\nPandas describe() 输出：")
print(response_times_pd.describe())
```

### 解释四分位数：一个例子

我们来看看`response_times`的例子结果（NumPy和Pandas由于插值方法不同，结果略有差异，但道理一致。这里我们使用NumPy的结果：Q1=158.75，中位数=185.0，Q3=235.0）。

- **Q1 = 158.75 毫秒：** 25%的响应时间小于或等于158.75毫秒。
- **中位数 (Q2) = 185.0 毫秒：** 一半的响应时间低于185毫秒，一半高于185毫秒。
- **Q3 = 235.0 毫秒：** 75%的响应时间小于或等于235.0毫秒。这也表示响应时间中最高的25%的值*高于*235.0毫秒。
- **IQR = 76.25 毫秒 (235.0 - 158.75)：** 中间50%的响应时间落在76.25毫秒的范围内，具体在158.75毫秒到235.0毫秒之间。

这比仅有平均值或标准差能提供更清晰的整体情况。我们可以看到集中趋势（中位数）和大部分数据的离散程度（IQR），这有助于我们发现潜在的偏斜或不寻常的值（例如500毫秒的响应时间）。

这里是一个简单的可视化，显示了四分位数如何划分数据点：



![响应时间四分位数视图](plots/2413-0.json)



> 样本响应时间的直方图，带有垂直线，显示第一四分位数 (Q1)、中位数 (Q2) 和第三四分位数 (Q3)。开始、Q1、中位数、Q3和结束之间的每个部分理想情况下包含25%的数据点（尽管分箱效应可能会改变每条柱的精确计数）。

理解百分位数和四分位数能让你对数据分布有更细致的认识。它们是基本知识点，直接用于创建和解读箱线图等可视化图表，我们将在下一步讨论。

## 参考资料

- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  提供了统计概念的实用应用，包括四分位数和IQR，与数据科学和机器学习相关。
- [numpy.percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html) — NumPy Developers (2024)
  NumPy官方文档，详细说明了使用该库计算百分位数函数的方法和参数。
- [pandas.Series.quantile](https://pandas.pydata.org/docs/reference/api/pandas.Series.quantile.html) — pandas-dev (2024)
  Pandas Series计算分位数的官方文档，是理解数据分布的重要方法。
