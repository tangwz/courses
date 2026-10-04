---
course: "probability-statistics-essentials-ml"
chapter: "descriptive-statistics-datasets"
lesson: "descriptive-stats-pandas"
sourceId: 1408
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-3-descriptive-statistics-datasets/descriptive-stats-pandas"
title: "使用 Pandas 计算描述性统计量"
description: "在 Python 中使用 Pandas 库高效计算描述性统计量。"
order: 8
plots: ["plots/1408-0.json"]
sourceHash: "73047e5ab8a10147030c752296a9b1a94f8bfcdde0d869f1c1578dc0fe2479f6"
sourceCorrections: []
---

尽管理解均值、方差、相关性和其他描述性统计量的公式是基本的，但对于任何大于小规模数据集的数据而言，手动计算很快就会变得不切实际。这时，Python 的数据分析库，特别是 Pandas，就成了数据科学家和机器学习 (machine learning)实践者的必需工具。Pandas 提供了高效且易用的函数，用于对通常存储在 Series 或 DataFrame 对象中的数据计算大量描述性统计量。

假设您的数据已加载到 Pandas DataFrame 中。如果您正在跟着操作，可以这样创建一个示例 DataFrame：

```python
import pandas as pd
import numpy as np

# 创建示例数据
data = {'ExamScore': [78, 85, 92, 65, 72, 88, 95, 81, 76, 80, np.nan, 83],
        'StudyHours': [5, 6, 8, 3, 4, 7, 9, 5.5, 4.5, 5, 2, 6],
        'SleepHours': [7, 6.5, 7.5, 8, 7, 6, 7, 7.5, 8, 6.5, 9, 7]}
df = pd.DataFrame(data)

print(df)
```

这会创建一个 DataFrame `df`，其中包含不同学生的考试分数、学习时长和睡眠时长，包括一个缺失的考试分数（`np.nan`）。

### `.describe()` 方法：快速概览

通常，使用 Pandas 审视数值数据集的第一步是使用 `.describe()` 方法。它为 DataFrame 中每个数值列（或单个 Series）提供了集中趋势、离散程度和分布形状的简洁概括。

```python
# 获取数值列的汇总统计
summary_stats = df.describe()
print(summary_stats)
```

运行此代码将产生类似于以下内容的输出：

```
       ExamScore  StudyHours  SleepHours
count  11.000000   12.000000   12.000000
mean   81.363636    5.833333    7.208333
std     8.737508    1.991495    0.793394
min    65.000000    2.000000    6.000000
25%    77.000000    4.500000    6.500000
50%    81.000000    5.750000    7.000000
75%    86.500000    7.000000    7.500000
max    95.000000    9.000000    9.000000
```

注意以下几点：

- **`count`**：显示非缺失值的数量。`ExamScore` 有 11 个，反映了 `np.nan`。
- **`mean`**：平均值。
- **`std`**：标准差，衡量离散程度。
- **`min`, `max`**：最小值和最大值。
- **`25%`, `50%`, `75%`**：这些是四分位数（百分位数）。第 50 个百分位数是中位数。

`.describe()` 方法非常适合快速了解数据的分布和规模。

### 计算单独的统计量

虽然 `.describe()` 很方便，但您通常需要特定的统计量。Pandas 为此提供了专用方法。您可以将它们应用于整个 DataFrame（为每个列计算统计量）或单个 Series（特定列）。

#### 集中趋势度量

```python
# 计算每列的均值
means = df.mean()
print("均值:\n", means)

# 计算 'ExamScore' 列的中位数
median_score = df['ExamScore'].median()
print(f"\n考试分数中位数: {median_score}")

# 计算 'SleepHours' 的众数
# 如果有多个值出现频率最高，众数可能返回多个值
modes_sleep = df['SleepHours'].mode()
print("\n睡眠时长众数:\n", modes_sleep)
```

这些函数（`.mean()`、`.median()`、`.mode()`）默认自动处理缺失值（由 `skipna=True` 参数 (parameter)控制）。

#### 离散度和位置度量

```python
# 计算每列的方差
variances = df.var()
print("方差:\n", variances)

# 计算 'StudyHours' 的标准差
std_study = df['StudyHours'].std()
print(f"\n学习时长标准差: {std_study:.4f}")

# 计算最小值和最大值
min_values = df.min()
max_values = df.max()
print("\n最小值:\n", min_values)
print("\n最大值:\n", max_values)

# 计算 'ExamScore' 的特定百分位数（例如，第 10 和第 90 个）
p10 = df['ExamScore'].quantile(0.10)
p90 = df['ExamScore'].quantile(0.90)
print(f"\n考试分数第 10 个百分位数: {p10}")
print(f"考试分数第 90 个百分位数: {p90}")

# 计算 'ExamScore' 的四分位距（IQR）
q1 = df['ExamScore'].quantile(0.25)
q3 = df['ExamScore'].quantile(0.75)
iqr = q3 - q1
print(f"考试分数的 IQR: {iqr}")
```

`.quantile(q)` 方法多功能，可以查找任何百分位数，其中 $q$ 在 0 到 1 之间。范围可以通过用 `.max()` 的结果减去 `.min()` 的结果来简单计算。

#### 分布形状度量

偏度和峰度分别告诉您分布的不对称性和尖峭程度。

```python
# 计算每列的偏度
skewness = df.skew()
print("偏度:\n", skewness)

# 计算 'StudyHours' 的峰度
kurt_study = df['StudyHours'].kurt() # 费舍尔定义（正态分布为 0）
# kurt_study = df['StudyHours'].kurtosis() # 与 .kurt() 相同
print(f"\n学习时长的峰度: {kurt_study:.4f}")
```

正偏度表示尾部延伸向更高值，而负偏度表示尾部延伸向更低值。峰度衡量“尾部厚度”；更高的峰度意味着与正态分布相比，有更多异常值或更厚的尾部。

#### 相关性分析

为了解数值变量对之间的线性关系，请在 DataFrame 上使用 `.corr()` 方法。

```python
# 计算列之间的成对相关性
correlation_matrix = df.corr()
print("\n相关矩阵:\n", correlation_matrix)
```

这将输出一个相关矩阵，其中每个单元格 $(i, j)$ 包含列 $i$ 和列 $j$ 之间的皮尔逊相关系数。对角线元素始终为 1（变量与自身的关联）。

```
              ExamScore  StudyHours  SleepHours
ExamScore      1.000000    0.946434   -0.394137
StudyHours     0.946434    1.000000   -0.417979
SleepHours    -0.394137   -0.417979    1.000000
```

从中我们可以看到 `ExamScore` 和 `StudyHours` 之间存在强正相关（$0.95$），这表明学习时间更多的学生往往获得更高的分数。`StudyHours` 和 `SleepHours` 之间存在中等负相关，这可能表明在这个样本中，更多的学习时间可能与较少的睡眠相关。请记住，相关性不代表因果关系！

### 描述性统计量的可视化

尽管数值概括很重要，但数据可视化通常能提供更深刻的理解。Pandas 与 Matplotlib 集成，可以直接从 DataFrames 或 Series 进行快速绘图。对于更自定义或高级的图表，通常使用 Seaborn 或 Plotly 等库。

以下是在计算统计量后，如何使用 Plotly 快速可视化 `ExamScore` 分布的方法：



![考试分数分布](plots/1408-0.json)



> 直方图显示了样本数据中考试分数的频率分布。

该直方图通过视觉方式展示分数分布的形状，补充了数值统计量（如均值、中位数、偏度）。

有效地使用 Pandas 可以让您快速地从原始数据过渡到有意义的统计概括，为机器学习 (machine learning)工作流程中的进一步分析、可视化和模型构建奠定基础。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  以实践方法讲解与数据科学相关的统计概念，包括对描述性统计及其应用的审阅。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 582
  使用 Pandas 的权威资源，涵盖 Python 中的数据结构、处理和统计方法。
