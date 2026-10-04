---
course: "intro-eda-course"
chapter: "univariate-analysis"
lesson: "practice-univariate-exploration"
sourceId: 1164
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis/practice-univariate-exploration"
title: "练习：单变量数据分析"
description: "进行单变量分析，并为数值变量和类别变量创建可视化图表。"
order: 8
plots: ["plots/1164-0.json", "plots/1164-1.json", "plots/1164-2.json"]
sourceHash: "6553bb33105bea09eb2815b6f74d2c904dc1fb7fc92b0049158e93b0c540f625"
sourceCorrections: []
---

此练习将单变量分析方法应用到一个真实数据集中。它涵盖了如何使用Python库为数值变量和类别变量计算描述性统计量并生成可视化图表。

### 准备工作：加载数据

我们将使用'penguins'数据集，它可以通过Seaborn库方便地获取。此数据集包含不同企鹅物种的测量数据。首先，确保你已安装并导入Seaborn和Pandas。然后，加载数据集：

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
penguins_df = sns.load_dataset('penguins')

# 显示基本信息和前几行，以了解数据
print("数据集信息：")
penguins_df.info()

print("\n前5行：")
print(penguins_df.head())

# 在本次练习中，简单处理缺失值
# (在实际情况中，你会使用第2章的方法)
penguins_df.dropna(inplace=True)
print("\n删除NaN值后的数据集信息：")
penguins_df.info()
```

这个初始输出向我们显示了列、它们的数据类型（例如数值测量值的float64，类别字符串的object），并确认存在一些缺失值，为了本练习的简便，我们已将其删除。

### 分析数值变量：`flipper_length_mm`

我们来查看企鹅的鳍肢长度。

#### 描述性统计

我们可以使用`.describe()`方法对此特定列（Series）进行快速统计汇总。

```python
# 计算flipper_length_mm的描述性统计量
flipper_stats = penguins_df['flipper_length_mm'].describe()
print("\n鳍肢长度（毫米）的描述性统计量：")
print(flipper_stats)

# 单独计算中位数（通常很有用）
flipper_median = penguins_df['flipper_length_mm'].median()
print(f"\n鳍肢中位数长度：{flipper_median} 毫米")

# 计算偏度
flipper_skew = penguins_df['flipper_length_mm'].skew()
print(f"鳍肢长度的偏度：{flipper_skew:.2f}")
```

**解读：** `.describe()`的输出提供了计数、均值、标准差（$\sigma$）、最小值、最大值和四分位数（第25百分位数或Q1，第50百分位数或中位数，第75百分位数或Q3）。中位数给我们提供了对异常值不敏感的中心点。偏度值（接近0表示近似对称，正值表示右偏，负值表示左偏）可以快速检查分布的形态。在此处，偏度接近0表明在处理缺失数据后，鳍肢长度的分布相当对称。

#### 分布可视化：直方图

直方图非常适合可视化分布的形态、集中趋势和离散程度。

```python
# 设置绘图风格（可选，为了美观）
sns.set_style("whitegrid")

# 为flipper_length_mm创建直方图
plt.figure(figsize=(8, 5))
sns.histplot(data=penguins_df, x='flipper_length_mm', kde=True, bins=15, color='#4dabf7')
plt.title('企鹅鳍肢长度分布')
plt.xlabel('鳍肢长度 (毫米)')
plt.ylabel('频数')
plt.show()
```



![企鹅鳍肢长度分布](plots/1164-0.json)



> 直方图显示了企鹅鳍肢长度的频数分布。曲线（KDE）提供了分布的平滑估计。

**解读：** 直方图直观地证实了偏度值所暗示的近似对称、大致呈钟形的分布。大多数鳍肢长度集中在中心（均值/中位数附近）。`kde=True`参数 (parameter)添加了核密度估计曲线，提供了分布的平滑轮廓。`bins=15`指定了要使用的条形数量；尝试此数量有时可以显示分布的不同特性。

#### 汇总统计可视化：箱线图

箱线图能有效比较汇总统计量（中位数、四分位数、范围），并根据IQR规则识别潜在异常值。

```python
# 为flipper_length_mm创建箱线图
plt.figure(figsize=(6, 4))
sns.boxplot(data=penguins_df, x='flipper_length_mm', color='#96f2d7')
plt.title('企鹅鳍肢长度的箱线图')
plt.xlabel('鳍肢长度 (毫米)')
plt.show()
```



![企鹅鳍肢长度的箱线图](plots/1164-1.json)



> 箱线图汇总了企鹅鳍肢长度的分布情况。箱体表示IQR（Q1到Q3），内部的线是中位数，晶须延伸显示数据范围（不包括异常值，通常为1.5 \* IQR）。晶须之外的点是潜在异常值。

**解读：** 箱线图清楚地显示了中位数（约200毫米）、IQR（箱体本身，根据`.describe()`大约为193毫米到209毫米），以及由晶须指示的整体范围。在这个特定图中，删除NaN值后，似乎没有点被标记 (token)为异常值，这表明根据标准IQR规则（Q3 + 1.5*IQR或Q1 - 1.5*IQR），没有极端值。

### 分析类别变量：`species`

现在我们来查看`species`列。

#### 频数和比例

对于类别数据，我们想知道每个类别中有多少观测值。

```python
# 计算species的频数
species_counts = penguins_df['species'].value_counts()
print("\n物种频数：")
print(species_counts)

# 计算比例（百分比）
species_proportions = penguins_df['species'].value_counts(normalize=True) * 100
print("\n物种比例（%）：")
print(species_proportions)
```

**解读：** `.value_counts()`方法列出了每个独特的物种以及属于该物种的企鹅数量。将`normalize=True`设置为真会把这些计数转换为比例（或乘以100后的百分比），显示数据集中每个物种的相对频率。我们看到在此清洗过的数据集中，阿德利企鹅物种最为常见。

#### 频数可视化：条形图

条形图是比较不同类别频数的理想工具。

```python
# 为物种计数创建条形图
plt.figure(figsize=(7, 5))
# 使用countplot直接计数并绘图
sns.countplot(data=penguins_df, x='species', palette=['#ff8787', '#74c0fc', '#74b816'], order=species_counts.index)
plt.title('各物种企鹅数量')
plt.xlabel('物种')
plt.ylabel('数量')
plt.show()
```



![各物种企鹅数量](plots/1164-2.json)



> 条形图显示了数据集中每种企鹅的计数。

**解读：** 条形图提供了物种数量的即时视觉比较，确认了在我们的处理数据中，阿德利企鹅物种的代表性最高，其次是金图企鹅，然后是帽带企鹅。`order=species_counts.index`参数 (parameter)确保条形按频率降序绘制，这通常有助于提高可读性。使用特定的`palette`可以控制颜色。

### 练习总结

在本次练习中，你对企鹅数据集中的数值变量（`flipper_length_mm`）和类别变量（`species`）应用了单变量分析方法。你计算了重要的描述性统计量，并生成了直方图、箱线图和条形图，以图表形式表示它们的分布和频数。这种逐个查看变量的过程是你在查看变量之间关系之前，理解数据基本特征的根本方式。

## 参考资料

- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用 Python 的 Pandas、NumPy 和 Matplotlib 库进行数据处理和分析的综合指南，直接适用于代码示例。
- [Seaborn Documentation](https://seaborn.pydata.org/index.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn 库的官方参考资料，提供详细指南和 API 文档，用于 Python 中的统计数据可视化，用于直方图、箱线图和条形图。
- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  提供数据科学相关统计概念的实用方法，包括本章讨论的描述性统计和数据可视化方法。
