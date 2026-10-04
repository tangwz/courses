---
course: "intermediate-python-programming-ml"
chapter: "data-visualization-matplotlib-seaborn"
lesson: "seaborn-distributions-relationships"
sourceId: 2187
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-distributions-relationships"
title: "分布与关系的可视化"
description: "使用Seaborn有效地可视化数据分布和变量之间的关系。"
order: 7
plots: ["plots/2187-0.json", "plots/2187-1.json", "plots/2187-2.json", "plots/2187-3.json"]
sourceHash: "d5a108ef6c570b2f0c577d162187d376dce8365f44d9df2029ea96bec587a856"
sourceCorrections: []
---

理解数据如何分布以及不同变量之间如何关联，对数据分析的初步阶段（EDA）和为机器学习 (machine learning)模型准备数据来说是基础。Seaborn，一个强大的数据可视化库，提供了专门的函数，使这些方面的可视化变得直接且有意义。主要内容是使用Seaborn来有效地查看单变量分布和多变量关系。

### 单变量分布的可视化

在考察变量关系之前，了解单个变量的特性通常是很有帮助的。值是如何散布的？它们是对称的、偏斜的吗？存在异常值吗？Seaborn为此提供了多种图表类型。

#### 直方图和密度图

直方图是一种常见的方式，通过对数据进行分箱并显示每个箱的计数来可视化连续变量的频率分布。Seaborn的`histplot`函数简化了此过程，并可以选择性地叠加核密度估计 (KDE)。KDE图对直方图进行平滑处理，提供对潜在概率密度函数的估计。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 样本数据模拟
np.random.seed(42)
data = pd.DataFrame({
    'feature_A': np.random.normal(loc=5, scale=2, size=200),
    'feature_B': np.random.exponential(scale=5, size=200),
    'category': np.random.choice(['Group 1', 'Group 2', 'Group 3'], size=200, p=[0.4, 0.3, 0.3])
})

# 带有KDE的直方图
plt.figure(figsize=(8, 5))
sns.histplot(data=data, x='feature_A', kde=True, color='#339af0')
plt.title('Distribution of Feature A (Histogram with KDE)')
plt.xlabel('Feature A Value')
plt.ylabel('Frequency / Density')
plt.show()

# 独立的KDE图
plt.figure(figsize=(8, 5))
sns.kdeplot(data=data, x='feature_B', fill=True, color='#20c997')
plt.title('Density Plot of Feature B')
plt.xlabel('Feature B Value')
plt.ylabel('Density')
plt.show()
```

直方图显示清晰的分箱，而KDE则提供分布形状的更平滑表示。当您想查看特定范围内的频率计数时，请使用`histplot`；当您对分布的整体形状和平滑度更感兴趣时，请使用`kdeplot`。



![特征A的直方图和KDE](plots/2187-0.json)



> 特征A的计数直方图，叠加了其核密度估计。

#### 箱线图和提琴图

箱线图（或箱须图）使用四分位数总结分布。箱体代表第25 ($Q_1$) 和第75 ($Q_3$) 百分位数之间的四分位距 (IQR)，一条线表示中位数 ($Q_2$)。须线通常延伸到距箱体1.5倍IQR的距离，超出须线的点通常被认为是潜在的异常值。

提琴图结合了箱线图和KDE。“提琴”形状的宽度表示数据在不同值处的密度。这使您可以看到汇总统计数据*以及*分布的形状，包括箱线图可能隐藏的潜在多峰（多模态 (multimodal)）。

```python
# 箱线图
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, y='feature_A', color='#ffc078')
plt.title('Box Plot of Feature A')
plt.ylabel('Feature A Value')
plt.show()

# 提琴图
plt.figure(figsize=(8, 5))
sns.violinplot(data=data, y='feature_B', color='#b197fc')
plt.title('Violin Plot of Feature B')
plt.ylabel('Feature B Value')
plt.show()
```



![特征B分布的比较](plots/2187-1.json)



> 箱线图和提琴图显示了特征B的分布汇总和密度形状。

使用箱线图可以简洁地汇总集中趋势和离散程度，尤其是在比较多个组时。当您需要更全面地了解分布形状时（例如，在汇总统计数据之外识别多个峰值或偏斜），请使用提琴图。

### 变量间关系的可视化

机器学习 (machine learning)通常涉及理解变量如何互相影响。一个变量增加时另一个变量也会增加吗？不同的类别是否呈现不同的模式？Seaborn在可视化这些关系方面表现出色。

#### 两个数值变量之间的关系

散点图是考察两个连续变量之间关系的常用工具。每个点代表一个观测值，根据其在两个轴上的值进行绘制。Seaborn的`scatterplot`使这变得容易，并允许使用颜色 (`hue`)、大小 (`size`) 或样式 (`style`) 引入第三个变量。

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='feature_A', y='feature_B', hue='category', palette=['#fa5252', '#4c6ef5', '#82c91e'])
plt.title('Relationship between Feature A and Feature B by Category')
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.legend(title='Category')
plt.show()
```



![特征A与特征B的散点图](plots/2187-2.json)



> 特征A和特征B之间的关系，按类别着色。

`jointplot`函数将散点图与每个变量的直方图或KDE图结合在边距中。这同时提供了关系和各个分布的视图。

```python
sns.jointplot(data=data, x='feature_A', y='feature_B', kind='scatter', color='#7048e8') # kind 可以是 'kde' (核密度估计), 'hist' (直方图), 'reg' (回归)
plt.suptitle('Joint Distribution of Feature A and Feature B', y=1.02) # 调整标题位置
plt.show()
```

#### 多个数值变量之间的关系

当处理两个以上数值变量时，为每对变量单独创建散点图可能会很繁琐。Seaborn的`pairplot`可使此过程自动化。它生成一个网格，其中对角线显示每个变量的分布（使用`histplot`或`kdeplot`），非对角线单元格显示每对变量的散点图。这对于快速了解数据集中变量之间的成对关系非常有用。

```python
# 创建另一个数值特征用于演示
data['feature_C'] = data['feature_A'] * 0.5 + np.random.normal(0, 1, 200)

# 仅选择数值列用于pairplot
numerical_data = data[['feature_A', 'feature_B', 'feature_C']]

sns.pairplot(numerical_data)
plt.suptitle('Pairwise Relationships Between Numerical Features', y=1.02)
plt.show()

# 按类别着色的pairplot
# sns.pairplot(data, hue='category', palette=['#fa5252', '#4c6ef5', '#82c91e'])
# plt.suptitle('按类别着色的成对关系', y=1.02)
# plt.show()
```

请注意，当特征数量很多时，`pairplot`可能会变得计算量大且视觉上杂乱。

#### 涉及分类变量的关系

通常，您需要比较不同类别之间的数值变量，或可视化两个分类变量之间的关系。

- **数值变量 vs. 分类变量：** 箱线图和提琴图在此处非常有效。通过为`x`（或`y`）轴指定一个分类变量，为另一个轴指定一个数值变量，您可以并排比较分布。

  ```python
  plt.figure(figsize=(10, 6))
  sns.boxplot(data=data, x='category', y='feature_A', palette=['#f783ac', '#91a7ff', '#8ce99a'])
  plt.title('Distribution of Feature A across Categories (Box Plot)')
  plt.xlabel('Category')
  plt.ylabel('Feature A')
  plt.show()

  plt.figure(figsize=(10, 6))
  sns.violinplot(data=data, x='category', y='feature_B', palette=['#f783ac', '#91a7ff', '#8ce99a'])
  plt.title('Distribution of Feature B across Categories (Violin Plot)')
  plt.xlabel('Category')
  plt.ylabel('Feature B')
  plt.show()
  ```

  Seaborn还提供了`stripplot`和`swarmplot`，它们为每个类别绘制单独的数据点，有助于可视化密度和重叠，尤其适用于较小的数据集。`swarmplot`调整点的位置以避免重叠，而`stripplot`允许点重叠（通常与透明度或`jitter`结合使用）。
- **分类变量 vs. 分类变量：** 要理解两个分类变量之间的关系，通常会查看频率计数。Seaborn中的`countplot`可以显示单个分类变量内的计数，或使用`hue`参数 (parameter)按另一个分类变量分组的计数。为了可视化联合频率或从中派生的指标（例如外部计算的相关性），通常使用热力图。

#### 用于相关矩阵的热力图

热力图非常适合可视化矩阵状数据，其中颜色代表值。数据分析中一个常见应用是可视化数值特征的相关矩阵。Pandas DataFrame具有`.corr()`方法来计算成对相关性，Seaborn的`heatmap`可以有效地显示此矩阵。

```python
# 计算数值特征的相关矩阵
correlation_matrix = numerical_data.corr()

print("相关矩阵:")
print(correlation_matrix)

plt.figure(figsize=(7, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
# 常用配色方案选项: 'viridis', 'plasma', 'coolwarm', 'Blues', 'Reds'
# annot=True 在单元格上显示值
# fmt=".2f" 将注释格式化为两位小数
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```



![相关性热力图](plots/2187-3.json)



> 热力图显示了数值特征之间的成对相关系数。在此特定的“viridis”配色方案中，颜色越深可能表示相关性越弱，颜色越浅则表示正相关性越强。

热力图可以快速显示哪些变量具有强正向（接近1）或负向（接近-1）线性关系，以及哪些变量相对不相关（接近0）。

通过使用这些Seaborn函数，您可以对数据结构有重要的了解，识别模式、异常以及变量互相影响的情况，这对于高效的特征工程和模型构建是不可或缺的。请选择最适合变量类型（数值型、分类型）以及您希望回答的关于其分布或关系的具体问题的图表类型。

## 参考资料

- [Seaborn: statistical data visualization](https://seaborn.pydata.org/) — Michael Waskom and the Seaborn development team (2024)
  提供所有 Seaborn 函数（包括用于可视化分布和关系的函数）的全面文档、教程和示例。
- [Matplotlib: Visualization with Python](https://matplotlib.org/) — John Hunter, Matplotlib development team (2024)
  Python 中静态、交互式和动画可视化的基础库；对于 Seaborn 图形的深度定制至关重要。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHED9-aeE6Eb220-UEQDaYnwjGW6Br6gKueYRKB8tLZRa7GQlmzcfSUc-l3e3FVcr5grcLJ1b2w3YtBg7M3WaNqf_zhlhF-4hjUywdOvVxjzJ0VXDMH1S39UAUshT5r8bMBIaYAcND-x0sSr1oMwebjh70iGxTDGEUu8Rw3LoSiQuAdubo9Vn6PtyPiyw==) — Wes McKinney (2017)
  Publisher: O'Reilly Media
  Python 数据分析生态系统的权威指南，包含对 Pandas 的广泛介绍以及使用 Matplotlib 和 Seaborn 进行数据可视化的实用示例。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供创建有效数据可视化的原则和最佳实践，为实践编码技能提供了理论基础。
