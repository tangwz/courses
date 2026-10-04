---
course: "intermediate-python-programming-ml"
chapter: "data-visualization-matplotlib-seaborn"
lesson: "seaborn-advanced-plots"
sourceId: 2183
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-4-data-visualization-matplotlib-seaborn/seaborn-advanced-plots"
title: "使用 Seaborn 创建高级图表"
description: "使用 Seaborn 生成复杂的统计可视化图表，如热图、配对图、小提琴图和计数图。"
order: 6
plots: ["plots/2183-0.json", "plots/2183-1.json"]
sourceHash: "a5b7dd45dd09e93bbe6b5de10886791a74ce7c465ee369ac28195f4cb1736153"
sourceCorrections: []
---

尽管 Matplotlib 提供了 Python 中绘图的基础工具，但 Seaborn 提供了一个更高级的接口，专门用于制作信息丰富且美观的统计图形。Seaborn 是在 Matplotlib 之上构建的，它简化了生成复杂可视化的过程，这在数据分析和机器学习 (machine learning)中很常见，对于在 Matplotlib 中需要大量定制的图表，Seaborn 通常只需一次函数调用即可完成。

本节介绍几种 Seaborn 中的高级图表类型，它们对于分析数据集中变量间的关系和结构特别有用。这些图表有助于发现仅凭简单图表可能不明显的规律。

### 热图

热图非常适合将矩阵式数据可视化，其中单个值由颜色表示。它们常用于显示相关矩阵，以紧凑的视觉格式展示多个变量间的相关系数。暖色通常表示正相关，冷色表示负相关，颜色的深浅代表强度。

要创建热图，通常从二维数组或 Pandas DataFrame 开始。例如，计算 DataFrame 的相关矩阵会得到一个非常适合热图的结构。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 生成示例数据
np.random.seed(42)
data = pd.DataFrame(np.random.rand(10, 5), columns=[f'Var{i}' for i in range(1, 6)])
data['Var3'] = data['Var1'] * 2 + np.random.normal(0, 0.1, 10)
data['Var5'] = -data['Var2'] * 1.5 + np.random.normal(0, 0.2, 10)

# 计算相关矩阵
correlation_matrix = data.corr()

# 创建热图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title('相关矩阵热图')
plt.show()
```



![相关矩阵热图](plots/2183-0.json)



> 相关矩阵可视化为热图。颜色深浅表示相关性的强度，注释显示具体的系数值。

在 `sns.heatmap()` 函数中：

- 第一个参数 (parameter)是数据矩阵（例如 `correlation_matrix`）。
- `annot=True` 在单元格上显示数据值。
- `cmap` 设置颜色映射（例如 'viridis', 'coolwarm', 'YlGnBu'）。
- `fmt=".2f"` 将注释文本格式化为两位小数。

### 配对图

在进行探索性数据分析（EDA）时，同时理解多个数值变量之间的关系通常是必要的。配对图（也称为散点图矩阵）提供一个轴网格，其中数据集中的每个变量都与所有其他变量进行绘图。对角线轴通常显示每个变量的单变量分布（直方图或核密度估计）。

该图对于快速查看双变量关系以及识别不同变量组合中潜在的相关性或规律非常有用。Seaborn 的 `sns.pairplot()` 函数使得生成这些网格变得简单直接。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 从 Seaborn 加载示例数据集
iris = sns.load_dataset('iris')

# 创建配对图
# 'hue' 根据 'species' 类别为点上色
sns.pairplot(iris, hue='species', palette='viridis')
plt.suptitle('Iris 数据集中的两两关系', y=1.02) # 调整标题位置
plt.show()
```

> 代码为 Iris 数据集生成了一个配对图。非对角线图是散点图，显示特征对之间的关系，并按物种着色。对角线图显示每个物种各特征的分布（KDE）。

`sns.pairplot()` 函数将 DataFrame 作为输入。

- `hue` 参数 (parameter)功能强大；它允许您根据分类变量（例如 `iris` 数据集中的 'species'）为点上色，这样可以很容易地看出不同组在变量对之间是否呈现不同的聚类。
- `palette` 控制用于 `hue` 变量的配色方案。

虽然信息量非常大，但请注意，为具有大量变量的数据集生成配对图可能会变得计算密集且视觉混乱。

### 小提琴图

小提琴图是一种可视化不同类别数值数据分布的方法。它们类似于箱线图，但通过在每侧加入核密度估计（KDE）来提供更多信息。这使您可以查看分布的形状，包括潜在的多峰性（多个峰值），这在标准箱线图中是隐藏的。

`sns.violinplot()` 函数用于创建这些图。它通常将一个分类变量用于 x 轴，一个数值变量用于 y 轴。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载 tips 数据集
tips = sns.load_dataset("tips")

# 创建小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x="day", y="total_bill", data=tips, palette="coolwarm")
plt.title('每日总账单金额的分布')
plt.xlabel('星期几')
plt.ylabel('总账单金额 ($)')
plt.show()
```



![每日总账单金额的分布](plots/2183-1.json)



> 小提琴图显示了每周各天的总账单金额分布。小提琴的宽度表示不同账单金额处数据点的密度。内部元素显示中位数和四分位距，类似于箱线图。

`sns.violinplot()` 的重要参数 (parameter)：

- `x`, `y`: 定义坐标轴的变量。
- `data`: 包含数据的 DataFrame。
- `palette`: 设置 x 轴上不同类别的配色方案。
- 您还可以添加 `hue` 参数，根据另一个分类变量进一步划分小提琴。

这些高级 Seaborn 图表（热图、配对图和小提琴图）提供了强大的方法，通过可视化从数据中获得更细致的理解。它们通常比基本图表更能有效地呈现复杂关系、分布和潜在问题（如异常值或偏斜数据），使它们成为机器学习 (machine learning)项目数据分析阶段的重要工具。

## 参考资料

- [Seaborn: statistical data visualization](https://seaborn.pydata.org/) — Michael Waskom (Ongoing)
  学习和使用Seaborn的主要资源，包含其API和本节介绍的各种图表类型。
- [Matplotlib Documentation](https://matplotlib.org/) — John Hunter, The Matplotlib development team (Ongoing)
  理解Seaborn所依赖的基础绘图库。
- [Python for Data Analysis](https://wesmckinney.com/book) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  一本实用指南，广泛涵盖使用Pandas进行数据处理以及使用Matplotlib和Seaborn进行数据可视化。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  提供创建有效且信息丰富的可视化图表的一般原则，补充了技术操作指南。
