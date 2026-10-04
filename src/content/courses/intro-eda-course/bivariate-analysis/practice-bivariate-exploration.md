---
course: "intro-eda-course"
chapter: "bivariate-analysis"
lesson: "practice-bivariate-exploration"
sourceId: 1177
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-4-bivariate-analysis/practice-bivariate-exploration"
title: "动手实践：双变量分析"
description: "应用双变量分析技术，计算相关性，并创建比较可视化图表。"
order: 7
plots: ["plots/1177-0.json", "plots/1177-1.json", "plots/1177-2.json", "plots/1177-3.json"]
sourceHash: "490453037f66971c58078ed0c16f0fb84fd3e6e74c3109bdc210a52d35a9f3e4"
sourceCorrections: []
---

实际操作示例使用 Pandas、Matplotlib 和 Seaborn 等 Python 库对示例数据集进行双变量分析。这些示例展示了如何探索变量间的关系。进行这些实践操作需要一个已加载的 Pandas DataFrame。

在我们的示例中，假设我们有一个名为 `df` 的 DataFrame，它包含客户信息，具有 `Age`（年龄）、`Annual_Income_k$`（年收入，单位千美元）、`Spending_Score`（消费得分，根据消费行为分配的1-100分值）、`Gender`（性别）和 `Education`（教育程度）等列。

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 'df' 是您已加载的 DataFrame
# 示例：df = pd.read_csv('customer_data.csv')

# 让我们回顾一下数据结构
print("数据样本:")
print(df.head())
print("\n数据信息:")
df.info()

# 设置绘图风格以保持一致性
sns.set_theme(style="whitegrid")
```

`df.head()` 和 `df.info()` 的输出将确认列名及其数据类型，为我们的分析奠定基础。

### 分析数值变量间的关系

一个常见的任务是理解两个定量测量之间如何关联。收入是否随年龄增长？收入和消费得分之间是否存在关联？

#### 散点图

散点图非常适合可视化两个数值变量之间的关系。让我们考察 `Annual_Income_k$` 和 `Spending_Score` 之间的关系。

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Income_k$', y='Spending_Score', hue='Gender', palette=['#1c7ed6', '#f06595'])
plt.title('年收入与消费得分')
plt.xlabel('年收入（千美元）')
plt.ylabel('消费得分 (1-100)')
plt.show()
```



![年收入与消费得分](plots/1177-0.json)



> 散点图显示客户年收入与消费得分之间的关系。点按性别着色。

该图有助于我们直观地查看分布情况。我们可能会观察到一些簇群，例如低收入低消费得分的客户，或高收入高消费得分的客户。添加性别的 `hue` 参数 (parameter)使我们能够观察不同群体之间关系是否存在差异。

#### 相关性分析

散点图直观地展示了关系，而相关系数量化 (quantization)了数值变量之间的*线性*关联。珀尔逊相关系数 ($r$) 的范围是从 -1（完全负线性相关）到 +1（完全正线性相关），0 表示无线性相关。

让我们计算 DataFrame 中数值列之间的相关性。

```python
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_cols].corr()

print("相关系数矩阵:")
print(correlation_matrix)
```

这将输出一个显示成对相关系数的表格。例如，我们可能会发现 `Age` 和 `Annual_Income_k$` 之间存在微弱的正相关，或者 `Age` 和 `Spending_Score` 之间存在中度的负相关。

#### 相关系数热力图

热力图提供相关系数矩阵的图形表示，便于识别强相关或弱相关，尤其是在涉及许多变量时。

```python
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('数值特征的相关系数矩阵')
plt.show()
```



![相关系数矩阵热力图](plots/1177-1.json)



> 热力图可视化年龄、年收入和消费得分之间的珀尔逊相关系数。注释显示相关值。

颜色表示相关性的强度和方向（例如，红色表示正相关，蓝色表示负相关），注释提供了精确的数值。

### 分析数值变量与类别变量间的关系

通常，我们需要比较由类别变量定义的各个群体间的数值测量。例如，`Annual_Income_k$` 的平均值在不同 `Gender` 群体或不同 `Education` 级别之间是否存在显著差异？

#### 比较图 (箱线图、小提琴图)

箱线图和小提琴图对于可视化类别变量的每个类别中数值变量的分布非常有效。

让我们使用箱线图比较不同 `Education` 级别的 `Annual_Income_k$`。

```python
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='Education', y='Annual_Income_k$', palette='Blues')
plt.title('按教育水平划分的年收入分布')
plt.xlabel('教育水平')
plt.ylabel('年收入（千美元）')
plt.xticks(rotation=45)
plt.show()
```



![按教育水平划分的年收入分布](plots/1177-2.json)



> 箱线图比较不同教育水平的年收入分布。

该图显示了每个教育类别中年收入的中位数（中心线）、四分位距（IQR，即箱体）和潜在异常值（晶须外的点）。我们可以直观地评估这些群体之间年收入的集中趋势和离散程度是否不同。小提琴图（`sns.violinplot`）也可以类似地使用，它能提供关于分布密度形状的信息。

### 分析类别变量间的关系

为了理解两个类别变量之间是否存在关联，我们可以使用频数统计和可视化方法。例如，在我们的客户群中，`Gender` 和 `Education` 级别之间是否存在关系？

#### 交叉制表

交叉制表（或列联表）显示了一个类别变量相对于另一个类别变量的频数分布。Pandas 提供了 `crosstab` 函数来实现这一点。

```python
cross_tab = pd.crosstab(df['Gender'], df['Education'])

print("交叉制表: 性别与教育程度")
print(cross_tab)
```

这会输出一个表格，其中行代表一个类别（例如，性别），列代表另一个类别（例如，教育程度），单元格中包含每个组合的出现次数。您也可以通过在 `pd.crosstab` 中使用 `normalize` 参数 (parameter)来显示比例。

#### 堆叠或分组柱状图

将交叉制表可视化通常能使关系更清晰。分组或堆叠柱状图适用于此。我们可以使用 Seaborn 的 `countplot` 并带上 `hue` 参数。

```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Education', hue='Gender', palette=['#1c7ed6', '#f06595'])
plt.title('按性别划分的教育水平分布')
plt.xlabel('教育水平')
plt.ylabel('数量')
plt.xticks(rotation=45)
plt.legend(title='性别')
plt.show()
```



![按性别划分的教育水平分布](plots/1177-3.json)



> 分组柱状图显示了按性别区分的每个教育水平的客户数量。

此图可以对不同类别之间的计数进行直接比较。例如，我们可以查看我们数据集中拥有“硕士”学位的个体比例在“男性”和“女性”客户之间是否存在差异。堆叠柱状图（通过在 `sns.countplot` 中设置 `dodge=False` 或在 Pandas 绘图中使用 `kind='bar', stacked=True` 实现）可以替代地显示每个教育水平内部的相对比例。

通过应用这些技术，即散点图、相关性分析、比较图、交叉制表和类别柱状图，您可以系统地研究数据集中成对变量之间的关系。这些分析为您的数据底层结构提供了有价值的见解，是 EDA 过程的重要组成部分。

## 参考资料

- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media
  提供了使用 NumPy、Pandas、Matplotlib 和 Seaborn 进行数据处理、分析和可视化的全面指导，包含绘图和统计探索的专门章节。
- [Seaborn: Statistical data visualization](https://seaborn.pydata.org/) — Michael Waskom and the Seaborn development team (2023)
  Seaborn 库的官方用户指南和教程，详细介绍了用于双变量分析的各种统计绘图函数，如散点图、热力图、箱线图和计数图。
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The Pandas Development Team (2023)
  Pandas 库的官方文档，对于数据加载、处理以及计算相关矩阵和交叉表至关重要。
- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  解释了与数据分析和探索性数据分析（EDA）相关的统计概念和技术，包括不同类型的相关性、散点图以及比较组间分布，通常附有 Python 示例。
