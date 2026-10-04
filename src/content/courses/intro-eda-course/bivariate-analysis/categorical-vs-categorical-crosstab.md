---
course: "intro-eda-course"
chapter: "bivariate-analysis"
lesson: "categorical-vs-categorical-crosstab"
sourceId: 1172
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-4-bivariate-analysis/categorical-vs-categorical-crosstab"
title: "分类变量与分类变量：交叉制表"
description: "使用列联表（交叉制表）来分析两个分类变量之间的关系。"
order: 5
plots: []
sourceHash: "de5aa1d1c23eab7d77c1c82db844c721eba5859950597b2339404aed86a10935"
sourceCorrections: []
---

在分析两个变量之间的关系时，所采用的方法很大程度上取决于所涉及的变量类型。我们已经看到散点图和相关性如何帮助理解数值变量与数值变量之间的关系。现在，我们来考察如何研究两个*分类*变量之间的关联。简单的散点图不适合这里，因为绘制离散类别不能显示太多规律。相反，我们依赖于对类别组合内部的频率计数。

对此，主要工具是**交叉制表**，也常被称为**列联表**。交叉制表本质上是一个表格，它汇总了两个分类变量的每个类别组合的观测频率（计数）。它使我们能够查看一个分类变量的分布如何随另一个变量的不同层面而变化。

### 使用 Pandas `crosstab`

探索两个分类变量之间的关联是数据分析中的一个任务。为了实现此分析，Pandas 库提供了一个直观的函数 `pd.crosstab()`。其基本语法是传入你想要比较的两个分类 Series（列）。

```python
import pandas as pd

# 假设 'df' 是你的 DataFrame
# 'category_col_1' 和 'category_col_2' 是你的分类列的名称
contingency_table = pd.crosstab(df['category_col_1'], df['category_col_2'])

print(contingency_table)
```

让我们用一个例子来说明。设想一个数据集 `survey_df`，包含列 `satisfaction_level`（'低', '中', '高'）和 `region`（'北', '南', '东', '西'）。

```python
# 样本数据创建（用于说明）
data = {'satisfaction_level': ['Low', 'Medium', 'High', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Medium'],
        'region': ['North', 'South', 'North', 'East', 'West', 'South', 'West', 'North', 'East', 'South']}
survey_df = pd.DataFrame(data)

# 创建交叉制表
cross_tab_results = pd.crosstab(survey_df['satisfaction_level'], survey_df['region'])

print(cross_tab_results)
```

运行此代码将产生类似于下表的输出：

```
region              East  North  South  West
satisfaction_level                          
High                   1      1      1     0
Low                    0      2      0     1
Medium                 1      0      2     1
```

**解释原始计数：**

此表中的每个单元格显示属于该特定类别组合的观测数量。例如：

- '东'地区有1个“高”满意度的观测值。
- '北'地区有2个“低”满意度的观测值。
- '西'地区有0个“高”满意度的观测值。

The表格也隐含地包含行总计（例如，总“高”满意度观测值）和列总计（例如，总“北”地区观测值），尽管 `pd.crosstab` 默认不显示它们。如果需要，你可以使用 `margins=True` 参数 (parameter)添加这些总计。

### 正则化 (regularization)交叉制表以更好比较

尽管原始计数提供信息，但比较总计数不同的类别之间的分布可能很困难。例如，如果“北”地区的总受访者数量远多于“西”地区，仅仅比较“高”满意度的原始计数可能会产生误导。

为解决此问题，我们可以将表格归一化 (normalization)以显示比例或百分比。`pd.crosstab()` 中的 `normalize` 参数 (parameter)控制此行为：

- `normalize='index'`: 计算每行的比例（行总和为1）。这显示了*列变量*在*行变量*的每个类别中的分布情况。
- `normalize='columns'`: 计算每列的比例（列总和为1）。这显示了*行变量*在*列变量*的每个类别中的分布情况。
- `normalize='all'`: 根据总计计算比例（所有单元格值总和为1）。这显示了总数据集中落入每个单元格组合的比例。

**范例：按行归一化（'index'）**

让我们看看满意度水平在各个地区如何分布：

```python
# 按行归一化（显示每个满意度水平的地区分布）
row_normalized_tab = pd.crosstab(survey_df['satisfaction_level'], survey_df['region'], normalize='index')

# 乘以100并四舍五入以查看百分比（可选）
print((row_normalized_tab * 100).round(1))
```

输出可能如下所示：

```
region              East  North  South  West
satisfaction_level                          
High                33.3   33.3   33.3   0.0
Low                  0.0   66.7    0.0  33.3
Medium              25.0    0.0   50.0  25.0
```

> 解释：在“高”满意度的人群中，33.3%在东部，33.3%在北部，33.3%在南部。在“低”满意度的人群中，66.7%在北部，33.3%在西部。

**范例：按列归一化（'columns'）**

现在，让我们看看每个区域内的满意度分布：

```python
# 按列归一化（显示每个地区内的满意度分布）
col_normalized_tab = pd.crosstab(survey_df['satisfaction_level'], survey_df['region'], normalize='columns')

# 乘以100并四舍五入以查看百分比（可选）
print((col_normalized_tab * 100).round(1))
```

输出可能如下所示：

```
region              East  North  South  West
satisfaction_level                          
High                50.0   33.3   33.3   0.0
Low                  0.0   66.7    0.0  50.0
Medium              50.0    0.0   66.7  50.0
```

> 解释：在“东”地区，50%的人拥有“高”满意度，50%的人拥有“中等”满意度。在“北”地区，33.3%的人拥有“高”满意度，66.7%的人拥有“低”满意度。

### 查找关联

通过查看这些表格（包括原始计数和归一化 (normalization)版本），你可以开始识别潜在关联。提出以下问题：

- 变量A的某个特定类别是否比偶然情况下更频繁地与变量B的特定类别一起出现？
- 变量A的分布在变量B的不同类别之间看起来是否明显不同？

在我们的例子中，归一化表格表明了一些潜在模式：“低”满意度似乎更多地集中在“北”和“西”地区，而“高”满意度在“西”地区似乎缺失。“中等”满意度在“南”地区最为普遍。

请记住，交叉制表显示了你在特定样本数据中观察到的模式。建立统计显著性通常需要正式的假设检验（如卡方独立性检验），这些通常在统计建模中更详细地介绍，而不是在初始的EDA中。对于分析，识别这些潜在关系是主要目标。

交叉制表提供了紧凑的数值汇总。为了使这些模式在视觉上更明显，通常使用堆叠条形图或分组条形图等方法，我们将在接下来讨论。

## 参考资料

- [pandas.crosstab](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html) — pandas development team (2024)
  Pandas `crosstab` 函数的官方文档，详细说明其用法、参数和创建列联表的示例。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  学习使用 Pandas 进行数据处理的常见资料，其中包含列联表和数据聚合的实际示例。
- [OpenIntro Statistics](https://www.openintro.org/book/os/) — David Diez, Mine Çetinkaya-Rundel, and Christopher Barr (2023)
  Publisher: OpenIntro; Pages: Chapter 3: Foundations for Inference
  为统计分析中理解列联表、边际分布和条件分布提供了概念基础。
