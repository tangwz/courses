---
course: "probability-statistics-essentials-ml"
chapter: "descriptive-statistics-datasets"
lesson: "practice-summarizing-dataset"
sourceId: 1417
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-3-descriptive-statistics-datasets/practice-summarizing-dataset"
title: "实践：总结数据集"
description: "应用描述性统计方法，使用 Python 分析和总结一个示例数据集。"
order: 9
plots: ["plots/1417-0.json", "plots/1417-1.json", "plots/1417-2.json", "plots/1417-3.json"]
sourceHash: "5e50cbb9f4f14128b215818ffccf32d50f0d06519faf1149e285950461f47013"
sourceCorrections: []
---

使用 Python 将描述性统计应用于示例数据集，以获得初步认识，之后再进行更复杂的模型构建。Pandas 库在 Python 中进行数据处理和分析不可或缺，将用于执行数据总结。

### 环境设置

首先，请确保您已安装所需的库。如果没有，通常可以使用 pip 安装它们：

```bash
pip install pandas numpy scipy matplotlib seaborn plotly
```

现在，让我们导入将在 Python 脚本或 Jupyter Notebook 中使用的库：

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
```

### 加载和检查数据

对于本次练习，我们将使用一个模拟数据集，它代表了一些测量值。可以想象，这些可能是传感器读数、用户活动指标，或您可能遇到的任何一组观测数据。让我们直接使用 Pandas 和 NumPy 创建这个数据集。

```python
# 设置随机种子以保证结果可重现
np.random.seed(42)

# 生成数据
n_samples = 150
data = {
    'feature_A': np.random.normal(loc=50, scale=15, size=n_samples),
    'feature_B': np.random.gamma(shape=2, scale=10, size=n_samples) + 20, # 偏态分布
    'feature_C': 0.7 * np.random.normal(loc=50, scale=15, size=n_samples) + np.random.normal(loc=0, scale=5, size=n_samples) + 10, # 与 feature_A 相关
    'category': np.random.choice(['Type X', 'Type Y', 'Type Z'], size=n_samples, p=[0.4, 0.35, 0.25])
}
df = pd.DataFrame(data)

# 确保典型特征没有负值
df['feature_A'] = df['feature_A'].clip(lower=0)
df['feature_B'] = df['feature_B'].clip(lower=0)
df['feature_C'] = df['feature_C'].clip(lower=0)

print("数据集维度:", df.shape)
print("\n前5行:")
print(df.head())
print("\n数据类型和非空计数:")
df.info()
```

`df.head()` 的输出让我们快速查看了前几行，而 `df.info()` 告诉我们条目数量、列名、每列非空值的计数以及每列的数据类型。在这种情况下，我们看到 150 个条目和 4 列，没有缺失值。`feature_A`、`feature_B` 和 `feature_C` 是数值型 (`float64`)，而 `category` 是类别型 (`object`)。

### 整体概括统计

Pandas 中的 `.describe()` 方法非常适合快速获取数值列的统计概括。

```python
# 获取数值列的概括统计
summary_stats = df.describe()
print("\n概括统计:")
print(summary_stats)
```

此输出提供了我们讨论过的几个重要统计量：

- `count`：非缺失观测值的数量。
- `mean`：平均值。
- `std`：标准差，衡量离散程度。
- `min`：最小值。
- `25%`：第一四分位数 (Q1)。
- `50%`：中位数（第二四分位数，Q2）。
- `75%`：第三四分位数 (Q3)。
- `max`：最大值。

查看输出，我们已经可以得到一些观察结果：

- `feature_A` 的平均值约为 50，接近其中位数，表明分布相对对称。其标准差约为 14。
- `feature_B` 的平均值（约 40）明显大于其中位数（约 36）。这表明是右偏分布。与 `feature_A` 相比，其范围（最大值 - 最小值）相当大。
- `feature_C` 的特征与 `feature_A` 有些相似，其平均值接近其中位数。

### 计算特定度量

尽管 `.describe()` 很实用，但有时我们需要特定的统计量，或者为了清晰起见，或对于非数值列（如众数），我们希望单独计算它们。

#### 集中趋势

让我们计算看起来偏态分布的 `feature_B` 的平均值、中位数和众数。

```python
# feature_B 的集中趋势
mean_b = df['feature_B'].mean()
median_b = df['feature_B'].median()
mode_b = df['feature_B'].mode() # 如果有多个值具有相同的最高频率，众数可以返回多个值

print(f"\n特征 B - 平均值: {mean_b:.2f}")
print(f"特征 B - 中位数: {median_b:.2f}")
print(f"特征 B - 众数: {mode_b.tolist()}") # 将众数显示为列表

# 类别列的众数
mode_category = df['category'].mode()
print(f"\n类别 - 众数: {mode_category.tolist()}")
```

如预料，由于右偏，`feature_B` 的平均值被拉高至高于中位数。众数代表最常出现的值。对于类别特征，'Type X' 是最常见的类别。

#### 离散程度

让我们查看 `feature_A` 和 `feature_B` 的离散程度。

```python
# feature_A 的离散程度
variance_a = df['feature_A'].var()
std_dev_a = df['feature_A'].std()
range_a = df['feature_A'].max() - df['feature_A'].min()
iqr_a = df['feature_A'].quantile(0.75) - df['feature_A'].quantile(0.25)

print(f"\n特征 A - 方差: {variance_a:.2f}")
print(f"特征 A - 标准差: {std_dev_a:.2f}")
print(f"特征 A - 范围: {range_a:.2f}")
print(f"特征 A - 四分位距 (IQR): {iqr_a:.2f}")

# feature_B 的离散程度
variance_b = df['feature_B'].var()
std_dev_b = df['feature_B'].std()
range_b = df['feature_B'].max() - df['feature_B'].min()
iqr_b = df['feature_B'].quantile(0.75) - df['feature_B'].quantile(0.25)

print(f"\n特征 B - 方差: {variance_b:.2f}")
print(f"特征 B - 标准差: {std_dev_b:.2f}")
print(f"特征 B - 范围: {range_b:.2f}")
print(f"特征 B - 四分位距 (IQR): {iqr_b:.2f}")
```

比较标准差（A 为 $14.06$，B 为 $14.49$）并不能立即显示形状上的差异，但比较范围（$64.65$ vs. $76.56$）和 IQR（$17.50$ vs. $16.26$）开始表明 `feature_B` 在一侧（右侧，如偏度所示）有更多极端值。四分位距通常比范围或标准差对异常值更具抵抗力。

#### 形状：偏度和峰度

让我们使用偏度和峰度来量化 (quantization)形状。我们可以使用 `scipy.stats` 模块。

```python
# feature_A 的形状
skew_a = stats.skew(df['feature_A'])
kurt_a = stats.kurtosis(df['feature_A']) # Fisher 定义（正态分布 == 0）

print(f"\n特征 A - 偏度: {skew_a:.2f}")
print(f"特征 A - 峰度: {kurt_a:.2f}")

# feature_B 的形状
skew_b = stats.skew(df['feature_B'])
kurt_b = stats.kurtosis(df['feature_B'])

print(f"\n特征 B - 偏度: {skew_b:.2f}")
print(f"特征 B - 峰度: {kurt_b:.2f}")
```

- `feature_A` 的偏度接近 0 (-0.12)，这证实了其相对对称性。峰度也接近 0 (-0.22)，表明其峰值与正态分布相似。
- `feature_B` 具有正偏度 (1.02)，证实了我们之前观察到的右偏（尾部延伸向右）。正峰度 (1.15) 表明与正态分布相比，其尾部略重且峰值更尖。

#### 相关性分析

现在，让我们检查数值特征之间的线性关系。

```python
# 计算相关矩阵
correlation_matrix = df[['feature_A', 'feature_B', 'feature_C']].corr()

print("\n相关矩阵:")
print(correlation_matrix)
```

相关矩阵显示了每对变量之间的皮尔逊相关系数。

- 对角线元素始终为 1（变量与自身的相关性）。
- 我们看到 `feature_A` 和 `feature_C` 之间存在很强的正相关（约 0.70），这与我们的数据生成过程一致。
- `feature_B` 与 `feature_A`（约 0.05）和 `feature_C`（约 0.09）的相关性较弱。

请记住，相关性衡量的是*线性*关联。低相关性不一定表示不存在任何关系，只是不存在线性关系。并且，非常重要的一点是，相关性不代表因果关系。即使 A 和 C 相关，我们也不能仅凭此值就断定 A *导致* C 或反之。

### 概括的可视化

数值概括很有效，但可视化通常能提供更直观的理解。

#### 直方图

直方图有助于显示单个数值变量的分布。

```python
# feature_A 的直方图
fig_hist_a = px.histogram(df, x='feature_A', nbins=20, title='Distribution of Feature A',
                         color_discrete_sequence=['#339af0']) # 蓝色
fig_hist_a.update_layout(bargap=0.1)
fig_hist_a.show()

# feature_B 的直方图
fig_hist_b = px.histogram(df, x='feature_B', nbins=20, title='Distribution of Feature B',
                         color_discrete_sequence=['#20c997']) # 青色
fig_hist_b.update_layout(bargap=0.1)
fig_hist_b.show()
```



![特征 A 的分布](plots/1417-0.json)



> `feature_A` 的直方图显示了一个大致呈钟形、对称的分布，中心位于 50 附近。



![特征 B 的分布](plots/1417-1.json)



> `feature_B` 的直方图清晰地显示出右偏，大多数值集中在左侧，尾部延伸向更高值。

#### 箱线图

箱线图非常适合比较分布或概括单个分布的四分位数、中位数和潜在异常值。

```python
# 所有数值特征的箱线图
fig_box = px.box(df, y=['feature_A', 'feature_B', 'feature_C'], title='Box Plots of Numerical Features',
                 color_discrete_sequence=['#339af0', '#20c997', '#7048e8']) # 蓝色, 青色, 紫罗兰色
fig_box.show()
```



![数值特征的箱线图](plots/1417-2.json)



> 箱线图直观地比较了特征的中位数（箱内线）、四分位距（箱体本身）和范围（触须）。异常值可能被绘制为单独的点。请注意 `feature_B` 较高的中位数和更长的上触须/异常值，这表明存在右偏。

#### 散点图

散点图有助于显示两个数值变量之间的关系。让我们绘制高度相关的 `feature_A` 和 `feature_C`。

```python
# feature_A vs feature_C 散点图
fig_scatter = px.scatter(df, x='feature_A', y='feature_C', title='Feature A vs Feature C',
                        trendline='ols', # 添加普通最小二乘回归线
                        color_discrete_sequence=['#be4bdb']) # 葡萄紫
fig_scatter.show()
```



![特征 A vs 特征 C](plots/1417-3.json)



> 散点图显示 `feature_A` 和 `feature_C` 之间存在正线性趋势，证实了之前计算的相关系数。点集中在回归线附近。

### 结论

在本实践部分，我们将本章的描述性统计原理应用于一个示例数据集。使用 Pandas、SciPy 和 Plotly，我们计算了集中趋势、离散程度和形状的度量，计算了相关性，并创建了直方图、箱线图和散点图等可视化。

总结数据集的过程是任何数据分析或机器学习 (machine learning)项目中的重要的第一步。它有助于您：

- 理解变量的基本属性和分布。
- 找出潜在问题，如偏度或异常值。
- 发现变量之间的关系。
- 有效地传达数据的重要特征。

有了这些概括性认识，您可以更好地准备选择合适的数据预处理方法，选择适合的机器学习模型，并解释后续分析的结果。

## 参考资料

- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas创建者撰写的基础书籍，详细介绍了如何使用Python、Pandas和NumPy进行数据处理、分析和总结。
- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce and Andrew Bruce (2020)
  Publisher: O'Reilly Media
  本书涵盖了数据科学中常用的统计概念，包括描述性统计，为实践者量身定制。
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The Pandas Development Team (2024)
  Pandas库的官方文档，提供了关于数据结构、函数和方法（用于数据加载、检查和计算描述性统计）的全面指南。
- [SciPy Reference Guide: scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) — The SciPy Developers (2024)
  SciPy统计函数的官方文档，包括用于计算偏度和峰度的方法，以量化数据分布形状。
- [Plotly Python Open-Source Graphing Library](https://plotly.com/python/) — Plotly Technologies Inc. (2024)
  使用Plotly在Python中创建交互式可视化的官方文档，对于通过视觉方式理解数据分布和关系至关重要。
