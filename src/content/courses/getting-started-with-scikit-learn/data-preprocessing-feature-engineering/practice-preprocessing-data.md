---
course: "getting-started-with-scikit-learn"
chapter: "data-preprocessing-feature-engineering"
lesson: "practice-preprocessing-data"
sourceId: 1888
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/practice-preprocessing-data"
title: "动手实践：数据预处理"
description: "使用 Scikit-learn 对数据集应用缩放、编码和插补技术。"
order: 8
plots: ["plots/1888-0.json"]
sourceHash: "6dc0aa3159f1e472e16776b865c078ba9b83f46c339b4fc0d48a482f8cdbe629"
sourceCorrections: []
---

数据预处理包括数值特征缩放、分类特征编码以及缺失值处理。一个具体实例将演示如何使用Scikit-learn的转换器将这些技术应用到一个小型、有代表性的数据集中。

假设我们有一个包含员工信息的数据集，包括他们的年龄、薪水、部门和经验水平。我们的目标是为潜在的机器学习 (machine learning)模型准备这些数据。

首先，我们使用 Pandas 来创建这个示例数据集：

```python
import pandas as pd
import numpy as np

# 创建示例数据
data = {
    'Age': [25, 45, 30, 55, 22, 38, np.nan, 41, 29, 50],
    'Salary': [50000, 80000, 62000, 110000, 45000, 75000, 58000, np.nan, 60000, 98000],
    'Department': ['HR', 'Engineering', 'Sales', 'Engineering', 'Sales', 'HR', 'Sales', 'Engineering', 'HR', 'Sales'],
    'Experience_Level': ['Entry', 'Senior', 'Mid', 'Senior', 'Entry', 'Mid', 'Mid', 'Senior', 'Entry', 'Senior']
}
df = pd.DataFrame(data)

print("原始 DataFrame:")
print(df)
```

我们的 DataFrame 看起来是这样的：

```
   Age    Salary   Department Experience_Level
0  25.0   50000.0           HR            Entry
1  45.0   80000.0  Engineering           Senior
2  30.0   62000.0        Sales              Mid
3  55.0  110000.0  Engineering           Senior
4  22.0   45000.0        Sales            Entry
5  38.0   75000.0           HR              Mid
6   NaN   58000.0        Sales              Mid
7  41.0       NaN  Engineering           Senior
8  29.0   60000.0           HR            Entry
9  50.0   98000.0        Sales           Senior
```

我们可以立即发现几个需要预处理的地方：

1. **缺失值：** 'Age' 和 'Salary' 列包含 `NaN` 条目。
2. **数值特征：** 'Age' 和 'Salary' 是数值型特征，可能从缩放中受益，特别是如果我们计划使用对特征大小敏感的算法（如 KNN 或正则化 (regularization)回归）。
3. **分类特征：** 'Department' 是名义型（无内在顺序），而 'Experience\_Level' 是序数型（有明确顺序：Entry < Mid < Senior）。它们需要数值编码。

### 处理缺失值

我们首先使用插补来处理缺失值。我们将使用 `SimpleImputer` 用平均年龄填充缺失的 'Age'，并用中位薪水填充缺失的 'Salary'（中位数通常更适用于像薪水这样可能偏斜的数据）。

```python
from sklearn.impute import SimpleImputer

# 用平均值填充 'Age'
mean_imputer = SimpleImputer(strategy='mean')
# 注意：插补器需要2D数组状输入，因此使用 df[['Age']]
df['Age'] = mean_imputer.fit_transform(df[['Age']])

# 用中位数填充 'Salary'
median_imputer = SimpleImputer(strategy='median')
df['Salary'] = median_imputer.fit_transform(df[['Salary']])

print("\n插补后的 DataFrame:")
print(df)
```

输出显示 `NaN` 值已被替换：

```
DataFrame after Imputation:
        Age    Salary   Department Experience_Level
0  25.000000   50000.0           HR            Entry
1  45.000000   80000.0  Engineering           Senior
2  30.000000   62000.0        Sales              Mid
3  55.000000  110000.0  Engineering           Senior
4  22.000000   45000.0        Sales            Entry
5  38.000000   75000.0           HR              Mid
6  37.222222   58000.0        Sales              Mid # 插补后的年龄（均值）
7  41.000000   68500.0  Engineering           Senior # 插补后的薪水（中位数）
8  29.000000   60000.0           HR            Entry
9  50.000000   98000.0        Sales           Senior
```

注意 `fit_transform` 的用法。`fit` 步骤从列中的非缺失值计算出统计量（均值或中位数），而 `transform` 步骤则使用计算出的统计量填充缺失的条目。

### 缩放数值特征

现在，我们来缩放 'Age' 和 'Salary' 列。我们将使用 `StandardScaler`，它将数据转换为零均值和单位方差（$z = (x - \mu) / \sigma$）。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_cols = ['Age', 'Salary']

# 拟合并转换数值列
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\n数值特征缩放后的 DataFrame:")
print(df[numerical_cols].head()) # 为简洁起见，只显示缩放后的列
```

输出将显示缩放后的值：

```
DataFrame after Scaling Numerical Features:
        Age    Salary
0 -1.213538 -1.007992
1  0.770603  0.439769
2 -0.717503 -0.475179
3  1.762674  1.887530
4 -1.510762 -1.248957
```

这些值现在表示每个特征与均值的标准差。例如，第一位员工的年龄在插补后比平均年龄低约 1.21 个标准差。

让我们可视化 'Salary' 缩放前后的分布：

````python
# 存储缩放前的原始薪水用于对比图
original_salary = median_imputer.transform(pd.DataFrame(data)['Salary']) # 获取插补后的原始规模薪水

# 创建图表数据
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(x=original_salary.flatten(), name='原始薪水', marker_color='#1f77b4', opacity=0.75))
fig.add_trace(go.Histogram(x=df['Salary'], name='缩放后的薪水', marker_color='#ff7f0e', opacity=0.75))

# 更新布局以提高清晰度
fig.update_layout(
    title_text='薪水标准化前后的分布',
    xaxis_title_text='值',
    yaxis_title_text='计数',
    barmode='overlay', # 叠加直方图
    legend_title_text='特征'
)
fig.update_traces(opacity=0.7) # 调整透明度

# 转换为 JSON 字符串以供显示
import plotly.io as pio
plotly_json = pio.to_json(fig)
print(f"\n```plotly\n{plotly_json}\n```")
````



![薪水标准化前后的分布](plots/1888-0.json)



> 直方图显示了薪水的分布。蓝色条表示原始薪水（插补后），橙色条表示标准化后的薪水。注意分布的形状得以保留，但 x 轴上的比例显著变化，缩放后的版本以零为中心。

### 编码分类特征

接下来，我们处理分类列。

**名义特征：部门**
'Department' 没有内在顺序，因此独热编码是合适的。它为每个类别创建新的二元列。

```python
from sklearn.preprocessing import OneHotEncoder

# 选择分类列
cat_col = ['Department']
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # 为可读性生成密集数组

# 拟合并转换
one_hot_encoded = one_hot_encoder.fit_transform(df[cat_col])

# 创建带有有意义列名的新 DataFrame
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(cat_col))

print("\n独热编码后的 'Department'（前 5 行）：")
print(one_hot_df.head())

# 我们可以删除原始的 'Department' 列并连接新的列
df = df.drop(cat_col, axis=1)
df = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
```

输出显示了新的二元列：

```
One-Hot Encoded 'Department' (first 5 rows):
   Department_Engineering  Department_HR  Department_Sales
0                     0.0            1.0               0.0
1                     1.0            0.0               0.0
2                     0.0            0.0               1.0
3                     1.0            0.0               0.0
4                     0.0            0.0               1.0
```

**序数特征：经验水平**
'Experience\_Level' 有一个顺序（'Entry' < 'Mid' < 'Senior'）。我们可以使用 `OrdinalEncoder` 并指定顺序。

```python
from sklearn.preprocessing import OrdinalEncoder

# 定义期望的顺序
exp_order = ['Entry', 'Mid', 'Senior']
ordinal_encoder = OrdinalEncoder(categories=[exp_order]) # 以列表形式传入顺序

# 拟合并转换
df['Experience_Level'] = ordinal_encoder.fit_transform(df[['Experience_Level']])

print("\n序数编码 'Experience_Level' 后的 DataFrame:")
print(df.head()) # 现在显示完整的 DataFrame
```

经过所有预处理步骤后的最终 DataFrame 看起来是这样的（显示前 5 行）：

```
DataFrame after Ordinal Encoding 'Experience_Level':
        Age    Salary  Experience_Level  Department_Engineering  Department_HR  Department_Sales
0 -1.213538 -1.007992               0.0                     0.0            1.0               0.0
1  0.770603  0.439769               2.0                     1.0            0.0               0.0
2 -0.717503 -0.475179               1.0                     0.0            0.0               1.0
3  1.762674  1.887530               2.0                     1.0            0.0               0.0
4 -1.510762 -1.248957               0.0                     0.0            0.0               1.0
```

我们的数据现在已经完全数值化、已缩放并且没有缺失值。它现在以一种更好的格式用于许多 Scikit-learn 机器学习 (machine learning)模型的输入。

这个动手练习演示了如何使用 Scikit-learn 的转换器应用单个预处理步骤。请记住，在实际机器学习工作流程中，仅在训练数据上拟合转换器以避免数据泄露非常重要，然后使用*相同*的已拟合转换器来转换您的训练和测试数据。在第六章中，您将学习如何使用 Scikit-learn Pipelines 将这些步骤优雅而安全地串联起来。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2023)
  解释了使用 Scikit-learn 转换器进行数据预处理的各种方法，包括缺失值处理、特征缩放和编码。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  涵盖了使用 Scikit-learn 示例进行数据预处理和特征工程的实用方法（第3版）。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEruhS14FXrKmj-NMEe_lmcHcISY0CTlxhDheognY1d5yZ5x_SaRtyf4bnZZt63GDLGD5kgtceCGx182ABUCtMLJc8zy6W8pklLYInnsb4A4VBJu7UWISEUL9-vdmzgRJGAEj182ueMxZns-cXse8D0JfOE5B68K3tQGCf4HFUVLwh0xC_Unp5VSHUwFv3bGe5sClVFAJcSzHPFIiM=) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  提供了一本特征工程指南，详细介绍了缺失值处理、特征缩放和编码。
