---
course: "intro-feature-engineering"
chapter: "handling-missing-data"
lesson: "practice-imputing-missing-data"
sourceId: 1322
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data/practice-imputing-missing-data"
title: "动手实践：填充缺失数据"
description: "使用Scikit-learn和Pandas将各种填充技术应用于数据集。"
order: 8
plots: ["plots/1322-0.json"]
sourceHash: "e15dbbf46d989a137334d534bf6a381d06193c2f0f1e57614ff5fbaa6bbab051"
sourceCorrections: []
---

使用Pandas和Scikit-learn在一个样本数据集上实现各种填充方法，提供了使用Python数据科学工具集的实践经验。这些方法的应用对于为机器学习 (machine learning)模型准备数据很重要。

首先，让我们通过导入必要的库并创建一个包含缺失值的样本DataFrame来设置我们的环境。

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer # 启用IterativeImputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个包含缺失值的样本DataFrame
data = {
    'Age': [25, 30, np.nan, 35, 40, 45, 50, np.nan, 55],
    'Salary': [50000, 60000, 75000, np.nan, 80000, 90000, 110000, 65000, np.nan],
    'Experience': [1, 5, 3, 10, 15, 20, np.nan, 8, 30],
    'Department': ['HR', 'IT', 'Finance', 'IT', np.nan, 'HR', 'Finance', 'IT', 'Finance'],
    'Rating': [3.5, 4.0, 4.5, 3.0, np.nan, 4.2, 3.8, 4.8, 3.9]
}
df = pd.DataFrame(data)

print("包含缺失值的原始DataFrame：")
print(df)
print("\n每列的缺失值数量：")
print(df.isnull().sum())
```

我们的样本`df`在数值型（`Age`、`Salary`、`Experience`、`Rating`）和类别型（`Department`）列中包含缺失值（`np.nan`）。

### 简单填充策略

如前所述，简单填充涉及使用基本统计量替换缺失值。Scikit-learn的`SimpleImputer`是一个方便的工具。

#### 均值和中位数填充（数值型特征）

我们通常将均值用于正态分布数据，将中位数用于偏态数据或包含异常值的数据。让我们对不同列应用这两种方法进行演示。

```python
# 使用均值填充'Age'
mean_imputer = SimpleImputer(strategy='mean')
# 需要重塑，因为SimpleImputer期望2D数组
df['Age_mean_imputed'] = mean_imputer.fit_transform(df[['Age']])

# 使用中位数填充'Salary'（通常更适合薪资数据）
median_imputer = SimpleImputer(strategy='median')
df['Salary_median_imputed'] = median_imputer.fit_transform(df[['Salary']])

# 使用中位数同时填充'Experience'和'Rating'
num_cols_median = ['Experience', 'Rating']
median_imputer_multi = SimpleImputer(strategy='median')
# 对原始列进行拟合
median_imputer_multi.fit(df[num_cols_median])
# 转换并创建新列
df[['Experience_median_imputed', 'Rating_median_imputed']] = median_imputer_multi.transform(df[num_cols_median])

print("\n均值/中位数填充后的DataFrame：")
print(df[['Age', 'Age_mean_imputed', 'Salary', 'Salary_median_imputed', 'Experience', 'Experience_median_imputed', 'Rating', 'Rating_median_imputed']].head())
```

观察原始列中的`NaN`值是如何在对应的新列中被计算出的均值或中位数替换的。

#### 众数填充（类别型特征）

对于像`Department`这样的类别型特征，最常见的值（众数）常用于填充。

```python
# 使用众数填充'Department'
mode_imputer = SimpleImputer(strategy='most_frequent')
df['Department_mode_imputed'] = mode_imputer.fit_transform(df[['Department']])

print("\n众数填充后的DataFrame：")
print(df[['Department', 'Department_mode_imputed']].head(6)) # 显示包含原始NaN的行
```

缺失的部门被列中最常见的部门填充。

### 创建缺失值指示器

有时，一个值缺失的事实本身就包含信息。我们可以使用指示器特征来捕获此信息。`SimpleImputer`可以自动完成，或者我们可以使用`MissingIndicator`。

```python
# 使用SimpleImputer并设置add_indicator=True
median_imputer_indicator = SimpleImputer(strategy='median', add_indicator=True)
imputed_with_indicator = median_imputer_indicator.fit_transform(df[['Salary']]) # 使用原始Salary

# 输出是一个NumPy数组：第0列是填充后的数据，第1列是指示器
df['Salary_median_imputed_si'] = imputed_with_indicator[:, 0]
df['Salary_missing_indicator_si'] = imputed_with_indicator[:, 1].astype(int) # 将布尔值转换为整数

# 直接使用MissingIndicator
indicator = MissingIndicator(features='all') # 检查所有特征
missing_indicators = indicator.fit_transform(df[['Age', 'Salary', 'Experience', 'Department', 'Rating']])
# 转换为DataFrame以便清晰显示
indicator_df = pd.DataFrame(missing_indicators, columns=[f'{col}_missing' for col in df.columns if df[col].isnull().any()], index=df.index)

# （可选，用于查看）与原始df合并
df_with_indicators = pd.concat([df, indicator_df], axis=1)

print("\n包含Salary填充和指示器（来自SimpleImputer）的DataFrame：")
print(df[['Salary', 'Salary_median_imputed_si', 'Salary_missing_indicator_si']].head())

print("\n显示所有生成的缺失指示器（来自MissingIndicator）的DataFrame：")
print(df_with_indicators[['Age', 'Age_missing', 'Salary', 'Salary_missing', 'Experience', 'Experience_missing', 'Department', 'Department_missing', 'Rating', 'Rating_missing']].head(6))
```

这些二元指示器列明确地指示了数据最初缺失的位置，这可能对某些模型有用。

### 多变量填充技术

多变量方法使用其他特征的信息来估计缺失值，可能带来比简单策略更准确的填充结果。

#### KNN填充器

`KNNImputer`使用训练集中$k$个最近邻的平均值来填充缺失值。邻居是根据*非*缺失的特征来识别的。这要求用于填充的所有特征都是数值型的。我们首先需要对`Department`列进行编码（例如，使用独热编码，这将在下一章中介绍）或将其排除。为简化起见，这里我们仅对数值特征进行联合填充。

```python
from sklearn.preprocessing import MinMaxScaler

# KNNImputer对特征缩放敏感，因此请先进行缩放
numerical_cols = ['Age', 'Salary', 'Experience', 'Rating']
df_numerical = df[numerical_cols].copy()

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=numerical_cols)

# 应用KNNImputer
knn_imputer = KNNImputer(n_neighbors=3) # 使用3个邻居
df_knn_imputed_scaled = pd.DataFrame(knn_imputer.fit_transform(df_scaled), columns=numerical_cols)

# 逆变换以将数据恢复到原始比例
df_knn_imputed = pd.DataFrame(scaler.inverse_transform(df_knn_imputed_scaled), columns=numerical_cols)

# （可选）将填充后的列添加回原始df以进行比较
df['Age_knn_imputed'] = df_knn_imputed['Age']
df['Salary_knn_imputed'] = df_knn_imputed['Salary']
df['Experience_knn_imputed'] = df_knn_imputed['Experience']
df['Rating_knn_imputed'] = df_knn_imputed['Rating']

print("\nKNN填充后的DataFrame（并排显示原始数据和填充数据）：")
# 显示原始数据缺失的行，以查看填充值
missing_rows_idx = df[df[numerical_cols].isnull().any(axis=1)].index
print(df.loc[missing_rows_idx, ['Age', 'Age_knn_imputed', 'Salary', 'Salary_knn_imputed', 'Experience', 'Experience_knn_imputed', 'Rating', 'Rating_knn_imputed']])
```

请注意，KNN填充需要仔细考虑邻居数量（$k$）和所使用的距离度量。通常建议事先对特征进行缩放。

#### 迭代填充器

`IterativeImputer`将每个包含缺失值的特征建模为其他特征的函数，并使用迭代方法估计缺失值。它循环预测每个特征的缺失值，基于所有其他特征，直到估计值稳定下来。

```python
# IterativeImputer通常也更适合缩放后的数据
# 我们可以重用KNN示例中的缩放数据

iterative_imputer = IterativeImputer(max_iter=10, random_state=0) # max_iter控制迭代次数
df_iterative_imputed_scaled = pd.DataFrame(iterative_imputer.fit_transform(df_scaled), columns=numerical_cols)

# 逆变换
df_iterative_imputed = pd.DataFrame(scaler.inverse_transform(df_iterative_imputed_scaled), columns=numerical_cols)

# 将填充后的列添加回原始df
df['Age_iterative_imputed'] = df_iterative_imputed['Age']
df['Salary_iterative_imputed'] = df_iterative_imputed['Salary']
df['Experience_iterative_imputed'] = df_iterative_imputed['Experience']
df['Rating_iterative_imputed'] = df_iterative_imputed['Rating']

print("\n迭代填充后的DataFrame（并排显示原始数据和填充数据）：")
print(df.loc[missing_rows_idx, ['Age', 'Age_iterative_imputed', 'Salary', 'Salary_iterative_imputed', 'Experience', 'Experience_iterative_imputed', 'Rating', 'Rating_iterative_imputed']])
```

`IterativeImputer`通常更复杂，但计算量可能比`KNNImputer`更大。

### 比较填充方法

填充方法的选择取决于数据特性、缺失机制（如果已知）以及机器学习 (machine learning)模型的具体要求。

让我们可视化'Salary'特征在不同填充前后的分布，以查看影响。

```python
# 准备绘图数据
salary_data = pd.DataFrame({
    'Original': df['Salary'],
    'Median Imputed': df['Salary_median_imputed'],
    'KNN Imputed': df['Salary_knn_imputed'],
    'Iterative Imputed': df['Salary_iterative_imputed']
})

# 熔化DataFrame以便Seaborn绘图
salary_melted = salary_data.melt(var_name='Imputation Method', value_name='Salary')

# 创建图表
plt.figure(figsize=(12, 6))
sns.kdeplot(data=salary_melted, x='Salary', hue='Imputation Method', fill=True, common_norm=False, palette="viridis")
plt.title('不同填充方法后的薪资分布')
plt.xlabel('薪资')
plt.ylabel('密度')
plt.show()
```



![薪资分布比较](plots/1322-0.json)



> 使用小提琴图比较不同填充方法后的薪资分布。原始数据包含空值。中位数填充在 median 值处添加点。KNN和迭代填充基于其他特征提供了可能更详细的估计值。

- **简单填充（均值/中位数/众数）**：快速，易于实现。不利用特征之间的关系。可能扭曲方差和相关性。均值对异常值敏感，中位数更可靠。众数适用于类别型数据。
- **指示器特征**：保留关于缺失的信息。可以与任何填充方法一起使用。
- **KNN填充器**：考虑特征关系。比简单方法计算成本更高。对缩放和 $k$ 的选择敏感。需要数值数据。
- **迭代填充器**：通过建模特征通常提供准确的填充。如果底层估计器支持，可以处理不同数据类型（尽管默认的BayesianRidge适用于数值型数据）。计算量可能很大。对缩放敏感。

最佳方法通常涉及实验和评估对下游模型性能的影响。请考虑填充准确性、计算成本以及数据集中可能引入的失真之间的权衡。请记住，只在训练数据上拟合填充器，并使用拟合好的填充器转换训练和测试数据集，以防止数据泄露。这通常最好使用Scikit-learn管道来管理。

## 参考资料

- [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) — Scikit-learn Developers (2024)
  Scikit-learn 缺失值插补转换器及其用法的官方指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  涵盖数据预处理（包括使用 Python 和 Scikit-learn 的各种插补策略）的实用指南。
- [Flexible Imputation of Missing Data, Second Edition](https://stefvanbuuren.name/fimd/) — Stef van Buuren (2018)
  Publisher: CRC Press; DOI: [10.1201/9780429052822](https://doi.org/10.1201/9780429052822)
  关于多重插补理论与实践的权威著作，适用于 MICE 等方法。
