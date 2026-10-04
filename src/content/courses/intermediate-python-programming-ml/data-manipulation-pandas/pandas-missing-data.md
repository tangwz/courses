---
course: "intermediate-python-programming-ml"
chapter: "data-manipulation-pandas"
lesson: "pandas-missing-data"
sourceId: 2148
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/pandas-missing-data"
title: "处理缺失数据"
description: "学习在DataFrame中检测、移除或填充缺失值（NaN）的策略。"
order: 4
plots: []
sourceHash: "17decbcc0fa39a53c6daaeb0f078b7ec73b04e8537223d676c255cc8edcc4c13"
sourceCorrections: []
---

数据集通常不完整。数据可能因未收集、处理过程中丢失或对某些观测值不适用而缺失。Pandas通常使用特殊的浮点值`NaN`（非数字）来表示这些缺失值。忽视缺失数据可能导致计算错误、分析偏差以及机器学习 (machine learning)模型性能不佳。因此，识别并妥善处理缺失值是数据准备中的一个基本步骤。

### 识别缺失数据

Pandas提供了直接的方法来检查缺失值。`isnull()`方法（及其别名`isna()`）返回一个与原始DataFrame或Series形状相同的布尔类型，`True`表示数据缺失（`NaN`），`False`表示其他情况。相反，`notnull()`对于非缺失值返回`True`。

```python
import pandas as pd
import numpy as np

# 包含缺失值的示例DataFrame
data = {'col1': [1, 2, np.nan, 4, 5],
        'col2': [np.nan, 7, 8, 9, 10],
        'col3': [11, 12, 13, np.nan, np.nan],
        'col4': ['A', 'B', 'C', 'D', np.nan]}
df = pd.DataFrame(data)

print("原始DataFrame:")
print(df)

print("\n检查缺失值 (isnull()):")
print(df.isnull())

print("\n检查非缺失值 (notnull()):")
print(df.notnull())
```

虽然查看布尔掩码对于小型数据集很有用，但通常更实用的是获取每列的缺失值数量。通过在`isnull()`之后链式调用`sum()`方法可以实现这一点，因为在求和时，`True`值被视为1，`False`被视为0。

```python
# 计算每列的缺失值数量
print("\n每列缺失值数量:")
print(df.isnull().sum())

# DataFrame中缺失值的总数量
print("\nDataFrame中缺失值总数量:")
print(df.isnull().sum().sum())
```

这个摘要能快速地告诉你哪些列有缺失数据以及缺失的数量。

### 处理缺失数据的策略

处理`NaN`值有两种主要策略：移除它们或替换它们（填充）。选择哪种方法取决于具体情况、缺失数据的数量以及对您的分析或模型可能产生的影响。

#### 1. 丢弃缺失值

最简单的方法是使用`dropna()`方法移除包含`NaN`值的行或列。

- **丢弃行：** 默认情况下，`dropna()`会移除任何包含至少一个`NaN`值的行。如果只有一小部分行存在缺失数据，并且移除它们不会引入显著偏差，这种方法通常是合适的。

  ```python
  # 丢弃包含任何NaN值的行（默认行为）
  df_dropped_rows = df.dropna()
  print("\n丢弃任何NaN值行后的DataFrame:")
  print(df_dropped_rows)
  ```
- **丢弃列：** 如果列包含缺失值，可以通过将`axis`参数 (parameter)设置为1（或'columns'）来丢弃整列。如果一列有非常高比例的缺失值或被认为对分析不重要，这可能会很有用。

  ```python
  # 丢弃包含任何NaN值的列
  df_dropped_cols = df.dropna(axis=1)
  print("\n丢弃任何NaN值列后的DataFrame:")
  print(df_dropped_cols)
  ```
- **控制丢弃行为：** `how`参数修改此行为：

  - `how='any'`: 如果存在*任何* `NaN`值，则丢弃该行/列（默认）。
  - `how='all'`: 仅当*所有*值都为`NaN`时，才丢弃该行/列。

  `thresh`参数提供了更精确的控制：它指定了行/列需要保留的*非缺失*值的最小数量。

  ```python
  # 保留至少有3个非NaN值的行
  df_thresh = df.dropna(thresh=3)
  print("\n保留至少有3个非NaN值行后的DataFrame:")
  print(df_thresh)

  # 丢弃所有值都为NaN的行（有用，但此处不适用）
  df_all_nan = df.dropna(how='all')
  # print(df_all_nan) # 在此情况下输出将与原始df相同
  ```

**注意：** 丢弃数据，特别是行，会导致信息丢失。如果缺失值的分布并非随机，丢弃它们可能会使剩余的数据集产生偏差。在移除数据之前，务必考虑其影响。

#### 2. 填充缺失值（插补）

除了移除数据，您还可以用替代值替换`NaN`。这被称为插补。`fillna()`方法用于此目的。

- **用常数值填充：** 您可以用特定值替换所有`NaN`，例如0、"Unknown"或在特征上下文 (context)中合理的某个值。

  ```python
  # 将所有NaN填充为0
  df_filled_zero = df.fillna(0)
  print("\n将NaN填充为0后的DataFrame:")
  print(df_filled_zero)

  # 以不同方式填充特定列中的NaN
  fill_values = {'col1': df['col1'].mean(), 'col2': 0, 'col3': -1, 'col4': 'Unknown'}
  df_filled_specific = df.fillna(value=fill_values)
  print("\n按列用特定值填充NaN后的DataFrame:")
  print(df_filled_specific)
  ```
- **向前填充和向后填充：** 对于时间序列等有序数据，将最后一个已知有效观测值向前传播（`method='ffill'`）或将下一个已知有效观测值向后传播（`method='bfill'`）可能是合适的。

  ```python
  # 向前填充
  df_ffill = df.fillna(method='ffill')
  print("\n向前填充后的DataFrame:")
  print(df_ffill)

  # 向后填充
  df_bfill = df.fillna(method='bfill')
  print("\n向后填充后的DataFrame:")
  print(df_bfill)
  ```
- **用统计量填充：** 对于数值数据，一种常用的方法是用列的均值、中位数或众数填充`NaN`。当数据存在异常值时，中位数通常优于均值，因为它对极端值不那么敏感。众数可用于分类特征。

  ```python
  # 用col1的均值填充col1中的NaN
  mean_col1 = df['col1'].mean()
  df_filled_mean = df.copy() # 创建一个副本以避免隐式修改原始df
  df_filled_mean['col1'].fillna(mean_col1, inplace=True)
  print("\n用col1均值填充col1中NaN后的DataFrame:")
  print(df_filled_mean)

  # 用col3的中位数填充col3中的NaN
  median_col3 = df['col3'].median()
  df_filled_median = df.copy()
  df_filled_median['col3'].fillna(median_col3, inplace=True)
  print("\n用col3中位数填充col3中NaN后的DataFrame:")
  print(df_filled_median)
  ```

  **重要提示：** 在为机器学习 (machine learning)准备数据时，您应仅在*训练*数据集上计算统计量（如均值或中位数），然后使用这些值来填充训练和测试数据集中的缺失数据。这可以防止测试集的信息泄露到训练过程中。我们将在第五章中讨论数据划分。
- **分组特定填充：** 有时，全局均值或中位数不具有代表性。例如，如果填充缺失的薪资，平均薪资可能因职位角色而显著不同。您可以结合使用`groupby()`和`transform()`来根据组级统计量填充缺失值。

  ```python
  # 示例：基于组均值填充NaN
  data_groups = {'Group': ['A', 'A', 'B', 'B', 'A', 'B'],
                 'Value': [10, np.nan, 20, 22, 12, np.nan]}
  df_groups = pd.DataFrame(data_groups)

  # 计算组均值并在每组内填充NaN
  df_groups['Value_filled'] = df_groups.groupby('Group')['Value'] \
                                     .transform(lambda x: x.fillna(x.mean()))

  print("\n进行分组特定均值填充后的DataFrame:")
  print(df_groups)
  ```

  `transform()`将一个函数（此处为用组均值填充NaN）应用于每个组，并返回一个与原始DataFrame索引对齐 (alignment)的Series。

### 选择合适的策略

处理缺失数据的最佳方法取决于：

1. **缺失数据的数量：** 如果只有极小部分数据缺失（<1-5%），丢弃可能是可接受的。如果大部分缺失，通常更推荐填充，或者该特征可能需要完全丢弃。
2. **数据性质：** 数据是数值型、类别型还是时间序列？这会影响是否适合使用均值/中位数、众数或向前/向后填充。
3. **缺失机制：** 数据*为什么*会缺失？是随机的，还是存在某种模式？如果处理不当（例如，简单地丢弃行），非随机缺失可能会导致结果偏差。可视化缺失数据模式有时有助于理解这一点。
4. **分析目标：** 一些机器学习 (machine learning)模型本身可以处理缺失数据，而另一些则需要完整的数据集。您的分析或模型对填充方法的敏感度也起着作用。

处理缺失数据通常是一个迭代过程。您可以尝试不同的策略，并评估它们对后续分析或模型性能的影响。Pandas提供了所需的灵活工具，能够有效实施这些不同方法。

## 参考资料

- [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) — The pandas development team (2024)
  Pandas DataFrame和Series中检测、删除和填充缺失值的官方指南，详细介绍了`isnull`、`dropna`和`fillna`方法。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://books.google.com/books?id=bE87DwAAQBAJ) — Wes McKinney (2017)
  Publisher: O'Reilly Media
  关于使用Pandas进行数据操作的基础著作，全面涵盖数据清洗和缺失数据处理策略。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  提供了关于机器学习数据预处理的实用见解，包括处理缺失值以提高模型性能。
