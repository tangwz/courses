# Pandas DataFrame 用于数据准备

来源：[原文](https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-1-dsa-ml-foundations/pandas-dataframes-ml)

[返回章节目录](README.md) · [返回课程目录](../README.md)

NumPy 支持高效的数值计算。机器学习 (machine learning)工作流通常涉及处理结构化表格数据——类似于电子表格或数据库表的各类数据集，其中行表示观察值，列表示特征。Pandas 库，特别是其 `DataFrame` 对象，是处理此类结构化表格数据的主要工具。Pandas 在内部使用 NumPy 构建，提供高级数据结构和操作工具，旨在提高数据分析和准备任务的清晰度和效率。

可以将 Pandas `DataFrame` 视为一种二维、大小可变且可能包含异构数据类型的表格数据结构，具有标签轴（行和列）。您可以将其想象成这样：

> DataFrame 由带标签的行（索引）、带标签的列和实际数据组成，这些数据通常在内部存储为 NumPy 数组，从而实现对不同数据类型的高效操作。

Pandas 擅长简化必要的数据准备步骤，在将数据输入机器学习模型之前，这些步骤是不可或缺的。

### 加载和检查数据

常见的首个步骤是从文件（如 CSV）加载数据。Pandas 使这一过程变得简单直观：

```python
import pandas as pd

# 从 CSV 文件加载数据
try:
    # 假设 'dataset.csv' 文件存在，包含 'feature1'、'feature2'、'target' 等列
    df = pd.read_csv('dataset.csv')
    print("数据加载成功。")
except FileNotFoundError:
    # 如果文件未找到，则创建示例 DataFrame
    print("未找到 dataset.csv。正在创建示例 DataFrame。")
    data = {'feature1': [1.0, 2.5, 0.8, 4.2],
            'feature2': ['A', 'B', 'A', 'C'],
            'target': [0, 1, 0, 1]}
    df = pd.DataFrame(data)

# 显示前 5 行
print("前 5 行：")
print(df.head())

# 获取 DataFrame 的简洁汇总信息
print("\nDataFrame 信息：")
df.info()

# 获取数值列的描述性统计信息
print("\n描述性统计：")
print(df.describe())
```

`.head()` 方法能快速显示前几行，`.info()` 提供包含数据类型和每列非空值数量的汇总（这对于查找缺失数据很必要），而 `.describe()` 则提供数值列的统计汇总（计数、均值、标准差、最小值、最大值、四分位数）。

### 选择和筛选数据

机器学习 (machine learning)通常需要处理特定的数据子集，例如选择输入特征或根据条件筛选样本。Pandas 提供直观的方法来完成这些操作，可以使用标签 (`.loc`) 或整数位置 (`.iloc`)：

```python
# 选择单列（返回 Pandas Series）
feature1_data = df['feature1']
print("\n已选择 'feature1' 列 (Series)：")
print(feature1_data.head())

# 选择多列（返回 DataFrame）
features = df[['feature1', 'feature2']]
print("\n已选择 'feature1' 和 'feature2' 列 (DataFrame)：")
print(features.head())

# 根据索引标签选择行（例如，索引为 0 和 2 的行）
# 注意：.loc 使用标签。如果索引是默认整数，则它使用这些整数。
rows_0_2 = df.loc[[0, 2]]
print("\n索引为 0 和 2 的行：")
print(rows_0_2)

# 根据整数位置选择行（前 3 行）
first_3_rows = df.iloc[0:3] # 注意：切片不包含结束索引
print("\n使用 iloc 的前 3 行：")
print(first_3_rows)

# 根据条件筛选行
# 选择 'feature1' 大于 1.0 的行
filtered_df = df[df['feature1'] > 1.0]
print("\nfeature1 大于 1.0 的行：")
print(filtered_df)
```

这些选择方法通常高效，特别是与迭代 Python 列表或字典相比。

### 处理缺失数据

"数据集经常包含缺失值（通常表示为 `NaN`，即非数字）。大多数机器学习 (machine learning)算法无法直接处理缺失数据，这使得处理缺失数据成为一个必要的预处理步骤。Pandas 提供便捷的工具："

```python
# 创建包含缺失值的示例 DataFrame
data_missing = {'col1': [1, 2, None, 4, 5],
                'col2': [None, 'B', 'C', 'D', None]}
df_missing = pd.DataFrame(data_missing)
print("\n包含缺失值的 DataFrame：")
print(df_missing)

# 检查缺失值（返回布尔型 DataFrame）
print("\n缺失值检查：")
print(df_missing.isnull())

# 统计每列的缺失值数量
print("\n每列的缺失值数量：")
print(df_missing.isnull().sum())

# 选项 1：删除包含任何缺失值的行
df_dropped_rows = df_missing.dropna()
print("\n删除包含 NaN 的行后的 DataFrame：")
print(df_dropped_rows)

# 选项 2：填充缺失值（例如，数值型数据用均值填充，或用特定值填充）
# 计算 'col1' 的均值 - 使用原始 df_missing 进行计算
col1_mean = df_missing['col1'].mean()
df_filled = df_missing.copy() # 在副本上操作

# 用 'col1' 的均值填充其中的 NaN
df_filled['col1'] = df_filled['col1'].fillna(col1_mean)
# 用占位符（如 'Unknown'）填充 'col2' 中的 NaN
df_filled['col2'] = df_filled['col2'].fillna('Unknown')

print(f"\n填充 NaN 后的 DataFrame（col1 使用均值={col1_mean:.2f} 填充）：")
print(df_filled)
```

选择删除还是填充取决于缺失数据的数量和具体问题。填充（插补）通常能保留更多数据，但需要仔细权衡插补策略。Pandas 的操作，例如 `.fillna()`，通常实现得高效。

### 数据转换与特征工程

Pandas 使得修改现有列或创建新列变得容易，这对于特征工程非常重要：

```python
# 对列应用函数（例如，对 'feature1' 进行平方）
df['feature1_squared'] = df['feature1'].apply(lambda x: x**2)
print("\n包含 'feature1_squared' 的 DataFrame：")
print(df.head())

# 将分类值映射为数值（简单示例）
# 实际操作中，请使用独热编码等技术（将在后续或机器学习课程中介绍）
mapping = {'A': 0, 'B': 1, 'C': 2}
# 在映射前检查 'feature2' 是否存在
if 'feature2' in df.columns:
    df['feature2_mapped'] = df['feature2'].map(mapping)
    print("\n包含 'feature2_mapped' 的 DataFrame：")
    print(df.head())
else:
    print("\nDataFrame 中未找到 'feature2'，跳过映射。")
```

许多 Pandas 操作是*向量 (vector)化*的，意味着它们可以一次性处理整个数组（列），而无需显式的 Python 循环。这利用了底层的 NumPy 数组和 C 实现，从而与纯 Python 中的逐行处理相比，带来显著的性能提升。

### 分组与聚合

Pandas 的 `.groupby()` 功能允许基于列值进行高效的数据聚合。这对于了解类别内的数据分布或创建聚合特征很有用。

```python
# 按 'feature2' 分组，并计算每个组中 'feature1' 的均值
if 'feature2' in df.columns:
    grouped_means = df.groupby('feature2')['feature1'].mean()
    print("\n'feature1' 按 'feature2' 分组后的均值：")
    print(grouped_means)
else:
     print("\nDataFrame 中未找到 'feature2'，跳过分组。")
```

### 为何机器学习 (machine learning)要用 Pandas DataFrame？

尽管 Python 列表和字典可以存储数据，但它们在表格数据操作方面缺乏 Pandas DataFrame 的专业功能和性能优化：

1. **效率：** 使用 NumPy 进行数值数据的快速向量 (vector)化操作。能在同一结构中处理异构数据类型。
2. **功能：** 丰富多样的内置函数，用于数据加载、清理、转换、合并、重塑和聚合。
3. **处理缺失数据：** 集成工具，用于检测和处理缺失值。
4. **基于标签的索引：** 使用有意义的标签进行直观的数据选择和对齐 (alignment)。
5. **集成性：** 可与其他科学计算 Python 库集成，如 Matplotlib（可视化）、Scikit-learn（机器学习）和 Statsmodels（统计建模）。

使用 Pandas DataFrame 可以高效且有效地准备数据，为后续的机器学习建模步骤奠定良好基础。了解其结构和核心操作对于任何使用 Python 处理结构化数据的机器学习实践者来说都不可或缺。随着我们学习的推进，我们将看到这些操作的性能特点（通常与 NumPy 和底层算法有关）如何影响机器学习流程的整体速度。

## 参考资料

- [Pandas documentation](https://pandas.pydata.org/docs/) — The Pandas Development Team (2025)
  Pandas所有功能和数据结构的权威综合参考，包括DataFrame、数据加载、操作和清洗。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由Pandas的创建者撰写，本书详细解释了Pandas DataFrame及其在数据分析和准备中的使用，并提供了实际示例。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  在机器学习工作流程的背景下，提供了使用Pandas进行数据准备的实践指导，展示了在模型训练前如何使用DataFrame。

---

[上一节](04-NumPy%20%E6%95%B0%E7%BB%84%EF%BC%9A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%95%B0%E5%80%BC%E5%A4%84%E7%90%86%E7%9A%84%E6%A0%B9%E6%9C%AC.md) · [下一节](06-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%97%AE%E9%A2%98%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E7%9A%84%E5%AF%B9%E5%BA%94.md)
