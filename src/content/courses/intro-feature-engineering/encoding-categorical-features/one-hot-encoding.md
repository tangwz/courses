---
course: "intro-feature-engineering"
chapter: "encoding-categorical-features"
lesson: "one-hot-encoding"
sourceId: 1327
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-3-encoding-categorical-features/one-hot-encoding"
title: "标称特征的独热编码"
description: "实现独热编码并讨论其优点和潜在的缺点（维度）。"
order: 3
plots: []
sourceHash: "e77788736810ede2d0f97f05770a8d7461f19af97afec33b73242f4eec9a5ea9"
sourceCorrections: []
---

标称类别特征表示没有内在顺序或等级的类别。例如颜色（“红色”、“蓝色”、“绿色”）、城市名称（“伦敦”、“东京”、“纽约”）或产品类型（“电子产品”、“服装”、“杂货”）。机器学习 (machine learning)算法通常需要数值输入，因此我们需要一种方法将这些标签转换为数字，而不会强加人为的顺序。为此，最常用且直接的方法之一是独热编码 (OHE)。

独热编码背后的基本思想是为原始标称特征中存在的每个独特类别创建新的二元特征。对于数据集中的每个观测（行），这些新二元特征中只有一个会是“1”（热），表示该特定类别的存在，而所有其他新创建的二元特征都将是“0”。

假设一个名为“颜色”的特征有三个独特值：“红色”、“绿色”和“蓝色”。独热编码将此单列转换为三个新列：“Color\_Red”、“Color\_Green”和“Color\_Blue”。

- 如果原始值为“红色”，则新行将是“Color\_Red” = 1，“Color\_Green” = 0，“Color\_Blue” = 0。
- 如果原始值为“绿色”，则新行将是“Color\_Red” = 0，“Color\_Green” = 1，“Color\_Blue” = 0。
- 如果原始值为“蓝色”，则新行将是“Color\_Red” = 0，“Color\_Green” = 0，“Color\_Blue” = 1。

这种转换有效地将类别信息传递给算法，使用数值（0/1）来表达，而没有暗示类别之间的任何序数关系。

> 使用独热编码转换“颜色”特征。每个独特类别都获得其自己的二元列。

### 使用 Pandas 实现

Pandas 提供了方便的函数 `get_dummies()`，可以直接在数据框或 Series 上执行独热编码。

```python
import pandas as pd

# 示例数据框
data = {'ID': [1, 2, 3, 4],
        'Color': ['Red', 'Green', 'Blue', 'Green'],
        'Value': [10, 15, 5, 12]}
df = pd.DataFrame(data)

print("原始数据框:")
print(df)

# 对“Color”列应用独热编码
df_encoded = pd.get_dummies(df, columns=['Color'], prefix='Color', prefix_sep='_')

print("\n独热编码后的数据框:")
print(df_encoded)
```

**输出:**

```
Original DataFrame:
   ID  Color  Value
0   1    Red     10
1   2  Green     15
2   3   Blue      5
3   4  Green     12

DataFrame after One-Hot Encoding:
   ID  Value  Color_Blue  Color_Green  Color_Red
0   1     10           0            0          1
1   2     15           0            1          0
2   3      5           1            0          0
3   4     12           0            1          0
```

`pd.get_dummies()` 的重要参数 (parameter):

- `data`: 要编码的数据框或 Series。
- `columns`: 要编码的列名列表。如果为 None，则尝试编码所有对象或类别数据类型的列。
- `prefix`: 要添加到新列名开头的字符串或字符串列表。有助于标识原始特征。
- `prefix_sep`: 前缀和类别名称之间的分隔符字符串（默认为“\_”）。
- `drop_first`: 一个布尔值（默认为 False）。如果设置为 `True`，它会为每个特征丢弃第一个类别级别。这在一些线性模型中很有用，可以避免多重共线性，因为当所有其他类别列都为 0 时，被丢弃类别的信息已被隐含地包含。例如，如果“Color\_Blue”被丢弃，则“Color\_Green=0”和“Color\_Red=0”的行将表示颜色是蓝色。

### 使用 Scikit-learn 实现

Scikit-learn 的 `OneHotEncoder`（来自 `sklearn.preprocessing`）在构建机器学习 (machine learning)管道时通常更受青睐，特别是它能够在训练和测试阶段之间一致地处理数据。

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 示例数据框（同前）
data = {'ID': [1, 2, 3, 4],
        'Color': ['Red', 'Green', 'Blue', 'Green'],
        'Value': [10, 15, 5, 12]}
df = pd.DataFrame(data)

# 选择类别列——注意：OneHotEncoder 期望 2D 数组状输入
categorical_features = df[['Color']]

# 初始化编码器
# sparse_output=False 返回一个密集 NumPy 数组（更易于查看）
# handle_unknown='ignore' 可以避免测试数据中出现未见过类别时的错误
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# 拟合编码器到数据（学习类别）并转换数据
encoded_data = encoder.fit_transform(categorical_features)

# 获取编码器生成的新特征名称
encoded_feature_names = encoder.get_feature_names_out(['Color'])

# 使用编码后的特征创建新数据框
df_encoded_sklearn = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df.index)

# 与原始非类别特征合并
df_final = pd.concat([df.drop('Color', axis=1), df_encoded_sklearn], axis=1)

print("\nScikit-learn OneHotEncoder 后的数据框:")
print(df_final)

# 示例：转换新数据（可能包含未见过类别“Yellow”）
new_data = pd.DataFrame({'Color': ['Green', 'Red', 'Yellow']})
new_encoded = encoder.transform(new_data[['Color']])
print("\n编码后的新数据:")
print(pd.DataFrame(new_encoded, columns=encoded_feature_names))
```

**输出:**

```
DataFrame after Scikit-learn OneHotEncoder:
   ID  Value  Color_Blue  Color_Green  Color_Red
0   1     10         0.0          0.0        1.0
1   2     15         0.0          1.0        0.0
2   3      5         1.0          0.0        0.0
3   4     12         0.0          1.0        0.0

Encoded new data:
   Color_Blue  Color_Green  Color_Red
0         0.0          1.0        0.0
1         0.0          0.0        1.0
2         0.0          0.0        0.0
```

请注意，当使用 `handle_unknown='ignore'` 时，未见过类别“Yellow”的结果是全零。`handle_unknown` 的其他选项包括 `'error'`（默认值，引发错误）或使用 `infrequent_if_exist` 分配特定的编码向量 (vector)。

`sklearn.preprocessing.OneHotEncoder` 的重要参数 (parameter):

- `categories`: 手动指定类别（默认为“auto”从数据中学习）。
- `drop`: `{'first', 'if_binary'}` 或 None（默认）。类似于 Pandas 中的 `drop_first`，用于为每个特征丢弃一个类别。`'if_binary'` 仅为具有两个类别的特征丢弃第一个类别。
- `sparse_output`: 布尔值（默认为 True）。是否返回稀疏矩阵（高维度时内存高效）或密集 NumPy 数组。对于更易于检查或较小的数据集，设置为 `False`。
- `handle_unknown`: `{'error', 'ignore', 'infrequent_if_exist'}`。如何处理 `transform` 期间遇到但 `fit` 期间未见的类别。`'ignore'` 为该特征输出全零。

### 独热编码的优点

1. **无任意排序：** 它正确表示标称数据，而不会强加任何序数关系。
2. **算法兼容性：** 生成的二元特征几乎所有机器学习 (machine learning)算法都能轻松理解。
3. **可解释性（低基数）：** 当类别数量较少时，生成的特征非常易于解释（例如，“Color\_Red”清楚地表示红色）。

### 缺点与注意事项

1. **维度灾难：** 主要缺点是 OHE 会为*每个*独特类别引入一个新列。如果特征具有许多独特值（高基数），这将显著增加数据集中的列数（维度）。这可能导致：
   - 内存使用量增加。
   - 模型训练时间更长。
   - 潜在的过拟合 (overfitting)，特别是如果数据集大小相对于特征数量不够大（Hughes 现象）。
   - 对某些难以处理高维稀疏数据的算法而言，性能会下降。
2. **多重共线性：** 如果保留所有类别列（即 `drop_first=False` 或 `drop=None`），则生成的特征将是完全多重共线的。任何一列的值都可以由其他列的值完美预测（如果一个类别不存在，其对应的 OHE 列之和将为 0；否则为 1）。尽管许多算法（特别是基于树的算法，如随机森林或梯度提升）能够较好地处理这种情况，但线性模型（如线性回归、逻辑回归）可能会出现系数稳定性和可解释性问题。使用 `drop_first=True` 或 `drop='first'` 可以缓解此问题。
3. **处理未见过类别：** 如果在测试集或预测数据中出现训练（`fit`）期间不存在的新类别，`pd.get_dummies`（默认情况下）和 `sklearn.OneHotEncoder`（使用 `handle_unknown='error'`）都会导致问题（要么缺少列，要么出错）。使用 Scikit-learn 的 `handle_unknown='ignore'` 或 `'infrequent_if_exist'` 可以处理这种情况。

### 何时使用独热编码

OHE 是以下情况的可靠默认选择：

- 标称类别特征（这类特征的类别没有顺序）。
- 基数低到中等的特征（例如，少于 15-20 个独特值，尽管这取决于具体情况）。
- 对特征的大小或顺序敏感的算法（例如，线性模型、基于距离的算法如 KNN）。

对于基数非常高的特征，替代编码方法，如目标编码、哈希或分箱（稍后讨论），可能更合适来管理维度。

## 参考资料

- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) — scikit-learn developers (2023)
  Scikit-learn OneHotEncoder的官方文档，提供参数、使用方法和机器学习管道示例的详细信息。
- [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) — pandas development team (2023)
  Pandas get_dummies函数的官方文档，详细说明其在DataFrame和Series中进行独热编码的用法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的教材，涵盖机器学习基础知识，包括数据预处理和独热编码等特征工程技术。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本关于特征工程的专业资源，为包括独热编码在内的各种编码策略提供实践指南和理论背景。
