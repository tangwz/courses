---
course: "intro-feature-engineering"
chapter: "handling-missing-data"
lesson: "simple-imputation"
sourceId: 1313
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data/simple-imputation"
title: "简单填充策略：均值、中位数、众数"
description: "对数值和分类特征应用均值、中位数和众数等基本填充技法。"
order: 3
plots: ["plots/1313-0.json"]
sourceHash: "edc4665c767d0202384ea8f6c1afe59b442ed5bf30b49717678eac2f02707041"
sourceCorrections: []
---

在数据集中识别出缺失值后，下一步是决定如何处理它们。正如之前讨论的，大多数机器学习 (machine learning)算法都需要完整的数据集。虽然删除含有缺失值的行或列（完全删除或成对删除）是一种方法，但这通常会导致大量数据丢失，尤其当缺失值普遍存在时。填充（即用估计值填补缺失值的过程）提供了一种保留数据点的方法。

最简单的填充技术一次处理一个变量，使用该变量已观测值的汇总统计量。这些单变量方法，特别是均值、中位数和众数填充，易于理解和实现，使它们成为一种常见的初步处理方法。

### 均值填充

均值填充将数值列中的所有缺失值（$NaN$）替换为该列非缺失值的算术平均值。

假设我们有一个包含一些缺失项的特征向量 (vector) $x_j$。均值 $\bar{x}_j$ 仅使用已观测值计算。对于每个缺失项 $x_{ij}$（其中 $i$ 是样本索引，$j$ 是特征索引），我们进行替换：

$x_{ij} \leftarrow \bar{x}_j$

**使用 Pandas 实现：**

你可以使用 Pandas 轻松进行均值填充。假设 `df` 是你的 DataFrame，'Age' 是一个包含缺失值的列：

```python
import pandas as pd
import numpy as np

# 示例 DataFrame
data = {'Age': [25, 30, np.nan, 35, 40, np.nan, 55],
        'Salary': [50000, 60000, 75000, np.nan, 80000, 95000, 120000]}
df = pd.DataFrame(data)

# 计算 'Age' 列的均值（不包括 NaN）
mean_age = df['Age'].mean()
print(f"年龄均值: {mean_age:.2f}")

# 用均值填充 'Age' 列中的缺失值
df['Age_mean_imputed'] = df['Age'].fillna(mean_age)
print("\n均值填充 'Age' 列后的 DataFrame:")
print(df)
```

**优点：**

- **简单：** 非常易于实现和理解。
- **保留样本量：** 保留所有观测值，避免了因删除而导致的数据丢失。
- **保留均值：** 填充后列的均值与原始观测值的均值保持一致。

**缺点：**

- **降低方差：** 用一个常数（均值）替换缺失值会人为地减少特征的变异性。这可能会影响模型性能，特别是对于对变异性敏感的算法。
- **扭曲关系：** 它削弱了填充后的特征与其他特征之间的相关性和协方差，因为填充值未能保留原始关系。
- **对异常值敏感：** 均值对极端值敏感。如果存在异常值，均值可能无法很好地代表数据的中心趋势，从而可能导致不佳的填充效果。
- **影响分布：** 它扭曲了特征的原始概率分布，通常会将分布的尾部拉向中心。

当缺失数据量较小且变量具有大致对称分布、没有明显异常值时，通常考虑使用均值填充。

### 中位数填充

中位数填充类似于均值填充，但它不使用均值，而是使用列中已观测值的中位数（数据排序后的中间值）来替换缺失项。

对于特征 $x_j$，中位数 $median(x_j)$ 从已观测值中计算。缺失项 $x_{ij}$ 替换为：

$x_{ij} \leftarrow median(x_j)$

**使用 Pandas 实现：**

```python
# 计算 'Age' 列的中位数
median_age = df['Age'].median()
print(f"\n年龄中位数: {median_age}")

# 用中位数填充 'Age' 列中的缺失值
df['Age_median_imputed'] = df['Age'].fillna(median_age)
print("\n中位数填充 'Age' 列后的 DataFrame:")
print(df)
```

**优点：**

- **对异常值的鲁棒性：** 中位数比均值受极端值的影响小。这使得中位数填充对于偏态分布或存在明显异常值的特征来说，是更好的选择。
- **简单：** 与均值填充一样，它易于实现。
- **保留样本量：** 保留所有观测值。

**缺点：**

- **降低方差：** 类似于均值填充，它减少了特征的方差。
- **扭曲关系：** 它也会扭曲与其他特征的相关性和协方差。
- **影响分布：** 会显著改变特征分布的形状。

对于数值特征，尤其是在处理偏态数据或潜在异常值时，中位数填充通常比均值填充更受青睐。

### 众数填充

均值和中位数适用于数值数据，但不适用于分类特征。对于分类数据（或有时是具有少量唯一值的离散数值数据），会使用众数填充。它将缺失值替换为众数，即该列中出现频率最高的值。

对于特征 $x_j$，众数 $mode(x_j)$ 从已观测值中确定。缺失项 $x_{ij}$ 替换为：

$x_{ij} \leftarrow mode(x_j)$

**使用 Pandas 实现：**

Pandas 的 `.mode()` 方法返回一个 Series（因为可能存在多个众数）。我们通常使用 `[0]` 选择第一个。

```python
# 包含分类特征的示例 DataFrame
data_cat = {'Color': ['Red', 'Blue', 'Green', np.nan, 'Blue', 'Red', 'Blue', np.nan],
            'Size': ['M', 'L', 'S', 'M', np.nan, 'L', 'M', 'S']}
df_cat = pd.DataFrame(data_cat)

# 计算 'Color' 列的众数
mode_color = df_cat['Color'].mode()[0]
print(f"\n颜色众数: {mode_color}")

# 用众数填充 'Color' 列中的缺失值
df_cat['Color_mode_imputed'] = df_cat['Color'].fillna(mode_color)

# 计算 'Size' 列的众数
mode_size = df_cat['Size'].mode()[0]
print(f"尺码众数: {mode_size}")

# 用众数填充 'Size' 列中的缺失值
df_cat['Size_mode_imputed'] = df_cat['Size'].fillna(mode_size)

print("\n众数填充后的 DataFrame:")
print(df_cat)
```

**优点：**

- **适用于分类数据：** 它是非数值特征最直接的填充方法。
- **简单：** 易于实现。
- **保留样本量：** 保留所有观测值。

**缺点：**

- **增加众数频率：** 人为地增加了众数类别的频率，可能导致分析或模型偏向该类别。
- **扭曲关系：** 与均值和中位数填充一样，它忽略了特征之间的关系。
- **可能引入偏差：** 如果某个类别占据压倒性优势，用众数填充会强化这种优势。

众数填充是分类特征的标准简单技术。

### 使用 Scikit-learn 的 SimpleImputer

虽然 Pandas 的 `.fillna()` 便于快速填充，但 Scikit-learn 提供了 `SimpleImputer` 类，该类在将填充整合到机器学习 (machine learning)流程中时特别有用。它确保从训练数据中学习到的填充策略能够一致地应用于任何新数据（如测试集），从而防止数据泄露。

`SimpleImputer` 支持均值、中位数、众数（'most\_frequent'）和常数值填充。

```python
from sklearn.impute import SimpleImputer

# --- 使用 SimpleImputer 进行均值填充 ---
# 需要重塑，因为 SimpleImputer 期望二维数组状输入
age_column = df[['Age']] # 选择 'Age' 列，保持其为 DataFrame

imputer_mean = SimpleImputer(strategy='mean')

# 在数据上拟合填充器（学习均值）
imputer_mean.fit(age_column)

# 转换数据（应用填充）
df['Age_sklearn_mean'] = imputer_mean.transform(age_column)

# --- 使用 SimpleImputer 进行中位数填充 ---
imputer_median = SimpleImputer(strategy='median')
imputer_median.fit(age_column) # 拟合学习中位数
df['Age_sklearn_median'] = imputer_median.transform(age_column) # 转换应用填充

# --- 使用 SimpleImputer 进行众数填充 ---
# 需要单独处理分类数据，通常在编码之后
# 为了演示，让我们将其应用于 'Size' 列（假设它被独立处理）
size_column = df_cat[['Size']]

# 注意：SimpleImputer 最适用于数值或编码后的分类数据。
# 对于直接对字符串进行众数填充，Pandas 通常更简单。
# 如果 'Size' 被数值编码（例如，S=0, M=1, L=2），SimpleImputer 将直接起作用。
# 让我们模拟对 'Color' 进行众数填充以作说明（拟合/转换）
color_column = df_cat[['Color']]
imputer_mode = SimpleImputer(strategy='most_frequent')
imputer_mode.fit(color_column) # 学习到 'Blue' 是众数
df_cat['Color_sklearn_mode'] = imputer_mode.transform(color_column)


print("\nScikit-learn 均值/中位数填充后的 DataFrame (原始 DF):")
print(df[['Age', 'Age_sklearn_mean', 'Age_sklearn_median']].head())

print("\nScikit-learn 众数填充后的 DataFrame (分类 DF):")
print(df_cat[['Color', 'Color_sklearn_mode']].head())
```

在 `Pipeline` 对象中使用 `SimpleImputer`（在后续关于机器学习工作流的讨论中会涉及）是构建模型的标准做法。

### 简单填充的局限性

均值、中位数和众数填充是快速简便的基线方法。然而，它们的主要缺点是它们是**单变量**的——它们只考虑包含缺失值的列中的信息，忽略了与其他特征的潜在关系或相关性。这通常会导致：

- **特征分布的扭曲。**
- **方差的减少。**
- **变量之间相关性的削弱。**

考虑一个身高和体重相关的数据集。仅使用平均体重来填充缺失的体重，会忽略身高较高的人体重往往较重这一事实。简单的均值填充可能会给一个很高的人分配一个平均体重，这很可能不准确。

下图说明了均值填充如何扭曲我们之前示例中 'Age' 特征的分布。



![年龄分布：原始数据与均值填充对比](plots/1313-0.json)



> 'Age' 特征在均值填充前后的分布。请注意，填充数据中均值（37.0）处出现了尖峰，这改变了原始分布的形状。

虽然简单的填充方法提供了一种快速解决方案，但它们往往无法捕捉数据的底层结构。更复杂的技法，例如 KNN 填充和迭代填充，会借鉴其他特征的数据来进行更准确的估计，我们将在后续进行学习。

## 参考资料

- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.routledge.com/Feature-Engineering-for-Machine-Learning-Principles-and-Techniques-for-Data/Chio-Fan/p/book/9781138612140) — Ching Chio, C. Fan (2019)
  Publisher: Chapman and Hall/CRC
  一本关于特征工程的实用指南，涵盖数据预处理和缺失值处理。
- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn, Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  一本广受引用的预测建模书籍，其中包含关于数据预处理和插补的详细章节。
- [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) — scikit-learn developers (2023)
  scikit-learn中`SimpleImputer`类的官方文档，详细说明了其在均值、中位数和众数插补中的使用。
