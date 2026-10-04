---
course: "getting-started-with-scikit-learn"
chapter: "data-preprocessing-feature-engineering"
lesson: "handling-missing-values"
sourceId: 1884
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/handling-missing-values"
title: "处理缺失值"
description: "处理缺失数据的策略，包括填充方法。"
order: 6
plots: []
sourceHash: "2720686997f4e561408afda4b1ab584655f7d68481ade93a5bbaa7a30129a43b"
sourceCorrections: []
---

数据集很少是完整的。缺失值，通常表示为 `NaN`（非数字）、`None` 或其他占位符，由于数据录入错误、传感器故障或受访者跳过问题而很常见。Scikit-learn 中的大多数机器学习 (machine learning)算法无法直接处理缺失值，如果在训练或预测时遇到它们，将引发错误。因此，处理缺失数据是预处理流程中的必要一步。

有几种方法可以处理缺失值，每种方法都有其优缺点。

### 识别缺失值

在决定策略之前，您首先需要识别缺失值所在的位置。Pandas DataFrame 为此提供了便利的方法：

```python
import pandas as pd
import numpy as np

# 含有缺失值的示例 DataFrame
data = {'Age': [25, 30, np.nan, 35, 40],
        'Salary': [50000, 60000, 75000, np.nan, 90000],
        'City': ['New York', 'London', 'Paris', 'Tokyo', np.nan]}
df = pd.DataFrame(data)

print("DataFrame：")
print(df)

# 检查缺失值（返回布尔型 DataFrame）
print("\n缺失值检查：")
print(df.isnull())

# 获取每列缺失值的数量
print("\n每列缺失值数量：")
print(df.isnull().sum())
```

此输出有助于量化 (quantization)每个特征中缺失数据问题的程度。

### 基本策略：删除

一种直接的方法是简单地删除含有缺失值的数据点或特征。

1. **行删除 (Listwise Deletion):** 删除包含*任何*缺失值的整行。这易于实现，但可能导致数据大量丢失，特别是当缺失值普遍存在或单个列中有很多缺失条目时。这会降低分析的统计效力，并且如果缺失不是完全随机的，可能会引入偏差。您可以使用 Pandas 中的 `df.dropna()` 来完成此操作。
2. **列删除:** 如果一个特征（列）有非常高比例的缺失值（例如，> 50-70%），它可能不会包含太多有用的信息。在这种情况下，可以考虑使用 `df.drop()` 删除整个列。然而，这应谨慎进行，因为即使是含有许多缺失值的列也可能保留一些预测能力。

删除通常不是首选方法，除非缺失数据量非常小或某一列明显无法使用。

### 填充策略

填充涉及用替代估计值来填补缺失值。这可以保持数据集的大小，但需要仔细考虑填充方法，以避免引入偏差或扭曲数据分布。Scikit-learn 在其 `impute` 模块中提供了 `SimpleImputer` 类，用于基本的填充任务。

`SimpleImputer` 的工作方式与其他 Scikit-learn 转换器类似，遵循熟悉的 `fit` 和 `transform` 模式。

#### 均值填充

此策略将数值列中的缺失值替换为该列中观测值的均值。

- **优点：** 易于实现。保留了列的均值。
- **缺点：** 降低列的方差。可能受异常值严重影响。扭曲变量之间的关系。仅适用于数值数据。

#### 中位数填充

此策略将数值列中的缺失值替换为观测值的中位数。

- **优点：** 易于实现。与均值填充相比，对异常值不那么敏感。
- **缺点：** 降低方差。扭曲变量之间的关系。仅适用于数值数据。当存在异常值时，通常优于均值填充。

#### 最常见值（众数）填充

此策略将缺失值替换为列中最常见的值（众数）。

- **优点：** 可用于数值和类别特征。
- **缺点：** 可能引入偏差，特别是当某个值出现频率非常高时。对于数值数据，它可能引入不自然符合分布的值。降低方差。

#### 固定值填充

此策略将缺失值替换为用户指定的固定常量值（例如，0、-1 或“缺失”）。

- **优点：** 简单。允许您明确标记 (token)原始缺失的值，这本身可能包含信息。
- **缺点：** 常量的选择会影响模型性能。如果常量值是特征正常范围的一部分，它可能会人为地创建或扭曲模式。

### 在 Scikit-learn 中使用 SimpleImputer

让我们看看如何使用 `SimpleImputer` 应用这些策略。我们将使用之前创建的示例 DataFrame。

```python
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# 示例 DataFrame（同上）
data = {'Age': [25, 30, np.nan, 35, 40],
        'Salary': [50000, 60000, 75000, np.nan, 90000],
        'City': ['New York', 'London', 'Paris', 'Tokyo', np.nan]}
df = pd.DataFrame(data)

# 分离数值列和类别列用于填充
df_numeric = df[['Age', 'Salary']]
df_categorical = df[['City']]

# --- 均值填充（针对数值）---
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# 在数据上进行拟合以学习均值
mean_imputer.fit(df_numeric)
# 转换数据（替换 NaN）
df_numeric_mean_imputed = mean_imputer.transform(df_numeric)
print("\n均值填充后的数值数据：")
# 转换回 DataFrame 以便清晰显示
print(pd.DataFrame(df_numeric_mean_imputed, columns=df_numeric.columns))

# --- 中位数填充（针对数值）---
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
median_imputer.fit(df_numeric)
df_numeric_median_imputed = median_imputer.transform(df_numeric)
print("\n中位数填充后的数值数据：")
print(pd.DataFrame(df_numeric_median_imputed, columns=df_numeric.columns))

# --- 最常见值填充（针对类别）---
# 注意：SimpleImputer 适用于数值表示。
# 对于类别数据，您通常会在编码*之后*应用此方法，
# 或者在编码之前直接使用 Pandas 的 fillna。
# 这里我们直接应用它进行演示，它会将字符串视为“对象”。
most_frequent_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
most_frequent_imputer.fit(df_categorical)
df_categorical_mf_imputed = most_frequent_imputer.transform(df_categorical)
print("\n最常见值填充后的类别数据：")
print(pd.DataFrame(df_categorical_mf_imputed, columns=df_categorical.columns))

# --- 固定值填充（例如，针对数值）---
constant_imputer_num = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
constant_imputer_num.fit(df_numeric)
df_numeric_const_imputed = constant_imputer_num.transform(df_numeric)
print("\n固定值 (0) 填充后的数值数据：")
print(pd.DataFrame(df_numeric_const_imputed, columns=df_numeric.columns))

# --- 固定值填充（例如，针对类别）---
constant_imputer_cat = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Unknown')
constant_imputer_cat.fit(df_categorical)
df_categorical_const_imputed = constant_imputer_cat.transform(df_categorical)
print("\n固定值 ('Unknown') 填充后的类别数据：")
print(pd.DataFrame(df_categorical_const_imputed, columns=df_categorical.columns))
```

**关于 `fit` 和 `transform` 的重要说明：** 就像缩放器和编码器一样，填充器必须*只*在训练数据上拟合。从训练数据中学到的统计量（例如均值、中位数、众数）随后用于转换训练数据*和*测试数据。这可以防止数据泄露，即测试集中的信息无意中影响预处理步骤。在使用 Scikit-learn Pipeline 中的填充器时，这会自动处理，我们将在稍后介绍。

选择正确的填充策略取决于数据的性质（数值型与类别型）、特征的分布、缺失值的百分比以及您计划使用的机器学习 (machine learning)算法的特定要求。尽管 `SimpleImputer` 涵盖了基本方法，Scikit-learn 也提供了更完善的填充器，例如 `KNNImputer`（使用 K 近邻算法估计值）和 `IterativeImputer`（将每个含有缺失值的特征建模为其他特征的函数），这些方法可能带来更精确的结果，但它们的计算成本更高。

## 参考资料

- [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) — scikit-learn developers (2023)
  Scikit-learn 缺失值填充模块的官方文档，涵盖了 `SimpleImputer` 以及 `KNNImputer` 和 `IterativeImputer` 等更高级的方法。
- [Working with missing data](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html) — The pandas development team (2023)
  Pandas DataFrame 中处理缺失数据的官方文档，包括缺失值的识别和基本操作。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的机器学习指南，全面介绍了数据预处理技术，例如使用 Scikit-learn 管理缺失值。
