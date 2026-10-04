---
course: "getting-started-with-scikit-learn"
chapter: "intro-setup-scikit-learn"
lesson: "data-representation"
sourceId: 1831
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-1-intro-setup-scikit-learn/data-representation"
title: "Scikit-learn 中的数据表示"
description: "了解 Scikit-learn 期望数据如何格式化，通常使用 NumPy 数组或 Pandas DataFrames。"
order: 4
plots: []
sourceHash: "18d81dff4cf78aef9bd9f4c5f7127fbd52eebfd58ddca4e0af6a4055eea50384"
sourceCorrections: []
---

机器学习 (machine learning)算法从数据中学习模式。为了有效地使用 Scikit-learn，理解该库期望数据如何格式化是很重要的。Scikit-learn 具有统一的 API 设计，主要处理存储在特定结构中的数值数据。

### 特征矩阵 (X)

输入数据，通常称为特征、样本或实例，通常表示为二维数组结构。按照惯例，这个特征矩阵被称为 `X`。

- **形状：** `X` 的期望形状是 `(n_samples, n_features)`。
  - `n_samples`：数据集中独立观测值或数据点的数量（行）。
  - `n_features`：为每个样本测量的特征或属性的数量（列）。
- **数据类型：** Scikit-learn 最适合处理数值数据，通常是64位浮点数 (`numpy.float64`)。尽管有些算法可以处理整数，但将数据转换为浮点数通常是一个稳妥的默认做法。处理分类（非数值）特征需要特定的预处理步骤，我们将在第四章中介绍。
- **支持的格式：**
  - **NumPy 数组：** 最基本的格式。对数值运算很高效。
  - **Pandas DataFrames：** 非常方便，特别是在最初处理带有标签列和混合数据类型时。Scikit-learn 估计器通常直接接受 DataFrames，通常在内部将它们转换为 NumPy 数组。列名通常被算法本身忽略，但对分析和预处理很有用。
  - **SciPy 稀疏矩阵：** 用于大多数值为零的数据集。对于高维稀疏数据（例如文本分析），这种格式内存效率高。

让我们看一个简单的例子。想象一个包含3个样本和2个特征（例如身高和体重）的数据集。

使用 NumPy：

```python
import numpy as np

# 3个样本，2个特征
X_np = np.array([
    [170.0, 75.0], # 样本 1：身高=170，体重=75
    [165.5, 68.2], # 样本 2：身高=165.5，体重=68.2
    [181.2, 88.9]  # 样本 3：身高=181.2，体重=88.9
])

print(f"NumPy 数组的形状: {X_np.shape}")
# 输出：NumPy 数组的形状：(3, 2)
print(f"数据类型: {X_np.dtype}")
# 输出：数据类型：float64
```

使用 Pandas：

```python
import pandas as pd

# 3个样本，2个特征
X_pd = pd.DataFrame({
    'Height (cm)': [170.0, 165.5, 181.2],
    'Weight (kg)': [75.0, 68.2, 88.9]
})

print("Pandas DataFrame：")
print(X_pd)
print(f"\nDataFrame 的形状: {X_pd.shape}")
# 输出：
# Pandas DataFrame：
#    身高 (cm)  体重 (kg)
# 0        170.0         75.0
# 1        165.5         68.2
# 2        181.2         88.9
#
# DataFrame 的形状：(3, 2)

# Scikit-learn 通常使用底层 NumPy 表示
print(f"\n底层 NumPy 数组的形状: {X_pd.values.shape}")
# 输出：底层 NumPy 数组的形状：(3, 2)
print(f"底层 NumPy 数组的数据类型: {X_pd.values.dtype}")
# 输出：底层 NumPy 数组的数据类型：float64
```

### 目标变量 (y)

在监督学习 (supervised learning)中（第二章和第三章），除了特征 `X` 之外，我们还需要目标变量，它代表我们想要预测的值。按照惯例，目标变量称为 `y`。

- **形状：** 目标变量通常表示为一维数组（向量 (vector)）。其期望的形状是 `(n_samples,)`。`y` 中样本的数量必须与 `X` 中样本的数量精确匹配，这一点很重要。
- **数据类型：**
  - **回归：** 对于回归任务（预测连续值），`y` 应该是一个数值数组（通常是浮点数）。
  - **分类：** 对于分类任务（预测离散类别），`y` 可以是数值（表示类别的整数，例如 0、1、2），也可以包含表示类别标签的字符串/对象（例如 '垃圾邮件'、'非垃圾邮件'）。Scikit-learn 通常在内部将非数值标签转换为整数。
- **支持的格式：**
  - **NumPy 数组：** 非常常见。
  - **Pandas Series：** Pandas Series（DataFrame 的单列）也容易被接受。

延续我们的例子，让我们添加一个目标变量，也许表示每个人是否打篮球（1表示是，0表示否）——这是一个分类任务。

使用 NumPy：

```python
# 3个样本的目标
y_np = np.array([1, 0, 1]) # 样本 1：是，样本 2：否，样本 3：是

print(f"目标数组的形状: {y_np.shape}")
# 输出：目标数组的形状：(3,)
print(f"数据类型: {y_np.dtype}")
# 输出：数据类型：int64
```

使用 Pandas：

```python
# 3个样本的目标
y_pd = pd.Series([1, 0, 1], name='Plays Basketball')

print("Pandas Series：")
print(y_pd)
print(f"\nSeries 的形状: {y_pd.shape}")
# 输出：
# Pandas Series：
# 0    1
# 1    0
# 2    1
# 名称: 是否打篮球, dtype: int64
#
# Series 的形状：(3,)
```

### 数据表示总结

| 组件 | 惯例 | 典型结构 | 形状 | 数据类型（常见） |
| --- | --- | --- | --- | --- |
| 特征 | `X` | 二维数组/DataFrame | `(n_samples, n_features)` | `float64` |
| 目标变量 | `y` | 一维数组/Series | `(n_samples,)` | `float64` (回归)   `int` 或 `object` (分类) |

主要结论是，Scikit-learn 主要期望你的数据是具有特定形状的数值数组：`X` 是一个二维数组，其中行是样本，列是特征；而 `y`（用于监督学习 (supervised learning)）是一个一维数组，存储每个样本的目标值。尽管 Pandas DataFrame 增加了便利性，但理解底层的 `(n_samples, n_features)` 结构很重要。遵循这些惯例可确保与库中广泛的工具兼容。在下一节中，我们将了解如何加载一些已经遵循这些惯例的预打包数据集。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广受认可的书籍，为机器学习算法的数据准备提供了实用指导，包括Scikit-learn所需的数据格式。第3版。
- [Sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html) — SciPy Developers (2023)
  详细说明SciPy中稀疏矩阵的各种格式和功能，这对于高效处理高维稀疏数据集至关重要。
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — NumPy Developers (2023)
  提供NumPy数组对象、数组创建和数据类型的基础信息，这些是Scikit-learn中使用的数据结构的基础。
