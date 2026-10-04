---
course: "linear-algebra-essentials-ml"
chapter: "vector-spaces-subspaces-linear-independence"
lesson: "practice-analyzing-feature-vector-sets"
sourceId: 1509
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-4-vector-spaces-subspaces-linear-independence/practice-analyzing-feature-vector-sets"
title: "实践操作：分析特征向量集"
description: "使用计算工具检查线性独立性并计算表示特征集的矩阵的秩。"
order: 8
plots: []
sourceHash: "ab9b04125685933886dfe3572040f1c9786fb5e68f28b382b6143e92aa88627a"
sourceCorrections: []
---

向量 (vector)空间、线性独立性、基和秩是理解数据集中结构的基本概念，特别是当数据点表示为特征向量时。分析这些向量集有助于找出多余信息，并了解特征空间的实际维度。Python的NumPy库是一个进行数值计算的常用工具，用于执行这些分析。

### 将特征集表示为矩阵

假设我们有一个小型数据集，其中包含多个数据点，每个数据点由若干特征描述。我们可以将这些特征向量 (vector)组织成一个矩阵。通常，每*行*表示一个数据点，每*列*表示一个特征。然而，当分析*特征本身*的线性独立性或它们所张成的维度时，将特征向量排列为矩阵的*列*通常更为方便。我们将采用这种约定来分析特征间的关系。

假设我们有3个数据点，每个点有4个特征。我们可以将*特征*表示为列向量：

$\mathbf{f}_1 = \begin{bmatrix} 1 \\ 2 \\ 0 \\ 1 \end{bmatrix}, \quad \mathbf{f}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{f}_3 = \begin{bmatrix} 1 \\ 3 \\ 1 \\ 1 \end{bmatrix}, \quad \mathbf{f}_4 = \begin{bmatrix} 2 \\ 4 \\ 0 \\ 2 \end{bmatrix}$

我们可以将这些列向量组合成矩阵$A$：

$A = \begin{bmatrix} 1 & 0 & 1 & 2 \\ 2 & 1 & 3 & 4 \\ 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 2 \end{bmatrix}$

让我们使用NumPy创建这个矩阵：

```python
import numpy as np

# 特征向量作为列
A = np.array([
    [1, 0, 1, 2],
    [2, 1, 3, 4],
    [0, 1, 1, 0],
    [1, 0, 1, 2]
])

print("特征矩阵A：\n", A)
```

### 检查线性独立性

特征向量 (vector)间的线性独立性有其意义。如果一组特征向量是线性相关的，这意味着至少一个特征可以表示为其他特征的线性组合。这表明我们的特征中存在冗余。例如，拥有“摄氏温度”和“华氏温度”这两个特征并没有增加任何新的信息，因为一个可以从另一个完美预测；它们是线性相关的（在中心化之后）。冗余特征有时会给机器学习 (machine learning)算法带来问题，例如线性回归中的多重共线性，这会导致系数估计不稳定。

A检查矩阵列线性独立性的一种实际方法是计算它的**秩**。矩阵的秩是矩阵中线性独立列（或行）的最大数量。

- 如果秩等于列数，则列是线性独立的。
- 如果秩小于列数，则列是线性相关的。

### 使用NumPy计算矩阵的秩

NumPy的`linalg`模块提供了`matrix_rank`函数来计算矩阵的秩。它通常使用奇异值分解（SVD，我们稍后会介绍）等方法来可靠地确定秩，即使存在小的数值误差。

让我们计算特征矩阵`A`的秩：

```python
# 计算矩阵A的秩
rank_A = np.linalg.matrix_rank(A)
num_features = A.shape[1] # 列数（特征数）

print(f"矩阵A：\n{A}")
print(f"特征数（列数）：{num_features}")
print(f"矩阵A的秩：{rank_A}")

if rank_A < num_features:
    print("特征向量（列）是线性相关的。")
else:
    print("特征向量（列）是线性独立的。")
```

执行这段代码将输出：

```
矩阵A：
[[1 0 1 2]
 [2 1 3 4]
 [0 1 1 0]
 [1 0 1 2]]
特征数（列数）：4
矩阵A的秩：2
特征向量（列）是线性相关的。
```

秩为2，这小于特征数（4）。这证实了我们的推断：特征向量 (vector)是线性相关的。仔细观察矩阵$A$，我们可以看到$\mathbf{f}_3 = \mathbf{f}_1 + \mathbf{f}_2$ 和 $\mathbf{f}_4 = 2 \mathbf{f}_1$。这种冗余意味着特征$\mathbf{f}_3$和$\mathbf{f}_4$没有在$\mathbf{f}_1$和$\mathbf{f}_2$中已有的信息上增加任何独特的方向信息。这些特征所张成的维度只有2，这由秩所体现。

### 示例：线性独立特征

现在，让我们考虑另一组我们预期是线性独立的特征向量 (vector)。

```python
# 另一组特征向量（列）
B = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1] # 添加了第四个数据点/维度
])

rank_B = np.linalg.matrix_rank(B)
num_features_B = B.shape[1]

print(f"\n矩阵B：\n{B}")
print(f"特征数（列数）：{num_features_B}")
print(f"矩阵B的秩：{rank_B}")

if rank_B < num_features_B:
    print("B的特征向量（列）是线性相关的。")
else:
    print("B的特征向量（列）是线性独立的。")
```

这可能会输出：

```
矩阵B：
[[1 0 0]
 [0 1 0]
 [0 0 1]
 [1 1 1]]
特征数（列数）：3
矩阵B的秩：3
B的特征向量（列）是线性独立的。
```

此处，秩（3）等于特征数（3），表明这些特征向量是线性独立的。这些特征都不能表示为其他特征的线性组合。

### 在机器学习 (machine learning)中的解释

为什么进行这种分析？

1. **特征选择/工程：** 识别线性依赖有助于发现冗余特征。移除它们可以简化模型，降低计算成本，有时还能在不丢失信息的情况下提高数值稳定性。对于矩阵$A$，我们可能只需要保留特征$\mathbf{f}_1$和$\mathbf{f}_2$，因为它们构成了列空间（由所有四个特征张成的空间）的一个基。
2. **理解数据维度：** 秩告诉我们特征所张成子空间的有效维度。这与主成分分析（PCA）等降维方法密切相关，这些方法旨在找到一个捕捉大部分数据方差的低维基。
3. **模型稳定性：** 如前所述，特征之间的高度相关性或线性依赖性（多重共线性）可以使某些模型（如线性回归）的参数 (parameter)估计对数据中的微小变化非常敏感。计算秩是诊断此类问题的第一步。

### NumPy函数总结

在本次实践练习中，我们使用了：

- `np.array()`：从列表的列表创建矩阵。
- `A.shape[1]`：获取列数（在我们设置中指特征数）。
- `np.linalg.matrix_rank()`：计算矩阵的秩，这是我们检查列间线性独立性的主要工具。

通过使用这些工具，你可以将向量 (vector)空间和线性独立性的抽象知识应用于特征数据集的具体分析，获得有助于机器学习 (machine learning)中预处理步骤和模型构建的见解。

## 参考资料

- [numpy.linalg.matrix_rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html) — NumPy Developers (2024)
  NumPy 库中 `matrix_rank` 函数的官方文档，详细说明了其用法和实现细节。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习领域的标准著作，讨论了特征关系（如多重共线性）的统计含义，这些与特征集中的线性相关性直接相关。
