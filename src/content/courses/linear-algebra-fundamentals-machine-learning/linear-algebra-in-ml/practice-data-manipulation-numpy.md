---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "linear-algebra-in-ml"
lesson: "practice-data-manipulation-numpy"
sourceId: 7836
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-6-linear-algebra-in-ml/practice-data-manipulation-numpy"
title: "实践：使用NumPy处理数据"
description: "实践加载一个简单数据集并将其表示为NumPy矩阵以进行分析。"
order: 5
plots: []
sourceHash: "64bee03a299bf03ce06e9b64fc05a88b8ec3ebb5f5b484f509737f9342a652b7"
sourceCorrections: []
---

应用理论知识处理实际数据集是机器学习 (machine learning)的基础。准备用于算法的数据集涉及理解其如何加载和组织。这个动手练习涵盖了机器学习工作流中一个常见的初始步骤：将原始数据转换为NumPy矩阵。

### 一个简单数据集

假设我们有一个用于预测房价的小数据集。数据包含房屋面积、卧室数量和最终售价。在典型的Python应用中，这些数据可能最初是列表的列表形式。

```python
import numpy as np

# 每个内部列表代表一栋房屋：[房屋面积, 卧室数量, 价格]
raw_data = [
    [1500, 3, 320000],
    [2100, 4, 450000],
    [1200, 2, 250000],
    [1800, 3, 380000]
]
```

尽管这种格式易于阅读，但它并未针对机器学习 (machine learning)模型所需的数学运算进行优化。为此，我们需要将其转换为NumPy数组，这是Python中数值数据的标准格式。

### 转换为NumPy矩阵

使用`np.array()`函数，从列表的列表创建NumPy矩阵（或更准确地说，是一个二维`ndarray`）非常直接。

```python
# 将列表的列表转换为二维NumPy数组
house_data_matrix = np.array(raw_data)

print(house_data_matrix)
```

这将产生以下输出：

```
[[  1500      3 320000]
 [  2100      4 450000]
 [  1200      2 250000]
 [  1800      3 380000]]
```

现在我们的数据已整理成结构化网格。每一行是一个观测值（一栋房屋），每一列代表一个特定属性。这就是几乎所有机器学习 (machine learning)算法都预期的那种数据矩阵格式。

### 分离特征与目标

在监督学习 (supervised learning)中，我们区分*特征*（用于做出预测的输入）和*目标*（我们希望预测的值）。

- **特征矩阵 ($X$):** 一个矩阵，其中行是观测值，列是特征。在我们的例子中，特征是“房屋面积”和“卧室数量”。
- **目标向量 (vector) ($y$):** 一个向量，包含我们希望为每个观测值预测的值。这里，它是“价格”。

我们可以使用NumPy强大的切片功能将`house_data_matrix`分离为`X`和`y`。

> 原始数据矩阵被分为特征矩阵`X`和目标向量`y`。

以下是如何在代码中执行此划分：

```python
# 为特征选择所有行 (:) 和直到索引 2（不包含）的列
X = house_data_matrix[:, :2]

# 为目标选择所有行 (:) 和仅最后一个列（索引 2）
y = house_data_matrix[:, 2]

print("特征矩阵 X:")
print(X)
print("\n目标向量 y:")
print(y)
```

输出确认了划分：

```
Feature Matrix X:
[[1500    3]
 [2100    4]
 [1200    2]
 [1800    3]]

Target Vector y:
[320000 450000 250000 380000]
```

### 查看维度

一种常规做法是检查矩阵和向量 (vector)的`shape`。这有助于确认你的数据结构正确，并有助于预防后续步骤中的错误。

```python
# 获取特征矩阵和目标向量的维度
print("X 的形状:", X.shape)
print("y 的形状:", y.shape)
```

输出将是：

```
Shape of X: (4, 2)
Shape of y: (4,)
```

这告诉我们：

- `X` 有 4 行（观测值）和 2 列（特征）。
- `y` 是一个长度为 4 的向量，对应于 4 个观测值。

你现在已成功将原始数据转换为机器学习 (machine learning)所需的精确结构。特征矩阵`X`和目标向量`y`，分别对应于我们在线性回归中讨论的方程 $Ax = b$ 中的矩阵`A`和向量`b`。你在前面章节中学到的所有矩阵和向量运算，现在可以直接应用于`X`和`y`来训练模型、识别模式或降低维度。

## 参考资料

- [NumPy User Guide: Array creation routines](https://numpy.org/doc/stable/user/basics.creation.html) — NumPy Developers (2024)
  官方文档，说明如何从各种数据结构创建 NumPy 数组，直接适用于将原始数据转换为矩阵。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，介绍如何使用 Python 库为机器学习模型准备和组织数据（特征矩阵 X 和目标向量 y）。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679901](https://doi.org/10.1017/9781108679901)
  提供机器学习的数学基础，包括线性代数概念如何应用于数据表示（矩阵和 X, y 向量）。
