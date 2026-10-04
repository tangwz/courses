---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "eigenvalues-and-eigenvectors"
lesson: "numpy-finding-eigenvalues-eigenvectors"
sourceId: 7831
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-5-eigenvalues-and-eigenvectors/numpy-finding-eigenvalues-eigenvectors"
title: "动手实践：使用NumPy求解特征值"
description: "使用NumPy的`linalg.eig`函数计算矩阵的特征值和特征向量。"
order: 5
plots: ["plots/7831-0.json"]
sourceHash: "fe725524b47c254b5d0166e12754bc3c2176a4c5b62107a6adb757d4ad858f07"
sourceCorrections: []
---

手动求解特征方程有助于培养直觉，但对于机器学习 (machine learning)中遇到的矩阵而言并不实用。对于计算任务，可以利用NumPy等数值计算库高效地找到特征值和特征向量 (vector)。

### 使用 `numpy.linalg.eig`

NumPy的线性代数模块`linalg`包含`eig()`函数，它正是我们所需要的。它接受一个方阵作为输入，并返回两个NumPy对象：

1. 一个包含特征值的一维数组。
2. 一个二维数组，其中每个**列**是与第一个数组中相同索引处的特征值对应的特征向量 (vector)。

让我们从一个简单的2x2矩阵开始。考虑变换矩阵 $A$：


$$
A = \begin{pmatrix} 4 & -2 \\ 1 & 1 \end{pmatrix}
$$


现在，我们使用NumPy来找到它的特征值和特征向量。

```python
import numpy as np

# 定义我们的方阵
A = np.array([[4, -2],
              [1,  1]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("特征值：")
print(eigenvalues)

print("\n特征向量（每列为一个特征向量）：")
print(eigenvectors)
```

运行此代码会产生以下输出：

```
特征值：
[3. 2.]

特征向量（每列为一个特征向量）：
[[0.89442719 0.70710678]
 [0.4472136  0.70710678]]
```

从这个结果中，我们可以识别出我们的特征值-特征向量对：

- 第一个特征值是 $\lambda_1 = 3$，对应的特征向量是 $v_1 = \begin{pmatrix} 0.894 \\ 0.447 \end{pmatrix}$。
- 第二个特征值是 $\lambda_2 = 2$，对应的特征向量是 $v_2 = \begin{pmatrix} 0.707 \\ 0.707 \end{pmatrix}$。

请注意，NumPy返回的特征向量是**单位向量**，这表示它们的长度（L2范数）为1。这是数值计算库中的一种标准做法，因为特征向量的方向才是主要考量，而不是其大小。

### 验证特征方程

为了确认我们的结果，我们可以检查它们是否满足定义方程 $Av = \lambda v$。让我们为第一个特征值 $\lambda_1 = 3$ 及其特征向量 (vector) $v_1$ 进行测试。

首先，我们从`eigenvectors`矩阵中分离出特征向量。记住，它是第一列。

```python
# 分离第一个特征值和特征向量
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0] # 第一列

print("Lambda 1：", lambda1)
print("向量 1：", v1)
```

现在，我们来计算方程的两边。

```python
# 方程的左边：A @ v1
left_side = A @ v1

# 方程的右边：lambda1 * v1
right_side = lambda1 * v1

print("\nA @ v1 =", left_side)
print("lambda1 * v1 =", right_side)
```

输出将是：

```
A @ v1 = [2.68328157 1.34164079]
lambda1 * v1 = [2.68328157 1.34164079]
```

这两个结果是相同的，确认我们找到了有效的特征值和特征向量对。您可以对第二对（$\lambda_2=2$ 和 $v_2$）执行相同的检查，以进一步验证结果。

### 特征向量 (vector)的几何视角

我们可以将这种关系可视化，以更好地理解“方向不变”的含义。以下图表显示了原始特征向量 $v_1$（蓝色）和变换后的向量 $A v_1$（橙色）。



![特征向量的变换](plots/7831-0.json)



> 原始特征向量 $v_1$ 和经过变换 $A v_1$ 得到的向量。两个向量都位于从原点延伸的完全相同的直线上。变换只是将向量按其特征值（$\lambda_1 = 3$）的因子进行了缩放。

正如图表所示，将矩阵 $A$ 作用于其特征向量 $v_1$ 并没有使其旋转。它只是将其拉伸，使其指向相同的方向。新向量的长度是原始向量长度的三倍，这与特征值 $\lambda_1 = 3$ 完全对应。这就是特征方程的几何意义的体现。

计算线性变换主轴的这种能力是一个有用的工具。在最后一章中，我们将看到这个过程如何构成主成分分析（PCA）的核心，这是一种在机器学习 (machine learning)中广泛使用的降维技术。

## 参考资料

- [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) — NumPy Developers (2023)
  使用NumPy计算特征值和特征向量的官方文档。
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) — Gilbert Strang (2019)
  Publisher: Wellesley-Cambridge Press
  一本涵盖线性代数概念（包括特征值和特征向量）的教科书，具有在机器学习和数据科学中的直接应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书提供了机器学习的数学基础，包括与特征值和特征向量相关的线性代数主题。
