---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "scalars-vectors-matrices-introduction"
lesson: "introduction-to-matrices"
sourceId: 7805
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction/introduction-to-matrices"
title: "矩阵：以网格形式组织数据"
description: "介绍矩阵作为二维数字数组，用于表示数据集和变换。"
order: 4
plots: []
sourceHash: "37cc4b0aafa7f77aecf80248318c7ba026d15b1fe5be499bfd999c5831854c4e"
sourceCorrections: []
---

虽然向量 (vector)非常适合表示带有多个特征的单个数据点，但机器学习 (machine learning)模型很少只用一个例子进行训练。我们需要一种方法来整理整个数据点集合，为此我们选用矩阵。

矩阵是由行和列组成的数字矩形网格。你可以把它看作向量的一种扩展。如果向量是一个数字列表，那么矩阵就是将多个列表堆叠在一起。这种结构对于机器学习中表示数据集必不可少。

### 矩阵的构成

假设我们要建立一个预测房价的模型。我们收集了几栋房屋的数据。每栋房屋都是一个样本，对于每个样本，我们记录了几个特征，例如它的面积、卧室数量和房龄。我们可以用一个特征向量 (vector)来表示每栋房屋：

- **房屋 1:** `[1500, 3, 10]` (1500 平方英尺, 3 卧室, 10 年房龄)
- **房屋 2:** `[2100, 4, 5]` (2100 平方英尺, 4 卧室, 5 年房龄)
- **房屋 3:** `[1200, 2, 20]` (1200 平方英尺, 2 卧室, 20 年房龄)

为了整理这个完整的数据集，我们可以将这些行向量堆叠在一起，形成一个矩阵。

> 整理成矩阵的数据集。每一行对应一个数据样本（一栋房屋），每一列对应一个特定的特征。

在这个矩阵中：

- **行**水平排列，表示单个数据样本或观测值。我们的矩阵有 3 行。
- **列**垂直排列，表示数据的不同特征。我们的矩阵有 3 列。

矩阵的**维度**或**形状**由其行数和列数给出。我们将其描述为“$m 	imes n$”矩阵，其中 $m$ 是行数，$n$ 是列数。例如，表示房屋数据的矩阵可能是一个 $3 	imes 3$ 矩阵。

### 矩阵的数学记法

在线性代数中，我们通常用大写字母表示矩阵，例如 $A$。要指代矩阵中的特定元素，我们使用下标。矩阵 $A$ 中第 $i$ 行第 $j$ 列的元素表示为 $A_{ij}$ 或 $a_{ij}$。请记住约定：**先行后列。**

对于我们的房屋数据矩阵 $A$：


$$
A = \begin{bmatrix} 1500 & 3 & 10 \\ 2100 & 4 & 5 \\ 1200 & 2 & 20 \end{bmatrix}
$$


- 元素 $A_{1,1}$ 是 $1500$ (第一行，第一列)。
- 元素 $A_{2,3}$ 是 $5$ (第二行，第三列)。
- 元素 $A_{3,1}$ 是 $1200$ (第三行，第一列)。

### 机器学习 (machine learning)中的标准约定

这种每行一个样本、每列一个特征的排列方式是机器学习中的标准约定。这个数据矩阵通常称为**特征矩阵**，并常用变量 $X$ 表示。

我们想要预测的值，例如每栋房屋的价格，通常存储在一个单独的列向量 (vector)中。这通常称为**目标向量**，并用变量 $y$ 表示。

对于我们的例子，特征矩阵 $X$ 就是我们已经看到的矩阵，而目标向量 $y$ 可能包含相应的价格：


$$
X = \begin{bmatrix} 1500 & 3 & 10 \\ 2100 & 4 & 5 \\ 1200 & 2 & 20 \end{bmatrix}, \quad y = \begin{bmatrix} 300000 \\ 450000 \\ 220000 \end{bmatrix}
$$


这种特征 ($X$) 和目标 ($y$) 的结构化表示是几乎所有监督机器学习算法的输入。通过将数据整理成矩阵和向量，我们可以运用线性代数的强大运算来分析关联并建立预测模型。接下来，我们将准备好环境，使用 Python 创建和操作这些对象。

## 参考资料

- [Introduction to Linear Algebra](http://math.mit.edu/linearalgebra) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本广泛使用的教材，涵盖矩阵和向量的定义、属性及运算，是理解线性代数的基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第2章为机器学习提供了出色的线性代数介绍，解释了如何使用矩阵表示数据并执行基本操作。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  一本全面的资源，涵盖机器学习的数学基础，包括关于矩阵及其在数据表示中作用的专门部分。
