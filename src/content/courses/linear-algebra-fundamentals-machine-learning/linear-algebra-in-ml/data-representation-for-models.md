---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "linear-algebra-in-ml"
lesson: "data-representation-for-models"
sourceId: 7832
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-6-linear-algebra-in-ml/data-representation-for-models"
title: "模型的数据表示"
description: "回顾如何使用特征向量和数据矩阵来组织信息，以用于机器学习算法。"
order: 1
plots: []
sourceHash: "68fe8404e85dac8350c8cb7b2d2b425f558bce5f50c563e963320fd7520b2322"
sourceCorrections: []
---

机器学习 (machine learning)的根本在于找出数据中的规律。但为了让计算机处理这些数据，数据必须转换为结构化的数值形式。这正是线性代数提供核心框架的地方。它为我们提供了将信息组织成整齐的矩形数字阵列的工具，以便算法进行操作。

### 从观测值到特征向量 (vector)

考虑一个用于预测房价的简单数据集。对于每栋房子，我们可能会记录几个属性：其面积（平方英尺）、卧室数量和浴室数量。在机器学习 (machine learning)中，这些属性被称为**特征**。一栋房子，连同其所有特征，代表一个**观测值**或**样本**。

对于机器学习模型而言，一栋具体的房子，比如有1,500平方英尺、3间卧室和2间浴室，不是描述。它是一个空间中的点。我们可以将这栋房子表示为一个**特征向量**，它只是其数值特征的有序列表。


$$
\text{房子\_1} = \begin{bmatrix} 1500 \\ 3 \\ 2 \end{bmatrix}
$$


向量中的顺序很重要。我们必须确定一个一致的序列，例如[面积，卧室数量，浴室数量]。我们数据集中每一栋房子都将按照这种相同结构转换为向量。这使我们能够比较不同的房子并一致地对它们执行数学运算。

### 从向量 (vector)集合到数据矩阵

数据集很少只包含一个观测值。通常，我们有数百、数千甚至数百万个观测值。将我们所有单个的特征向量堆叠在一起，就得到了一个**数据矩阵**，它几乎总是用一个大写字母表示，例如 $X$。

在机器学习 (machine learning)中，按照惯例，数据矩阵中的每一行对应一个观测值（一栋房子），每一列对应一个特征（例如，面积）。

如果我们有三栋房子的数据：

1. 1,500 平方英尺，3间卧室，2间浴室
2. 2,100 平方英尺，4间卧室，3间浴室
3. 1,200 平方英尺，2间卧室，2间浴室

我们可以堆叠它们的特征向量来形成数据矩阵 $X$：


$$
X = \begin{bmatrix}
1500 & 3 & 2 \\
2100 & 4 & 3 \\
1200 & 2 & 2
\end{bmatrix}
$$


这个矩阵现在包含了我们整个特征数据集。这个矩阵的维度说明了我们的数据集。如果矩阵 $X$ 的形状是 $m \times n$，这意味着我们有 $m$ 个观测值和 $n$ 个特征。在我们的例子中，$X$ 是一个 $3 \times 3$ 矩阵。

### 用目标向量 (vector)表示标签

在许多机器学习 (machine learning)任务中，称为监督学习 (supervised learning)，我们的目标是预测一个特定的结果。对于我们的房价数据集来说，这将是每栋房子的价格。这个结果被称为**标签**或**目标**。

就像我们处理特征一样，我们将每个观测值的标签收集到一个向量中，通常命名为 $y$。如果我们三栋房子的价格是 $300,000、$450,000 和 $250,000，我们的目标向量$y\$ 将是：


$$
y = \begin{bmatrix}
300000 \\
450000 \\
250000
\end{bmatrix}
$$


$y$ 中的第一个条目对应 $X$ 中的第一行，第二个对应第二行，依此类推。这种对齐 (alignment)对于模型学习 $X$ 中的特征与 $y$ 中的结果之间的关系具有重大意义。

> 一个图示，说明了原始数据如何被分割和组织成特征矩阵 $X$ 和目标向量 $y$，然后它们作为机器学习模型的输入。

### 这种表示的强大之处

以这种方式组织数据可能看起来只是一个简单的组织步骤，但它是所有现代机器学习 (machine learning)的立足点。以下是它如此有效的原因：

- **计算效率：** NumPy 和 TensorFlow 等现代计算库经过高度优化，可以对矩阵执行计算。对整个矩阵执行一次操作比单独遍历每个数据点要快。这种做法，称为**向量 (vector)化**，是使机器学习在大型数据集上变得实用的根本。
- **数学简洁性：** 这种表示允许我们用紧凑而优雅的数学符号表达复杂的学习算法。正如您将在下一节看到的，线性回归的整个问题可以用简单的方程 $Ax = b$ 来概括。这种抽象使得推理 (inference)和实现算法更加容易。
- **普遍性：** 这种结构是普遍的。无论您的原始数据包含图像、文本、用户点击还是传感器读数，第一步几乎总是将其转换为数值矩阵。图像变成像素值矩阵，文本可以转换为表示单词计数的矩阵。一旦采用这种形式，相同的线性代数工具包就可以应用于不同的问题。

在 NumPy 中，创建这些结构非常直接。我们的房屋数据矩阵和目标向量将如下所示：

```python
import numpy as np

# 数据矩阵 X (3个观测值, 3个特征)
X = np.array([
    [1500, 3, 2],
    [2100, 4, 3],
    [1200, 2, 2]
])

# 目标向量 y (3个标签)
y = np.array([300000, 450000, 250000])

print("数据矩阵 X:\n", X)
print("\n目标向量 y:\n", y)
```

通过将数据组织成向量和矩阵，我们将一个实际问题转化为一个数学问题。这使我们能够运用线性代数的全部能力来找出规律、进行预测，并构建定义机器学习的智能系统。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  本书提供了使用Python库在机器学习任务中准备和表示数据的实用指导，包括特征、观测值和目标变量，以向量和矩阵的形式。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本基础教科书的第二章详细解释了向量、矩阵和线性代数运算的数学原理，这对于理解机器学习和深度学习模型中的数据表示至关重要。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  这是一本经典教科书，系统地介绍了向量、矩阵和向量空间的核心概念，它们是机器学习中用于组织和处理数据的基本数学工具。
- [NumPy ndarray object](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy Developers (2024)
  Publisher: NumPy Project
  NumPy官方文档中关于其基本N维数组对象的内容，它对于在Python中使用NumPy进行机器学习应用中的高效数值数据表示和操作至关重要。
