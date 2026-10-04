---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "scalars-vectors-matrices-introduction"
lesson: "why-linear-algebra-for-ml"
sourceId: 7802
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction/why-linear-algebra-for-ml"
title: "线性代数对机器学习有何作用？"
description: "了解线性代数为何是理解和构建机器学习模型的必要条件。"
order: 1
plots: []
sourceHash: "42c3b4173d8c62505d5fb470e620cc9ea3e9d4122af0b46decd2a32f7a77a158"
sourceCorrections: []
---

机器学习 (machine learning)的核心是发现数据中的规律。但无论信息是客户评价、股票价格，还是狗的照片，计算机要处理它，首先必须将其转化成它能理解的语言：数字语言。线性代数正是这种语言的语法。它提供了一套强大高效的工具，用于整理和处理这些数字，使其成为几乎所有机器学习模型的计算支柱。

### 用数字表示数据

在算法学习之前，我们需要一套组织数据的方法。考虑一个实际例子：预测房价。一栋房屋的信息可能包括其面积（1,500平方英尺）、卧室数量（3间）和房龄（20年）。我们可以用一个有序的数字列表来表示这栋房屋，这在线性代数中称为**向量 (vector)**：


$$
\text{house\_vector} = [1500, 3, 20]
$$


这个向量不仅仅是一个列表。它是三维空间中的一个点，每个维度对应一个特征。另一栋房屋，例如一栋2,100平方英尺、4间卧室、5年房龄的房屋，也只是同一空间中的另一个点：$[2100, 4, 5]$。

当我们收集数千栋房屋的数据时，可以将这些向量堆叠起来，形成一个数字网格。这个网格就是**矩阵**。矩阵中的每一行代表一栋房屋（一个样本），每一列代表一个特征（面积、卧室数量、房龄）。这个矩阵就成为机器学习 (machine learning)模型处理的核心对象。

### 高效地进行数据计算

一旦数据被组织成向量 (vector)和矩阵，我们需要进行计算。我们可能需要调整模型参数 (parameter)，计算两个项目间的相似度，或者转换数据以凸显重要规律。

如果没有线性代数，你将不得不编写循环来逐一遍历数据集中的每个数字。这速度慢、效率低，并使代码难以阅读。线性代数提供了一种方式，能用一行代码表示整个数据集上的复杂计算。像矩阵乘法这样的运算使我们能够同时处理数千个数据点。这不仅仅是为了方便；它也是现代机器学习 (machine learning)在计算上可行的原因。NumPy等库经过高度优化，能够以极快的速度执行这些向量和矩阵运算。

### 机器学习 (machine learning)模型的蓝图

许多机器学习算法不仅仅被线性代数*支持*；它们更是由线性代数*定义*的。

- **线性回归：** 将一条线拟合到一组数据点的经典问题，表示为求解方程 $Ax = b$。其中，$A$是我们的数据矩阵，$b$是我们希望预测的结果向量 (vector)（例如，房价），而求解向量$x$则能得到我们模型的参数 (parameter)。
- **神经网络 (neural network)：** 神经网络是一系列相互连接的层。数据从一层传递到下一层的过程涉及矩阵与向量的乘法，这充当线性变换，随后是一个“激活函数 (activation function)”。训练神经网络的全部就是为这些矩阵找到合适的数字。
- **主成分分析（PCA）：** 这种常用的降维技术找出数据中最主要的“方向”。这些方向就是数据协方差矩阵的特征向量，一个我们将在第五章中讨论的核心内容。

下图展示了原始数据如何转换成数字格式，以便机器学习算法在线性代数支持下进行预测。

> 原始信息被转换成特征向量，随后汇集到数据矩阵中。机器学习算法使用线性代数对该矩阵进行运算，并生成最终输出或预测。

总而言之，学习线性代数并非可有可无的学术练习。它是你理解如何表示数据、算法的底层运行方式以及如何高效实现它们所需的实用工具集。在接下来的章节中，我们将从最基本的部分开始构建这个工具集，从最简单的对象：标量、向量和矩阵开始。

## 参考资料

- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679989](https://doi.org/10.1017/9781108679989)
  本书为理解现代机器学习算法所需的数学原理,包括线性代数,提供了全面的基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 2: Linear Algebra
  第二章专门介绍线性代数概念,以理解机器学习和深度学习模型(如神经网络)。
- [NumPy Reference](https://numpy.org/doc/stable/reference/index.html) — The NumPy Developers (2025)
  NumPy 的官方参考文档,详细介绍了数组对象和优化过的线性代数例程,这些对于机器学习中的高效计算至关重要。
- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本经典的教科书,为线性代数奠定了坚实的数学基础,涵盖了向量、矩阵和方程组等基本概念。
