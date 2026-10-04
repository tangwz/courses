---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "linear-algebra-in-ml"
lesson: "linear-regression-matrix-problem"
sourceId: 7833
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-6-linear-algebra-in-ml/linear-regression-matrix-problem"
title: "线性回归的矩阵问题"
description: "用线性代数记号 $Ax = b$ 表述一个简单的线性回归问题。"
order: 2
plots: ["plots/7833-0.json"]
sourceHash: "facc875cd767a59895b7676771903cdf448047950967dfbac17b4fd41088da81"
sourceCorrections: []
---

机器学习 (machine learning)中最常见的任务之一是预测。给定一组输入特征，我们希望预测一个输出值。这其中最简单的形式是线性回归，我们尝试找出一条最能拟合数据的直线。初看之下，这可能像是统计学问题，但其公式表述和解法纯粹是线性代数。

### 从直线到方程组

直线方程为 $y = mx + b$。其中：

- $y$ 是我们想预测的值 (因变量)。
- $x$ 是我们的输入特征 (自变量)。
- $m$ 是直线的斜率，决定其倾斜程度。
- $b$ 是 y 截距，即直线与垂直轴相交的点。

在机器学习 (machine learning)中，我们常用不同的记号。我们可以将方程写作 $y = w_1x_1 + w_0$，其中 $w_1$ 是特征 $x_1$ 的权重 (weight)（斜率），$w_0$ 是偏置 (bias)项（截距）。我们的目的是找到权重 ($w_1$ 和 $w_0$) 的合适值，使直线尽可能地拟合我们的数据。

设想一下，我们有一个小数据集，用于根据房屋面积（平方英尺）预测房价。

| 面积 (平方英尺) | 价格 (千美元) |
| --- | --- |
| 1500 | 300 |
| 2000 | 410 |
| 1200 | 270 |
| 1800 | 350 |

如果我们的直线要完美地经过每个点，我们就需要满足一个方程组：

$300 = w_1(1500) + w_0$
$410 = w_1(2000) + w_0$
$270 = w_1(1200) + w_0$
$350 = w_1(1800) + w_0$

这个难点显而易见。一条直线极不可能穿过所有这四个点。数据含有噪声。与其寻找一个完美解，我们寻找能使总误差最小的直线。



![房价数据和可能的拟合](plots/7833-0.json)



> 找出最能代表蓝色数据点所示关系的红线是线性回归的目的。

### 构建矩阵方程

这就是线性代数提供巧妙而有力地表示此情形的方法。正如我们在第四章所见，一个线性方程组可以写成矩阵形式 $Ax = b$。我们来组合这些组成部分。

向量 (vector) $b$（在此情形中为 $y$）包含我们的目标值，即房价。


$$
y = \begin{bmatrix} 300 \\ 410 \\ 270 \\ 350 \end{bmatrix}
$$


向量 $x$ 包含我们正在寻找的未知参数 (parameter)：截距 $w_0$ 和斜率 $w_1$。


$$
x = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}
$$


矩阵 $A$ 是一个很有趣的部分。它常被称为“设计矩阵”。每一行对应一个数据点，每一列对应一个特征。我们的方程形式是 $w_1 \cdot (\text{尺寸}) + w_0 \cdot 1$。因此，特征的第一列将是房屋尺寸，第二列将表示截距 $w_0$ 的常数项。为使数学运算成立，我们将第二列填充为 1。


$$
A = \begin{bmatrix} 1500 & 1 \\ 2000 & 1 \\ 1200 & 1 \\ 1800 & 1 \end{bmatrix}
$$


等等，我们为什么要添加一列 1 呢？我们来检查一下执行矩阵-向量乘法 $Ax$ 时会发生什么：


$$
Ax = \begin{bmatrix} 1500 & 1 \\ 2000 & 1 \\ 1200 & 1 \\ 1800 & 1 \end{bmatrix} \begin{bmatrix} w_1 \\ w_0 \end{bmatrix} = \begin{bmatrix} 1500 \cdot w_1 + 1 \cdot w_0 \\ 2000 \cdot w_1 + 1 \cdot w_0 \\ 1200 \cdot w_1 + 1 \cdot w_0 \\ 1800 \cdot w_1 + 1 \cdot w_0 \end{bmatrix}
$$


\text{请注意：我们在此重新排列了 A 和 x 中的列，以匹配 (尺寸) \* w1 + 1 \* w0 的形式，这是一种常见做法。}

这种乘法完美地重构了我们原始的方程组。我们整个情形现在可以表述为寻找使 $Ax$ 尽可能接近 $y$ 的向量 $x$。

完整的情形表达为：


$$
Ax \approx y
$$


### 矩阵表述的优点

这种表示法不止是记号上的简便。它有几个明显的好处：

1. **紧凑性：** 我们用一个简洁的方程表示包含数千个观测值和数十个特征的数据集。
2. **通用性：** 如果我们想添加另一个特征，比如卧室数量，该怎么办？我们只需在矩阵 $A$ 中为该特征添加一列，并在向量 (vector) $x$ 中添加一个权重 (weight)。方程 $Ax \approx y$ 保持不变。这使得这种方法非常易于扩展。
3. **计算效率：** 像 NumPy 这样的数值库经过高度优化，可以执行矩阵运算。使用矩阵代数解决此问题比编写循环逐一迭代方程要快得多。

我们现在已将线性回归表达为一个线性代数问题。我们正在寻找最能解方程 $Ax = y$ 的向量 $x$。由于完美解很少存在，下一步（我们在此不进行求解）涉及使用矩阵运算，如转置和逆，以找到使预测值 ($Ax$) 与实际值 ($y$) 之间误差最小的向量 $x$。这种方法正式称为求解“正规方程”，是直接应用你在前几章学到的工具。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebras/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本经典的线性代数教材，为线性回归所需的方程组、最小二乘法和矩阵运算提供了基础处理。（第5版）
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  一本全面介绍机器学习数学基础的教材，其中包含专门讨论使用矩阵代数进行线性回归的章节。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本权威的统计学习方法教材，其中包含使用矩阵表示法和正规方程组详细推导线性回归的内容。（第2版）
