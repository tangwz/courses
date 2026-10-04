---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "vector-operations"
lesson: "vector-norms-l1-l2"
sourceId: 7811
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-2-vector-operations/vector-norms-l1-l2"
title: "向量范数：度量长度"
description: "介绍向量范数，特指用于度量向量长度的L1（曼哈顿）和L2（欧几里得）范数。"
order: 4
plots: ["plots/7811-0.json", "plots/7811-1.json"]
sourceHash: "5d86c2fe112a6132cff252e814774867f04e7134132831c73369e5440479c7bd"
sourceCorrections: []
---

尽管我们常把向量 (vector)看作空间中指向箭头的形式，但一个核心问题是：“这个箭头有多长？”这种“长度”或“大小”是向量的一个基本属性，在线性代数中，我们使用一个称为**范数**的函数来度量它。范数接收一个向量作为输入，并返回一个表示其大小的单一非负数。

理解范数之所以有帮助，是因为它们是许多机器学习 (machine learning)应用的根本，从度量模型预测中的误差到正则化 (regularization)模型以防止过拟合 (overfitting)。两种最常见的向量范数是L2范数和L1范数。

### L2范数：欧几里得距离

L2范数是大多数人直观上所认为的“距离”。它计算向量 (vector)从原点到其终点的直线长度。你可能从几何学中记得这是勾股定理。对于一个二维向量 `v = [x, y]`，其L2范数是 $\sqrt{x^2 + y^2}$。这个思想可以扩展到任意维数。

向量 $v$ 具有 $n$ 个分量时的L2范数公式为：


$$
||v||_2 = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$


双竖线 $|| \cdot ||$ 是范数的标准记法，下标2表示这是L2范数。由于它非常常见，下标常被省略，所以如果你看到 $||v||$，它通常指代L2范数。

让我们以向量 $v = [3, 4]$ 为例。它的L2范数是：


$$
||v||_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$


这表示向量的长度为5个单位。从几何上看，它构成了一个边长分别为3和4的直角三角形的斜边。



![Interactive chart](plots/7811-0.json)



> 向量 $[3, 4]$ 的L2范数是从原点出发的直线距离，即5。

一个有趣的性质是L2范数与点积直接相关。一个向量的平方L2范数等于它与自身的点积： $||v||_2^2 = v \cdot v$。

### L1范数：曼哈顿距离

L1范数提供了一种不同的度量向量 (vector)长度的方式。它不是度量直接的“直线距离”，而是将每个分量的绝对值相加。它常被称为**曼哈顿范数**或**出租车距离**。想象你身处一个拥有网格状街道布局的城市。你无法穿过建筑物（对角线移动）；你必须沿着街道（坐标轴）移动。

向量 $v$ 的L1范数公式为：


$$
||v||_1 = |v_1| + |v_2| + \dots + |v_n| = \sum_{i=1}^{n} |v_i|
$$


让我们使用相同的向量 $v = [3, 4]$。它的L1范数计算如下：


$$
||v||_1 = |3| + |4| = 7
$$


从几何上看，这是你只沿着x轴然后沿着y轴移动，从原点到达点 $(3, 4)$ 所走的总距离。



![Interactive chart](plots/7811-1.json)



> L1范数度量沿着坐标轴的路径。对于向量 $[3, 4]$，L1距离是7，而L2路径是5。

### 为什么不同的范数在机器学习 (machine learning)中有其应用价值

你可能想知道为什么我们需要不止一种方式来度量距离。在机器学习中，不同的范数具有不同的性质，使它们适用于特定任务。

- **L2范数：** 这是最常用于度量误差的范数。例如，在线性回归中，模型通常通过最小化误差向量 (vector)（预测值与实际值之间的差值）的L2范数来训练。这被称为最小化“平方误差和”。L2范数也用于一种称为**岭回归**的正则化 (regularization)方法，该方法会惩罚大的模型权重 (weight)，鼓励权重值小且分散的解。
- **L1范数：** L1范数也用于度量误差和进行正则化。一种称为**Lasso回归**的正则化方法使用L1范数来惩罚权重。L1范数有一个独特的性质：它倾向于产生**稀疏**解，这意味着它促使许多模型权重变为精确的零。这通过告知你哪些特征对预测没有影响，从而有效地执行特征选择。

### 使用NumPy计算范数

正如你所预料的，NumPy使用 `numpy.linalg.norm()` 函数提供了一种简单高效的方法来计算向量 (vector)范数。此函数默认计算L2范数，但你可以使用 `ord` 参数 (parameter)指定其他范数。

让我们用一个新向量 $v = [-5, 12]$ 来实际操作。

首先，计算L2范数：
$||v||_2 = \sqrt{(-5)^2 + 12^2} = \sqrt{25 + 144} = \sqrt{169} = 13$

接下来，计算L1范数：
$||v||_1 = |-5| + |12| = 5 + 12 = 17$

现在，让我们用NumPy验证这些结果。

```python
import numpy as np

# 创建我们的示例向量
v = np.array([-5, 12])

# 计算L2范数（默认）
l2_norm = np.linalg.norm(v)

# 通过设置 ord=1 计算L1范数
l1_norm = np.linalg.norm(v, ord=1)

print(f"向量: {v}")
print(f"L2 范数 (欧几里得): {l2_norm}")
print(f"L1 范数 (曼哈顿): {l1_norm}")
```

**输出：**

```
Vector: [-5 12]
L2 Norm (Euclidean): 13.0
L1 Norm (Manhattan): 17.0
```

NumPy的结果与我们的手动计算完全符合。能够计算向量的“大小”是你在使用更复杂的机器学习 (machine learning)算法时将经常用到的一项核心能力。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本广泛使用的教材，全面而易懂地介绍了线性代数，涵盖向量范数及其数学性质。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: 68-79 (Chapter 3 on Linear Methods for Regression); DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本统计学习的权威书籍，解释了L1和L2范数在机器学习正则化（Lasso和Ridge回归）中的实际应用和理论基础。
- [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) — NumPy Developers (2023)
  NumPy官方文档，说明了`linalg.norm`函数的用法，包括计算各种向量和矩阵范数的详细信息。
