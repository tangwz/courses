---
course: "linear-algebra-essentials-ml"
chapter: "vectors-in-machine-learning"
lesson: "vector-norms-measuring-length"
sourceId: 1453
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning/vector-norms-measuring-length"
title: "向量范数：衡量长度"
description: "理解使用L1（曼哈顿）和L2（欧几里得）范数衡量向量长度的不同方法。"
order: 5
plots: ["plots/1453-0.json"]
sourceHash: "1e2a34de15b954698c5be4729faefef2e5ac2e1ed3bd9f874dafda69b8c583ac"
sourceCorrections: []
---

在机器学习 (machine learning)中，我们常需要量化 (quantization)向量 (vector)的“大小”或“量级”。虽然我们能直观感知二维或三维空间中的长度，但表示数据特征的向量可能存在于高得多的维度中。我们需要一种统一的方法来衡量这种长度。在此，**范数**这个思想便派上用场。范数是一种函数，它将一个向量作为输入，并返回一个表示其量级的非负标量值。可以把它看作是长度这一想法的正式推广。

不同的范数以不同方式衡量长度，范数的选择对机器学习算法有重要影响，尤其是在正则化 (regularization)和误差计算等方面。接下来，我们来查看最常见的范数。

### $L_2$ 范数（欧几里得范数）

最常见的范数是$L_2$范数，也称为欧几里得范数。它对应于向量 (vector)的标准几何长度，计算方法是其分量平方和的平方根。对于一个n维向量 $v = [v_1, v_2, ..., v_n]$，$L_2$范数的定义如下：

$||v||_2 = \sqrt{v_1^2 + v_2^2 + ... + v_n^2} = \sqrt{\sum_{i=1}^{n} v_i^2}$

这本质上是勾股定理在高维空间 (high-dimensional space)中的推广。它表示从原点到向量所代表点的最短直线距离。

**例子：**
考虑向量 $v = [3, 4]$。
其 $L_2$ 范数为：
$||v||_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

```python
import numpy as np

v = np.array([3, 4])
l2_norm = np.linalg.norm(v) # 默认范数为L2

print(f"向量: {v}")
print(f"L2 范数: {l2_norm}")
# 输出:
# 向量: [3 4]
# L2 范数: 5.0
```

> $L_2$ 范数衡量的是从原点 (0,0) 到点 (3,4) 的直接欧几里得距离。

在机器学习 (machine learning)中，$L_2$范数经常被使用：

- 计算数据点之间的欧几里得距离（下一节将涉及）。
- 衡量误差向量的量级。
- 在像岭回归这样的正则化 (regularization)方法中，它惩罚大的系数，以防止过拟合 (overfitting)。

### $L_1$ 范数（曼哈顿范数）

另一个重要的范数是$L_1$范数，常被称为曼哈顿范数或出租车范数。它不是将分量平方，而是将它们的绝对值相加：

$||v||_1 = |v_1| + |v_2| + ... + |v_n| = \sum_{i=1}^{n} |v_i|$

“曼哈顿范数”这个名称源于在网格状城市规划中导航的思路。你只能沿着网格线（就像城市街区一样）移动，而不能斜向移动。$L_1$范数表示沿着这些轴线从原点到向量 (vector)终点所走的总距离。

**例子：**
对于相同的向量 $v = [3, 4]$。
其 $L_1$ 范数为：
$||v||_1 = |3| + |4| = 3 + 4 = 7$

```python
import numpy as np

v = np.array([3, 4])
l1_norm = np.linalg.norm(v, ord=1)

print(f"向量: {v}")
print(f"L1 范数: {l1_norm}")
# 输出:
# 向量: [3 4]
# L1 范数: 7.0
```

> $L_1$ 范数衡量的是分量绝对值之和，就像沿着网格线从 (0,0) 移动到 (3,0) 然后再到 (3,4)。

$L_1$范数在机器学习 (machine learning)中有其独特的性质：

- 与$L_2$范数相比，它对异常值不那么敏感，因为它不对分量值进行平方。
- 在正则化 (regularization)（如LASSO回归）中使用时，它常促使*稀疏性*。这意味着它倾向于将不那么重要的特征系数推向零，从而有效进行特征选择。

### 比较 $L_1$ 和 $L_2$ 范数

主要区别在于它们如何处理分量的大小。$L_2$范数对值进行平方，从而对大分量施加重罚。$L_1$范数取绝对值，对偏差进行线性处理。

当考虑所有范数为1的向量 (vector)时，这种差异在视觉上很明显。对于$L_2$范数，这形成一个圆形（在二维中）或超球体（在更高维中）。对于$L_1$范数，这形成一个菱形（在二维中）或交叉多面体（在更高维中）。



![Interactive chart](plots/1453-0.json)



> 二维空间中，表示 $||v||_2 = 1$（蓝色圆形）和 $||v||_1 = 1$（红色菱形）的几何形状。$L_1$ 菱形的“角”对应于坐标轴，这有助于它在优化问题中倾向于稀疏解（即某些分量恰好为零）。

### 其他范数（$L_p$ 和 $L_\infty$）

$L_1$和$L_2$范数是更广泛的$L_p$范数家族的特殊情况，其定义为：

$||v||_p = \left( \sum_{i=1}^{n} |v_i|^p \right)^{1/p}$

其中 $p \ge 1$。

另一个有时会遇到的范数是$L_\infty$范数（或最大范数），它是$L_p$范数当$p$趋于无穷时的极限。它简单地返回向量 (vector)分量中的最大绝对值：

$||v||_\infty = \max_i |v_i|$

尽管$L_1$和$L_2$在许多机器学习 (machine learning)应用中是主力，但了解一般的$L_p$家族能提供更广阔的视角。

### 范数在实践中的重要性

理解向量 (vector)范数十分重要，因为它们提供了衡量以下内容的方法：

- **特征的量级：** 特定特征向量的“强度”如何？
- **误差衡量：** 预测值与实际值之间的差异有多大（常表示为误差向量）？用于误差计算的不同范数会导致不同的模型行为（例如，最小化均方误差（$L_2$）与最小化平均绝对误差（$L_1$））。
- **正则化 (regularization)：** 范数用于惩罚模型的复杂度。$L_2$正则化（岭回归）会缩小系数，而$L_1$正则化（LASSO）可以将系数缩小到零，从而执行特征选择。
- **距离度量：** 正如我们接下来将看到的，范数是计算向量间距离的依据，这对像k近邻（k-NN）这样的算法来说非常重要。

能够有效地计算这些范数，通常使用NumPy等库，是实现和理解许多机器学习 (machine learning)算法的核心能力。

## 参考资料

- [Introduction to Linear Algebra](https://dokumen.pub/introduction-to-linear-algebra-6-ed-1733146679-9781733146678.html) — Gilbert Strang (2023)
  Publisher: Wellesley-Cambridge Press; Pages: 440
  提供线性代数中向量范数及其性质的基础理解。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  统计学习领域的权威教材，详细讨论了$L_1$（LASSO）和$L_2$（岭回归）正则化及其与向量范数的关系。
- [\`numpy.linalg.norm\`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) — NumPy Developers (2023)
  用于计算各种向量和矩阵范数（包括$L_1$和$L_2$）的NumPy函数官方文档。
