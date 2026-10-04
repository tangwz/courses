---
course: "linear-algebra-essentials-ml"
chapter: "vectors-in-machine-learning"
lesson: "calculating-distances-vectors"
sourceId: 1455
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning/calculating-distances-vectors"
title: "计算向量间的距离"
description: "使用向量范数计算特征空间中数据点之间的距离。"
order: 6
plots: ["plots/1455-0.json"]
sourceHash: "1b451e8c96760587a6042f18c7682380ef5c0efd416699d972dc93b2518d0315"
sourceCorrections: []
---

在上一节中，我们学习了向量 (vector)范数，它提供了一种测量单个向量的大小或“长度”的方法。这项内容是理解两个向量在向量空间中“相距多远”的基本原理。计算向量间的距离是机器学习 (machine learning)中一项频繁的操作，常用于衡量表示为向量的数据点之间的相似性或差异性。

考虑一个简单的分类任务：根据其特征（如词频）识别新邮件是否为垃圾邮件。新邮件和现有邮件可以表示为高维空间 (high-dimensional space)中的向量。为了对新邮件进行分类，可以在这个空间中找到与它“最近”的现有邮件。这个思想是诸如 k-近邻（k-NN）等算法中一项主要思想。类似地，聚类算法根据数据点彼此的接近程度对它们进行分组。

### 定义向量 (vector)距离

在向量空间中，两个向量 $u$ 和 $v$ 之间的距离通常被定义为它们差向量 $u - v$ 的范数。正如可以使用各种范数来测量单个向量的长度一样，也有相应的方法来测量两个向量之间的距离。

#### 欧几里得距离 ($L_2$ 距离)

测量两个向量之间距离的最常用方法是欧几里得距离，它源自 $L_2$ 范数。它表示由特征空间中向量定义的点之间的直线距离。如果 $u = (u_1, u_2, ..., u_n)$ 和 $v = (v_1, v_2, ..., v_n)$，则欧几里得距离 $d_2(u, v)$ 为：

$d_2(u, v) = ||u - v||_2 = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$

这本质上是推广到多个维度的勾股定理。如果您可以实际绘制向量，这就是您会用尺子测量的距离。

#### 曼哈顿距离 ($L_1$ 距离)

另一种有用的距离度量方式是曼哈顿距离（也称为城市街区距离或出租车距离），它源自 $L_1$ 范数。它通过累加向量分量的绝对差值来测量距离。设想在一个城市网格中穿行，您只能沿着水平或垂直街道移动；曼哈顿距离就是总的行进距离。

$d_1(u, v) = ||u - v||_1 = \sum_{i=1}^{n} |u_i - v_i|$

在处理高维空间 (high-dimensional space)或具有不同单位或比例的特征时，曼哈顿距离有时比欧几里得距离更受青睐，因为它对单个维度中大的差异的敏感度较低（欧几里得距离会对差异进行平方处理）。

### 几何展示

考虑二维空间中的两个向量 (vector)，$u = (2, 5)$ 和 $v = (6, 2)$。



![欧几里得距离与曼哈顿距离](plots/1455-0.json)



> 欧几里得距离（蓝线）是直线路径。曼哈顿距离（橙色路径）沿网格线，累加水平和垂直部分。

欧几里得距离是 $d_2(u, v) = \sqrt{(6-2)^2 + (2-5)^2} = \sqrt{4^2 + (-3)^2} = \sqrt{16 + 9} = \sqrt{25} = 5$。
曼哈顿距离是 $d_1(u, v) = |6-2| + |2-5| = |4| + |-3| = 4 + 3 = 7$。

### 使用 NumPy 计算距离

NumPy 使计算这些距离变得简单。我们用于向量 (vector)范数的 `numpy.linalg.norm` 函数可以直接计算差向量的范数。

```python
import numpy as np

# 定义两个向量 (作为NumPy数组)
u = np.array([2, 5])
v = np.array([6, 2])

# 计算差向量
difference = u - v
print(f"差向量 (u - v): {difference}") # 输出: [-4  3]

# 计算欧几里得 (L2) 距离
l2_distance = np.linalg.norm(difference) # 默认范数为 L2
# 或者: np.linalg.norm(difference, ord=2)
print(f"欧几里得 (L2) 距离: {l2_distance}") # 输出: 5.0

# 计算曼哈顿 (L1) 距离
l1_distance = np.linalg.norm(difference, ord=1)
print(f"曼哈顿 (L1) 距离: {l1_distance}") # 输出: 7.0

# 高维向量的例子
feature_vec1 = np.array([0.1, 1.5, -2.3, 0.8])
feature_vec2 = np.array([0.3, 1.0, -2.0, 1.1])

diff_features = feature_vec1 - feature_vec2

l2_dist_features = np.linalg.norm(diff_features)
l1_dist_features = np.linalg.norm(diff_features, ord=1)

print(f"\n特征向量 1: {feature_vec1}")
print(f"特征向量 2: {feature_vec2}")
print(f"特征之间的欧几里得距离: {l2_dist_features:.4f}")
print(f"特征之间的曼哈顿距离: {l1_dist_features:.4f}")
```

### 距离在机器学习 (machine learning)中为何重要

理解如何计算向量 (vector)之间的距离对许多机器学习任务非常重要：

1. **相似度测量：** 距离越小通常表明数据点之间有更高的相似度。这用于推荐系统（寻找品味相似的用户）和分类（k-NN）。
2. **聚类：** 像 K-均值这样的算法通过最小化聚类内点之间的距离和最大化聚类之间的距离来对数据点进行分组。
3. **异常值检测：** 与所有其他点或聚类中心距离显著的数据点可能被识别为异常值。
4. **损失函数 (loss function)：** 在某些回归模型中，预测输出向量与实际目标向量之间的距离（例如，欧几里得距离）用作衡量误差（损失）的指标，模型在训练期间会尝试最小化该误差。

通过将数据表示为向量并使用距离度量，我们可以定量分析数据点之间的关系，使算法能够学习模式、做出预测并有效地对信息进行分组。欧几里得、曼哈顿或其他距离度量之间的选择通常取决于具体问题、数据的性质以及特征空间的维度。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://doi.org/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  在统计学习算法（如K-近邻和聚类）的背景下，涵盖了各种距离度量，解释了它们在模式识别中的作用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=oP8QEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了机器学习算法实现的实际示例，其中应用了向量距离，有助于理解它们在现实世界ML系统中的作用。
- [numpy.linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) — NumPy Developers (2024)
  用于计算向量范数和距离的NumPy函数官方文档，详细说明了其参数和用法。
