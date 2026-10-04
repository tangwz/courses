---
course: "linear-algebra-essentials-ml"
chapter: "eigenvalues-eigenvectors"
lesson: "significance-pca"
sourceId: 1518
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-5-eigenvalues-eigenvectors/significance-pca"
title: "主成分分析（PCA）中的作用"
description: "理解协方差矩阵的特征向量如何代表PCA中的主成分。"
order: 6
plots: ["plots/1518-0.json"]
sourceHash: "45d3af2d1cd0d0d83fa83bcea8ceb494f5ba6512a71a8f348daa9457d6fa84b3"
sourceCorrections: []
---

当应用线性变换时，特征值和特征向量 (vector)会显示空间中方向保持不变（除了缩放）的特点。此性质对于理解数据的内在结构非常有帮助，这是主成分分析（PCA）的根本。

PCA是机器学习 (machine learning)和数据分析中的一种基本方法，主要用于**降维**。其主要目的是为数据找到一组新的坐标轴，称为主成分。选择这些新轴的目的是使数据中最大的方差沿第一个轴被捕获，剩余的最大方差沿第二个轴（与第一个轴正交）被捕获，依此类推。这样我们就能用更少的维度表示数据，同时最大程度地减少信息丢失。

### 协方差矩阵：数据离散程度的度量

为了找到这些最大方差的方向，我们首先需要量化 (quantization)数据集中不同特征如何共同变化。这正是**协方差矩阵**（$C$）的作用。对于一个包含 $p$ 个特征的数据集，协方差矩阵是一个 $p \times p$ 对称矩阵。它的对角线元素 $C_{ii}$ 是第 $i$ 个特征的方差；非对角线元素 $C_{ij}$ 是第 $i$ 个特征和第 $j$ 个特征之间的协方差。

该矩阵总结了数据的离散程度和线性关系。

### 作为最大方差方向的特征向量 (vector)

这里有一个重要关联：**数据协方差矩阵 $C$ 的特征向量指向数据中最大方差的方向。** 这些方向就是主成分。此外，**与每个特征向量关联的特征值 $\lambda$ 量化 (quantization)了沿该特定方向的方差大小**。

回想定义 $Ax = \lambda x$。在PCA中，我们的矩阵 $A$ 就是协方差矩阵 $C$。特征向量 $x$（在PCA文献中常表示为 $u$ 或 $v$）就是主成分。当我们对协方差矩阵 $C$ 的一个特征向量 $u$ 进行变换时，结果就是该向量按其特征值 $\lambda$ 进行缩放：$Cu = \lambda u$。

这为何重要？事实表明，找到能使数据投影到其上（$u^T C u$）的方差最大化的方向 $u$（一个单位向量），会直接得到特征向量方程。使此方差最大的方向 $u$ *就是* 对应于 $C$ 的最大特征值的特征向量。所获得的最大方差等于该最大特征值 $\lambda_{max}$。

### 主成分：新的坐标轴

因此，数据的主成分就是其协方差矩阵 $C$ 的特征向量 (vector)，通常按特征值从高到低排序。

1. **第一主成分 (PC1)：** 与最大特征值（$\lambda_1$）关联的特征向量。此方向捕获了数据集中最多的方差。
2. **第二主成分 (PC2)：** 与第二大特征值（$\lambda_2$）关联的特征向量。此方向与PC1正交（垂直），并捕获了最大的*剩余*方差。
3. **后续主成分：** 每一个后续主成分都与其所有前一个主成分正交，并捕获了剩余数据中可能的最大方差。

### 主成分的可视化

想象一个形成椭圆形云团的二维数据点的散点图。此数据协方差矩阵的特征向量 (vector)将与该椭圆的轴对齐 (alignment)。



![主成分与数据方差对齐](plots/1518-0.json)



> 散点图显示二维数据点。PC1（红线）指示最大方差的方向，这由具有最大特征值的特征向量决定。PC2（橙色虚线）与PC1正交，捕获次大的方差。

### PCA如何进行降维

特征值说明了每个主成分（特征向量 (vector)）捕获的方差*大小*。通过将特征向量根据其对应的特征值按降序排列，我们就能知道哪些方向对描述数据离散程度最“重要”。

要将数据从 $p$ 维降到更少的 $k$ 维（其中 $k < p$），PCA 包括以下步骤：

1. 计算原始数据的协方差矩阵 $C$。
2. 计算 $C$ 的特征值和特征向量。
3. 选择与 $k$ 个最大特征值对应的 $k$ 个特征向量。这些构成了我们降维子空间的新基。
4. 通过将原始数据点投影到所选的 $k$ 个特征向量上，将它们变换到这个新子空间。

结果是一个 $k$ 维数据集，它保留了原始 $p$ 维数据中最重要的方差。前 $k$ 个主成分保留的总方差比例可以通过求前 $k$ 个特征值的和并除以所有特征值的和来计算。

### 总结

特征值和特征向量 (vector)对主成分分析非常重要，因为：

- 协方差矩阵的**特征向量**定义了主成分：数据沿其展现最大方差的正交轴。
- **特征值**衡量这些主成分方向上的方差大小。

这种关联使得PCA能够系统地辨识数据中最有用的方向，从而有效地进行降维，同时保留数据的基本结构。理解这种关联对于任何在机器学习 (machine learning)中应用或理解PCA结果的人都很有意义。在接下来的实践练习中，您将学习如何使用NumPy等库进行这些计算。

## 参考资料

- [On Lines and Planes of Closest Fit to Systems of Points in Space](https://doi.org/10.1080/14786440109462720) — Karl Pearson (1901)
  Journal: Philosophical Magazine, Series 6; Publisher: Taylor & Francis; Volume: 2; Pages: 559-572; DOI: [10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720)
  一篇奠基性论文，从几何角度引入了主轴概念，这是主成分分析的前身。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  统计学习领域的知名教材，在更广阔的机器学习背景下对PCA作为降维技术进行了全面阐述。
- [Linear Algebra and Learning from Data](https://math.mit.edu/~gs/learningfromdata/) — Gilbert Strang (2019)
  Publisher: Wellesley-Cambridge Press
  一本将线性代数概念（如特征值和特征向量）与它们在包括PCA在内的机器学习算法中的应用直接联系起来的教材。
- [Principal Component Analysis (PCA)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHgRorUeSBHfHf6yiQogHJLKzi0Ww4E5AFFjH2Os3fK9M7ayO3cLwOK__CKLyRdVctgeIwFpj8VT5e3N9bHFVe48xHfp14mEjKdCV5bwiCvy6H6SGAMKUsqg8qLKgcaL4LjPCHZn3j0ERIfakQmmMdNeQs=) — Andrew Ng (2020)
  Journal: Stanford CS229 Machine Learning Course Notes; Publisher: Stanford University; Pages: 12
  来自一个非常有影响力的机器学习课程的课堂笔记，对PCA的数学基础和应用提供了实用且直观的解释。
