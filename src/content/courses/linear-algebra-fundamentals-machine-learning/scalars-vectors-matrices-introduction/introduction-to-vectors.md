---
course: "linear-algebra-fundamentals-machine-learning"
chapter: "scalars-vectors-matrices-introduction"
lesson: "introduction-to-vectors"
sourceId: 7804
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-fundamentals-machine-learning/chapter-1-scalars-vectors-matrices-introduction/introduction-to-vectors"
title: "向量：空间中的点"
description: "了解什么是向量，它们如何表示大小和方向，以及它们的几何视图。"
order: 3
plots: ["plots/7804-0.json"]
sourceHash: "63ad54f4456447038b22d9af48f0be18ea5de68f7802ffb50691ee7fd2fa2256"
sourceCorrections: []
---

标量只给我们单个信息，但多数数据并非如此简单。想象一下，要为房地产网站描述一栋房子。你不会只列出它的价格。你会列出它的面积、卧室数量和浴室数量。像这样有顺序的数字列表就是一个**向量 (vector)**。

向量是存储具有多个属性的数据的一种基本方式。如果一栋房子有1500平方英尺，3间卧室和2间浴室，我们可以用向量 $\mathbf{x}$ 来表示它：


$$
\mathbf{x} = [1500, 3, 2]
$$


向量中的每个数字都是一个**元素**或**分量**。顺序很重要。向量 $[3, 2, 1500]$ 将代表一栋完全不同的房子，一栋有3平方英尺、2间卧室和1500间浴室的房子，这没有什么意义。

### 几何视图

向量 (vector)作为数字的简单载体，具有强大的几何意义。为简单起见，我们只考虑一个有两个分量的向量，例如 $\mathbf{v} = [3, 4]$。我们可以在二维（2D）坐标系中将其可视化。

有两种常见的思考方式：

1. **作为点：** 向量 $\mathbf{v} = [3, 4]$ 可以表示空间中一个点的坐标。你沿x轴移动3个单位，沿y轴移动4个单位。
2. **作为箭头：** 向量也可以看作是一个从原点 $(0, 0)$ 开始，指向点 $(3, 4)$ 的箭头。

这种箭头视图尤其有益，因为它赋予了向量两个明显的属性：**大小**（箭头有多长？）和**方向**（箭头指向何处？）。



![向量 v = \[3, 4\] 在二维空间中](plots/7804-0.json)



> 向量 $\mathbf{v} = [3, 4]$ 被可视化为从原点出发的箭头和二维平面中的一个点。

在机器学习 (machine learning)中，我们经常处理多维度的数据，远超我们容易可视化的两或三维。例如，一个表示28x28像素灰度图像的向量将有 $28 \times 28 = 784$ 个分量。即使我们无法绘制784维空间，长度和方向的几何观念仍然适用，并且它们是许多算法的核心。

### 记法

在教科书和数学论文中，你会看到几种常用的向量 (vector)记法。

- **小写粗体字母：** 向量通常用小写粗体字母表示，例如 $\mathbf{v}$ 或 $\mathbf{x}$。这是我们将遵循的惯例。
- **列向量与行向量：** 默认情况下，在线性代数中，向量通常被假定为**列向量**。

列向量将其分量垂直排列：


$$
\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$


**行向量**则将其分量水平排列，就像我们目前为止所写的那样：


$$
\mathbf{v} = [3, 4]
$$


当我们开始将向量与矩阵相乘时，这种区别就显得关键。目前，只需认识到这两种形式都存在。向量中分量的数量决定了它的**维度**。向量 $[3, 4]$ 是二维的，而 $[1500, 3, 2]$ 是三维的。

向量提供了一种封装单个数据点特征的方式。但常见的数据集包含许多数据点。为了组织一整个向量集合，我们需要下一个构成要素：矩阵。

## 参考资料

- [Introduction to Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  经典教材，介绍向量及其几何解释和基本运算，是学习线性代数的标准参考书。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679933](https://doi.org/10.1017/9781108679933)
  涵盖线性代数基础知识，直接关注机器学习应用，从向量定义和属性开始。
- [18.06SC Linear Algebra](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) — Gilbert Strang (2011)
  Publisher: MIT OpenCourseWare
  Gilbert Strang教授的免费在线课程，涵盖线性代数基础主题，包括向量及其几何方面。
