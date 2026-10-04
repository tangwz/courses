---
course: "linear-algebra-essentials-ml"
chapter: "vector-spaces-subspaces-linear-independence"
lesson: "basis-dimension"
sourceId: 1503
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-4-vector-spaces-subspaces-linear-independence/basis-dimension"
title: "基与维数"
description: "定义向量空间的基并理解维数的意义。"
order: 5
plots: ["plots/1503-0.json"]
sourceHash: "62f67c2cf0fc41398aaff77aa60289b93fd31c229d0e0600c58768df9db620db"
sourceCorrections: []
---

在前几节中，我们研究了向量 (vector)如何进行组合（线性组合），以及一组向量如何生成整个空间或子空间（张成空间）。我们还考察了集合中的向量何时是多余的（线性相关），以及何时每个向量都提供独特的方向信息（线性无关）。现在，我们将这些知识结合起来，以阐明描述向量空间最有效的方法：使用一个*基*。

可以把基视为向量空间的基本构成单元或坐标系统。它是达到该空间中*任何*点所需的最小方向集合，且这些方向中的任何一个都不能用其他方向表示出来。

### 什么是基？

在向量 (vector)空间$V$中，一组向量$B = \{ \mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n \}$被称为$V$的**基**，如果它满足以下两个条件：

1. **线性无关性：** $B$中的向量是线性无关的。这意味着集合中没有任何向量可以写成集合中其他向量的线性组合。每个向量都提供独特的方向信息。方程$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_n\mathbf{v}_n = \mathbf{0}$的唯一解是$c_1 = c_2 = \dots = c_n = 0$。
2. **张成性质：** 集合$B$张成向量空间$V$。这意味着$V$中的每个向量$\mathbf{w}$都可以表示为$B$中向量的线性组合。也就是说，对于任何$\mathbf{w} \in V$，都存在标量$c_1, c_2, \dots, c_n$，使得$\mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_n\mathbf{v}_n$。

如果一组向量张成空间但不线性无关，则它包含冗余向量。如果它线性无关但不能张成空间，则无法描述该空间内的所有向量。基实现了恰当的平衡。

#### 例子：标准基

最常见的例子是$\mathbb{R}^n$的标准基。

- 对于$\mathbb{R}^2$，标准基是$\{ \mathbf{e}_1, \mathbf{e}_2 \} = \{ [1, 0]^T, [0, 1]^T \}$。这两个向量显然线性无关（彼此都不是对方的倍数），并且任何向量$[x, y]^T$都可以写成$x\mathbf{e}_1 + y\mathbf{e}_2$。
- 对于$\mathbb{R}^3$，标准基是$\{ \mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3 \} = \{ [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T \}$。同样，它们是线性无关的，并且任何向量$[x, y, z]^T$都可以写成$x\mathbf{e}_1 + y\mathbf{e}_2 + z\mathbf{e}_3$。



![Interactive chart](plots/1503-0.json)



> $\mathbb{R}^2$中的标准基向量$\mathbf{e}_1$（蓝色）和$\mathbf{e}_2$（红色）。它们是正交的，长度为单位，并张成整个二维平面。

#### 非标准基

一个向量空间可以有许多不同的基。例如，在$\mathbb{R}^2$中，集合$B' = \{ [1, 1]^T, [1, -1]^T \}$也是一个基。

1. **线性无关性：** 令$c_1[1, 1]^T + c_2[1, -1]^T = [0, 0]^T$。这得到方程组$c_1 + c_2 = 0$和$c_1 - c_2 = 0$。唯一解是$c_1 = 0, c_2 = 0$。因此，它们是线性无关的。
2. **张成性：** 我们需要表明任何向量$[x, y]^T$都可以写成$c_1[1, 1]^T + c_2[1, -1]^T$。这导致方程组$c_1 + c_2 = x$和$c_1 - c_2 = y$。求解后得到$c_1 = (x+y)/2$和$c_2 = (x-y)/2$。由于我们可以为任何$x, y$找到$c_1, c_2$，所以该集合张成$\mathbb{R}^2$。

由于两个条件都成立，$B'$是$\mathbb{R}^2$的一个有效基。

### 表示的唯一性

基的定义带来一个重要特性，即空间$V$中的每个向量 (vector)都可以表示为基向量的*唯一*线性组合。如果表示不唯一，就意味着可以通过不止一种方式组合基向量以获得相同的目标向量，这将与线性无关性要求相悖。与基$B$相关的这组唯一系数$(c_1, c_2, \dots, c_n)$构成了向量$\mathbf{w}$相对于该基的*坐标*。

### 维数

向量 (vector)空间的一个基本属性是，**给定向量空间的所有基都包含相同数量的向量**。这个相同的数量被称为向量空间的**维数**，通常表示为$\text{dim}(V)$。

- $\mathbb{R}^2$的维数是2，因为它的基（如标准基或$B'$)包含两个向量。
- $\mathbb{R}^3$的维数是3。
- 通常，$\mathbb{R}^n$的维数是$n$。

$\mathbb{R}^3$中通过原点的平面是维数为2的子空间。$\mathbb{R}^3$中通过原点的直线是维数为1的子空间。只包含零向量$\{\mathbf{0}\}$的空间维数为0（根据规定，它有一个空基）。

如果一个向量空间不能由有限组向量张成，它被称为*无限维*的。函数空间通常属于此类，但在大多数涉及特征向量的典型机器学习 (machine learning)情况中，我们处理的是有限维向量空间。

### 与机器学习 (machine learning)的关联

理解基和维数在机器学习中具有多方面作用，原因如下：

1. **特征空间解释：** 数据集所有可能的特征向量 (vector)集合通常形成一个向量空间（或其子集）。这个空间（或相关子空间）的维数对应于特征的数量，更准确地说，是描述数据所需的*独立*特征的数量。
2. **降维：** 像主成分分析（PCA）这样的技术明确地寻找一个低维基（子空间的基），该基能够捕捉原始高维数据中最重要的变化。目的是使用更少的维度（基向量）有效表示数据。
3. **识别冗余：** 如果一组特征向量线性相关，这意味着某些特征可以表示为其他特征的组合。它们不会向新的方向扩展数据的张成空间。找到这些特征向量所张成空间的基，有助于识别一组核心的、非冗余的特征或潜在因素。维数告诉我们驱动数据的固有独立因素数量。
4. **模型复杂度：** 输入特征空间的维数通常会影响机器学习模型所需的复杂程度（例如，线性模型中的参数 (parameter)数量）。理解实际维数可以指导模型选择和正则化 (regularization)方法。

本质上，基和维数为量化 (quantization)特征空间的“大小”和结构提供了一种正式途径，有助于我们分析数据、降低复杂性并构建更有效的模型。在接下来的部分中，我们将查看与矩阵相关的特定子空间以及它们的维数如何关联。

## 参考资料

- [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/) — Gilbert Strang (2016)
  Publisher: Wellesley-Cambridge Press
  一本基础教科书，涵盖了线性代数的基本概念，包括基、维度、线性独立性和张成空间等，并提供了清晰的解释和示例。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679989](https://doi.org/10.1017/9781108679989)
  全面且易懂地介绍了机器学习所需的数学概念，其中包含针对机器学习背景解释的向量空间、基和维度的专门章节。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第二章为机器学习所需的线性代数概念（包括向量空间、线性独立性、基和维度）提供了易懂的介绍，为更高级的主题奠定了基础。
