---
course: "introduction-to-machine-learning"
chapter: "unsupervised-learning-clustering"
lesson: "limitations-of-k-means"
sourceId: 1515
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-5-unsupervised-learning-clustering/limitations-of-k-means"
title: "K-均值算法的局限性"
description: "简要讨论K-均值算法的一些局限性及假设。"
order: 6
plots: ["plots/1515-0.json", "plots/1515-1.json"]
sourceHash: "fa281c136c630f60e53af1607504a939121d43ab02a690ed13f438df463942cf"
sourceCorrections: []
---

虽然K-均值是一种常用且通常有效的将数据分组的算法，但了解它的缺点很有用。熟悉这些局限性有助于您判断何时使用K-均值算法是合适的，以及如何理解其结果。

### 对初始质心位置的敏感性

K-均值算法通过随机放置初始的 $K$ 个质心来开始。这些质心的起始位置会显著影响最终的聚类结果。想象一下从地图上不同地点开始寻宝；根据您的起点和路线，您可能会发现不同的宝藏，不一定是最大的那个。

同样地，K-均值算法可能会收敛到所谓的**局部最优解**，这表示它找到了一个相对于其起始点而言不错的聚类方案，但可能不是**全局最优解**（给定数据和 $K$ 值下整体上最好的聚类方案）。由于这种随机初始化，在相同数据上多次运行K-均值算法可能会产生不同的聚类。



![不同的初始质心可能导致不同的结果](plots/1515-0.json)



> 根据K-均值算法是使用质心1还是质心2作为起点，蓝色数据点的最终聚类分配结果可能会有所不同。

一种常见的缓解方法是多次运行K-均值算法（例如，10次或更多次），每次使用不同的随机初始化。大多数软件实现，例如Python中的Scikit-learn，都内置了自动执行此操作的选项（通常通过 `n_init` 等参数 (parameter)控制）。实现会根据诸如簇内平方和（惯性）之类的指标选择最佳运行结果，以求得更一致且可能更好的结果。

### 需要提前指定K值

“也许K-均值算法最主要的一个实际问题是，您必须在运行算法*之前*告知它要找出多少个簇（$K$）。在许多情况下，最佳簇的数量事先是不知道的。”

选择错误的 $K$ 值可能导致不佳的结果：

- 如果 $K$ 过小，不同的组可能会被合并到一个簇中。
- 如果 $K$ 过大，自然形成的组可能会被不必要地分成多个小簇。

如前一节（“选择簇的数量 (K)”）所述，肘部法则等方法可以提供指导，但它们通常是启发式的，可能无法给出单一、明确的答案。选择 $K$ 值通常需要一些尝试和专业知识。

### 对球形和大小相近簇的假设

K-均值算法通过最小化数据点与其分配到的质心之间的距离来工作。从数学角度看，它试图最小化每个簇内部的方差。这个过程固有的假设是簇具有以下特点：

1. **球形：** 簇预计大致呈圆形（在二维空间中）或超球形（在更高维度中）。K-均值算法难以处理细长、弯曲或不规则形状的簇。
2. **大小相近：** 算法倾向于生成包含相似数量点的簇。它可能在识别大小差异很大的簇时遇到问题。
3. **密度相近：** K-均值算法隐含地假设簇的密度大致相同（点在每个簇中分布得相似）。

考虑以下K-均值算法可能表现不佳的数据分布情况：



![K-均值假设与实际情况](plots/1515-1.json)



> K-均值算法对左侧数据（球形、大小相近）效果良好。它难以处理右侧的细长簇以及两个密度/大小不同的簇，常常不自然地将它们分开或放置错质心。

由于K-均值将每个点分配到*最近的*单个质心，它有效地使用质心之间等距的边界来划分数据空间。这个过程自然会创建凸形（如沃罗诺伊图）。如果数据的基础结构是非凸的（如月牙形或环形），K-均值将无法准确地识别它。

### 对异常值的敏感性

异常值是远离大多数其他点的数据点。由于K-均值使用簇中点的平均位置来计算质心，它可能对异常值敏感。单个异常值可以大幅度地将质心拉向自身，可能扭曲簇并错误分类本应属于其他地方的点。

想象一下计算一个小镇的平均收入；如果一个亿万富翁搬进来，平均收入会飙升，即使大多数人的收入没有变化。同样地，一个异常值可以大幅度地移动K-均值质心。

识别和处理异常值的预处理步骤，或者使用对异常值不那么敏感的聚类算法（如K-中心点算法或DBSCAN，这些超出了本入门课程的范围），在存在异常值时可能是必需的。

了解这些局限性并非旨在劝退您使用K-均值算法。它仍然是一个基本且计算高效的聚类算法，适用于许多任务。然而，认识到其假设和潜在问题使您能够更有效地应用它，并识别出何时其他无监督学习 (supervised learning) (unsupervised learning)技术可能更合适的情况。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  清晰地解释了K-Means算法、其原理及局限性，包括初始化问题和对聚类形状的假设。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher Bishop (2006)
  Publisher: Springer; DOI: [10.1007/b139369](https://doi.org/10.1007/b139369)
  详细阐述了K-Means算法，包括其目标函数，并讨论了其对初始化和几何假设的敏感性。
- [k-means++: The Advantages of Careful Seeding](https://doi.org/10.1145/1283383.1283494) — David Arthur, Sergei Vassilvitskii (2007)
  Journal: Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms; Publisher: Society for Industrial and Applied Mathematics; Pages: 1027-1035; DOI: [10.1145/1283383.1283494](https://doi.org/10.1145/1283383.1283494)
  介绍了广泛使用的K-Means++初始化算法，通过解决K-Means对初始中心点放置的敏感性，显著提高了其结果的质量和一致性。
