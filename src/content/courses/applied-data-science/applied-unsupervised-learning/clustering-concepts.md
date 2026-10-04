---
course: "applied-data-science"
chapter: "applied-unsupervised-learning"
lesson: "clustering-concepts"
sourceId: 1877
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-4-applied-unsupervised-learning/clustering-concepts"
title: "理解聚类方法要点"
description: "回顾无监督聚类算法的主要思想。"
order: 1
plots: ["plots/1877-0.json"]
sourceHash: "0dbd8a5e2ce84beb143092035e5cb5b7ca53e3ce0a1ca521a32f9dd733926ed6"
sourceCorrections: []
---

如前所述，无监督学习 (supervised learning) (unsupervised learning)帮助我们发现数据中固有的模式，无需依赖预先存在的标签。聚类是无监督学习中的一种基本方法，专门用于将相似数据点归集到一起。可以将其视为根据数据点本身的特征，自动将数据整理成不同的类别。

聚类的主要目标是将数据集划分，使得同一组（或*簇*）内的点彼此之间比与其他组的点更相似。这在许多情况下很有用，例如：

- **客户细分：** 根据相似的购买行为或人口统计学信息对客户进行分组，以便进行精准营销。
- **文档分析：** 按主题对文章或文档进行分组。
- **图像分割：** 根据颜色或纹理对图像中的像素进行分组。
- **异常检测：** 识别不明显属于任何簇的点。
- **数据概括：** 理解数据集中主要的原型或概况。

### 衡量相似度：聚类方法的依据

我们如何判断两个数据点是否“相似”？这通常通过在特征空间中计算的距离或相似度量来量化 (quantization)。度量的选择非常重要，并且很大程度上取决于数据本身的特性。

#### 欧几里得距离

数值数据最常用的度量是欧几里得距离。它表示特征空间中两点之间的直线距离。对于$n$维空间中的两点$p = (p_1, p_2, ..., p_n)$和$q = (q_1, q_2, ..., q_n)$，欧几里得距离计算如下：


$$
d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}
$$


此度量直观易懂，但对特征的尺度很敏感。如果某个特征的范围远大于其他特征（例如，薪水与工作年限相比），它可能会在距离计算中占据主导地位。因此，如第1章所述，在应用使用欧几里得距离的聚类算法之前，特征缩放（如标准化或归一化 (normalization)）通常是必要的预处理步骤。

#### 曼哈顿距离

另一个常用度量是曼哈顿距离（或L1距离）。它通过对坐标的绝对差求和来衡量距离，类似于在城市网格中导航，只能沿着水平或垂直路径移动。


$$
d(p, q) = \sum_{i=1}^{n}|p_i - q_i|
$$


曼哈顿距离与欧几里得距离相比，对异常值不太敏感，在某些高维情况或特征表示不同方向或数量时，可能更受青睐。

#### 其他度量

尽管欧几里得距离和曼哈顿距离是数值数据的常用选择，但针对不同数据类型还存在其他度量：

- **余弦相似度：** 通常用于表示为向量 (vector)的文本数据（如TF-IDF），衡量的是两个向量之间夹角的余弦值，而非它们的大小差异。它在高维数据中表现良好。
- **汉明距离：** 用于分类数据，计算对应符号不同的位置数量。

选择合适的距离度量是聚类过程中的重要首步。

### 聚类算法的主要方法

聚类算法在定义簇和寻找簇的方式上有所不同。主要类别如下：

- **基于质心的聚类：** 这类算法通过一个中心向量 (vector)或*质心*来表示簇。最著名的例子是K-Means。数据点被分配到其质心最近的簇（通常使用欧几里得距离）。算法通常迭代地更新质心位置和点分配，以最小化点到其分配质心的距离。
- **基于密度的聚类：** 这些方法将簇定义为数据点的密集区域，这些区域由较低点密度的区域分隔。DBSCAN是一个常用示例。这类算法可以发现任意形状的簇，并且能有效处理噪声（不属于任何密集区域的点）。它们通常不需要预先指定簇的数量，而是依赖于密度参数 (parameter)。
- **层次聚类：** 这类算法构建簇的层次结构。*凝聚式*（自底向上）方法从每个点作为一个簇开始，并迭代地合并最接近的簇对。*分裂式*（自顶向下）方法从所有点在一个簇中开始，并递归地将其分裂。结果通常可视化为称为树状图的树形图，它允许通过在某个级别切割树来选择簇的数量。

下图显示了一个简单数据集，我们可能会凭视觉识别出潜在的群组，聚类算法旨在自动找出这些群组。



![用于聚类的示例数据分布](plots/1877-0.json)



> 示例：无标签的二维数据点，其中似乎存在明显的群组。聚类算法旨在正式识别这些群组。

### 聚类面临的挑战

聚类功能强大，但也伴随一些固有的挑战：

- **确定簇的数量（K）：** 对于K-Means等算法，需要预先指定簇的数量($K$)，但最优数量通常未知。像“肘部法则”或“轮廓分析”（我们稍后可能会提及）这样的方法可以帮助指导此选择，但这通常涉及一些领域知识或实验。
- **对初始化和参数 (parameter)的敏感性：** 某些算法（尤其是K-Means）可能会根据初始起始条件（例如，初始质心位置）产生不同的结果。基于密度的方法取决于定义“密度”的参数，这可能需要调整。
- **处理高维数据：** 随着特征数量的增加（高维度），距离度量可能变得意义不大（“维度灾难”）。点可能看起来等距，使得难以形成明显的簇。降维技术（如PCA，本章稍后讨论）在聚类之前可能有所帮助。
- **结果评估：** 由于无监督学习 (supervised learning) (unsupervised learning)中没有真实标签，评估簇的“质量”是主观的，通常依赖于*内部*验证指标（衡量簇的凝聚度和分离度，如轮廓系数）或基于领域知识的解释。

理解这些要点为有效应用特定聚类算法提供了依据。在接下来的章节中，我们将实现并学习两种广泛使用的算法：K-Means和DBSCAN。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  介绍统计学习方法，包括聚类等无监督技术，并提供实用示例。
- [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](http://www.aaai.org/Library/KDD/1996/kdd96-037.php) — Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996)
  Journal: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96); Publisher: AAAI Press; Pages: 226-231; DOI: [10.1145/300146.300150](https://doi.org/10.1145/300146.300150)
  介绍了DBSCAN算法，该算法根据密度可达性定义聚类，并能识别任意形状的聚类和异常点。
- [Clustering](https://scikit-learn.org/stable/modules/clustering.html) — scikit-learn developers (2023)
  Publisher: scikit-learn project
  scikit-learn中各种聚类算法的指南，包含其原理和参数说明。
