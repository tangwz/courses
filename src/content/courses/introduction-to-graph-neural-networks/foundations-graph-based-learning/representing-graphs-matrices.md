---
course: "introduction-to-graph-neural-networks"
chapter: "foundations-graph-based-learning"
lesson: "representing-graphs-matrices"
sourceId: 7615
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning/representing-graphs-matrices"
title: "图的表示：邻接矩阵与特征矩阵"
description: "学习表示图拓扑和属性的标准数值方法，包括邻接矩阵和节点特征矩阵。"
order: 4
plots: []
sourceHash: "568986030115c5cba5deaba7730e126c3d65fa3cf61718532e2c57bb7d2a0006"
sourceCorrections: []
---

要将机器学习 (machine learning)模型应用于图数据，我们首先需要将节点和边的抽象结构转换为算法可以处理的数值格式。正如我们将图像表示为像素值网格，或将文本表示为数值向量 (vector)序列一样，我们需要一种标准的方法来对图进行编码。这主要通过两个矩阵来实现：捕获图拓扑结构的**邻接矩阵**，以及保存每个节点属性的**特征矩阵**。

### 表示图结构：邻接矩阵

表示图中连接关系最直接的方法是使用**邻接矩阵**，通常记为 $A$。对于一个拥有 $N$ 个节点的图，邻接矩阵是一个大小为 $N \times N$ 的方阵。

填充该矩阵的规则很简单。对于无权图，第 $i$ 行第 $j$ 列的元素 $A_{ij}$ 为：


$$
A_{ij} = \begin{cases} 
1 & \text{如果节点 } i \text{ 与节点 } j \text{ 之间存在边} \\
0 & \text{否则}
\end{cases}
$$


按照惯例，节点通常不被视为与自身相连，因此对角线元素 $A_{ii}$ 通常设置为 0。

以下面这个包含四个人的简单社交网络图为例。

> 一个包含四个节点（0-3）的无向图，节点代表个人，边代表友谊关系。

该 4 节点图对应的邻接矩阵 $A$ 为：


$$
A = \begin{pmatrix}
0 & 1 & 1 & 0 \\
1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 \\
0 & 1 & 1 & 0
\end{pmatrix}
$$


请注意该矩阵的几个特性：

- **对称性：** 因为该图是无向的（小安和小本之间的友谊与小本和小安之间的友谊相同），所以矩阵是对称的，即 $A_{ij} = A_{ji}$。如果图是有向的（例如表示“关注”关系），矩阵则不一定对称。
- **稀疏性：** 该矩阵中的 0 比 1 多。对于大多数大型网络（如社交网络或引用网络），节点仅与所有其他节点中的一小部分相连。这导致邻接矩阵高度稀疏，存储效率较低。在实际应用中，图数据通常以**邻接表**（或 COO 格式）等格式存储，仅列出相连的节点对。尽管如此，邻接矩阵仍然是开发图神经网络 (neural network)理论的标准数学表示方式。

对于边具有不同强度（如交互频率）的**有权图**，矩阵条目 $A_{ij}$ 将包含边的权重 (weight)，而不仅仅是 1。

### 表示节点属性：特征矩阵

在大多数应用中，节点本身包含有用的信息。社交网络中的用户拥有个人资料（年龄、位置），生物网络中的蛋白质具有化学性质。这些信息存储在**节点特征矩阵**中，通常记为 $X$。

对于一个拥有 $N$ 个节点且每个节点有 $F$ 个特征的图，特征矩阵 $X$ 的维度为 $N \times F$。矩阵的每一行 $i$ 对应于节点 $i$，并包含其特征向量 (vector)。

让我们为示例图中的每个人分配两个特征：年龄和所属组别（编码为 0 或 1）。

- **节点 0 (小安)：** 年龄 25，组别 0
- **节点 1 (小本)：** 年龄 30，组别 0
- **节点 2 (小克)：** 年龄 22，组别 1
- **节点 3 (小丹)：** 年龄 28，组别 1

这些信息可以组织成一个 $4 \times 2$ 的特征矩阵 $X$：


$$
X = \begin{pmatrix}
25 & 0 \\
30 & 0 \\
22 & 1 \\
28 & 1
\end{pmatrix}
$$


第一列代表年龄，第二列代表组别。该矩阵在学习开始前提供了每个节点的初始状态或属性。图神经网络 (neural network)的目标通常是利用这些特征以及图结构，学习出更具表达力的节点表示。

### 整体架构

邻接矩阵 $A$ 和节点特征矩阵 $X$ 共同构成了一个带属性图的完整数值表示。它们是几乎所有图神经网络 (neural network)模型的核心输入。

- $A$ ($N \times N$) 告知模型**谁与谁相连**。
- $X$ ($N \times F$) 告知模型**每个节点是什么样的**。

图神经网络的核心操作（我们将在下一章中分析）涉及使用由 $A$ 定义的结构来传播和转换 $X$ 中包含的信息。这使得每个节点都能从其邻居那里学习，并整合自身属性和局部网络背景信息。

另外值得注意的是，在某些图中，边也可能具有特征。例如在分子图中，代表化学键的边可以有不同的类型（单键、双键）。这些信息通常存储在单独的边特征张量中。目前，我们将专注于 $A$ 和 $X$ 这一基础组合。

## 参考资料

- [Introduction to Graph Theory](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQERfQEadCtfSc7CWYzC-mUH-86b9-AN2RF7iTVrHI7ogVIIHqRYtNTEeAibvK_Qz8wm7iS3Zs5EO1m1at3CSYuElvD-XKJ54nJ4ZYa3XPhHnBTIcpLS2BBlhG6DDsJr) — Douglas B. West (2001)
  Publisher: Prentice Hall
  一本经典的图论教科书，涵盖图的基本概念，包括邻接矩阵和其他图表示方法。
- [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) — William L. Hamilton (2020)
  Journal: Synthesis Lectures on Artificial Intelligence and Machine Learning; Publisher: Morgan and Claypool; Volume: 14; Pages: 1-159; DOI: [10.2200/S00998ED1V01Y202003AIM007](https://doi.org/10.2200/S00998ED1V01Y202003AIM007)
  一本关于图表示学习的权威书籍，从图结构和节点特征的数值编码开始。
- [Dive into Deep Learning](https://d2l.ai/chapter_gnns/index.html) — Aston Zhang, Zachary C. Lipton, Mu Li, Alex Smola, and others (2024)
  Publisher: Cambridge University Press
  一本易于访问的在线教科书，其中专门的图神经网络章节解释了如何将图数值化表示作为输入。
- [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/8940833) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2970760](https://doi.org/10.1109/TNNLS.2020.2970760)
  一份被广泛引用的综述，全面概述了图神经网络，包括将图表示为输入数据的基本方法。
