---
course: "data-structures-algorithms-ml"
chapter: "graphs-for-ml"
lesson: "graph-embeddings"
sourceId: 5309
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-4-graphs-for-ml/graph-embeddings"
title: "图嵌入用于节点表示"
description: "了解将图中节点表示为低维向量以用于机器学习任务的理念。"
order: 5
plots: ["plots/5309-0.json"]
sourceHash: "4f2c3857d3655ba2cd96193b583aa08c7d3cceecb936cfe693bd50ef85cd6c3e"
sourceCorrections: []
---

图结构本身，以及BFS和DFS等遍历算法，为分析关系数据提供了有力的工具。然而，许多标准机器学习 (machine learning)算法（如scikit-learn中的算法）期望输入是固定大小的数值向量 (vector)或矩阵。我们如何弥合这一差距，并将分类、回归或聚类等方法直接应用于图中编码的丰富信息？使用简单的图统计量，如节点度或中心性度量，可以作为一个开始，但这些往往未能捕获节点之间更复杂的关联模式和相似性。

这正是**图嵌入 (embedding)**的作用所在。图嵌入方法的基本目标是学习一个映射函数，将图中的每个节点（有时也包括边或整个子图）转换为低维向量，通常称为嵌入。其主要目的是使这些学习到的向量在嵌入空间中**保持图的结构**。简单来说，图中“接近”或“相似”的节点，在向量空间中也应对应彼此接近的向量，这通常通过余弦相似度或欧几里得距离等度量来衡量。

### 图中的“相似性”意味着什么？

相似性的定义可能因具体的嵌入 (embedding)方法和应用而异：

1. **一阶邻近度：** 直接连接的节点被认为是相似的。如果节点 A 和节点 B 共享一条边，它们的嵌入 $v_A$ 和 $v_B$ 应该彼此接近。
2. **二阶邻近度：** 共享许多共同邻居的节点被认为是相似的，即使它们没有直接连接。如果节点 A 和 B 都连接到节点 X、Y 和 Z，它们共享一个相似的局部邻域结构，并且它们的嵌入应该反映这一点。
3. **结构等价性：** 在图结构中扮演相似作用的节点（例如，两个作为不同社区之间桥梁的节点）可能被认为是相似的，即使它们在图中相距很远且没有共同邻居。
   "4. **同质性：** 在许多网络（如社交网络）中，连接的节点往往共享相似的属性或标签。嵌入可以旨在将具有相似属性的节点放置在彼此接近的位置。"

考虑这个简单的图：

> 节点A、B和C构成一个连接组，而D和E构成另一个。节点C与D之间有一个弱连接。

成功的嵌入方法旨在将这些节点映射到向量 (vector)空间中（例如用于可视化的二维空间），从而保持邻近关系：



![模拟二维节点嵌入](plots/5309-0.json)



> 节点A、B和C被映射到嵌入空间中彼此靠近的向量，反映了它们在图中的邻近性。D和E也彼此接近，但与A-B-C群组相距较远。C的位置可能受到其与D链接的轻微影响。

### 学习嵌入 (embedding)的方法

存在多种生成图嵌入的方法。虽然本节的范围不包括对每种方法进行详细研究，但了解其主要思想是有益的：

1. **矩阵分解：** 这些方法通常从图的矩阵表示开始，例如邻接矩阵 $A$，或拉普拉斯矩阵 $L$ 等相关矩阵。应用类似于推荐系统中使用的技术（如奇异值分解SVD）来将此矩阵分解为低维因子。这些因子作为节点嵌入。其直觉是，矩阵分解过程隐式捕获了节点之间的潜在关系。
2. **基于随机游走的方法：** 受自然语言处理（NLP）中Word2Vec等方法的启发，DeepWalk和node2vec等方法模拟从图中每个节点开始的短随机游走。随机游走生成节点序列（例如，$A \rightarrow C \rightarrow B \rightarrow A \rightarrow C \rightarrow D$）。这些序列被视为“句子”，其中节点是“单词”。然后使用Skip-gram或Continuous Bag-of-Words (CBOW)等方法来学习嵌入，使得在这些游走中频繁共同出现的节点被映射到彼此接近的向量 (vector)。这些方法在捕获局部邻域结构（邻近性）方面很有效。Node2vec引入了参数 (parameter)来偏置 (bias)游走，允许在捕获同质性（相似节点）和结构等价性（作用相似节点）之间进行插值。
3. **图神经网络 (neural network)（GNN）：** 这是一种基于深度学习 (deep learning)的较新且功能强大的方法体系。GNN直接在图结构上操作。在典型的GNN层中，每个节点聚合来自其邻居的信息（特征或嵌入），将其与自身的当前表示结合，并通过神经网络转换来更新其表示。通过堆叠这些层，节点的嵌入可以包含来自多跳之外节点的信息。GNN可以学习复杂的模式，并能自然地整合节点和边的特征。它们通常生成嵌入作为中间表示，用于下游任务，如节点分类或链接预测。GNN本身是一个重要主题，通常在专门的课程或进阶材料中讲授。

### 优点与用途

一旦拥有这些节点嵌入 (embedding)（向量 (vector)），您就可以将它们用作各种机器学习 (machine learning)任务的特征：

- **节点分类：** 预测节点的标签或属性（例如，对社交网络中的用户进行分类，识别蛋白质功能）。节点嵌入作为标准分类器（例如，逻辑回归、SVM、前馈神经网络 (neural network)）的输入特征。
- **链接预测：** 预测两个当前未连接的节点之间是否可能存在边（例如，在社交网络中推荐朋友，推荐产品）。特征可以通过组合两个候选节点的嵌入来获得（例如，拼接、元素级乘积）。
- **社区检测/聚类：** 使用标准聚类算法（例如，K-Means、DBSCAN），根据节点在嵌入空间中的相似性将其分组到簇中。具有相似结构作用的节点或密集子图内的节点通常在嵌入空间中最终彼此接近。
- **图分类：** 为整个图生成一个嵌入（例如，通过池化节点嵌入），以对不同的图进行分类（例如，区分不同类型的分子）。
- **可视化：** 嵌入，尤其是在降维到2或3维（使用PCA或t-SNE等方法）时，提供了一种可视化大型图高层结构的方式。

### 考虑事项

虽然功能强大，但选择和使用图嵌入 (embedding)方法涉及几个考虑事项：

- **方法的选择：** 不同方法捕获图结构的不同方面。随机游走适用于局部邻域，矩阵分解可能捕获更全局的结构，而GNN提供灵活性，但需要更多数据和计算资源。
- **超参数 (parameter) (hyperparameter)：** 嵌入维度、随机游走参数（长度、游走次数）、GNN架构选择以及优化参数都显著影响学习嵌入的质量。
- **可伸缩性：** 某些方法比其他方法更适合大规模图。随机游走方法和一些GNN采样技术通常更受青睐，适用于非常大的网络。
- **评估：** 评估嵌入质量通常涉及衡量它们在下游机器学习 (machine learning)任务中的表现。

图嵌入起到了桥梁作用，使得我们能够将图中存储的复杂关系信息转换为适合广泛标准机器学习模型的格式，从而让图数据能够用于预测建模和分析。

## 参考资料

- [DeepWalk: Online Learning of Social Representations](http://doi.acm.org/10.1145/2623330.2623732) — Bryan Perozzi, Rami Al-Rfou, Steven Skiena (2014)
  Journal: KDD '14: Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 701-710; DOI: [10.1145/2623330.2623732](https://doi.org/10.1145/2623330.2623732)
  这篇基础性论文介绍了DeepWalk，这是一种开创性的基于随机游走的节点嵌入方法，通过将随机游走视为“句子”并应用类似Word2Vec的技术来学习节点嵌入。
- [node2vec: Scalable Feature Learning for Networks](http://doi.acm.org/10.1145/2939672.2939754) — Aditya Grover, Jure Leskovec (2016)
  Journal: KDD '16: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 855-864; DOI: [10.1145/2939672.2939754](https://doi.org/10.1145/2939672.2939754)
  这篇论文通过提出node2vec扩展了DeepWalk，该算法通过有偏随机游走学习节点嵌入，从而在类BFS和类DFS的邻域探索之间实现灵活的平衡。
- [Graph Neural Networks: A Review of Methods and Applications](https://doi.org/10.1016/j.aiopen.2021.01.001) — Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Maosong Sun, and Liqiang Nie (2021)
  Journal: AI Open; Volume: 1; Pages: 57-71; DOI: [10.1016/j.aiopen.2021.01.001](https://doi.org/10.1016/j.aiopen.2021.01.001)
  一份全面的综述，概述了各种图神经网络（GNN）模型、其应用和未来研究方向，对于理解这类强大的图嵌入技术非常有价值。
