---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-neural-networks-architectures-training"
lesson: "quantum-graph-neural-networks"
sourceId: 1066
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-5-quantum-neural-networks-architectures-training/quantum-graph-neural-networks"
title: "量子图神经网络 (QGNNs)"
description: "考察开发量子图神经网络的方法，运用量子计算原理。"
order: 4
plots: []
sourceHash: "308f640ebf33ead9a1d9f0108990bf0309b6524d24c3058c673ba1ce6d55f530"
sourceCorrections: []
---

图是表示实体间关系的基本结构，普遍存在于社交网络、分子结构、知识图谱及优化问题等。经典图神经网络 (neural network)（GNNs）已成为在图结构数据上学习表示并进行预测的有效工具。它们通常通过迭代汇聚节点局部邻域的信息来运作，从而有效进行消息传递。

量子图神经网络（QGNNs）旨在将这些思想应用于量子计算范畴，试图运用量子计算原理来处理图信息。其动机源于多个方面：

1. **表示能力：** 希尔伯特空间中的量子态，相较于经典向量 (vector)，为表示复杂图结构及节点/边特征提供了潜在更丰富的空间。量子纠缠可以捕捉节点间复杂的关联。
2. **量子动力学：** 受哈密顿量或酉算符控制的量子演化，或能为图上的信息传播和转换提供新颖机制，可能模拟或超过经典消息传递方案。
3. **问题映射：** 某些图问题，特别是那些与量子系统（如分子模拟）或特定优化任务有关的问题，可能更自然地映射到量子算法。

### 调整GNN思想以适应量子计算

开发QGNN需要在一个量子框架内重新思考经典GNN的核心组成部分：

1. **图编码：** 我们如何使用量子态或算符来表示图结构（节点、边、特征）？这是一项重要挑战，并与第二章讨论的数据编码策略相关联。常见方法包含：

   - 将节点特征编码到单独的量子比特态或寄存器中。
   - 在哈密顿量或参数 (parameter)化酉算符中表示图的邻接矩阵或拉普拉斯矩阵。
   - 使用量子随机游走，其中图结构决定行走者的运动。
2. **邻域汇聚/消息传递：** 信息如何在连接节点间传播？QGNNs可能使用以下方法代替经典汇聚函数（如求和或均值）：

   - **参数化量子电路（PQCs）：** 一种可学习的量子电路（第四章已有讨论），局部应用于相互作用的节点表示。电路的参数在训练期间进行优化。PQC的结构可能取决于图的连接性。
   - **哈密顿量演化：** 在从图结构导出的哈密顿量下演化编码图特征的量子态。
   - **量子相互作用算符：** 在表示连接节点的量子比特之间应用多量子比特门，以模拟互相影响或信息交换。
3. **更新机制：** 节点表示如何根据汇聚信息进行更新？这通常涉及应用另一个（可能参数化的）量子操作，或在混合方法中，通过测量提取经典信息并进行经典更新。
4. **读出/测量：** 如何提取最终预测（例如，节点标签、图属性）？这涉及测量相关量子比特并可能进行经典后处理。测量基的选择（第一章）在此处很重要。

### 潜在QGNN架构

鉴于该方面的试探性性质，几种QGNN架构模式正在出现，通常为混合量子-经典模型：

- **量子核GNNs：** 使用量子电路计算图子结构（例如，节点邻域）之间的核函数，随后输入到像SVM这样的经典核机器中（与第三章相关）。
- **变分量子GNNs：** 采用PQCs作为GNN中的层。例如，PQC可以作为经典邻域汇聚后应用的非线性转换函数，或者整个消息传递步骤可以封装在参数 (parameter)化量子演化中。训练包含使用第四章讨论的技术（例如，参数位移规则、量子自然梯度）优化PQC参数。
- **基于量子游走的GNNs：** 运用图结构上的量子游走来传播信息，并根据游走行为学习节点表示。

下面是一个图示，展示了混合QGNN层，其中量子电路处理汇聚的邻域信息。

> 混合QGNN层设计。经典汇聚为量子编码准备数据。参数化量子电路处理信息，随后进行测量以提取结果，供下一层或最终预测使用。电路参数$\theta$在训练期间进行优化。

### 挑战与考量

开发实用QGNN面临显著障碍：

- **可扩展性：** 编码大型复杂图需要大量量子比特资源和可能较深的电路，限制当前硬件能力并增加对噪声的敏感性（第七章）。
- **编码保真度：** 创建准确表示复杂图结构的高保真量子态并非易事。
- **训练难度：** 在QGNN架构中优化潜在复杂PQC的参数 (parameter)，可能受到诸如贫瘠高原等挑战（第四章）的影响，尤其对于大型图或深层电路。
- **经典竞争：** 经典GNN已高度先进和优化。证明QGNN具有明显优势，需要处理经典方法确实难以应对的问题，并表明量子方法能提供实际效益，同时考虑到量子计算的额外开销。
- **硬件限制：** 近期量子硬件噪声和有限的量子比特连接性严重限制了能有效实现的QGNN的大小和复杂度（第七章）。硬件感知设计非常重要。

### 应用与展望

尽管存在挑战，QGNNs代表了一个有趣的研究方向。潜在应用范围包含：

- **量子化学与材料科学：** 分析量子效应固有的分子图。
- **优化：** 解决可能与QAOA等量子算法（第四章）良好匹配的基于图的优化问题。
- **知识图谱推理 (inference)：** 捕捉复杂的关联模式，其可能受益于量子态的表达能力。

QGNN的发展与量子硬件、错误缓解、高效数据编码技术以及可扩展的变分电路训练算法的进展密切相关。尽管仍处于早期阶段，它们展示了量子计算如何增强我们从关系数据中学习的能力。

## 参考资料

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  提供了图卷积网络（GCN）的基础模型，启发了许多后续的图神经网络架构。
- [Quantum Graph Neural Networks](https://www.nature.com/articles/s41467-021-22002-0) — Guillaume Verdon, Michael Broughton, Juan Miguel Arrazola, and Nathan Killoran (2021)
  Journal: Nature Communications; Publisher: Nature Publishing Group; Volume: 12; Pages: 1983; DOI: [10.1038/s41467-021-22002-0](https://doi.org/10.1038/s41467-021-22002-0)
  提出了量子图神经网络的早期框架，展示了如何使用量子电路对图结构数据进行学习。
- [Quantum kernel methods for graph data](https://iopscience.iop.org/article/10.1088/2058-9565/ab7c0e) — Julien Gacon, Guillaume Verdon, Léo Sebbane, Michael Broughton, Juan Miguel Arrazola, and Nathan Killoran (2020)
  Journal: Quantum Science and Technology; Publisher: IOP Publishing; Volume: 5; Pages: 034007; DOI: [10.1088/2058-9565/ab7c0e](https://doi.org/10.1088/2058-9565/ab7c0e)
  介绍了专门用于图结构数据的量子核构建方法，与量子核GNN相关。
