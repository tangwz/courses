---
course: "graph-neural-networks-gnns"
chapter: "gnn-training-complexities"
lesson: "gnn-oversmoothing-problem"
sourceId: 2693
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/gnn-oversmoothing-problem"
title: "过平滑问题"
description: "理解深度GNN中节点特征变得难以区分的原因和结果。"
order: 1
plots: []
sourceHash: "58696237dd3da7c384abfe7311a14e90cd3e0588ed24d8a6bc4f04feb3a89073"
sourceCorrections: []
---

堆叠多层图神经网络 (neural network) (GNN) 看起来是增加模型能力和捕获更远距离关联的自然方式，但常常导致违反直觉的性能下降。这种现象被普遍称为**过平滑**。它指的是图中节点表示变得越来越相似，最终收敛到难以区分的值的趋势，当它们通过连续的GNN层时。

### 机制理解

从根本上讲，过平滑是许多GNN使用的标准消息传递机制的固有结果。回顾节点 $v$ 在第 $k+1$ 层的基本更新规则：


$$
h_v^{(k+1)} = \sigma \left( \text{更新}^{(k)} \left( h_v^{(k)}, \text{聚合}^{(k)} \left( \{ h_u^{(k)} : u \in \mathcal{N}(v) \} \right) \right) \right)
$$


`聚合`函数通常涉及某种形式的平均或加权和，这些和来自前一层邻居的特征 $h_u^{(k)}$。例如，在一个简化的图卷积网络 (GCN) 层中，聚合可以看作是将归一化 (normalization)邻接矩阵 $A_{norm}$（例如 $D^{-1/2}AD^{-1/2}$）应用于特征矩阵 $H^{(k)}$：


$$
H^{(k+1)} = \sigma(A_{norm} H^{(k)} W^{(k)})
$$


这种操作有效地将节点的特征与其邻居的特征进行平均。当这种平均过程在许多层（$k \rightarrow \infty$）中重复时，同一连通分量内的节点特征趋于收敛。直观地说，每个传播步骤都会混合相邻节点的特征。经过 $k$ 步后，节点的表示受到远至 $k$ 跳节点的影响。随着 $k$ 的增加，每个节点的感受野扩展，可能覆盖其连通分量的大部分，甚至全部。这种重复的局部平均作用类似于图信号（节点特征）上的低通滤波器，平滑了节点之间的差异。

考虑图上随机游走的类比。每个消息传递步骤都类似于随机游走中的一步。随着步数的增加，游走者位置的概率分布趋向于平稳分布，这种分布通常只取决于全局图属性（如节点度），而不是起始节点的独特特征。类似地，重复的聚合会洗去区分一个节点与另一个节点的特定局部邻域信息。

### 对学习的影响

过平滑的主要后果是节点表示辨别能力的丧失。如果同一连通分量内的所有节点在经过几层后具有几乎相同的嵌入 (embedding)，GNN就难以执行依赖于区分这些节点的任务，例如节点分类或链接预测。

- **性能下降：** 深度GNN在许多基准数据集上的表现可能不如浅层GNN（例如2-3层）。增加更多层会带来收益递减，甚至会损害准确性。
- **局部信息丢失：** 虽然更深层旨在捕获全局结构，但过平滑导致它们丢失了许多图任务所需的精细局部结构信息。
- **难以区分的嵌入：** 如果图中的节点在结构上接近，属于不同类别的节点最终可能具有非常相似的嵌入，这使得分类变得困难。

让我们来说明这种收敛。想象一个小型图，其中节点最初具有不同的特征（用颜色表示）。

> 最初不同的节点特征（颜色）在许多消息传递层之后变得同质化，这是由于重复的邻域平均。

这种同质化意味着网络有效地失去了利用初始层中编码的节点特有或局部结构信息的能力。虽然扩大感受野是期望的，但过平滑阻止模型有效利用从远处节点收集的信息，同时又不丢失局部上下文 (context)。

了解这种现象对于设计有效的深度GNN架构和训练策略非常重要。本章后面讨论的技术，例如残差连接、跳跃知识或注意力机制 (attention mechanism)，专门设计用于对抗这种过度平滑，并允许构建更深、更具表现力的GNN模型。

## 参考资料

- [DeeperGCN: All Deep GCNs Are Not Created Equal](https://ojs.aaai.org/index.php/AAAI/article/view/5879) — Qinqing Li, Zhichao Han, Xiaowen Wu (2020)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: AAAI Press; Volume: 34; Pages: 4786-4793; DOI: [10.1609/aaai.v34i04.5879](https://doi.org/10.1609/aaai.v34i04.5879)
  分析深度GNN中的过平滑问题，并提出架构解决方案以减轻其影响。
- [Measuring and Improving the Smoothness of Graph Neural Networks](http://proceedings.mlr.press/v119/oono20a.html) — Yuya Oono, Taiji Suzuki (2020)
  Journal: International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 119; Pages: 7356-7366; DOI: [10.5591/00742](https://doi.org/10.5591/00742)
  为理解和量化GNN中节点表示的平滑性提供了理论框架。
- [Is a Graph Convolutional Neural Network a Low-Pass Filter?](https://arxiv.org/abs/1905.09457) — Naoaki Okumoto, Taiji Maehara (2019)
  Journal: Proceedings of the NeurIPS Workshop on Graph Representation Learning; DOI: [10.48550/arXiv.1905.09457](https://doi.org/10.48550/arXiv.1905.09457)
  研究GCN的谱特性，解释消息传递如何充当低通滤波器，导致过平滑。
- [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/8786435) — Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, S. Yu Philip (2020)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2970482](https://doi.org/10.1109/TNNLS.2020.2970482)
  一篇广泛的GNN综述，其中讨论了过平滑问题及其应对策略。
