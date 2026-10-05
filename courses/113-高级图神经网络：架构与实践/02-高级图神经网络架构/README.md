# 第 2 章：高级图神经网络架构

来源：[原章节](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-2-advanced-gnn-architectures)

[返回课程目录](../README.md)

在上一章图表示和消息传递的基本原理之上，本章关注特定的高级图神经网络架构。我们将考察旨在提升表达能力、有效处理加权边以及借鉴其他成功深度学习方向中思路的模型。

在这里，你将学习：

*   图卷积网络（GCN）的详细分析，将其谱域推导与实际操作联系起来。
*   图注意力网络（GAT），侧重于自注意力机制 $a_{ij}$ 如何自适应地加权邻居信息，包括多头注意力策略。
*   Transformer 架构在图数据上的应用（图 Transformer），包括通过位置编码加入结构信息的方法。
*   高级谱域方法，如 ChebNet（其使用图滤波器 $g_\theta(\Lambda) \approx \sum_{k=0}^K \theta_k T_k(\tilde{\Lambda})$ 的多项式逼近），以及 CayleyNets。
*   精巧的空间域 GNN，包括对 GraphSAGE 的扩展和主邻域聚合（PNA）等技术。
*   对这些架构进行比较分析，评估其性能特点、可扩展性和理论能力。

本章包含实践实现指导，例如构建 GAT 层，以巩固对这些高级模型的理解。

## 小节

- 1. [图卷积网络 (GCN)](01-%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%20%28GCN%29.md)
- 2. [图注意力网络 (GAT)](02-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md)
- 3. [在GAT中实现多头注意力](03-%E5%9C%A8GAT%E4%B8%AD%E5%AE%9E%E7%8E%B0%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md)
- 4. [图Transformer](04-%E5%9B%BETransformer.md)
- 5. [进阶谱GNN (ChebNet, CayleyNets)](05-%E8%BF%9B%E9%98%B6%E8%B0%B1GNN%20%28ChebNet%2C%20CayleyNets%29.md)
- 6. [进阶空间GNNs (GraphSAGE变体, PNA)](06-%E8%BF%9B%E9%98%B6%E7%A9%BA%E9%97%B4GNNs%20%28GraphSAGE%E5%8F%98%E4%BD%93%2C%20PNA%29.md)
- 7. [比较架构选择与权衡](07-%E6%AF%94%E8%BE%83%E6%9E%B6%E6%9E%84%E9%80%89%E6%8B%A9%E4%B8%8E%E6%9D%83%E8%A1%A1.md)
- 8. [动手实践：GAT层实现](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9AGAT%E5%B1%82%E5%AE%9E%E7%8E%B0.md)
