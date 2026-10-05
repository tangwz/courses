# 第 3 章：GNN 基本架构

来源：[原章节](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures)

[返回课程目录](../README.md)

前一章确立了通用消息传递框架，为图神经网络的运行方式提供了蓝图。本章将从抽象的公式转向具体的知名架构，将这些原理付诸实践。我们将分析聚合函数和更新函数的不同选择如何产生具有不同行为和性能特征的模型。

你将学习三种基本模型背后的机制：

*   **图卷积网络 (GCN)：** 一种流行且高效的架构，通过简化的谱方法将卷积逻辑适配到图结构中。
*   **GraphSAGE：** 一种归纳式模型，通过学习通用的聚合函数并采用邻居采样，使其能够为训练期间未见过的节点生成嵌入。
*   **图注意力网络 (GAT)：** 一种引入掩码自注意力机制的架构，用于为邻域内的不同节点分配不同的权重。

针对每种模型，我们将讲解其数学公式，并讨论其主要优缺点。本章最后包含一个动手练习，通过从零开始构建 GCN，将理论公式直接转化为可运行的代码。

## 小节

- 1. [图卷积网络 (GCN)](01-%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%20%28GCN%29.md)
- 2. [图卷积的空间物理解读](02-%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%9A%84%E7%A9%BA%E9%97%B4%E7%89%A9%E7%90%86%E8%A7%A3%E8%AF%BB.md)
- 3. [GraphSAGE：邻域采样与聚合](03-GraphSAGE%EF%BC%9A%E9%82%BB%E5%9F%9F%E9%87%87%E6%A0%B7%E4%B8%8E%E8%81%9A%E5%90%88.md)
- 4. [GraphSAGE 的归纳学习](04-GraphSAGE%20%E7%9A%84%E5%BD%92%E7%BA%B3%E5%AD%A6%E4%B9%A0.md)
- 5. [图注意力网络 (GAT)](05-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md)
- 6. [GAT 中的注意力机制](06-GAT%20%E4%B8%AD%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 7. [GCN、GraphSAGE 与 GAT 的比较](07-GCN%E3%80%81GraphSAGE%20%E4%B8%8E%20GAT%20%E7%9A%84%E6%AF%94%E8%BE%83.md)
- 8. [动手实践：从零开始实现 GCN](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20GCN.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/quiz)
