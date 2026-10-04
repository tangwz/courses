# 第 3 章：应对GNN训练中的复杂性

来源：[原章节](https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities)

[返回课程目录](../README.md)

了解GNN架构很重要，但有效地训练它们，尤其是在大规模数据上，会带来独特的挑战。诸如过平滑（节点表示不理想地趋于一致）和过挤压（限制了信息在图上的传播）等问题，会大幅降低性能。在非常大的图上训练GNN还会带来大量计算和内存负担。

本章将侧重于讨论这些实际的训练复杂性。我们将考察过平滑和过挤压等问题的理论原理，并讨论应对它们的方法，例如架构修改和特定的训练技术。您还将学习将GNN训练扩展到海量数据集的策略，包括邻居采样（例如GraphSAGE）、图采样（例如GraphSAINT）和图聚类方法（例如Cluster-GCN）。最后，我们将提及这些模型的相关优化考量。

## 小节

- 1. [过平滑问题](01-%E8%BF%87%E5%B9%B3%E6%BB%91%E9%97%AE%E9%A2%98.md)
- 2. [缓解过平滑的技术](02-%E7%BC%93%E8%A7%A3%E8%BF%87%E5%B9%B3%E6%BB%91%E7%9A%84%E6%8A%80%E6%9C%AF.md)
- 3. [过度挤压问题](03-%E8%BF%87%E5%BA%A6%E6%8C%A4%E5%8E%8B%E9%97%AE%E9%A2%98.md)
- 4. [处理大型图：可扩展性挑战](04-%E5%A4%84%E7%90%86%E5%A4%A7%E5%9E%8B%E5%9B%BE%EF%BC%9A%E5%8F%AF%E6%89%A9%E5%B1%95%E6%80%A7%E6%8C%91%E6%88%98.md)
- 5. [邻域采样技术 (GraphSAGE)](05-%E9%82%BB%E5%9F%9F%E9%87%87%E6%A0%B7%E6%8A%80%E6%9C%AF%20%28GraphSAGE%29.md)
- 6. [图采样技术 (GraphSAINT, ShaDow-GNN)](06-%E5%9B%BE%E9%87%87%E6%A0%B7%E6%8A%80%E6%9C%AF%20%28GraphSAINT%2C%20ShaDow-GNN%29.md)
- 7. [子图与聚类方法 (Cluster-GCN)](07-%E5%AD%90%E5%9B%BE%E4%B8%8E%E8%81%9A%E7%B1%BB%E6%96%B9%E6%B3%95%20%28Cluster-GCN%29.md)
- 8. [图神经网络的优化策略](08-%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5.md)
- 9. [实践：应用可扩展GNN训练](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BA%94%E7%94%A8%E5%8F%AF%E6%89%A9%E5%B1%95GNN%E8%AE%AD%E7%BB%83.md)
