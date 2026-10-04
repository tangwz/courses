# 第 5 章：使用 PyTorch Geometric 实现 GNN

来源：[原章节](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric)

[返回课程目录](../README.md)

前面的章节说明了 GNN 的运行原理，包括从头开始实现消息传递层。虽然这很有启发性，但手动方法在构建和测试不同架构时并不实用。本章将转而使用 PyTorch Geometric (PyG)，这是一个简化图模型开发的专用库。PyG 提供了常用 GNN 层和数据处理程序的优化版本，使你能够把精力放在模型设计上，而非底层的实现细节。

在本章中，你将学会使用 PyTorch Geometric 的主要部分。我们将从 `Data` 对象开始，这是 PyG 表示整个图的方式。接着，你将看到如何加载标准图数据集，并使用 PyG 预置的层（例如 `GCNConv` 和 `GATConv`）来组建 GNN 模型。我们还会说明该库如何通过图的批量化处理来提升训练效率。

学完本章后，你将有能力为节点分类任务编写完整的训练和评估脚本。最后的动手操作环节将运用这些技能，在 Cora 引用网络这一标准基准数据集上构建 GNN。

## 小节

- 1. [PyTorch Geometric (PyG) 简介](01-PyTorch%20Geometric%20%28PyG%29%20%E7%AE%80%E4%BB%8B.md)
- 2. [PyG Data 对象](02-PyG%20Data%20%E5%AF%B9%E8%B1%A1.md)
- 3. [使用 PyG 数据集](03-%E4%BD%BF%E7%94%A8%20PyG%20%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 4. [使用 PyG GNN 层构建模型](04-%E4%BD%BF%E7%94%A8%20PyG%20GNN%20%E5%B1%82%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%9E%8B.md)
- 5. [针对大规模图数据的分批处理](05-%E9%92%88%E5%AF%B9%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%9B%BE%E6%95%B0%E6%8D%AE%E7%9A%84%E5%88%86%E6%89%B9%E5%A4%84%E7%90%86.md)
- 6. [PyG 中的完整训练脚本](06-PyG%20%E4%B8%AD%E7%9A%84%E5%AE%8C%E6%95%B4%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)
- 7. [动手实践：使用 PyG 在 Cora 数据集上进行节点分类](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20PyG%20%E5%9C%A8%20Cora%20%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%BF%9B%E8%A1%8C%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/quiz)
