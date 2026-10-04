# 图神经网络入门

来源：[图神经网络入门](https://apxml.com/zh/courses/introduction-to-graph-neural-networks)

本课程介绍图神经网络 (neural network)（GNN）。这类模型专门用于处理图结构数据。你将学习 GNN 的运作原理，包括消息传递机制以及图卷积网络（GCN）和图注意力网络（GAT）等具体架构。本课讲解构建 GNN 模型的完整流程：图数据表示、网络架构定义、模型训练以及在通用任务中的性能评估。实战部分侧重于通过 PyTorch Geometric 库进行开发。

预计学时：13 小时

先修要求：具备 Python 与机器学习入门知识

## 课程目录

### 1. [图学习基本原理](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/README.md)

- 1. [什么是图数据？](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%9B%BE%E6%95%B0%E6%8D%AE%EF%BC%9F.md)
- 2. [标准神经网络在图数据上的局限性](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/02-%E6%A0%87%E5%87%86%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9C%A8%E5%9B%BE%E6%95%B0%E6%8D%AE%E4%B8%8A%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 3. [常见图机器学习任务](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/03-%E5%B8%B8%E8%A7%81%E5%9B%BE%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%BB%BB%E5%8A%A1.md)
- 4. [图的表示：邻接矩阵与特征矩阵](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/04-%E5%9B%BE%E7%9A%84%E8%A1%A8%E7%A4%BA%EF%BC%9A%E9%82%BB%E6%8E%A5%E7%9F%A9%E9%98%B5%E4%B8%8E%E7%89%B9%E5%BE%81%E7%9F%A9%E9%98%B5.md)
- 5. [图的属性与度量](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/05-%E5%9B%BE%E7%9A%84%E5%B1%9E%E6%80%A7%E4%B8%8E%E5%BA%A6%E9%87%8F.md)
- 6. [NetworkX 库简介](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/06-NetworkX%20%E5%BA%93%E7%AE%80%E4%BB%8B.md)
- 7. [动手实践：加载并检查图数据集](01-%E5%9B%BE%E5%AD%A6%E4%B9%A0%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8A%A0%E8%BD%BD%E5%B9%B6%E6%A3%80%E6%9F%A5%E5%9B%BE%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- [章节测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-1-foundations-graph-based-learning/quiz)

### 2. [消息传递机制](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/README.md)

- 1. [邻域聚合思想](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/01-%E9%82%BB%E5%9F%9F%E8%81%9A%E5%90%88%E6%80%9D%E6%83%B3.md)
- 2. [通用 GNN 层：聚合与更新](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/02-%E9%80%9A%E7%94%A8%20GNN%20%E5%B1%82%EF%BC%9A%E8%81%9A%E5%90%88%E4%B8%8E%E6%9B%B4%E6%96%B0.md)
- 3. [常用的聚合函数](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/03-%E5%B8%B8%E7%94%A8%E7%9A%84%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0.md)
- 4. [更新函数与非线性](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/04-%E6%9B%B4%E6%96%B0%E5%87%BD%E6%95%B0%E4%B8%8E%E9%9D%9E%E7%BA%BF%E6%80%A7.md)
- 5. [置换不变性与置换等变性](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/05-%E7%BD%AE%E6%8D%A2%E4%B8%8D%E5%8F%98%E6%80%A7%E4%B8%8E%E7%BD%AE%E6%8D%A2%E7%AD%89%E5%8F%98%E6%80%A7.md)
- 6. [堆叠层以构建深度图神经网络 (GNN)](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/06-%E5%A0%86%E5%8F%A0%E5%B1%82%E4%BB%A5%E6%9E%84%E5%BB%BA%E6%B7%B1%E5%BA%A6%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28GNN%29.md)
- 7. [实践：使用 NumPy 实现简单的 GNN 层](02-%E6%B6%88%E6%81%AF%E4%BC%A0%E9%80%92%E6%9C%BA%E5%88%B6/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20NumPy%20%E5%AE%9E%E7%8E%B0%E7%AE%80%E5%8D%95%E7%9A%84%20GNN%20%E5%B1%82.md)
- [章节测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-2-the-message-passing-mechanism/quiz)

### 3. [GNN 基本架构](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/README.md)

- 1. [图卷积网络 (GCN)](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/01-%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%20%28GCN%29.md)
- 2. [图卷积的空间物理解读](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/02-%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%9A%84%E7%A9%BA%E9%97%B4%E7%89%A9%E7%90%86%E8%A7%A3%E8%AF%BB.md)
- 3. [GraphSAGE：邻域采样与聚合](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/03-GraphSAGE%EF%BC%9A%E9%82%BB%E5%9F%9F%E9%87%87%E6%A0%B7%E4%B8%8E%E8%81%9A%E5%90%88.md)
- 4. [GraphSAGE 的归纳学习](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/04-GraphSAGE%20%E7%9A%84%E5%BD%92%E7%BA%B3%E5%AD%A6%E4%B9%A0.md)
- 5. [图注意力网络 (GAT)](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/05-%E5%9B%BE%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%BD%91%E7%BB%9C%20%28GAT%29.md)
- 6. [GAT 中的注意力机制](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/06-GAT%20%E4%B8%AD%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 7. [GCN、GraphSAGE 与 GAT 的比较](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/07-GCN%E3%80%81GraphSAGE%20%E4%B8%8E%20GAT%20%E7%9A%84%E6%AF%94%E8%BE%83.md)
- 8. [动手实践：从零开始实现 GCN](03-GNN%20%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%20GCN.md)
- [章节测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-3-foundational-gnn-architectures/quiz)

### 4. [GNN 模型训练](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/README.md)

- 1. [为节点分类任务搭建 GNN](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/01-%E4%B8%BA%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E6%90%AD%E5%BB%BA%20GNN.md)
- 2. [图任务的损失函数](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/02-%E5%9B%BE%E4%BB%BB%E5%8A%A1%E7%9A%84%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)
- 3. [GNN 的训练循环](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/03-GNN%20%E7%9A%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
- 4. [图数据的切分：直推式与归纳式](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/04-%E5%9B%BE%E6%95%B0%E6%8D%AE%E7%9A%84%E5%88%87%E5%88%86%EF%BC%9A%E7%9B%B4%E6%8E%A8%E5%BC%8F%E4%B8%8E%E5%BD%92%E7%BA%B3%E5%BC%8F.md)
- 5. [节点分类的评估指标](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/05-%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
- 6. [GNN 中的过拟合与正则化](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/06-GNN%20%E4%B8%AD%E7%9A%84%E8%BF%87%E6%8B%9F%E5%90%88%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.md)
- 7. [实践：训练与评估你的 GCN](04-GNN%20%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83/07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%E4%BD%A0%E7%9A%84%20GCN.md)
- [章节测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/quiz)

### 5. [使用 PyTorch Geometric 实现 GNN](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/README.md)

- 1. [PyTorch Geometric (PyG) 简介](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/01-PyTorch%20Geometric%20%28PyG%29%20%E7%AE%80%E4%BB%8B.md)
- 2. [PyG Data 对象](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/02-PyG%20Data%20%E5%AF%B9%E8%B1%A1.md)
- 3. [使用 PyG 数据集](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/03-%E4%BD%BF%E7%94%A8%20PyG%20%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 4. [使用 PyG GNN 层构建模型](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/04-%E4%BD%BF%E7%94%A8%20PyG%20GNN%20%E5%B1%82%E6%9E%84%E5%BB%BA%E6%A8%A1%E5%9E%8B.md)
- 5. [针对大规模图数据的分批处理](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/05-%E9%92%88%E5%AF%B9%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%9B%BE%E6%95%B0%E6%8D%AE%E7%9A%84%E5%88%86%E6%89%B9%E5%A4%84%E7%90%86.md)
- 6. [PyG 中的完整训练脚本](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/06-PyG%20%E4%B8%AD%E7%9A%84%E5%AE%8C%E6%95%B4%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)
- 7. [动手实践：使用 PyG 在 Cora 数据集上进行节点分类](05-%E4%BD%BF%E7%94%A8%20PyTorch%20Geometric%20%E5%AE%9E%E7%8E%B0%20GNN/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20PyG%20%E5%9C%A8%20Cora%20%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%BF%9B%E8%A1%8C%E8%8A%82%E7%82%B9%E5%88%86%E7%B1%BB.md)
- [章节测验](https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/quiz)

## 学习目标

- **图数据表示**：使用邻接矩阵和特征矩阵为机器学习应用表示图结构数据。
- **GNN 运作机制**：讲解作为多数 GNN 架构底层的消息传递机制。
- **GNN 架构**：区分并实现 GCN、GraphSAGE 和 GAT 等常用模型。
- **模型训练**：针对节点分类等任务，搭建完整的 GNN 训练与评估流程。
- **实战开发**：通过 PyTorch Geometric 库高效构建并训练 GNN。
