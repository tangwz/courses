---
course: "introduction-to-graph-neural-networks"
chapter: "gnn-implementation-pytorch-geometric"
lesson: "introduction-pytorch-geometric"
sourceId: 7641
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/introduction-pytorch-geometric"
title: "PyTorch Geometric (PyG) 简介"
description: "PyTorch Geometric (PyG) 概述，这是一个基于 PyTorch 构建的库，用于轻松编写和训练适用于各种应用的 GNN。"
order: 1
plots: []
sourceHash: "9818b7230bf9b007a0d34cf40de63944109f5d2849d2059062c3f8356d515a36"
sourceCorrections: []
---

在实际的模型开发中，虽然从零开始构建消息传递层有助于理解底层机制，但这通常不是一种可扩展的方法。手动管理稀疏邻接矩阵并实现高效的聚合操作既复杂又容易出错。为了高效实现 GNN，我们需要一套专门的工具。PyTorch Geometric（通常简称为 PyG）为这些挑战提供了全面的解决方案。

PyTorch Geometric 是一个基于 PyTorch 构建的库，专为图和其他不规则数据结构的深度学习 (deep learning)而设计。由 Matthias Fey 开发的 PyG 在 PyTorch 熟悉的 API 基础上，增加了一系列针对图机器学习 (machine learning)优化的工具，使其成为开发 GNN 最流行的框架之一。

### 为什么要使用 PyTorch Geometric？

PyG 提供了一系列经过充分测试且高度优化的构建模块，无需为每个新的 GNN 架构重复造轮子。与手动实现相比，采用 PyG 这样的库具有几个显著优势。

#### 优化的性能

图操作，特别是大规模图上的消息传递，计算开销可能非常大。用 Python 进行的简单实现对于严肃的应用来说速度太慢。PyG 的核心例程使用了专门的 C++ 和 CUDA 内核，确保邻域聚合等操作在 CPU 和 GPU 上都能高效执行。这种性能优化使你能够在合理的时间内，在拥有数百万个节点和边的图上训练模型。

#### 模块化与易用性

PyG 直接集成到 PyTorch 生态系统中。其 `torch_geometric.nn` 模块中的 GNN 层被设计为标准的 PyTorch `nn.Module` 对象。这意味着你可以像构建 CNN 或 MLP 一样轻松地构建 GNN。例如，图卷积网络层可以用 `GCNConv` 表示，图注意力网络层可以用 `GATConv` 表示。你可以堆叠这些层，将它们与 `nn.Linear` 或 `nn.ReLU` 等标准 PyTorch 层组合，并以极少的样板代码构建复杂的架构。

#### 专门的数据处理

正如我们在第一章中所见，用稠密邻接矩阵表示图对于稀疏网络来说非常消耗内存且效率低下。PyG 通过专门的数据对象高效处理图结构，主要通过对边使用稀疏坐标 (COO) 格式。这种方法（我们将在下一节讨论）大幅减少了内存占用和计算开销。

### 是扩展而非替代

理解 PyG 不是一个独立的框架这一点非常重要。它是为 PyTorch 增加图处理能力的扩展。在大多数深度学习 (deep learning)工作流中，你仍然会使用 PyTorch 的核心组件：

- **张量 (Tensors)：** 所有节点特征、边索引和模型参数 (parameter)都是标准的 `torch.Tensor` 对象。
- **自动求导 (Autograd)：** PyTorch 的自动微分引擎会跟踪这些张量上的所有操作，从而实现 GNN 层的直接反向传播 (backpropagation)。
- **优化器 (Optimizers)：** 你将使用熟悉的优化器（如 `torch.optim.Adam`）来训练模型。
- **损失函数 (loss function) (Loss Functions)：** 标准损失函数（如 `torch.nn.CrossEntropyLoss`）可以直接与 GNN 模型的输出配合使用。

下图展示了 PyG 如何在 PyTorch 核心基础上构建，并增加了用于图数据、神经网络 (neural network)层和数据加载的专门模块。

> PyTorch Geometric 在 PyTorch 核心之上增加了专门的图模块。它利用 PyTorch 的张量和自动求导功能，同时提供用于图表示和 GNN 层的专用组件。

在本章中，我们将重点介绍 PyG 提供的主要组件。我们将从表示单个图的 `Data` 对象开始，然后学习使用 PyG 预构建的数据集和层来构建完整且高效的 GNN 模型。

## 参考资料

- [PyTorch Geometric: A Library for Deep Learning on Graphs](https://arxiv.org/abs/1903.02428) — Matthias Fey, Jan Eric Lenssen (2019)
  Journal: arXiv preprint arXiv:1903.02428; DOI: [10.48550/arXiv.1903.02428](https://doi.org/10.48550/arXiv.1903.02428)
  介绍PyTorch Geometric，概述其设计、主要特性和针对图深度学习的性能优化。
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) — PyTorch Geometric Developers (2024)
  PyTorch Geometric的官方资源，提供API、教程和示例，保持最新。
- [A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/9046288) — Zonghan Wu, Shirui Pan, Fangzhao Wu, Xindong Wu, and Chengqi Zhang (2021)
  Journal: IEEE Transactions on Neural Networks and Learning Systems; Publisher: IEEE; Volume: 32; Pages: 4-24; DOI: [10.1109/TNNLS.2020.2987520](https://doi.org/10.1109/TNNLS.2020.2987520)
  广泛回顾图神经网络，阐述PyG所实现的各种架构及其理论基础。
