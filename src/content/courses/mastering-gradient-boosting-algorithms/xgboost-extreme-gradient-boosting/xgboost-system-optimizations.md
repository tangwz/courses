---
course: "mastering-gradient-boosting-algorithms"
chapter: "xgboost-extreme-gradient-boosting"
lesson: "xgboost-system-optimizations"
sourceId: 1987
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/xgboost-system-optimizations"
title: "系统优化：缓存感知与并行处理"
description: "XGBoost 中缓存感知访问、块结构和并行处理的概述。"
order: 6
plots: []
sourceHash: "96be20f5a3ac5344552c60574bcfd8171bcd93919da68fde41b2c3757fa12fb3"
sourceCorrections: []
---

除了其广为人知的正则化 (regularization)目标函数和复杂的分割点查找算法之外，XGBoost出色的性能，尤其是在大型数据集上，也源于精心的系统设计和优化。这些优化侧重于有效利用多核处理器和CPU缓存等现代硬件资源。XGBoost采用的核心系统级方法是并行处理和缓存感知。

### 并行处理加快树构建速度

构建梯度提升树是一个迭代过程。在每次迭代中，会构建一棵新树来拟合伪残差。构建单棵树计算量最大的部分是为每个潜在节点查找所有特征的最佳分割点。

XGBoost 有效地并行化了这一分割点查找过程。考虑标准贪婪算法，其中对于每个节点，我们遍历所有特征，然后对于每个特征，遍历所有可能的分割点（或近似算法中的分位数）来计算潜在增益。

- **外层循环并行：** 为给定节点查找最佳分割点涉及独立评估所有特征。XGBoost 可以并行化这个外层循环，将不同特征分配给不同线程。每个线程计算其分配特征的最佳分割点和增益。然后通过同步步骤汇总结果，以查找所有特征中的整体最佳分割点。
- **数据结构支持：** 这种并行处理在很大程度上依赖于XGBoost的内部数据结构，通常称为`列块`。数据以内存压缩格式存储，其中每列（特征）都按其对应的特征值排序，并附带相关数据点的索引。这个预处理步骤使得每个处理一个特征的线程能够主要顺序地访问必要数据（特征值、梯度、Hessian），从而在分割点查找阶段实现高效的并行计算。

> XGBoost 在分割点查找过程中的并行处理。协调器将特征分配给不同线程，这些线程同时评估潜在分割点。结果被汇总以确定该节点的整体最佳分割点。

值得注意的是，XGBoost 主要并行化了*单棵树*的构建（树内并行），而不是同时构建多棵树（树间并行），因为提升过程本身是顺序的，每棵新树都依赖于前一棵树的残差。

### 缓存感知访问实现高效内存使用

现代 CPU 在很大程度上依赖于多级缓存（L1、L2、L3），这些缓存比主内存（RAM）快得多。访问缓存中已有的数据比从 RAM 获取数据要快得多。频繁访问随机分散内存位置的算法会面临高缓存未命中率问题，从而导致性能瓶颈。

在梯度提升中计算分割增益需要访问与每个潜在分割点对应的数据点的梯度和 Hessian 统计信息。朴素实现可能会以由考虑的特征值决定的非连续模式访问这些统计信息，从而导致缓存利用率低下。

XGBoost 通过使用**缓存感知预取**解决了这个问题。

1. **数据布局：** 前面提到的`列块`结构在这里也发挥着重要作用。通过在开始时*一次性*根据特征值排序数据并同时存储梯度/Hessian，XGBoost 在评估特定特征的分割点时可以更顺序地访问这些统计信息。
2. **小批量缓冲区：** XGBoost 在每个线程中内部分配小缓冲区。它将即将进行的分割点评估块所需的梯度和 Hessian 统计信息预取到这些缓冲区中。由于这些缓冲区很小，它们更有可能存放在 CPU 缓存中。然后，该块内分割点的后续计算可以直接从快得多的缓存中访问统计信息，最大程度地减少缓存未命中。

这种缓存感知方法确保 CPU 核将更多时间用于有效计算（计算分割增益），而不是等待从较慢的主内存中获取数据。

### 数据管理的块结构

`列块`结构是并行处理和缓存效率的根本。我们来总结一下它的优点：

- **压缩存储：** 减少内存占用。
- **排序的特征值：** 无论是使用精确算法还是近似算法，都能实现高效的分割点查找。
- **顺序访问：** 当评估某个特征的分割点时，相应的梯度和 Hessian 可以主要顺序地从已排序的块中访问，从而提高缓存局部性。
- **并行处理基础：** 允许不同块（特征）由不同线程独立处理。
- **核外支持：** 这种块结构可以扩展应用于大于主内存的数据集。块可以被压缩并存储在磁盘上，然后根据需要选择性地加载到内存中（块分片和压缩），使 XGBoost 能够处理有限内存下的海量数据集。

“通过在算法创新之外实施这些系统优化，XGBoost 与传统的梯度提升实现相比，实现了显著的加速。这使其成为一个高度实用且可扩展的工具，用于解决大量数据上的复杂机器学习 (machine learning)问题。理解这些优化有助于理解为何 XGBoost 经常提供显著的性能优势，以及其设计选择如何旨在有效利用硬件。”

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.acm.org/10.1145/2939672.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16); Publisher: Association for Computing Machinery; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  XGBoost 的原始论文，详细介绍了其可扩展的系统设计，包括并行性、缓存感知和列块结构。
- [Scalable Tree Boosting. XGBoost Documentation.](https://xgboost.readthedocs.io/en/stable/tutorials/model.html#system-for-scalable-tree-boosting) — XGBoost Contributors (2024)
  Publisher: XGBoost
  官方文档描述了 XGBoost 的系统优化，例如并行分割查找、缓存感知数据访问和核外计算。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture-a-quantitative-approach/hennessy/978-0-12-811905-1) — John L. Hennessy and David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  一本基础教科书，详细解释了计算机硬件，包括 CPU 缓存、内存层次结构及其对程序性能的影响。
- [Introduction to Parallel Computing](https://www.oreilly.com/library/view/introduction-to-parallel/0201648652/) — Ananth Grama, Anshul Gupta, George Karypis, and Vipin Kumar (2003)
  Publisher: Addison-Wesley
  一本关于并行计算的基本概念、算法和架构的综合性教科书，有助于理解 XGBoost 的并行处理。
