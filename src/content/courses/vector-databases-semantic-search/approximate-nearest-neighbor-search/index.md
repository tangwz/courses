---
course: "vector-databases-semantic-search"
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-3-approximate-nearest-neighbor-search"
sourceId: 736
chapter: "approximate-nearest-neighbor-search"
title: "近似最近邻 (ANN) 搜索"
order: 3
description: "理解 ANN 原理，比较算法（HNSW、IVF、LSH），学习索引调整，并评估性能权衡。"
hasQuiz: true
---

在庞大的高维向量集合中查找*精确*最近邻，计算成本可能非常高，对于交互式应用来说通常速度过慢。在处理向量数据库的规模时，对每次查询都执行穷举搜索通常不切实际。

本章将通过引入近似最近邻 (ANN) 搜索来应对这一挑战。您将了解到为何近似处理通常是必要的，以及 ANN 算法如何通过牺牲少量准确性来换取搜索速度和资源使用方面的大幅提升，从而提供一个实用的解决方案。

我们将涵盖：

*   高维空间中精确最近邻搜索的计算限制。
*   ANN 的核心原理，特别是搜索召回率与延迟等性能指标之间的权衡。
*   常见 ANN 算法的概述，包括分层可导航小世界 (HNSW)、倒排文件索引 (IVF) 和局部敏感哈希 (LSH)，并解释它们各自的工作方式。
*   用于构建和调整 ANN 索引的主要参数（例如，$ef\_construction$、$ef\_search$、$nlist$、$m$）及其影响。
*   使用相关指标评估 ANN 索引有效性和效率的方法。

本章最后将通过一个动手实践环节结束，您将尝试构建不同的 ANN 索引并观察由此产生的性能表现。
