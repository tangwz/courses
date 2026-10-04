# 第 3 章：近似最近邻 (ANN) 搜索

来源：[原章节](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-3-approximate-nearest-neighbor-search)

[返回课程目录](../README.md)

在庞大的高维向量集合中查找*精确*最近邻，计算成本可能非常高，对于交互式应用来说通常速度过慢。在处理向量数据库的规模时，对每次查询都执行穷举搜索通常不切实际。

本章将通过引入近似最近邻 (ANN) 搜索来应对这一挑战。您将了解到为何近似处理通常是必要的，以及 ANN 算法如何通过牺牲少量准确性来换取搜索速度和资源使用方面的大幅提升，从而提供一个实用的解决方案。

我们将涵盖：

*   高维空间中精确最近邻搜索的计算限制。
*   ANN 的核心原理，特别是搜索召回率与延迟等性能指标之间的权衡。
*   常见 ANN 算法的概述，包括分层可导航小世界 (HNSW)、倒排文件索引 (IVF) 和局部敏感哈希 (LSH)，并解释它们各自的工作方式。
*   用于构建和调整 ANN 索引的主要参数（例如，$ef\_construction$、$ef\_search$、$nlist$、$m$）及其影响。
*   使用相关指标评估 ANN 索引有效性和效率的方法。

本章最后将通过一个动手实践环节结束，您将尝试构建不同的 ANN 索引并观察由此产生的性能表现。

## 小节

- 1. [近似的需求](01-%E8%BF%91%E4%BC%BC%E7%9A%84%E9%9C%80%E6%B1%82.md)
- 2. [ANN 的核心思想](02-ANN%20%E7%9A%84%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3.md)
- 3. [算法概览：HNSW](03-%E7%AE%97%E6%B3%95%E6%A6%82%E8%A7%88%EF%BC%9AHNSW.md)
- 4. [算法概述：IVF](04-%E7%AE%97%E6%B3%95%E6%A6%82%E8%BF%B0%EF%BC%9AIVF.md)
- 5. [算法概述：LSH](05-%E7%AE%97%E6%B3%95%E6%A6%82%E8%BF%B0%EF%BC%9ALSH.md)
- 6. [索引参数与调优](06-%E7%B4%A2%E5%BC%95%E5%8F%82%E6%95%B0%E4%B8%8E%E8%B0%83%E4%BC%98.md)
- 7. [评估 ANN 性能](07-%E8%AF%84%E4%BC%B0%20ANN%20%E6%80%A7%E8%83%BD.md)
- 8. [动手实践：调整索引参数的试验](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%B0%83%E6%95%B4%E7%B4%A2%E5%BC%95%E5%8F%82%E6%95%B0%E7%9A%84%E8%AF%95%E9%AA%8C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-3-approximate-nearest-neighbor-search/quiz)
