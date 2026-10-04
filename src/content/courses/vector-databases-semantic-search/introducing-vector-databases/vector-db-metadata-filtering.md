---
course: "vector-databases-semantic-search"
chapter: "introducing-vector-databases"
lesson: "vector-db-metadata-filtering"
sourceId: 3531
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-2-introducing-vector-databases/vector-db-metadata-filtering"
title: "元数据过滤"
description: "将向量相似性搜索与基于关联元数据字段的过滤结合。"
order: 5
plots: []
sourceHash: "087d6f40f57afc4705872635d21e269a8dcc14bdbcc711059ab3b7bc0363f2a9"
sourceCorrections: []
---

“虽然使用相似性搜索查找最接近的向量 (vector)是一项强大的功能，但应用程序通常需要对结果进行更精细的控制。想象一下，你在产品数据库中搜索与你上传图片视觉上相似的鞋子。你可能会得到靴子、运动鞋和凉鞋的结果。但如果你只想要尺码为10、目前有库存的运动鞋怎么办？这时，元数据过滤就变得必不可少了。”

元数据指与每个向量关联的补充性结构化信息。这可以包括时间戳、类别、用户ID、产品属性（如尺码、颜色、品牌）、地理位置、访问权限或与你的应用场景相关的任何其他描述性数据。向量数据库通常将这些元数据与向量本身一同存储，通常在与向量相同的记录或文档结构中。

元数据过滤的主要目的是将向量相似性搜索的语义能力与传统的基于属性的过滤结合起来。你不再只是询问“找到与此查询向量最相似的向量”，而是可以问“找到与此查询向量最相似*且满足这些具体元数据条件的向量*”。

### 为何使用元数据过滤？

1. **提高相关性：** 过滤器将搜索结果缩小到精确匹配用户需求的项目，而不仅仅是语义相似度。搜索与给定摘要相似的研究论文是有用的，但按发表年份或作者进行过滤会使结果更有针对性。
2. **个性化：** 过滤器可用于根据用户个人资料、过往互动或权限定制搜索结果。例如，推荐与用户阅读历史相似的文章（向量 (vector)搜索），同时过滤掉他们已读过的文章（元数据过滤）。
3. **强制执行约束：** 应用程序可能需要强制执行业务规则或约束，例如根据可用性、价格范围或地理区域过滤产品。

### 过滤如何进行

向量 (vector)数据库通常与近似最近邻 (ANN) 搜索过程结合实现元数据过滤。主要有两种策略：

1. **预过滤（先过滤后搜索）：** 数据库首先识别所有元数据匹配指定过滤条件的向量。然后，ANN 搜索*只*在此向量子集内执行。如果过滤器具有高度选择性（即，它显著减少了候选数量），此方法可以显著加快速度，因为计算密集型 ANN 搜索在较小数据集上运行。然而，它要求底层索引结构能高效支持在搜索前进行过滤。
2. **后过滤（先搜索后过滤）：** 数据库首先执行ANN搜索，纯粹基于向量相似度找到前k个最相似的向量。然后，它过滤此初始结果集，移除任何元数据不匹配指定过滤条件的向量。此方法更简单，并且适用于任何ANN索引。然而，如果初始ANN搜索返回许多后来被过滤掉的候选，可能会效率低下，浪费计算资源。如果过滤器移除了初始前k个结果的很大一部分，你最终获得的结果数量可能会少于请求。

> 预过滤（ANN搜索前过滤）和后过滤（ANN搜索后过滤）工作流程的比较。

现代向量数据库通常采用复杂的索引技术，以实现高效的预过滤，或将过滤功能紧密集成到ANN搜索图遍历（如HNSW算法）中，旨在获得预过滤的性能优势，同时避免其局限性。

### 指定过滤器

指定过滤器的具体语法因不同的向量 (vector)数据库平台而异，但基本原理相似。你通常会将过滤表达式作为搜索查询的一部分提供，通常使用熟悉的逻辑运算符（AND、OR、NOT）和比较运算符（等于、大于、小于、在列表中等）应用于元数据字段。

例如，查询可能如下所示：

```
查找与 query_vector 相似的向量
其中 metadata.category = "运动鞋"
  并且 metadata.size = 10
  并且 metadata.in_stock = true
限制 10 条
```

### 性能考量

- **过滤器选择性：** 高度选择性（只匹配一小部分数据）的过滤器能让预过滤策略极大受益。选择性较低的过滤器在预过滤或后过滤下的性能可能类似。
- **索引类型：** 某些ANN索引类型比其他类型更适合高效的预过滤。数据库的实现细节在此处影响很大。
- **元数据复杂度：** 涉及多个字段和运算符的复杂过滤条件会增加开销。适当地索引元数据字段（类似于传统数据库）有时可以提升过滤性能。

“元数据过滤将向量 (vector)搜索从纯粹的相似性查询转变为多功能的查询机制，能够处理复杂需求。它弥合了语义理解与结构化数据约束之间的差距，使你能够基于向量数据构建更精确、相关且有用的应用程序。在后续的学习中，理解如何有效运用元数据对于优化搜索性能和相关性将很重要。”

## 参考资料

- [Filtering in Qdrant: How it works and why it’s important](https://qdrant.tech/articles/filtering/) — Kacper Łukawski (2022)
  Publisher: Qdrant
  详细解释了元数据过滤策略（预过滤、后过滤）及其在生产型向量数据库中的应用。
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://ieeexplore.ieee.org/document/8040446) — Yu. A. Malkov, D. A. Yashunin (2018)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 40; Pages: 2192-2202; DOI: [10.1109/TPAMI.2017.2762879](https://doi.org/10.1109/TPAMI.2017.2762879)
  描述了许多向量数据库使用的核心HNSW算法，为如何在这些结构上集成或应用过滤提供了背景。最初以预印本形式发布在arXiv上。
- [Filtering Nearest Neighbors for Hybrid Search](https://www.vldb.org/pvldb/vol16/p3676-li.pdf) — Haoxiang Li, Yandi Li, Minjia Zhang, Song Liu, Jian Li (2023)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 16; Pages: 3676-3689; DOI: [10.14778/3611477.3611502](https://doi.org/10.14778/3611477.3611502)
  讨论了将元数据过滤高效集成到近似最近邻搜索中的高级技术，涵盖了各种策略及其性能影响。
