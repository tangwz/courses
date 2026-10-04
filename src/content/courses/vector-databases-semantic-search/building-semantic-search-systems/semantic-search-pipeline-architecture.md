---
course: "vector-databases-semantic-search"
chapter: "building-semantic-search-systems"
lesson: "semantic-search-pipeline-architecture"
sourceId: 3562
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-4-building-semantic-search-systems/semantic-search-pipeline-architecture"
title: "语义搜索流程的架构"
description: "涉及的组成部分：数据获取、嵌入生成、索引、查询嵌入、搜索和排序。"
order: 2
plots: []
sourceHash: "49d86dce173a6393a561cace4fbea3fb971045f3398fd452292a6fd2251f0e44"
sourceCorrections: []
---

一个完整的语义搜索流程由多个主要组成部分构成。这些组成部分包括向量 (vector)嵌入 (embedding)的生成、向量数据库的结构，以及通过近似最近邻（ANN）搜索实现相似向量的查找。了解这些部分如何结合起来形成整体架构，对于构建能够根据含义而非仅仅精确关键词匹配进行搜索的应用非常重要。

语义搜索流程并非一个单一的整体；它是一系列处理步骤，通常包含独立的离线（索引）和在线（查询）阶段。

> 该图展示了语义搜索系统中的典型数据流，分为离线索引阶段和在线查询阶段。

我们来细分图中所示的每个阶段：

### 索引阶段 (离线)

此阶段在用户开始搜索之前发生。其目的是准备好您的数据，并使其在向量 (vector)数据库中可被搜索。

1. **原始数据获取：** 这是您收集想要进行搜索的数据的起点。这可以是文本文档、产品描述、图片、音频文件或任何其他可生成有含义嵌入 (embedding)的数据类型。
2. **数据预处理与分块：** 原始数据通常需要清洗（例如，移除无关字符、HTML标签等）。对于语义搜索而言，更重要的一点是，大型数据（如长文档）通常需要分解成更小、语义连贯的*块*。为什么？因为嵌入模型通常有输入长度限制，而且更重要的是，单个嵌入表示的是整个输入文本的含义。对整本书进行嵌入可能会过多地平均其含义；而对较小的段落或部分进行嵌入通常会生成更具体和有用的搜索向量。我们将在下一节讨论分块策略。与原始数据关联的元数据（例如，文档ID、作者、创建日期、产品类别）应被保留并与相应的块关联起来。
3. **嵌入生成：** 每个经过处理的数据块都会通过选定的嵌入模型（例如用于文本的Sentence-BERT，或用于图片/文本的CLIP）。这会将数据块转换为高维向量，捕捉其语义内容。这个过程可能需要大量计算，特别是在处理大型数据集时，因此批处理和潜在的GPU加速是常见的考量因素。
4. **向量数据库索引：** 生成的向量，连同其关联的元数据和每个块的唯一标识符，被载入向量数据库。数据库随后会为这些向量建立一个ANN索引（例如HNSW、IVF），如第3章所述。这种索引结构允许后续进行快速的近似相似性搜索。元数据也同时被索引，使得过滤搜索成为可能。

### 查询阶段 (在线)

此阶段在用户提交搜索查询时实时发生。

1. **用户查询输入：** 系统接收用户的查询（例如，一个句子、一个问题，甚至是一张图片）。
2. **查询预处理：** 与数据预处理类似，原始查询可能会经历一些清洗或规范化。
3. **查询嵌入 (embedding)生成：** 这是重要的一步。索引阶段使用的*完全相同的嵌入模型*必须应用于处理后的用户查询。这会将查询转换为与已索引数据向量 (vector)处于相同语义空间中的向量。使用不同的模型会导致向量不兼容，并产生无意义的搜索结果。
4. **ANN搜索：** 查询向量被发送到向量数据库。数据库使用其ANN索引，以高效方式在索引中查找与查询向量最接近（最相似，基于所选指标，如余弦相似度或欧氏距离）的向量。通常，您会请求最接近的 $k$ 个邻居。
5. **元数据过滤 (可选)：** 向量数据库通常允许将向量相似性搜索与基于向量旁边存储的元数据进行过滤相结合。这可以发生在向量搜索*之前*（预过滤，可能缩小搜索范围）或向量搜索*之后*（后过滤，优化前 $k$ 个候选结果）。例如，用户可能搜索“机器学习 (machine learning)文章”，但只希望得到过去一年内发表的结果。向量搜索找到语义相关的文章，而元数据过滤器则移除超出所需日期范围的文章。
6. **重排序 (可选)：** ANN搜索的原始相似度分数提供了一个初步排序。然而，您可能希望使用额外的信号来调整这个排序。例如，在*混合搜索*系统（本章稍后讨论）中，您可以将语义相似度分数与传统关键词相关性分数（如BM25）结合起来。您还可以考虑文档的新近度、受欢迎程度或用户偏好。重排序会接收来自ANN搜索（和可能的过滤）的初始候选集，并根据一个可能更复杂的关联函数重新排列它们。
7. **结果展示：** 最终的、经过排序的结果列表（通常是原始数据块或文档的引用）会被格式化并展示给用户。

这种模块化架构允许每个组件得到优化或可以被替换。例如，您可以尝试不同的嵌入模型或调整ANN索引参数 (parameter)，而无需重新设计整个流程。重要的是数据的流动，将其转换为向量，高效地进行索引，然后使用查询向量在该已索引空间中查找语义相似的项。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了一种广泛使用的生成高质量句子嵌入的方法，直接适用于管道中的嵌入生成步骤。
- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://doi.org/10.1109/TPAMI.2018.2889473) — Yu. A. Malkov, D. A. Yashunin (2018)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 42; Pages: 824-836; DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
  详细介绍了HNSW算法，这是矢量数据库中高效近似最近邻搜索的常用方法。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本经典教科书，为信息检索概念提供了坚实的基础，包括索引、查询和排序，有助于理解搜索系统的更广泛背景。
