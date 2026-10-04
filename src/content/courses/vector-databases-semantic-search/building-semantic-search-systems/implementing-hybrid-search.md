---
course: "vector-databases-semantic-search"
chapter: "building-semantic-search-systems"
lesson: "implementing-hybrid-search"
sourceId: 3573
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-4-building-semantic-search-systems/implementing-hybrid-search"
title: "实现混合搜索"
description: "将语义搜索结果与传统关键词搜索分数（例如BM25）结合起来。"
order: 6
plots: []
sourceHash: "3278d53f31c7969f5b8ac75d74275441b0afbbede9c3d1a5690da0cca1a10b31"
sourceCorrections: []
---

虽然语义搜索擅长根据含义查找相关结果，但有时难以处理精确的关键词匹配或特定术语（如产品代码、名称或行话），这些在嵌入 (embedding)空间中可能表现不佳。反之，传统关键词搜索（常使用BM25等算法）在检索包含特定术语的文档方面表现出色，但无法理解同义词、相关想法或自然语言表达的用户意图。

混合搜索通过结合这两种方法的优点，提供了一种实用的解决方案。它旨在检索同时具有语义关联 (semantic relationship)性和包含重要关键词的结果，从而带来更全面且通常更准确的搜索体验。

### 为何结合搜索方法？

请看以下情况：

1. **特定术语的准确性：** 用户搜索产品ID如“XZ-47b”时，期望结果包含该精确字符串。语义搜索可能返回相似产品，但如果ID在向量 (vector)上下文 (context)中不强，则可能错过精确匹配。关键词搜索在这方面表现优异。
2. **捕获意图：** 用户搜索“如何修理漏水的水龙头”时，受益于语义搜索理解水龙头修理的*想法*，从而检索到使用不同措辞（“滴水水龙头修理”、“阻止水槽泄漏”）的指南。单独的关键词搜索可能会错过未使用确切词语“漏水的水龙头”的相关文档。
3. **提高关联度：** 有时，一个文档既语义相似又包含查询中的特定关键词。混合方法可以提升此类文档的排名，认为它们高度相关。

通过合并向量相似度搜索和基于关键词的系统的结果，我们可以减轻每种单独方法的不足。

### 结果结合方案

混合搜索中的主要难点是将两组不同的结果（每组都有自己的关联度分数）合并成一个单一、连贯的排名列表。以下是常用做法：

1. **基于分数的合并：** 这涉及从语义搜索系统（向量 (vector)数据库）和关键词搜索系统（例如，使用BM25的Elasticsearch、OpenSearch）独立检索结果，然后结合它们的分数。

   - **标准化：** 不同系统的分数通常处于不同量级。向量相似度分数（如余弦相似度）可能在-1到1或0到1之间，而BM25分数则根据词频和文档长度变化很大。在结合之前，分数通常需要标准化到公共范围（例如0到1）。可以使用最小-最大缩放或基于排名的标准化等方法。
   - **加权组合：** 一旦标准化，分数可以使用加权和进行组合。对于两个系统都检索到的文档 $d$：
     
     $$
     \text{混合分数}(d) = \alpha \times \text{语义分数}(d) + (1 - \alpha) \times \text{关键词分数}(d)
     $$
     
     这里，$\alpha$ 是一个介于0和1之间的权重 (weight)因子。较高的 $\alpha$ 会使语义关联 (semantic relationship)度更受重视，而较低的 $\alpha$ 则强调关键词匹配。$\alpha$ 的最佳值通常需要根据具体数据集和用例进行实验和调整。仅由一个系统找到的文档可能会排在列表靠后的位置，或者仅根据找到它们的系统分配分数（可能带有惩罚）。
   - **倒数排名融合 (RRF)：** RRF是一种结合排名列表的技术，它基于每个文档的排名位置，而不是直接依赖可能无法比较的分数。对于每个文档 $d$，其RRF分数计算如下：
     
     $$
     \text{RRF分数}(d) = \sum_{i=1}^{N} \frac{1}{k + \text{排名}_i(d)}
     $$
     
     其中，$N$是搜索系统数量（混合搜索通常为2），$rank_i(d)$是文档$d$在系统$i$结果列表中的排名，$k$是一个常数（通常设为60，如原始论文所建议），用于减轻高排名对分数的主导影响。未被系统找到的文档被赋予无限排名（对总和贡献0）。RRF通常更受青睐，因为它避免了分数标准化和加权的复杂性。
2. **预过滤 / 后过滤：** 除了复杂的分数合并，更简单的做法是使用一个系统来为另一个系统过滤候选结果。

   - **关键词预过滤：** 首先执行关键词搜索，以获得包含基本术语的候选文档集。然后，*只*在此候选集中执行语义搜索。这可能更快，但可能会错过缺乏精确关键词但语义相关的文档。

- **语义预过滤：** 首先执行语义搜索，以获得一组相关文档。然后，在此集合内根据关键词的存在进行关键词匹配或重新排序。

### 实现流程

典型的基于分数的合并实现如下：

> 流程图展示了使用基于分数合并的常见混合搜索实现。用户查询并行处理以进行语义嵌入 (embedding)和关键词提取。然后，向量 (vector)数据库和关键词索引的结果进行合并和重排。

### 注意事项

- **性能：** 与单一搜索系统相比，并行执行两次搜索（向量 (vector)和关键词）并合并结果会增加延迟。缓存和优化很重要。
- **复杂度：** 实现和调整混合搜索比使用单一方法更复杂。它需要管理两种不同的索引和查询系统，并制定一个方案。
- **权重 (weight)/参数 (parameter)调整：** 找到正确的平衡（加权和中的 $\alpha$ 或RRF中的 $k$）通常需要在一个有代表性的数据集上评估搜索关联度（如下一节所述），并迭代调整参数。
- **系统集成：** 将向量数据库与现有关键词搜索系统（如Elasticsearch或Solr）集成需要仔细的基础设施规划。一些现代数据库或搜索平台提供内置的混合搜索功能，这可以使集成变得更简单。

混合搜索代表了一种有效的方法，可以发挥语义理解和精确术语匹配的互补优势，最终在许多应用中带来更有效、更令用户满意的搜索结果。

## 参考资料

- [Reciprocal Rank Fusion (RRF): A Multi-System Combiner](https://dl.acm.org/doi/10.1145/1879792.1879809) — Jeffrey A. Allan, Bruce Croft (2009)
  Journal: Proceedings of the Eighteenth Text REtrieval Conference (TREC 2009); Publisher: ACM; Pages: 111-118; DOI: [10.1145/1879792.1879809](https://doi.org/10.1145/1879792.1879809)
  这是介绍互惠排序融合（RRF）的原始论文，该方法用于结合来自多个搜索系统的排序列表。
- [Okapi at TREC-3: Automatic Query Construction and Other Experiments](https://trec.nist.gov/pubs/trec3/papers/london.pdf) — Stephen E. Robertson, Stephen Walker, Michael Hancock, Ann Beaulieu, Mike Gatford (1994)
  Journal: Proceedings of the Third Text REtrieval Conference (TREC 3); Publisher: National Institute of Standards and Technology (NIST); Pages: 109-122
  本文正式介绍了BM25排序函数，这是一种广泛用于基于关键词的信息检索算法。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本信息检索基础教科书，内容涵盖BM25等关键词模型以及结合搜索结果的策略。
