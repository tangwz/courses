---
course: "langchain-production-llm"
chapter: "production-data-retrieval"
lesson: "hybrid-search-implementation"
sourceId: 3895
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-4-production-data-retrieval/hybrid-search-implementation"
title: "混合搜索的实现"
description: "将密集向量搜索与传统稀疏方法 (例如 BM25) 结合，以提高相关性。"
order: 4
plots: []
sourceHash: "5a35add1c4af3910f91f9602ac87302ae1aa75687cc2ec8b88b45ed6dc002783"
sourceCorrections: []
---

#### LangChain 实现说明

LangChain 提供专门的抽象层，以便更好地支持混合搜索。举例来说：

- **`EnsembleRetriever`**：`langchain.retrievers` 中的这个类接受检索器列表 (例如 BM25 检索器和向量 (vector)存储检索器)，并整合它们的检索结果。它通常使用倒数排名融合 (RRF) 算法，根据文档在各个结果列表中的位置进行重新排序。你可以在 RRF 计算中分配权重 (weight)来优先处理特定检索器。

```python
# 使用 EnsembleRetriever 的示例 (假设检索器已初始化)
from langchain.retrievers import EnsembleRetriever
# 假设 bm25_retriever 和 faiss_retriever 已配置
# 示例：pip install rank_bm25, faiss-cpu

# 初始化稀疏检索器 (例如 BM25)
# ... 设置 bm25_retriever ...

# 初始化密集检索器 (例如 FAISS 向量存储)
# ... 设置 faiss_retriever ...

# 使用 EnsembleRetriever 结合它们 (使用加权倒数排名融合)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.4, 0.6] # 可选：应用于 RRF 计算的权重
)

# 使用混合检索器
query = "How to fix connection timeout?"
hybrid_results = ensemble_retriever.invoke(query)

print(hybrid_results)
```

- **向量存储集成**：一些向量数据库 (例如 Pinecone、Weaviate、带密集向量 (dense vector)的 Elasticsearch) 提供对混合搜索的内置支持，允许你通过它们的 API 同时使用稀疏关键词和密集向量进行查询。LangChain 集成通常会公开这些功能。请查阅你的特定向量存储集成的文档。

## 参考资料

- [Okapi at TREC-3](http://trec.nist.gov/pubs/trec3/papers/city.ps.gz) — Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, Mike Gatford (1994)
  Journal: Proceedings of The Third Text REtrieval Conference, TREC 1994; Publisher: National Institute of Standards and Technology (NIST); Volume: 500-225; Pages: 109–126
  描述了Okapi BM25排名函数，这是稀疏检索的基石，阐释了其在早期信息检索评估中的有效性。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410.pdf) — Nils Reimers, Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3981-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了Sentence-BERT，它是BERT的一种修改版，能够生成具有语义意义的句子嵌入，这是密集检索方法的基本技术。
- [Hybrid Search: An Overview](https://www.pinecone.io/learn/hybrid-search-overview/) — Adam Benayoun (2024)
  Publisher: Pinecone
  从向量数据库供应商的角度，提供了混合搜索的实用概述，解释了其组成部分、优势和常见的实现策略。
- [LangChain Documentation: EnsembleRetriever](https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/ensemble/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain `EnsembleRetriever` 的官方文档，详细说明了其用法以及如何结合来自多个检索器（包括稀疏和密集方法）的结果。
