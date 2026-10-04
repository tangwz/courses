---
course: "getting-started-with-llm-toolkit"
chapter: "embeddings-and-semantic-search"
lesson: "performing-semantic-search"
sourceId: 7771
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-5-embeddings-and-semantic-search/performing-semantic-search"
title: "进行语义搜索"
description: "实现语义搜索功能，以查找针对给定查询最相关的文档片段。"
order: 4
plots: []
sourceHash: "9de965b06bae7ca5fd283c500882c283ae1be269d67e639d91337aed22aa3903"
sourceCorrections: []
---

语义搜索是检索系统的核心组成部分。这种方式采用文本嵌入 (embedding)和向量 (vector)相似度，根据查询的含义和意图来查找信息。与依赖精确词语匹配的传统关键词搜索不同，语义搜索可以帮助你找到相关文档，即使这些文档不包含搜索查询中使用的确切词语。

该过程是你刚学到的工具的直接应用。为了找到给定查询最相关的文档，你将执行以下步骤：

1. 为用户查询生成一个嵌入。
2. 将此查询嵌入与知识库中所有文档片段的预计算嵌入进行比较。
3. 计算每个文档片段的相似度分数（使用余弦相似度）。
4. 将文档片段从高分到低分排序，以识别出最相关的。

### 批量计算相似度

虽然你可以循环遍历每个文档嵌入 (embedding)并逐一计算其与查询嵌入的相似度，但这种方式效率不高，特别是当文档数量很多时。更高效的方法是在一次优化操作中计算所有相似度。

`batch_similarity` 函数正是为此目的而设计的。它接收一个查询向量 (vector)和一系列文档向量，然后高效地计算查询与每个文档之间的相似度。

下面是如何实现一个基本的语义搜索功能：

```python
from kerb.embedding import embed, embed_batch, batch_similarity

# 1. 你的文档片段集合
documents = [
    "Python 是一种高级编程语言",
    "机器学习模型从数据中学习模式",
    "自然语言处理帮助计算机理解文本",
    "深度神经网络有多层",
    "数据科学结合了统计学和编程",
]

# 2. 预计算所有文档的嵌入
doc_embeddings = embed_batch(documents)

# 3. 定义一个搜索查询并生成其嵌入
query = "我想了解人工智能和神经网络"
query_embedding = embed(query)

# 4. 计算所有文档与查询的相似度分数
similarities = batch_similarity(query_embedding, doc_embeddings, metric="cosine")

# 5. 将文档及其分数结合，并按相关性排序
results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

# 显示前3个结果
print(f"查询: '{query}'\n")
print("前3个结果:")
for i, (doc, score) in enumerate(results[:3], 1):
    print(f"{i}. [{score:.4f}] {doc}")
```

请注意，排名靠前的结果“深度神经网络 (neural network)有多层”与查询没有太多共同关键词。然而，由于它们的嵌入在向量空间中距离很近，系统正确地将其识别为最相关的文档。

### 高效获取前K个结果

对整个相似度列表进行排序效果不错，但对于仅需要少量排名靠前结果的应用，有更直接的方法。`top_k_similar` 函数经过优化，可以在不排序整个集合的情况下找到 `k` 个最相似的向量 (vector)。它返回一个与得分最高的文档对应的索引列表。

这在RAG系统中尤其有用，在RAG系统中，通常只需要将3-5个最相关的片段传递给LLM。

```python
from kerb.embedding import embed, embed_batch, top_k_similar, cosine_similarity

documents = [
    "Python 是一种高级编程语言",
    "机器学习模型从数据中学习模式",
    "自然语言处理帮助计算机理解文本",
    "深度神经网络有多层",
    "数据科学结合了统计学和编程",
    "软件工程涉及系统设计与构建",
]
doc_embeddings = embed_batch(documents)

query = "编程语言和软件开发"
query_embedding = embed(query)

# 获取前3个最相似文档的索引
top_3_indices = top_k_similar(query_embedding, doc_embeddings, k=3)

print(f"查询: '{query}'\n")
print("使用 top_k_similar 的前3个结果:")
for rank, idx in enumerate(top_3_indices, 1):
    # 我们可以通过只计算排名靠前索引的相似度来获取分数
    similarity = cosine_similarity(query_embedding, doc_embeddings[idx])
    print(f"{rank}. [{similarity:.4f}] {documents[idx]}")
```

使用 `top_k_similar` 通常比 `batch_similarity` 加上完全排序更快，内存效率更高，使其成为大多数生产场景中进行检索的首选方法。

### 构建一个简单的搜索引擎

有了这些组件，你可以将逻辑封装到一个简单、可重用的搜索类中。这是构建更复杂检索系统的基本模式。该类将处理文档索引（预计算嵌入 (embedding)）和执行搜索。

```python
from kerb.embedding import embed, embed_batch, top_k_similar, cosine_similarity

class SimpleSearchEngine:
    """一个基本的语义搜索引擎。"""

    def __init__(self, documents: list[str]):
        self.documents = documents
        print(f"正在索引 {len(documents)} 个文档...")
        self.embeddings = embed_batch(documents)
        print("索引完成。")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """搜索相关文档。"""
        query_emb = embed(query)
        top_indices = top_k_similar(query_emb, self.embeddings, k=top_k)

        results = []
        for idx in top_indices:
            sim = cosine_similarity(query_emb, self.embeddings[idx])
            results.append({
                'document': self.documents[idx],
                'score': sim,
                'index': idx
            })
        return results

# 文档集合
documents = [
    "Python 是一种高级编程语言",
    "机器学习模型从数据中学习模式",
    "自然语言处理帮助计算机理解文本",
    "人工智能使机器能够思考",
    "软件工程涉及系统设计与构建"
]

# 创建并使用搜索引擎
engine = SimpleSearchEngine(documents)

# 执行搜索
search_query = "人工智能和计算机思维"
search_results = engine.search(search_query, top_k=2)

print(f"\n搜索: '{search_query}'")
for i, result in enumerate(search_results, 1):
    print(f"  {i}. [{result['score']:.4f}] {result['document']}")
```

这个简单的类演示了完整的端到端流程：通过创建嵌入来索引文档，然后利用这些嵌入在查询时查找相关信息。

### 使用阈值处理不相关查询

检索系统中一个常见难题是处理知识库中没有相关文档的查询。在这种情况下，语义搜索仍然会返回*最不相似*的文档，即使它们的相似度分数非常低。这可能导致LLM接收到不相关的内容，并生成质量不佳或错误的回复。

为了解决这个问题，你可以应用一个相似度分数阈值。如果没有文档达到最低分数，你可以得出结论，没有相关结果。

```python
# 假设 'doc_embeddings' 和 'documents' 已经定义
query = "量子计算" # 一个在我们的少量集合中不存在的话题
query_embedding = embed(query)

# 计算所有文档的相似度
similarities = batch_similarity(query_embedding, doc_embeddings)

# 根据阈值过滤结果
threshold = 0.3
relevant_results = [
    (doc, sim) for doc, sim in zip(documents, similarities) if sim > threshold
]

print(f"查询: '{query}'")
print(f"相似度高于 {threshold} 的结果:")

if relevant_results:
    for doc, sim in sorted(relevant_results, key=lambda x: x[1], reverse=True):
        print(f"  [{sim:.4f}] {doc}")
else:
    print(f"  没有找到高于阈值 {threshold} 的结果")
    best_match_score = max(similarities)
    best_match_doc = documents[similarities.index(best_match_score)]
    print(f"  (低于阈值的最佳匹配: [{best_match_score:.4f}] {best_match_doc})")
```

选择合适的阈值取决于具体的应用场景，并且通常需要根据你的数据集进行实验。一个好的起始范围通常在0.25到0.4之间。通过过滤掉低质量的结果，你可以确保传递给语言模型的内容既相关又有用，这是任何RAG系统性能的根本保证。

## 参考资料

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://doi.org/10.48550/arXiv.1908.10084) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Pages: 3972-3982; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  本文介绍了Sentence-BERT，它是BERT的改进版本，能够生成有语义意义的句子嵌入，适用于语义相似度搜索等任务。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  这本广泛使用的教科书提供了信息检索的基本概念，包括向量空间模型和排序，这些与理解语义搜索相关。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  本文介绍了检索增强生成（RAG）框架，它将预训练的检索器与生成器结合，以改进知识密集型NLP任务，与文中提到的RAG系统直接相关。
