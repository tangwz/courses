---
course: "advanced-vector-search-llms"
chapter: "hybrid-search-approaches"
lesson: "hands-on-hybrid-search-pipeline"
sourceId: 3883
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/hands-on-hybrid-search-pipeline"
title: "动手实践：构建混合搜索流程"
description: "实施混合搜索系统，结合向量和关键词检索的实践环节。"
order: 7
plots: []
sourceHash: "e83d63caa5fed4e36b1a27f004e24eb7f82c31b456ec452c3143653297dba367"
sourceCorrections: []
---

使用 Python 构建的混合搜索流程结合了关键词搜索（特别是BM25）与向量 (vector)相似度搜索。此流程整合了这些不同的检索方法，并合并它们的结果以提升相关性。

我们假设您有一个可运行的Python环境，其中包含用于向量操作的库，以及可能存在的向量数据库客户端或Faiss等库。您还需要关键词搜索的库。

### 前提条件与设置

在我们开始之前，请确保已安装所需的库。在本例中，我们将使用 `rank_bm25` 进行关键词搜索，`sentence-transformers` 用于生成嵌入 (embedding)（尽管在搜索阶段我们假定向量 (vector)是预先计算好的），以及 `numpy`。

```bash
pip install rank_bm25 numpy sentence-transformers
```

我们将使用一个文档数据集，其中每个文档都包含ID、文本内容和预先计算好的向量嵌入。

```python
# 示例数据结构（请替换为您的实际数据加载方式）
documents = [
    {"id": "doc1", "text": "Introduction to machine learning concepts.", "vector": [0.1, 0.2, ..., 0.5]},
    {"id": "doc2", "text": "Advanced deep learning techniques for NLP.", "vector": [0.3, 0.1, ..., 0.7]},
    {"id": "doc3", "text": "A guide to Python programming basics.", "vector": [0.8, 0.7, ..., 0.1]},
    {"id": "doc4", "text": "Optimizing machine learning models.", "vector": [0.2, 0.3, ..., 0.6]},
    # ... 更多文档
]

# 模拟将预计算的向量加载到结构中以供查找
vector_index_data = {doc["id"]: doc["vector"] for doc in documents}
document_texts = {doc["id"]: doc["text"] for doc in documents}

# 查询示例
query_text = "machine learning optimization"

# 假设我们有一个获取查询嵌入的函数
# 在实际场景中，您会使用与文档向量相同的模型
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # 示例模型

def get_query_embedding(query):
    return embedding_model.encode(query)

query_vector = get_query_embedding(query_text)
```

### 步骤1：实现关键词搜索 (BM25)

首先，让我们设置BM25组件。我们将对文档的文本内容进行索引。`rank_bm25` 库提供了一个直接的实现。

```python
import numpy as np
from rank_bm25 import BM25Okapi

# 对文档进行分词（本例中使用简单的空格分隔）
tokenized_corpus = [doc["text"].lower().split() for doc in documents]
doc_ids = [doc["id"] for doc in documents]

# 创建BM25索引
bm25 = BM25Okapi(tokenized_corpus)

def perform_keyword_search(query_text, top_n=10):
    """执行BM25搜索并返回排名的文档ID和分数。"""
    tokenized_query = query_text.lower().split()
    # 获取所有文档的分数
    doc_scores = bm25.get_scores(tokenized_query)

    # 获取前N个分数的索引
    top_indices = np.argsort(doc_scores)[::-1][:top_n]

    # 将索引映射回文档ID并创建结果列表
    results = []
    for index in top_indices:
        if doc_scores[index] > 0: # 只包含分数非零的文档
            results.append({"id": doc_ids[index], "score": doc_scores[index]})
    return results

# 执行关键词搜索
keyword_results = perform_keyword_search(query_text, top_n=5)
print("关键词搜索结果 (BM25):")
print(keyword_results)
# 预期输出（示例）:
# 关键词搜索结果 (BM25):
# [{'id': 'doc4', 'score': 0.937...}, {'id': 'doc1', 'score': 0.937...}, {'id': 'doc2', 'score': 0.0}] # 注意：分数很大程度上取决于语料库
```

这会给我们一个文档ID列表，它们按其与查询的BM25相关性分数进行排序。

### 步骤2：实现向量 (vector)搜索

接下来，我们模拟向量搜索组件。在实际系统中，这会涉及查询一个从 `vector_index_data` 构建的ANN索引（如Faiss、OpenSearch、Pinecone、Weaviate等中的HNSW）。在这里，我们将通过计算与所有向量的余弦相似度来模拟它（对于大型数据集效率不高，但足以演示混合方法的想法）。

```python
from numpy.linalg import norm

def cosine_similarity(vec_a, vec_b):
    """计算两个向量之间的余弦相似度。"""
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def perform_vector_search(query_vector, index_data, top_n=10):
    """执行向量搜索并返回排名的文档ID和分数。"""
    scores = []
    for doc_id, doc_vector in index_data.items():
        similarity = cosine_similarity(query_vector, doc_vector)
        scores.append({"id": doc_id, "score": similarity})

    # 按分数降序排序
    scores.sort(key=lambda x: x["score"], reverse=True)

    return scores[:top_n]

# 执行向量搜索
vector_results = perform_vector_search(query_vector, vector_index_data, top_n=5)
print("\n向量搜索结果:")
print(vector_results)
# 预期输出（示例）:
# 向量搜索结果:
# [{'id': 'doc4', 'score': 0.85...}, {'id': 'doc1', 'score': 0.78...}, {'id': 'doc2', 'score': 0.65...}, {'id': 'doc3', 'score': 0.12...}]
```

现在我们有了第二个文档ID列表，按语义相似度进行排序。请注意，分数范围不同（余弦相似度通常在[-1, 1]或[0, 1]之间，BM25通常为正且无上限），并且排名可能与关键词结果有显著差异。

### 步骤3：使用倒数排名融合 (RRF) 合并结果

如前所述，由于分数尺度和分布不同，简单地相加或平均分数会带来问题。倒数排名融合 (RRF) 提供了一种有效的方法来组合排名，而无需分数归一化 (normalization)。

文档 $d$ 的RRF分数计算公式如下：
$Score_{RRF}(d) = \sum_{i \in \text{结果列表}} \frac{1}{k + rank_i(d)}$
其中 $rank_i(d)$ 是文档 $d$ 在结果列表 $i$ 中的排名（从1开始），而 $k$ 是一个常数，控制着较低排名项的影响。$k$ 的一个常用值是60。

让我们实现RRF函数：

```python
def apply_rrf(results_lists, k=60):
    """应用倒数排名融合来组合多个排名列表。"""
    fused_scores = {}
    doc_data = {} # 用于存储原始分数或仅存储ID（如果需要）

    for results in results_lists:
        for rank, result in enumerate(results):
            doc_id = result["id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_data[doc_id] = {"id": doc_id} # 存储关联数据

            # 计算此列表的RRF贡献
            rrf_score = 1.0 / (k + rank + 1) # 排名是基于0的，所以加1
            fused_scores[doc_id] += rrf_score

    # 组合ID和分数，然后排序
    fused_results = [{"id": doc_id, "score": fused_scores[doc_id]} for doc_id in fused_scores]
    fused_results.sort(key=lambda x: x["score"], reverse=True)

    return fused_results

# 组合关键词和向量搜索结果
hybrid_results = apply_rrf([keyword_results, vector_results], k=60)

print("\n混合搜索结果 (RRF):")
print(hybrid_results)
# 预期输出（示例，取决于输入排名）：
# 混合搜索结果 (RRF):
# [{'id': 'doc4', 'score': 0.032...}, {'id': 'doc1', 'score': 0.032...}, {'id': 'doc2', 'score': 0.016...}, {'id': 'doc3', 'score': 0.015...}]
# 注意：RRF分数是相对的；排名是主要输出。
# 在本例中，doc4和doc1在两个列表中都排名靠前，获得了更高的RRF分数。
```

### 组合起来：混合流程工作原理

我们可以将整个过程封装到一个函数中，表示我们的混合搜索流程。

> 这是一个图表，说明了混合搜索流程。用户查询既用于关键词搜索，也用于生成向量 (vector)搜索的嵌入 (embedding)。然后将两者的结果合并（例如，使用RRF）以生成最终的排名列表。

```python
def hybrid_search_pipeline(query_text, top_n=10, rrf_k=60):
    """执行完整的混合搜索流程。"""

    print(f"正在为查询 '{query_text}' 执行混合搜索")

    # 1. 获取查询嵌入
    query_vector = get_query_embedding(query_text)

    # 2. 执行关键词搜索
    keyword_results = perform_keyword_search(query_text, top_n=top_n * 2) # 最初检索更多结果
    print(f" BM25 返回了 {len(keyword_results)} 个结果。")

    # 3. 执行向量搜索
    # 请将此占位符替换为对您实际向量搜索系统的调用
    vector_results = perform_vector_search(query_vector, vector_index_data, top_n=top_n * 2) # 最初检索更多结果
    print(f" 向量搜索返回了 {len(vector_results)} 个结果。")

    # 4. 应用RRF合并
    fused_results = apply_rrf([keyword_results, vector_results], k=rrf_k)
    print(f" 合并结果生成了 {len(fused_results)} 个候选。")

    # 5. 返回最终的前N个结果及文本
    final_results = []
    for result in fused_results[:top_n]:
        doc_id = result["id"]
        final_results.append({
            "id": doc_id,
            "text": document_texts.get(doc_id, "N/A"), # 获取文本以提供上下文
            "rrf_score": result["score"]
        })

    return final_results

# 使用示例
final_ranked_list = hybrid_search_pipeline(query_text, top_n=5)

print("\n最终混合排序列表:")
for i, item in enumerate(final_ranked_list):
    print(f"{i+1}. ID: {item['id']}, 分数: {item['rrf_score']:.4f}, 文本: {item['text'][:80]}...")
```

### 考量与后续步骤

- **性能：** 该流程执行两次独立的搜索查询（关键词和向量 (vector)），然后进行合并。总延迟大约是两次搜索的最大延迟加上合并开销（通常可以忽略不计）。对于生产系统，优化各个搜索组件（第二章）并可能并行化查询是很重要的。
- **参数 (parameter) `k` 的调整：** RRF常数 `k` 影响着对较低排名项给予的权重 (weight)。较小的 `k` 会给予靠前排名更高的重要性。其最优值可能取决于特定的数据集和查询模式，并可以通过离线评估（第五章）进行调整。
- **检索量：** 请注意，在合并之前，我们从每个系统检索了 `top_n * 2` 个结果。在合并之前从每个来源检索更多候选（例如，前50或100个）通常会带来更好的最终结果，但代价是合并计算量略有增加。
- **其他合并方法：** 尽管RRF很常见，但可以考虑其他方法，例如加权分数组合（如果分数可以有意义地归一化 (normalization)），或更复杂的基于两种搜索结果特征训练的排序学习模型。
- **评估：** 使用相关性指标（如NDCG、召回率）在真实数据集上对混合系统进行严格评估，并与纯向量和纯关键词搜索进行对比，详细内容在第五章。这有助于量化 (quantization)效益并找出可以改进的地方。

这个动手练习为构建混合搜索系统提供了一个基本蓝图。通过结合词法和语义检索的优势，您通常可以获得比单独使用任一方法显著更好的相关性，尤其是在RAG等LLM应用中常见的复杂信息需求场景下。

## 参考资料

- [The Probabilistic Relevance Framework: BM25 and Beyond](http://dx.doi.org/10.1561/1500000019) — Stephen E. Robertson and Hugo Zaragoza (2009)
  Journal: Foundations and Trends® in Information Retrieval; Publisher: Now Publishers; Volume: 3; Pages: 333-389; DOI: [10.1561/1500000019](https://doi.org/10.1561/1500000019)
  解释了BM25排序函数的理论基础和实践方面，这是关键词搜索的基石。
- [Abstracting over ranks using Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1390334.1390453) — Charlie Clarke, Nick Craswell, and Stephen Soboroff (2008)
  Journal: SIGIR '08: Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval; Publisher: Association for Computing Machinery (ACM); Pages: 747-748; DOI: [10.1145/1390334.1390453](https://doi.org/10.1145/1390334.1390453)
  介绍了互惠排序融合（RRF）作为一种无需分数归一化即可合并多个排序列表的方法。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://doi.org/10.18653/v1/D19-1410) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了Sentence-BERT框架，该框架被广泛用于生成高质量的句子嵌入，是许多LLM应用中向量相似性搜索的基础。
- [An Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本全面的教材，涵盖了核心信息检索主题，包括BM25等关键词搜索算法、向量空间模型和评估指标，这些是理解混合搜索的基础。
