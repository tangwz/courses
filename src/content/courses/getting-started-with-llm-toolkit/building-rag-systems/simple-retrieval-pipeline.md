---
course: "getting-started-with-llm-toolkit"
chapter: "building-rag-systems"
lesson: "simple-retrieval-pipeline"
sourceId: 7774
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-6-building-rag-systems/simple-retrieval-pipeline"
title: "构建一个简易的检索流程"
description: "使用前面章节介绍的模块，构建您的第一个端到端RAG流程。"
order: 2
plots: []
sourceHash: "e93b07e14198f9cbaa20af0560d002ee98ac33c98c923f0658a9833ab03cf9da"
sourceCorrections: []
---

构建一个可用的检索流程，涉及组合数据处理和嵌入 (embedding)。这个过程构成了检索增强生成（RAG）的“检索”部分。其目标是接收用户的查询，搜索您准备好的文档，并返回最相关的信息，以此作为大型语言模型的上下文 (context)。

检索流程包含几个按顺序运行的不同阶段：搜索、排序和格式化。对于我们的首次实现，我们将构建一个直接的流程，它执行搜索并将结果为LLM格式化。

> 检索流程接收用户查询，使用混合搜索机制找到相关文档，并将结果格式化成可注入LLM提示词 (prompt)的上下文字符串。

### 组装流程组件

构建检索流程涉及组合用于文档表示、嵌入 (embedding)和搜索的模块。在此示例中，我们将使用内存中的文档集合及其预先计算的嵌入。在生产系统中，这些通常会存储和管理在专门的向量 (vector)数据库中。

首先，我们来定义知识库并为搜索做准备。这包括创建 `Document` 对象并批量生成它们对应的嵌入。

```python
from kerb.retrieval import Document
from kerb.embedding import embed, embed_batch

# 1. 定义知识库
documents = [
    Document(
        id="doc1",
        content="Python is a high-level programming language known for its simplicity and readability."
    ),
    Document(
        id="doc2",
        content="Asynchronous programming in Python allows concurrent execution using async/await syntax."
    ),
    Document(
        id="doc3",
        content="Machine learning models learn patterns from data to make predictions."
    ),
    Document(
        id="doc4",
        content="REST APIs provide a standardized way for applications to communicate over HTTP."
    ),
]

# 2. 为文档生成嵌入
print("正在生成文档嵌入...")
doc_texts = [doc.content for doc in documents]
doc_embeddings = embed_batch(doc_texts)
print(f"已生成 {len(doc_embeddings)} 个嵌入。")
```

文档和嵌入准备就绪后，下一步是执行搜索。虽然您可以使用简单的 `keyword_search` 或 `semantic_search`，但 `hybrid_search` 是一种更有效的方法。此函数结合了关键词匹配（针对特定术语）和语义相似性（针对含义）的优点，为不同类型的查询提供更可靠的结果。

要使用 `hybrid_search`，您需要用户的查询文本、其嵌入、文档列表以及它们对应的嵌入。

```python
from kerb.retrieval import hybrid_search

# 3. 定义用户查询并生成其嵌入
query = "building scalable python applications"
query_embedding = embed(query)

# 4. 执行搜索
search_results = hybrid_search(
    query=query,
    query_embedding=query_embedding,
    documents=documents,
    document_embeddings=doc_embeddings,
    top_k=2,
    keyword_weight=0.4,
    semantic_weight=0.6
)

print(f"\n为查询 '{query}' 找到了 {len(search_results)} 个相关文档")
for result in search_results:
    print(f"  - ID: {result.document.id}, 评分: {result.score:.3f}")
    print(f"    内容: {result.document.content[:80]}...")
```

`hybrid_search` 的输出是一个 `SearchResult` 对象列表，每个对象包含一个 `Document` 和一个相关性评分。虽然这对于检查很有用，但它不是传递给LLM的正确格式。模型期望一个单独的文本字符串作为其上下文 (context)。

`results_to_context` 函数为您处理此转换。它接收 `SearchResult` 对象列表，并将其内容格式化为一个整洁、结构良好的字符串，在每个文档之间添加分隔符。

```python
from kerb.retrieval import results_to_context

# 5. 将搜索结果格式化为上下文字符串
context_string = results_to_context(search_results)

print("\n为LLM格式化的上下文:")
print("---------------------------")
print(context_string)
print("---------------------------")
```

这个格式化的 `context_string` 是我们简易检索流程的最终输出。它包含您的知识库中最相关的信息，准备好与用户查询组合，形成完整的提示词 (prompt)。LLM将使用此上下文生成一个有依据、准确且信息丰富的回答。

### 整合所有内容

让我们将这些步骤整合到一个单一、连贯的示例中，模拟从查询到最终提示词 (prompt)构建的完整RAG流程。这展现了不同模块如何一起工作，以检索和准备生成信息。

```python
from kerb.retrieval import Document, hybrid_search, results_to_context
from kerb.embedding import embed, embed_batch

# --- 1. 设置阶段 ---
# 在实际应用中，此操作只需执行一次，以构建您的索引。
documents = [
    Document(id="py-async", content="Asynchronous programming in Python uses async/await syntax for I/O-bound tasks."),
    Document(id="py-web", content="FastAPI and Flask are popular Python frameworks for building web APIs."),
    Document(id="ml-intro", content="Machine learning enables systems to learn from data without explicit programming."),
]

# 为知识库生成并存储嵌入
doc_embeddings = embed_batch([doc.content for doc in documents])

# --- 2. 检索阶段 ---
# 此操作针对每个用户查询执行。
user_query = "How do I build fast web services in Python?"

# 嵌入用户查询
query_embedding = embed(user_query)

# 执行混合搜索以找到相关文档
retrieved_results = hybrid_search(
    query=user_query,
    query_embedding=query_embedding,
    documents=documents,
    document_embeddings=doc_embeddings,
    top_k=2,
    keyword_weight=0.5,
    semantic_weight=0.5
)

# 将结果格式化为单个上下文字符串
context = results_to_context(retrieved_results)

# --- 3. 生成阶段 ---
# 为LLM构建最终提示词。
final_prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {user_query}

Answer:"""

print("\n--- LLM的最终提示词 ---")
print(final_prompt)
```

这个简易流程是任何RAG系统的基本构成。通过组装这些组件，您已成功构建了一个能将LLM回答基于您自身数据的系统。接下来的章节将介绍如何通过更先进的技术来改进此流程，例如重新排序以提高相关性，以及上下文 (context)管理以优化令牌限制。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  一篇开创性论文，介绍了检索增强生成（RAG）范式，将检索集成到生成过程中，以外部知识为大型语言模型提供基础。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本信息检索原理的奠基性教科书，涵盖了基于关键词的搜索和向量空间模型，这些对于构建混合检索系统中传统和语义搜索组件很重要。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Pages: 3972-3982; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  介绍了一种用于生成具有语义意义的句子嵌入的方法，该方法对于语义搜索和信息检索等任务非常有效，是RAG管道的核心组件。
