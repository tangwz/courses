---
course: "building-llm-apps-with-langchain"
chapter: "data-connection-rag"
lesson: "fetching-data-with-retrievers"
sourceId: 7432
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/fetching-data-with-retrievers"
title: "使用检索器获取数据"
description: "学习如何配置和使用检索器，根据用户查询获取最匹配的文档块。"
order: 5
plots: []
sourceHash: "f24b505716b9083b124dba617c94c03855de66098b5d8cdcfd34c9a90ce63ae8"
sourceCorrections: []
---

查询包含文档嵌入 (embedding)的向量 (vector)存储是寻找与用户问题对应信息的一种基本方式。虽然向量存储提供了直接查询的方式，但 LangChain 为此给出了一个更标准且通用的接口：`检索器`。

检索器是一个对象，它实现了一个通用接口，用于根据查询来获取文档。它的首要作用是接收一个字符串查询，并返回一个 `Document` 对象列表。这种抽象很有意义，因为它将您的应用程序逻辑与特定的数据源分离开。一个检索器可能从向量存储、传统 SQL 数据库，甚至 Web API 中获取数据，但您的应用程序代码与它的交互方式是相同的。

> 检索器在应用程序逻辑和不同底层数据源之间充当通用接口。

### 从向量 (vector)存储到检索器

实例化检索器的最普遍方法是直接从一个已有的向量存储对象进行。这个过程直截了当，通过 `as_retriever()` 方法完成。我们假设您已经从上一节初始化了一个 `vectorstore`。

```python
# 假设 'vectorstore' 是一个已初始化的 Chroma 或 FAISS 向量存储对象
retriever = vectorstore.as_retriever()

# 检索器现在已可使用
query = "What is the architecture of a RAG system?"
relevant_docs = retriever.invoke(query)

# 打印第一个检索到的文档内容
print(relevant_docs[0].page_content)
```

执行 `vectorstore.as_retriever()` 会创建一个 `VectorStoreRetriever` 实例。默认情况下，此检索器被设定为执行相似性搜索，即在嵌入 (embedding)空间中找到与查询向量最接近的文档向量。

### 自定义检索行为

默认的相似性搜索很有用，但在某些情况下，您可能需要对检索过程进行更多控制。`as_retriever()` 方法支持通过其参数 (parameter)进行自定义，特别就是 `search_type` 和 `search_kwargs`。

#### 控制文档数量

一个经常性的调整是更改要检索的文档数量。这由 `search_kwargs` 内的 `k` 参数控制。`k` 的值决定了多少文档作为上下文 (context)传递给 LLM。较小的 `k` 速度更快，使用的 token 更少，但可能会错过核心上下文。较大的 `k` 提供更多上下文，但会增加成本并可能引入噪音。

```python
# 创建一个检索器，它获取最匹配的5个文档
retriever_k5 = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

relevant_docs_k5 = retriever_k5.invoke(query)
print(f"Retrieved {len(relevant_docs_k5)} documents.")
# 预期输出：检索到 5 份文档。
```

#### 通过最大边际匹配度（MMR）提升丰富性

有时，排在前 `k` 位的文档彼此非常相似，给出重复信息。为获得更多不同种类的结果，您可以使用最大边际匹配度（MMR）搜索类型。MMR 的作用是，首先选择与查询最相似的文档，然后迭代选择后续文档，这些文档代表了与查询的相似性和与已选文档的不同性的最佳结合。这有助于对主题提供更宽广的看法。

```python
# 创建一个使用 MMR 选择文档的检索器
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr"
)

relevant_docs_mmr = retriever_mmr.invoke(query)
```

使用 `mmr` 对于复杂查询可能尤其有用，其中不同文档可以覆盖答案的不同方面。

#### 通过相似度分数进行过滤

另一种有助的技术是基于匹配度阈值过滤结果。这确保您只检索满足最低相似度分数的文档，有助于过滤掉不匹配的结果，特别是当用户查询的主题在您的知识库中未被充分涵盖时。

对于支持此功能的向量 (vector)存储（如 Chroma），您可以使用 `similarity_score_threshold` 搜索类型。

```python
# 创建一个检索器，它只返回相似度分数为 0.7 或更高的文档
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)

relevant_docs_threshold = retriever_threshold.invoke(query)
```

这种配置可以防止低质量或不相关的文档传递给 LLM，这可以提高最终答案的准确性。

配置好的功能性检索器，能够为特定查询获取最匹配且内容丰富的情境，至此我们已拥有构建 RAG 系统所需的所有组件。下一节将展示如何将此检索器与 LLM 和提示模板组合起来，构建一个完整的问答链。

## 参考资料

- [LangChain Conceptual Guide: Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/) — LangChain (2024)
  Publisher: LangChain
  提供了在LangChain框架中实现和定制检索器的实用指南。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  本文介绍了检索增强生成（RAG）框架，该框架为结合检索与语言模型提供了架构基础。
- [Using MMR for Diversity and Novelty in Information Retrieval](https://doi.org/10.1145/290941.291025) — Jaime Carbonell, Jade Goldstein (1998)
  Journal: SIGIR '98: Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval; Publisher: ACM Press; Pages: 335-336; DOI: [10.1145/290941.291025](https://doi.org/10.1145/290941.291025)
  介绍了最大边际相关性（MMR），这是一种选择既与查询相关又相互多样化文档的策略，与定制检索器行为相关。
