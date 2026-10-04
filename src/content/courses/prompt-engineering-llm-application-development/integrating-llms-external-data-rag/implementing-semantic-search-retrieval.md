---
course: "prompt-engineering-llm-application-development"
chapter: "integrating-llms-external-data-rag"
lesson: "implementing-semantic-search-retrieval"
sourceId: 3066
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/implementing-semantic-search-retrieval"
title: "实施语义搜索/检索"
description: "执行向量相似度搜索以找到相关文档块。"
order: 6
plots: ["plots/3066-0.json"]
sourceHash: "eaf0ff6b1d507ba8e027a2fc4ff2566fbf518c6e62215cd53fef29cace7886a0"
sourceCorrections: []
---

在文档被加载、分割成易于处理的块、嵌入 (embedding)到向量 (vector)中并在向量存储中建立索引之后，下一步是在用户提出查询时检索相关信息。这项活动构成了检索增强生成（RAG）中的“检索”部分，它根本上依赖于语义搜索的原理。

与匹配精确词语的关键词搜索不同，语义搜索寻找在上下文 (context)和语义上意义相似的文本，即使措辞不同。这是可能的，因为我们的文本嵌入模型已经将文档块的含义捕获为高维空间 (high-dimensional space)中的向量。含义相似的块在此向量空间中位置更接近。

### 通过向量 (vector)相似度寻找相关块

当用户提交查询时（例如，“第三季度报告的主要发现是什么？”），检索过程的第一步是使用与嵌入 (embedding)文档块时*相同*的文本嵌入模型，将此查询转换为嵌入向量。这很重要；比较由不同模型生成的向量通常没有意义。

有了查询向量后，我们使用向量存储来查找嵌入空间中与查询向量“最接近”的文档块向量。接近度通常使用相似度指标衡量。

#### 常见相似度指标

有几个指标可以量化 (quantization)两个向量之间的相似度。语义搜索中最常用的是**余弦相似度**。

- **余弦相似度：** 衡量两个向量之间夹角的余弦值。它的范围从 -1（完全相反）到 1（完全相同），其中 0 表示正交（无相似度）。对于通常经过归一化 (normalization)的文本嵌入，余弦相似度有效衡量方向相似度，与向量大小无关。更高的余弦相似度分数表示向量指向更相似的方向，表明语义关联 (semantic relationship)度更高。
  公式为：

  
  $$
  \text{余弦相似度}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
  $$
  

  其中 $\mathbf{A}$ 和 $\mathbf{B}$ 是向量，$\mathbf{A} \cdot \mathbf{B}$ 是点积，$\|\mathbf{A}\|$ 和 $\|\mathbf{B}\|$ 是它们的模。
- **欧几里得距离 (L2 距离)：** 衡量两个向量端点之间的直线距离。距离越小表示相似度越高。

  
  $$
  \text{欧几里得距离}(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
  $$
  
- **点积：** 有时直接使用，特别是在归一化向量中，此时它等同于余弦相似度。

大多数向量存储在语义搜索任务中默认使用余弦相似度，因为它通常在高维文本嵌入中表现良好。

### 执行搜索

向量 (vector)存储提供了执行这些相似度搜索的高效方法，通常被称为 k 最近邻（k-NN）搜索。给定一个查询向量，向量存储会快速识别其索引中与查询向量相似度最高（或距离最小）的 $k$ 个向量。

让我们用一个 Python 示例来说明。假设我们有一个 `vector_store` 对象（代表我们与 FAISS、ChromaDB、Pinecone 等的连接）和一个能够将文本转换为向量的 `embedding_function`：

```python
# 假设 vector_store 和 embedding_function 已初始化
user_query = "上个月更新了哪些安全协议？"

# 1. 嵌入用户查询
query_embedding = embedding_function.embed_query(user_query)

# 2. 在向量存储中执行相似度搜索
# 我们通常要求获取最相似的前 'k' 个文档。这里我们获取 4 个。
k = 4
search_results = vector_store.similarity_search_with_score(
    query=query_embedding,
    k=k
)

# 3. 处理结果
retrieved_docs = []
for doc, score in search_results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:200]}...") # 显示片段
    print("-" * 20)
    retrieved_docs.append(doc) # 'doc' 通常包含文本块和元数据

# 'retrieved_docs' 现在保存着 k 个最相关块的文本
```

在此示例中：

1. 我们嵌入 (embedding) `user_query` 以获得 `query_embedding`。
2. 我们在 `vector_store` 上调用 `similarity_search_with_score` 之类的方法。我们传入 `query_embedding` 并指定 `k=4`，这意味着我们希望获得 4 个最相似的文档块。向量存储库可能具有略微不同的方法名称（`search`、`query`、`get_nearest_neighbors`），但原理是相同的。
3. 结果通常包含原始文档块（或其引用）及其对应的相似度分数。更高的分数（对于余弦相似度）表明与查询的关联度更高。



![二维嵌入空间](plots/3066-0.json)



> 二维空间中的嵌入可视化。搜索旨在查找最接近查询点（红色）的文档点（蓝色）。在此，如果 k=3，则文档 1、文档 2 和文档 4 可能会被检索到。

### 选择文档数量 (k)

参数 (parameter) $k$，即要检索的文档数量，是一个重要的设置。

- 检索的文档过少（$k$ 过小）可能会遗漏回答查询所需的某些相关上下文 (context)。
- 检索的文档过多（$k$ 过大）可能会增加下一步发送给 LLM 的文本量，可能超出上下文窗口限制，增加成本和延迟，并可能引入无关噪声，干扰 LLM。

$k$ 的最佳值通常取决于具体的应用、文档块的长度以及预期查询的性质。这通常需要一些实验。通常从 3 或 5 这样的值开始，然后进行评估和调整。一些系统还实现了相关性分数阈值，只包含高于特定相似度分数的文档，而不是固定的数量 $k$。

在此检索步骤之后，我们有了一组根据语义相似度被认为与用户查询最相关的文档块。下一节将说明如何将此检索到的上下文与原始查询结合起来，形成用于 LLM 的新提示。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇论文介绍了检索增强生成（RAG），解释了如何将信息检索系统与预训练语言模型结合以提高知识密集型任务的性能。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers, Iryna Gurevych (2019)
  Journal: EMNLP 2019; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  本文介绍了Sentence-BERT，它是BERT网络的一个改进版本，生成语义有意义的句子嵌入，适用于语义相似性搜索等任务。
- [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/) — Roie Schwaber-Cohen (2023)
  Publisher: Pinecone
  介绍了向量数据库、其组成部分以及它们如何利用向量嵌入和相似性度量促进语义搜索。
- [Neural Language Models and Word Embeddings](https://web.stanford.edu/~jurafsky/slp3/8.pdf) — Dan Jurafsky and James H. Martin (2023)
  Publisher: Pearson
  自然语言处理教材中的一个章节，描述了词和句子嵌入的应用，包括向量空间模型和相似性度量。
