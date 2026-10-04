---
course: "getting-started-with-llm-toolkit"
chapter: "building-rag-systems"
lesson: "rag-system-anatomy"
sourceId: 7773
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-6-building-rag-systems/rag-system-anatomy"
title: "RAG 系统构成"
description: "回顾检索增强生成系统的完整架构，从数据到响应。"
order: 1
plots: []
sourceHash: "55cce321234ce25696cdbad4d38f8e98557b05eb55e61ae8573c8b627bed9bc1"
sourceCorrections: []
---

检索增强生成 (RAG)系统通过连接外部知识库来扩展标准大型语言模型。模型不再仅依赖其训练数据，而是可以获取相关且最新的信息来回答问题。这个过程并非单一一体；它是一个多阶段的流程，包含两个主要阶段：用于数据准备的离线**索引流程**和用于回答用户查询的在线**检索与生成流程**。

理解这种两部分架构是构建和排查 RAG 系统的核心。索引阶段用于准备您的知识，而检索和生成阶段则使用这些知识来创建有据可循的回答。

> RAG 系统两个主要阶段的概览。索引流程准备知识库，而检索流程则在推理 (inference)时使用它来回答查询。

### 索引流程

在回答任何问题之前，您需要从源文档中构建一个可搜索的知识库。这通常是离线进行的，包含一系列数据处理步骤。

#### 1. 载入文档

第一步是载入您的数据。这可以是文本文件、PDF、Markdown 文档的集合，或是从网站抓取的内容。在之前的章节中，您已经了解了 `document` 模块如何处理各种来源并将其转换为标准化格式。目的是将您的原始文本载入内存以便进行处理。

#### 2. 切分文档

大型文档太大，无法完全放入大型语言模型的上下文 (context)窗口。此外，对整个文档进行搜索效率不高，而且可能会返回嘈杂、不相关的信息。为解决此问题，您需要将文档分割成更小、更易于管理的部分，即“块”。

高效的切分对 RAG 系统的表现有很大作用。块应足够小，以包含具体、集中的信息，但也要足够大，以保留有意义的上下文。您在第四章中了解的递归字符切分或基于句子的分窗等策略，旨在创建语义连贯的块。

#### 3. 生成嵌入 (embedding)

拥有文档块后，您需要一种方法来根据意义而非仅仅关键词来搜索它们。这时嵌入的作用就在于此。使用嵌入模型，每段文本块都会被转换为一个数值向量 (vector)，代表其语义内容。这些向量是语义搜索的根基。`embed_batch` 函数非常适合高效地一次性为所有块生成嵌入。

#### 4. 存储到向量索引

拥有每个块的嵌入后，您需要一个地方来存储它们，以实现快速和可扩展的检索。**向量索引**（或向量数据库）是一种为此目的而设计的专门数据结构。它对高维向量进行索引，以便您可以快速找到与给定查询向量最相似的向量（及其对应的文本块）。虽然本课程为了演示使用简单的内存搜索，但生产系统通常会使用专门的向量数据库，例如 Pinecone、Weaviate 或 Chroma。

### 检索与生成流程

这是在线阶段，每当用户提交查询时就会执行。它使用预先构建的索引来查找相关信息并生成回答。

#### 1. 用户查询和查询嵌入 (embedding)

该过程始于用户的提问，例如，“如何在 Python FastAPI 应用中实现身份验证？”。就像文档块一样，这个查询会使用相同的嵌入模型转换为嵌入向量 (vector)。这确保了查询和文档在相同的向量空间中表示，使它们可以进行比较。

#### 2. 检索：搜索向量索引

然后使用查询向量来搜索向量索引。该索引执行相似性搜索（通常使用余弦相似性）来查找与查询向量最接近的文档块向量。这些块被认为是与用户问题语义上最相关的。

`retrieval` 模块提供了 `semantic_search` 等函数来执行此操作。您会检索到前 k 个最相关的结果，其中 $k$ 是一个可配置的数字（通常介于 3 到 5 之间）。

```python
# 在实际应用中，文档和嵌入会预先计算
# 并存储在向量索引中。
# 为演示目的，我们直接传递它们。

# 嵌入用户查询
query_embedding = embed("How do I implement authentication in Python?")

# 查找前 3 个最相关的文档块
retrieved_results = semantic_search(
    query_embedding=query_embedding,
    documents=all_document_chunks,
    document_embeddings=all_chunk_embeddings,
    top_k=3
)

# 'retrieved_results' 现在包含最相关的文本块
for result in retrieved_results:
    print(f"检索到的文档 (得分: {result.score:.2f}):")
    print(f"{result.document.content[:150]}...")
```

#### 3. 扩充提示词 (prompt)

这是检索增强生成 (RAG)中的“增强”部分。您不是将用户查询直接发送给大型语言模型，而是构建一个新的、更详细的提示词。这个提示词包含检索到的文档块作为上下文 (context)，以及原始查询。

这为大型语言模型提供了生成准确回答所需的具体信息。`results_to_context` 函数有助于将检索到的文档格式化为清晰的字符串，以便插入到提示词模板中。

```python
# 将检索到的搜索结果格式化为单个字符串
context_string = results_to_context(retrieved_results)

# 为大型语言模型创建扩充提示词
augmented_prompt = f"""根据以下上下文回答问题。

上下文：
{context_string}

问题：如何在 Python 中实现身份验证？

回答："""
```

#### 4. 生成回答

最后，扩充后的提示词会发送给一个大型语言模型。模型使用提供的上下文来生成一个基于您的文档信息的回答。因为大型语言模型在其提示词中直接拥有相关事实，所以它“产生幻觉 (hallucination)”或提供不正确信息的可能性大大降低。最终的回答是准确、相关并基于您的特定知识库的。

至此，RAG 系统的构成便讲解完毕。通过将数据准备与查询处理分开，您可以创建一个高效的系统，能够针对大量私有或最新信息回答问题。现在，我们继续构建您的第一个端到端检索流程。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）架构，阐述了其将外部知识与语言模型结合的核心组件。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Pages: 3971-3981; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  介绍了Sentence-BERT，一种生成语义上有意义的句子嵌入的方法，这对于RAG系统中的语义搜索和检索至关重要。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.44525) — Yunfan Gao, Yun Xiong, Xinyue Zhang, Jiaqi Guo, Qinyuan Ye, Xuanjing Huang, Donglin Wang, Jiawei Liu, ZHAO-YAN LI and Haizhou Shi (2023)
  Journal: arXiv preprint; Publisher: arXiv; DOI: [10.48550/arXiv.2312.44525](https://doi.org/10.48550/arXiv.2312.44525)
  对RAG系统进行全面综述，涵盖了其架构、组件、优化策略和当前研究方向。
