---
course: "building-llm-apps-with-langchain"
chapter: "data-connection-rag"
lesson: "vector-stores-embeddings"
sourceId: 7431
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/vector-stores-embeddings"
title: "向量存储和嵌入"
description: "了解文本嵌入的作用以及如何使用Chroma和FAISS等向量存储进行高效相似度搜索。"
order: 4
plots: ["plots/7431-0.json"]
sourceHash: "4d204c88bf61404fad135c04bcbc7a07094ae17d546cfb8760342ba2d0bfd98d"
sourceCorrections: []
---

在处理被拆分成小块的文档时，一个主要难点是高效找出与用户问题最相关的文本块。简单的关键词搜索往往不够用，因为它常无法识别同义词、语境和查询的真实含义。这需要一种*语义搜索*方式，这种方式能理解词语背后的意图。语义搜索通过将文本块转换成称为嵌入 (embedding)的数字形式来实现，从而可以对它们的含义进行数学比较。

### 文本嵌入 (embedding)的用途

嵌入是一个密集向量 (vector) (dense vector)，它本质上是一串数字，用来在多维空间中表示一段文本。这些向量由经过专门训练的模型生成，这些模型旨在捕捉语言的语义属性。一个良好构建的嵌入空间的显著特点是，含义相近的文本会有彼此靠近的向量。

比如，"猫"和"小猫"的向量会比"猫"和"飞机"的向量彼此更近。这一特性使我们能够通过在此空间中搜索与我们查询向量最接近的向量来查找相关文档。



![嵌入空间的简化2D视图](plots/7431-0.json)



> 语义相关的项在嵌入空间中聚集，而无关的项则距离较远。

LangChain为许多嵌入模型供应商提供了统一的接口。要生成嵌入，您需要为您选定的供应商实例化一个类，例如`OpenAIEmbeddings`或`HuggingFaceEmbeddings`，然后使用其`embed_query`方法。

```python
from langchain_openai import OpenAIEmbeddings

# 注意：需要设置 OPENAI_API_KEY 环境变量
embeddings_model = OpenAIEmbeddings()

query_embedding = embeddings_model.embed_query(
    "向量存储在RAG系统中如何使用？"
)

# 结果是一个浮点数列表
print(f"Vector dimension: {len(query_embedding)}")
print(f"First 5 elements: {query_embedding[:5]}")
```

```
向量维度: 1536
前5个元素: [-0.0123..., 0.0045..., -0.0218..., -0.0076..., 0.0091...]
```

### 从相似性到向量 (vector)存储搜索

一旦我们可以将文本转换为向量，就需要一种方法来衡量它们之间的“接近程度”。最常用的度量标准是**余弦相似度**，它衡量两个向量之间夹角的余弦值。余弦相似度为1表示向量指向完全相同的方向（高度相似），0表示它们正交（无关），而-1表示它们方向相反（不相似）。公式如下：


$$
\text{相似度} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
$$


虽然我们可以手动计算查询向量与每个文档块向量之间的相似度，但随着文档数量的增加，这种方法会变得计算成本高昂且缓慢。这就是**向量存储**发挥作用的地方。

向量存储是一种专门的数据库，其主要任务是：对大量向量集合执行极速的相似度搜索。它通过使用近似最近邻 (ANN) 搜索算法来实现此目的，这些算法无需将查询与每个条目进行比较，即可找到“最接近”的向量。

向量存储在RAG管道中承担两项主要工作：

1. **存储和索引：** 它存储您的文档块及其对应的嵌入 (embedding)向量。在此过程中，它会构建一个高效的索引以实现快速搜索。
2. **搜索和召回：** 当给定一个查询向量时，它会使用其索引快速召回 $k$ 个最相似的向量，并返回与它们关联的文档块。

### 在LangChain中实现向量 (vector)存储

LangChain集成了数十种向量存储，从Chroma和FAISS等轻量级内存库，到Pinecone和Weaviate等生产级独立数据库。对于开发而言，Chroma这样的内存选项是一个很好的起始点。

以下示例演示了完整的工作流程：获取文档块，将其嵌入 (embedding)，并将其索引到Chroma向量存储中。LangChain的向量存储集成通常提供一个便捷的`from_documents`类方法，可以为您处理这些步骤。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 上一步的示例文档块
documents = [
    Document(page_content="向量存储是一种数据库，旨在高效存储和检索向量嵌入。"),
    Document(page_content="LangChain与许多向量存储集成，包括Chroma、FAISS和Pinecone。"),
    Document(page_content="嵌入是文本的数字表示，用于捕捉语义。"),
    Document(page_content="向量搜索中最常用的相似度度量是余弦相似度。")
]

# 初始化嵌入模型
embeddings = OpenAIEmbeddings()

# 从文档创建Chroma向量存储
# 这将自动嵌入并索引文档
vectorstore = Chroma.from_documents(documents, embeddings)

# 现在，我们可以执行相似度搜索
query = "什么是嵌入？"
retrieved_docs = vectorstore.similarity_search(query)

# 结果是按相关性排序的Document对象列表
print(retrieved_docs[0].page_content)
```

```
嵌入是文本的数字表示，用于捕捉语义。
```

下图显示了将文档导入向量存储以及根据查询检索文档的完整数据流。

> 摄入管道处理并存储文档，而检索管道则使用存储的索引来为用户的查询查找相关信息。

通过创建和填充向量存储，我们已将静态文档集合转变为一个动态的、可搜索的知识库。下一节将显示如何将此搜索功能包装到一个`Retriever`中，`Retriever`是LangChain的一个标准组件，用于在更大的应用链中获取数据。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Volume: Volume 1 (Long and Short Papers); Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  介绍了BERT，一种生成上下文感知文本嵌入的突出模型，这在LLM应用程序的语义搜索中至关重要。
- [Vector stores](https://python.langchain.com/docs/concepts/#vector-stores) — LangChain Team (2024)
  Publisher: LangChain
  LangChain官方文档，关于如何集成各种向量数据库，涵盖存储和检索嵌入的实际实现。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）框架，展示了如何通过知识库（通过向量搜索）检索信息来提高LLM性能。
