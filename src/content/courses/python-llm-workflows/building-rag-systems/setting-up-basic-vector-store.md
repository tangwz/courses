---
course: "python-llm-workflows"
chapter: "building-rag-systems"
lesson: "setting-up-basic-vector-store"
sourceId: 5187
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/setting-up-basic-vector-store"
title: "搭建一个基础向量库"
description: "配置和使用Chroma或FAISS等简易向量库用于RAG。"
order: 4
plots: []
sourceHash: "1301a791e956ed1089520bbbbf5e0071a8e8710b40cbdc77bc7bea1b5e73e9be"
sourceCorrections: []
---

检索增强生成（RAG）通过在生成过程中从外部数据源检索相关信息来增强大型语言模型（LLMs）。实现这一检索的核心组件是**向量 (vector)库**，有时也称为向量数据库。使用Python搭建和操作基础向量库构成了你的RAG系统的一个根基部分。

### 为什么选择向量 (vector)库？

你可能会疑惑，为什么我们不能只使用标准数据库或搜索引擎。传统方法搜索的是精确的关键词匹配，而RAG则需要理解用户查询与数据源中信息之间的*语义含义*或*上下文 (context)相似性*。这就是向量嵌入 (embedding)和向量库的用武之地：

1. **存储嵌入：** 向量库专门用于存储高维数值向量（嵌入），这些向量代表文本段落、图像或其他数据类型的语义含义。
2. **高效相似性搜索：** 它们的主要功能是执行快速高效的相似性搜索。给定一个查询向量（代表用户的问题），向量库能够快速找到其索引中在嵌入空间中“最接近”的向量，通常使用近似最近邻（ANN）算法。这些最接近的向量对应于你原始数据中语义上最相关的信息片段。

可以把它想象成在图书馆整理书籍，不是仅仅通过书名或作者（关键词）来整理，而是通过它们讨论的根本主题和思想（语义含义）来整理。向量库能让你的应用程序快速找到最相关的“思想向量”。

### 选择一个基础向量 (vector)库

对于许多开发任务、原型或处理中等规模数据集的应用程序来说，简单、通常自包含的向量库就足够了，并且易于管理。Python生态系统中两个受欢迎的选择是：

- **ChromaDB：** 一个开源的嵌入 (embedding)数据库，设计宗旨是易于使用。它可以在内存中运行，持久化到磁盘，或以客户端/服务器模式操作。
- **FAISS (Facebook AI Similarity Search)：** 一个高度优化的库，用于稠密向量的高效相似性搜索和聚类。它通常被其他库用作后端引擎，但也可以直接使用。

我们的示例将主要使用ChromaDB，因为它有直接的API和最低的设置要求。

### 搭建和使用ChromaDB

让我们逐步完成安装ChromaDB、添加数据和执行相似性搜索的过程。

#### 1. 安装

首先，使用pip安装所需的库：

```bash
pip install chromadb
```

你可能还需要一个嵌入 (embedding)模型提供者库，这取决于你如何生成嵌入（例如，`sentence-transformers`、`openai`、`cohere`）。我们暂时假定你有一个函数或机制可以将文本转换为向量 (vector)。

#### 2. 初始化

你可以选择在内存中使用ChromaDB（脚本结束时数据会丢失），或者配置它将数据持久化到磁盘。

```python
import chromadb

# 内存客户端（最容易上手）
# client = chromadb.Client()

# 持久化客户端（将数据保存到 'my_chroma_db' 目录）
client = chromadb.PersistentClient(path="./my_chroma_db")

# 创建或获取一个集合（类似于SQL数据库中的表）
# 集合需要一个嵌入函数或显式的嵌入向量。
# 为简单起见，这里我们将使用Chroma默认基于SentenceTransformer的
# 嵌入函数。在实际使用中，你通常会配置一个特定的嵌入函数。
# 注意：首次使用时可能会下载模型。
try:
    collection = client.get_collection(name="my_documents")
    print("集合 'my_documents' 已经存在。")
except Exception:
    print("正在创建集合 'my_documents'...")
    # 如果使用Chroma的默认嵌入函数，请确保已安装sentence-transformers
    # 和torch（`pip install sentence-transformers torch`）
    # 或者提供你自己的嵌入函数: client.create_collection(name="my_documents", embedding_function=my_embedding_function)
    collection = client.create_collection(name="my_documents")
    print("集合已创建。")
```

这里，`PersistentClient`告诉ChromaDB将其数据保存到指定目录。`collection`是你存储向量、原始文本（文档）以及任何相关元数据的地方。Chroma需要知道如何嵌入文本，这可以通过提供一个`embedding_function`或直接添加预先计算好的向量来实现。Chroma的默认设置通常足以满足最初的试验。

#### 3. 添加数据（索引）

现在，我们向集合中添加一些文本文档。ChromaDB将自动使用其配置的嵌入函数（本例中为默认函数）将文本转换为向量，然后进行存储。

```python
# 要添加的数据
documents = [
    "埃菲尔铁塔是法国巴黎战神广场上的一座熟铁格子塔。",
    "中国长城是一系列由石头、砖块、夯土、木材和其他材料构成的防御工事。",
    "光合作用是植物和其他生物用来将光能转化为化学能的过程。",
    "Python编程语言广泛应用于Web开发、数据科学和人工智能方面。"
]
metadatas = [
    {"source": "wiki-地标", "topic": "建筑"},
    {"source": "wiki-地标", "topic": "建筑"},
    {"source": "wiki-生物学", "topic": "科学"},
    {"source": "wiki-编程", "topic": "技术"}
]
ids = ["doc1", "doc2", "doc3", "doc4"] # 每个文档的唯一ID

# 将文档添加到集合
# 如果没有明确提供，ChromaDB会自动处理嵌入生成。
try:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"已向集合添加 {len(ids)} 个文档。")
except ValueError as e:
    print(f"添加文档出错（可能存在重复项？）：{e}")
    # 处理多次运行脚本时ID可能已存在的情况
    # 为简单起见，这里我们只打印错误信息。
```

我们提供了文本内容（`documents`）、可选的`metadatas`（稍后用于过滤）以及每个条目的唯一`ids`。

#### 4. 查询（相似性搜索）

向量库的核心目的就是检索。让我们提出一个问题，并找到最相关的文档。

```python
query_text = "什么是流行的人工智能语言？"
n_results = 2 # 要检索的最相似文档数量

# 执行相似性搜索
results = collection.query(
    query_texts=[query_text],
    n_results=n_results,
    # 可选：include=['documents', 'distances', 'metadatas'] # 指定返回内容
)

print(f"\n查询：{query_text}")
print(f"找到 {len(results.get('ids', [[]])[0])} 个结果：")

# 处理并显示结果
retrieved_ids = results.get('ids', [[]])[0]
retrieved_docs = results.get('documents', [[]])[0]
retrieved_metadatas = results.get('metadatas', [[]])[0]
retrieved_distances = results.get('distances', [[]])[0]

for i in range(len(retrieved_ids)):
    print(f"  ID: {retrieved_ids[i]}")
    print(f"  距离：{retrieved_distances[i]:.4f}") # 距离越小 = 越相似
    print(f"  元数据：{retrieved_metadatas[i]}")
    print(f"  文档：{retrieved_docs[i]}")
    print("-" * 20)
```

ChromaDB接收`query_text`，自动使用与存储文档相同的过程对其进行嵌入，并执行相似性搜索（可能是在嵌入空间中的余弦相似度或欧几里得距离）以找到`n_results`个最接近的向量。结果包含ID、原始文档、元数据和表示相似度的距离分数（通常越低越好）。正如所料，关于AI语言的查询应该检索到关于Python的文档。

> 这是数据在ChromaDB等向量库中进行索引和查询时的流程图。文本被转换为嵌入并存储；查询被嵌入并与存储的向量进行比较，以找到相关信息。

### 与框架集成

LangChain和LlamaIndex等库通常提供对向量 (vector)库的更高层抽象。你通常只需配置一次向量库（如我们上面所做），然后将`collection`对象或基于它构建的专门`Retriever`对象传递给框架。

例如，在LangChain中，你可能会将一个Chroma集合包装到一个`Chroma`向量库对象中，并将其用作RAG链中的检索器：

```python
# 示例代码片段（假定已安装LangChain）
# from langchain_community.vectorstores import Chroma
# from langchain_core.embeddings import Embeddings # 基类

# 假定 'client' 和 'collection_name' 已如前定义
# 并且 'embedding_function' 是一个兼容LangChain的嵌入对象
# vector_store = Chroma(
#     client=client,
#     collection_name="my_documents",
#     embedding_function=embedding_function # 例如，OpenAIEmbeddings()
# )
# retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 'retriever' 现在可以在LangChain RAG链中使用
```

同样地，LlamaIndex使用向量库作为其`VectorStoreIndex`的一部分，以管理节点嵌入 (embedding)的存储和检索。

### 基础向量 (vector)库的考量

虽然它们非常适合上手，但请记住以下几点：

- **可扩展性：** 内存中或简单的基于文件的持久化可能难以应对数十亿向量或非常高的查询负载。
- **管理：** 你需要负责管理底层存储、备份和潜在的扩展需求。
- **功能：** 它们可能缺乏专用、托管向量数据库中的高级功能，例如实时更新、细粒度访问控制或复杂的过滤能力。

对于较小的项目或初期开发，像ChromaDB这样的存储提供了一种很好的平衡，兼顾了简单性和能力。随着你的需求增长，你可以考察更复杂的自托管选项（例如Weaviate、Qdrant）或托管云服务（例如Pinecone、Google Vertex AI Matching Engine、Azure AI Search）。

搭建并填充好基础向量库后，你就拥有了核心检索机制。下一步是将这个检索器集成到完整的RAG流程中，与大型语言模型结合，根据检索到的上下文 (context)生成信息丰富的回复。

## 参考资料

- [ChromaDB Documentation](https://docs.trychroma.com/) — Chroma (2024)
  Publisher: Chroma
  开源嵌入式数据库的官方文档。它提供了安装、配置和实际使用 ChromaDB API 以管理集合并执行相似性搜索的全面指南。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2018)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Volume: 1; Pages: 4171-4186; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  介绍了 BERT 模型，这是预训练上下文文本嵌入的里程碑。这项工作对于理解文本如何转化为捕获语义的高维向量，从而实现向量存储中的语义搜索是基础性的。
- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media
  这本书深入解释了现代 NLP 技术，包括基于 Transformer 的模型如何生成向量嵌入。它为在 RAG 等应用中使用这些嵌入，以及理解向量存储的作用提供了背景。
