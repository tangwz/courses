---
course: "prompt-engineering-llm-application-development"
chapter: "integrating-llms-external-data-rag"
lesson: "introduction-vector-stores-rag"
sourceId: 3064
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/introduction-vector-stores-rag"
title: "向量存储简介"
description: "向量数据库概览及其在存储和查询嵌入中的作用。"
order: 5
plots: []
sourceHash: "4a64bc99c8a676ce5b40f934285f556be3031afdc000f7847105883fa04499df"
sourceCorrections: []
---

您已成功将文本块转换为称为嵌入 (embedding)的数值表示。每个向量 (vector)都捕捉其相应文本片段的语义，将相似的构想在多维空间中放置得更近。

但现在您面临一个新的难题：如何高效地在可能数十万、数百万甚至数十亿的向量中进行查找？设想一下，您收到一个用户查询，将其嵌入，然后需要找到具有*最相似*嵌入的文档块。对数据集中每个向量进行暴力比较，会很快变得计算上不切实际。这正是向量存储旨在解决的问题。

### 向量 (vector)存储是什么？

向量存储，常被称为向量数据库，是一种专门的数据库类型，经过优化，用于存储、管理和查询大量高维向量，例如文本嵌入 (embedding)。与擅长结构化数据查找或关键词搜索的传统关系型或文档数据库不同，向量存储是为**相似性搜索**而构建的。它们的主要目的是在数据库中找到与给定查询向量“最近”或最相似的向量，实现语义检索而非仅是词汇匹配。

### 核心功能

向量 (vector)存储在RAG系统中执行多项基本功能：

1. **存储：** 它们提供持久性或内存中的位置来存放嵌入 (embedding)向量。重要的是，它们还允许您将相关的**元数据**与每个向量一同存放。这些元数据可能包含向量所表示的原始文本块、源文档的ID或文件名、页码、URL，或应用程序所需的任何其他上下文 (context)信息。如果没有这个指向原始数据的链接，单独的检索到的向量将不会很有用。
2. **索引（近似最近邻 - ANN）：** 这是实现高效搜索的关键所在。在高维空间 (high-dimensional space)中找到查询向量的*精确*最近邻（精确最近邻搜索）在计算上成本很高，通常与数据集大小呈线性或更差的扩展。向量存储通过使用\*\*近似最近邻（ANN）\*\*算法来解决这个问题。ANN算法构建复杂的索引结构（例如HNSW、IVF、LSH或根据特定向量存储而定的其他结构），从而实现大幅提速的搜索。它们牺牲了少量（通常可忽略不计）的准确性（这意味着它们可能偶尔会错过*绝对*最近的向量，但会找到非常接近的向量），以换取速度上的巨大提升，使得即使在庞大的数据集上也能进行搜索。
3. **查询：** 向量索引完成后，向量存储提供一个接口来执行相似性搜索。通常，您提供查询向量（用户输入的嵌入），并指定要检索的最近邻数量（$k$）。存储使用其ANN索引，根据所选的相似性度量，快速找出数据集中与查询向量最相似的靠前的$k$个向量。

### 相似性度量

向量 (vector)之间的“相似性”或“接近度”是如何测量的？向量存储通常支持多种距离度量：

- **余弦相似度：** 衡量两个向量之间夹角的余弦值。它对方向敏感但对幅度不敏感，这使其在捕捉文本嵌入 (embedding)中的语义相似性方面非常有效。值范围从-1（相反）到1（相同），值越接近1表示相似度越高。
  
  $$
  \text{余弦相似度}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
  $$
  
- **欧几里得距离（L2距离）：** 多维空间中两个向量终点之间的标准直线距离。距离越小表示相似度越高。
  
  $$
  d(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}
  $$
  
- **点积（内积）：** 也可以使用，如果向量经过归一化 (normalization)，有时与余弦相似度相关。

度量的选择通常取决于所用嵌入的特性（例如它们是否经过归一化）和具体任务。余弦相似度通常是基于文本的RAG的默认选项。

### 向量 (vector)存储的示例

向量存储的生态系统发展迅速。以下是一些示例，展示了可用的多样性：

- **库：**
  - **FAISS (Facebook AI Similarity Search)：** Meta AI 开发的高度优化的C++库（带Python绑定）。它提供多种ANN索引算法，常被用作其他系统中的核心引擎。它主要侧重于索引和搜索计算。
- **数据库（开源）：**
  - **Chroma：** 一个开源嵌入 (embedding)数据库，设计简洁易用，常用于开发和较小规模的部署。可内存运行或持久化到磁盘。
  - **Weaviate：** 一个开源的云原生向量数据库，具有GraphQL API和数据对象存储等功能。
  - **Milvus：** 一个高度可扩展的开源向量数据库，为大型生产环境中的性能而构建。
  - **Qdrant：** 一个专注于性能和可靠性的开源向量数据库，提供过滤和负载存储等功能。
- **托管云服务：**
  - **Pinecone：** 一项完全托管的向量数据库服务，提供简单的API用于索引和查询，无需管理基础设施。
  - 许多云提供商（如Google Cloud Vertex AI Matching Engine、Azure Cognitive Search、AWS OpenSearch与k-NN）也提供集成的向量搜索功能。

选择取决于规模、性能要求、易用性、部署模式（自托管 vs. 托管服务）以及应用程序所需的特定功能等因素。

> 此图表说明了流程：源文档被拆分并嵌入，嵌入及其相关元数据存储并索引在向量存储中。在查询时，用户查询被嵌入，向量存储使用其索引执行相似性搜索以找到最相关的向量，最后检索关联的元数据（原始文本块）作为上下文 (context)使用。

### 使用示例

这是一个简化的Python示例，展示了使用Chroma客户端的基本交互模式：

```python
# 使用类似 Chroma 客户端的示例

import chromadb
# 假设嵌入函数可用或嵌入已预先计算
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化客户端（例如，连接到本地 Chroma 实例）
client = chromadb.Client()

# 创建或获取一个集合（作用类似于向量的专用表）
# 如果需要，指定距离函数（默认为 L2）
collection = client.get_or_create_collection(
    name="my_documents_collection",
    metadata={"hnsw:space": "cosine"} # 示例：为 HNSW 索引指定余弦距离
)

# --- 索引阶段 ---
# 假设 'doc_chunks' 是字符串列表（您的文档片段）
# 假设 'doc_embeddings' 是列表/数组的列表（对应的嵌入）
# 假设 'doc_ids' 是每个文本块的唯一字符串列表
# 假设 'doc_metadata' 是字典列表 [{'source': 'doc1.pdf', 'chunk_id': 0}, ...]

# 将向量、元数据以及可能的原始文档添加到集合中
# 注意：嵌入计算可能在此处或之前进行
collection.add(
    # embeddings=doc_embeddings, # 提供预计算的嵌入 或
    documents=doc_chunks,       # 如果已配置，让 Chroma 计算嵌入
    metadatas=doc_metadata,
    ids=doc_ids                 # 需要唯一 ID
)

# --- 查询阶段 ---
query_text = "What is the role of vector stores in RAG?"
# 假设 query_embedding 是使用与 doc_embeddings 相同的模型计算的
# query_embedding = embedding_model.encode([query_text]).tolist()[0]

# 查询集合以找到前 5 个最相似的文档块
results = collection.query(
    query_texts=[query_text], # 提供查询文本 或
    # query_embeddings=[query_embedding], # 提供预计算的查询嵌入
    n_results=5, # 请求前 5 个结果
    # include=['metadatas', 'documents', 'distances'] # 指定要返回的内容
)

# 处理结果
retrieved_docs = results.get('documents', [[]])[0]
retrieved_metadatas = results.get('metadatas', [[]])[0]
retrieved_distances = results.get('distances', [[]])[0]

print("查询:", query_text)
for i, doc in enumerate(retrieved_docs):
    print(f"\n结果 {i+1}: (距离: {retrieved_distances[i]:.4f})")
    print(f" 元数据: {retrieved_metadatas[i]}")
    print(f" 文档块: {doc[:200]}...") # 打印片段

# 'retrieved_docs' 或由 'retrieved_metadatas' 识别的文本块
# 随后作为上下文传递给 LLM。
```

总之，向量 (vector)存储是RAG架构的重要组成部分。它们提供必要的基础设施，以高效地索引和搜索表示为向量的海量语义信息，从而能够获取增强大型语言模型能力所需的相关上下文 (context)。它们连接了您的外部数据与LLM的生成过程。

## 参考资料

- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih (2020)
  Journal: EMNLP 2020; DOI: [10.48550/arXiv.2004.04906](https://doi.org/10.48550/arXiv.2004.04906)
  介绍了密集段落检索（DPR），一种利用密集向量表示（嵌入）高效检索相关段落进行问答的方法，直接应用了语义搜索。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  引入检索增强生成（RAG）的基础论文，解释了如何将信息检索与生成相结合以增强大型语言模型。
