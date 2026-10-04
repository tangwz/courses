---
course: "python-llm-workflows"
chapter: "data-handling-llamaindex-basics"
lesson: "llamaindex-indexing-data"
sourceId: 5174
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics/llamaindex-indexing-data"
title: "高效检索的数据索引"
description: "了解 LlamaIndex 如何构造数据以实现快速和相关检索。"
order: 3
plots: []
sourceHash: "e3b541a6d5f06c77dd5484c4fc2c9a981d61ce6319cfa29ac4e964399567def0"
sourceCorrections: []
---

数据一旦转换为 LlamaIndex 的 `Document` 对象，还需要进一步组织才能供 LLM 高效使用。仅仅拥有这些原始数据是不足够的。想象一下，在海量、杂乱无章的书中寻找特定句子，与在整理完善的图书馆中查找它进行对比。索引就是为您的数据构建这种图书馆目录的过程。

LlamaIndex 中索引的目的是以一种方式组织您已加载的数据，使您之后提出查询时能够快速轻松地找到最相关的信息片段。这些相关信息最终将作为上下文 (context)提供给 LLM，从而实现我们稍后将构建的检索增强生成（RAG）过程。

### 从文档到节点

在检索阶段，LlamaIndex 通常不直接使用整个文档。相反，它将 `Document` 对象分解成更小、更易于管理的块，称为 `Nodes`（节点）。每个 `Node` 通常代表从原始 `Document` 中派生出的一段文本（例如一个段落或几句话），以及将其链接回其来源的元数据。

为什么要将文档分成小块？

1. **粒度:** 较小的块允许更精确的匹配。当用户提出问题时，您通常只需要一个特定的段落或句子作为上下文 (context)，而不是整个文档。
2. **效率:** 处理和嵌入 (embedding)较小的文本块在计算上所需的资源比处理大型文档要少。
3. **上下文窗口限制:** LLM 对一次能处理的文本量（即上下文窗口）有限制。提供较小、高度相关的块比用冗长、不那么相关的文本使模型过载更有效。

LlamaIndex 在索引过程中会自动处理这种分块，但如果需要，您可以自定义分块策略。

### 嵌入 (embedding)的作用

数据分解成 `Nodes` 后，LlamaIndex 通常会创建这些节点的数值表示，称为*嵌入*。嵌入是由机器学习 (machine learning)模型（嵌入模型）生成的向量 (vector)（数字列表），它们捕获文本的语义含义。含义相似的节点在向量空间中将拥有数学上彼此接近的嵌入。

可以把它想象成在地图上为每个 `Node` 分配坐标，其中讨论相似主题的 `Nodes` 位于彼此附近。当您之后查询索引时，LlamaIndex 会将您的查询嵌入到相同的向量空间中，并查找其嵌入与查询嵌入最接近的 `Nodes`。这个过程被称为*相似性搜索*。

常见的嵌入模型包括 OpenAI、Cohere 的模型，或者通过 Hugging Face 的 `transformers` 或 Sentence Transformers 等库获得的开源模型。LlamaIndex 与其中许多模型集成，如果您不指定，通常会使用默认模型。

### 创建索引

有了 `Nodes` 及其对应的嵌入 (embedding)，LlamaIndex 会构建一个 `Index`（索引）。索引是一种数据结构，它组织 `Nodes` 及其嵌入，以实现高效的查询。

LlamaIndex 中存在多种类型的索引，但用于相似性搜索的最常见和多功能的是 `VectorStoreIndex`。该索引存储 `Node` 嵌入（通常在称为向量 (vector)存储的专用数据库中），并允许快速搜索以找到与给定查询语义最相似的 `Nodes`。

让我们看看如何构建一个基本的 `VectorStoreIndex`。假设您已加载了一系列 `Document` 对象（如前一节所述，我们称之为 `documents`）：

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding # 或其他嵌入模型
# 使用 OpenAI 嵌入的示例（需要设置 OPENAI_API_KEY）
# 您可以用 LlamaIndex 支持的其他嵌入模型替换此项。
Settings.embed_model = OpenAIEmbedding()

# 从加载的文档创建索引
index = VectorStoreIndex.from_documents(documents)

print("索引创建成功！")
```

这个简单的命令在后台执行了几个步骤：

1. 它接收输入的 `documents`。
2. 它根据默认设置解析这些文档并将其分割成 `Nodes`。
3. 它使用配置的嵌入模型（此处通过 `Settings` 使用 `OpenAIEmbedding`）为每个 `Node` 生成嵌入。
4. 它将这些 `Nodes` 及其嵌入存储在内存中的向量存储结构中，从而创建 `VectorStoreIndex`。

下图展示了这一流程：

> 数据从原始文档到可查询的 LlamaIndex 索引的流程，包括加载、分割成节点、嵌入和存储在索引结构中。

数据索引完成后，它已为高效检索做好了结构化准备。下一步是学习如何对这个索引提出问题（查询），以找到 LLM 所需的相关信息。

### 持久化索引

构建索引，特别是生成嵌入 (embedding)，可能需要时间和计算资源，尤其是对于大型数据集。您通常不希望每次应用程序启动时都重新构建索引。LlamaIndex 允许您将索引持久化（保存）到磁盘，并在以后重新加载它。

```python
from llama_index.core import StorageContext, load_index_from_storage

# 定义索引将存储的路径
PERSIST_DIR = "./storage"

# 将索引保存到磁盘
index.storage_context.persist(persist_dir=PERSIST_DIR)
print(f"索引已保存到 {PERSIST_DIR}")

# 稍后从磁盘加载索引
# 创建一个指向持久化目录的默认存储上下文
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

# 加载索引（需要相同的嵌入模型配置）
loaded_index = load_index_from_storage(storage_context)
print("索引加载成功！")
```

通过持久化索引，您可以将可能耗时的索引过程与应用程序的查询阶段分离，从而大大加快启动速度。我们将在 RAG 章中介绍更高级的存储选项和向量 (vector)存储。

## 参考资料

- [LlamaIndex Documentation: Core Concepts - Data Indexing](https://docs.llamaindex.ai/en/stable/indexing/) — LlamaIndex team (2024)
  Publisher: LlamaIndex
  LlamaIndex 中数据索引的官方指南，详细介绍了文档如何处理成节点和索引以供检索。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://neurips.cc/paper_files/paper/2020/file/6b4932302373072417b658fd411710b7-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS; Volume: 33; Pages: 9459-9474; DOI: [10.5591/978-1-950305-24-1_223](https://doi.org/10.5591/978-1-950305-24-1_223)
  介绍了检索增强生成 (RAG)，为高效数据检索以增强大型语言模型 (LLM) 响应提供了基础背景。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410.pdf) — Nils Reimers, Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 3988-3996; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  描述了 Sentence-BERT，一个生成语义丰富句子嵌入的模型，与本节中“嵌入的作用”部分直接相关。
- [Approximate Nearest Neighbor Search in High Dimensions: A Survey](https://arxiv.org/pdf/1905.10519) — Xin Dong, Ben Moses, Yifan Li (2019)
  Journal: arXiv preprint arXiv:1905.10519
  一篇概述高效近似最近邻搜索技术和算法的综述，是向量存储和相似性搜索的基础。
