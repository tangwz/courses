---
course: "getting-started-rag"
chapter: "data-preparation-for-rag"
lesson: "chunk-metadata"
sourceId: 3649
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-3-data-preparation-for-rag/chunk-metadata"
title: "数据块的元数据关联"
description: "说明为数据块附加元数据（如源文档、页码）对于提供上下文和归属信息的重要性。"
order: 5
plots: []
sourceHash: "3869a94eb2b071fc1d221fd75c0492d08fcda644453ec63f856b38b7d38ea57e"
sourceCorrections: []
---

您已成功载入文档，并使用固定大小或内容感知等方法将它们划分为易于处理的数据块。但是，这些文本数据块脱离了其来源，缺少重要的上下文 (context)信息。想象一下，在一张纸上看到一个单独的句子。如果不知道它出自哪本书或页码，它的用处就会受到限制。这就是为每个数据块关联元数据变得非常必要的原因。

元数据，在RAG数据准备的背景下，是指关于每个数据块的结构化信息，这些信息本身不属于核心文本内容。您可以将其视为给文本片段添加标签，从而提供关于其来源和特征的必要背景信息。

### 元数据为何对RAG数据块很重要？

在数据块划分过程中附加相关元数据，这不仅仅是一种好习惯；它直接促成了有效RAG系统的几项核心功能：

1. **来源归属：** 这可以说是最重要的优点。当RAG系统检索数据块以回答查询时，关联的元数据使系统（最终是用户）能够将信息追溯到其原始源文档。这对于以下方面很重要：

   - **核实：** 用户可以查看原始来源，以确认生成答案的准确性和上下文 (context)。
   - **信任：** 了解信息来源可以增强对系统回应的信心。
   - **引用：** 对于需要参考文献的应用程序，元数据提供了必要的信息（例如文件名、页码），以便引用大型语言模型（LLM）使用的来源。我们将在第4章讨论生成器如何使用此功能。
2. **增强过滤和检索：** 虽然主要的检索机制通常依赖于查询嵌入 (embedding)和数据块嵌入之间的语义相似性，但元数据可以实现有效的预过滤或后过滤步骤。例如，您可以将搜索限定为某个日期之后修改的文档，或优先选择元数据中标记 (token)的特定部分的数据块，这可能会提高相关性和效率。设想一下，过滤搜索结果以只包含过去一年发布的信息；元数据使得这成为可能。
3. **调试和系统分析：** 在评估或排查RAG流程问题时，元数据非常有价值。如果系统提供了不正确或无意义的答案，检查检索到的数据块的元数据有助于找到问题所在。是否检索到了不相关的信息？它是否来自过时的文档？元数据提供了诊断检索组件内部问题所需的线索。
4. **上下文细化：** 尽管大型语言模型主要处理文本内容，但知道一个数据块源自，例如“文档A，第5页，章节：引言”，而不是“文档B，第50页，章节：高级示例”，有时可以提供一些额外的上下文提示，尽管这次于直接的文本含义。

### 常用的关联元数据字段

您选择存储的具体元数据将取决于您的数据源和应用程序要求，但常用且非常有用的字段包括：

- `source` 或 `file_path`：原始文件的名称或路径（例如，`product_manual_v2.pdf`，`api_docs/authentication.html`）。这通常是归属信息中最重要的部分。
- `document_id`：分配给原始文档的唯一标识符，如果文件名不唯一或随时间变化时很有用。
- `page_number`：对于PDF等页码上下文 (context)相关的文档（通常在载入时提取）。
- `chunk_id` 或 `chunk_seq_num`：标识数据块在原始文档中位置的标识符（例如，20个数据块中的第5个），有助于理解上下文的延续。
- `section` 或 `header`：如果您的文档结构允许（例如，在解析时提取章节或段落标题）。这需要更复杂的解析逻辑。
- `creation_date` 或 `last_modified`：与源文档关联的时间戳，有助于按时效性过滤或理解信息的及时性。

您还可以包含与您的知识库相关的特定领域元数据，例如作者、类别或访问级别。

### 将元数据与数据块和嵌入 (embedding)一起存储

这种元数据实际上是如何存储的呢？它通常不会嵌入到文本数据块本身内部，因为那样会干扰嵌入所捕获的语义。相反，当您为数据块的文本内容生成向量 (vector)嵌入并准备将其存储到向量数据库时，您也会包含关联的元数据。

大多数向量数据库正是为此设计的。除了表示文本的高维向量之外，它们还允许您存储与该向量对应的元数据字段的载荷或字典。此载荷与向量一起被索引，但它不直接影响相似性搜索中使用的距离计算（尽管某些数据库提供基于元数据的更高级过滤功能）。

插入向量数据库前的数据结构可能如下所示：

```python
# 插入向量数据库前的数据结构示例
chunk_data = {
  "vector_id": "chunk_abc_doc1_p3_001", # 此向量/数据块的唯一ID
  "embedding": [0.123, -0.045, ..., 0.912], # 实际的文本嵌入向量
  "text_content": "The core component responsible for finding relevant information is the retriever...", # 文本内容本身
  "metadata": {                   # 包含关联元数据的字典
      "source": "internal_docs/rag_architecture_v1.pdf",
      "page_number": 3,
      "document_id": "doc_rag_arch_v1",
      "chunk_seq_num": 1,
      "last_modified": "2023-10-26T10:00:00Z"
      # 在此处添加其他相关字段
  }
}
```

当检索器根据查询嵌入执行相似性搜索时，向量数据库不仅返回最接近查询的向量（及其对应的文本内容，通常通过 `vector_id` 检索），还会返回其关联的元数据字典。这种文本和上下文 (context)的组合随后被传递到RAG流程的下一阶段。

### 实施要点

元数据提取通常发生在文档载入阶段，我们已在本章前面讨论过。用于解析PDF（如`PyMuPDF`）、HTML（`BeautifulSoup`）或其他格式的库可能会自动提供文件名访问权限。提取页码、段落标题或修改日期通常需要这些库提供的特定功能或与文件系统的交互。

以下几点很重要：

1. **尽早获取：** 在载入过程中尽早提取元数据，以免文档结构信息丢失。
2. **传递：** 确保在将文档分割成数据块时，每个数据块都能从其父文档或文档内的特定位置继承或分配到正确的元数据。
3. **标准化：** 目标是不同数据源的元数据字段名称和格式保持一致，以简化后续处理。

数据准备阶段的仔细规划和实施，可确保这些有价值的上下文 (context)信息不会丢失。它将简单的文本片段转化为丰富、可追溯的知识内容，准备好被RAG系统的生成组件有效检索和应用，我们将在第4章详细讨论这个主题。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.55919/neurips.2020.00994](https://doi.org/10.55919/neurips.2020.00994)
  这篇基础性论文介绍了检索增强生成（RAG）范式，强调了从知识库中检索相关信息以改进语言模型响应的重要性。它支持了对准备充分的源数据的需求，其中上下文元数据可以增强检索效果。
- [Nodes, Documents, and Metadata](https://llamaindex.readthedocs.io/en/latest/module_guides/concepts/documents_and_nodes.html) — LlamaIndex Documentation (2024)
  Publisher: LlamaIndex
  这篇来自流行RAG框架的概念指南解释了文本如何被组织成节点（块），以及如何将重要的元数据与这些节点关联起来，以便进行有效的索引和检索，提供了实用的实施见解。
