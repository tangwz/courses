# RAG系统架构

来源：[原文](https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag/architecture-of-rag)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检索增强生成 (RAG)系统（RAG）通过结构化的多阶段流程运行，旨在使大型语言模型（LLM）的回复基于外部数据。此过程可分为两个主要阶段：离线索引阶段，在此阶段准备知识库；以及在线检索与生成阶段，实时响应用户查询。掌握这种两部分架构是构建高效RAG应用的基础。

下图展示了完整的流程，说明了数据在索引期间如何处理以及查询如何处理以生成回答。

> RAG工作流程包含一个离线索引流程用于准备数据，以及一个在线查询流程用于生成回答。检索器在向量 (vector)存储中查找相关背景信息，这些信息被整理成供LLM使用的提示。

我们来更详细地审视每个阶段。

## 索引流程

索引流程的目标是将非结构化文档集合转换为结构化、可搜索的知识库。这是一个准备步骤，通常在有新数据时运行一次或定期运行。它包含一系列数据处理操作。

### 1. 加载

此流程始于您的原始数据，这些数据可以有多种格式，例如PDF文件、网页、文本文件或数据库条目。`DocumentLoader` 用于将这些数据摄取为LangChain可处理的标准化格式。每个加载的来源都表示为一个 `Document` 对象，其中包含文本内容和相关的元数据。

### 2. 分块

大型文档通常会超出LLM的上下文 (context)窗口，并且不利于进行精确检索。如果您提供一个100页的文档作为特定问题的背景，模型可能难以找到相关信息。为此，`TextSplitters` 用于将大型 `Document` 对象拆分为更小、更易于处理的块。分块的技巧在于创建足够小以提高处理效率，但又足够大以保持其语义的块。

### 3. 嵌入 (embedding)

文本分块后，需要将其转换为机器可用于比较的格式。这就是嵌入的作用。嵌入模型（一个独立的神经网络 (neural network)）将每个文本块转换为高维数值向量 (vector)。这些向量的设计使得具有相似语义的文本块在向量空间中彼此靠近。例如，“如何制作披萨”的向量会比“股票市场趋势”的向量更接近“披萨面团配方”的向量。

### 4. 存储

索引的最后一步是将文本块及其对应的嵌入 (embedding)向量 (vector)存储在一个名为 `VectorStore` 的专用数据库中。这种类型的数据库经过优化，可对高维向量执行极快的相似性搜索。给定一个输入向量，向量存储可以高效地找到其索引中最相似的向量，通常使用近似最近邻 (ANN) 等算法。常见的向量存储包括FAISS、Chroma和Pinecone。

## 检索与生成流程

此流程在用户提交查询时启动。它使用预先构建的索引来获取相关信息并生成准确的回答。

### 1. 查询嵌入 (embedding)

用户的查询（一段文本字符串）会通过与索引阶段相同的嵌入模型。这将查询转换为向量 (vector)，$v_q$，使其与文档块向量 $v_d$ 处于相同的向量空间中。这一步很要紧，因为它使我们能够直接比较查询的含义与文档块的含义。

### 2. 检索

查询向量 (vector) $v_q$ 被发送到 `VectorStore`。存储执行相似性搜索，计算查询向量与索引中所有文档块向量之间的距离。返回相似性得分最高的“top-k”块（即向量空间中距离最小的块）。此组件由 `Retriever` 管理，`Retriever` 配置用于获取给定查询最相关的文档。

### 3. 增强与生成

检索到的文本块充当“增强”的背景信息。它们被传递给 `PromptTemplate` 组件，该组件将它们与原始用户查询组合成一个格式化的字符串。提示通常看起来像这样：

> "请根据以下背景信息回答问题。
>
> 背景信息：
> [检索到的块1文本]
> [检索到的块2文本]
> ...
>
> 问题：[原始用户查询]"

LLM随后 *仅* 基于提示中提供的信息来生成回答。这使得模型使用文档中真实、最新的数据，而不是仅仅依赖其内部预训练 (pre-training)的知识。结果是基于事实、准确且特定于您数据集的回复。

通过将此流程分解为这些不同阶段，RAG系统提供了一种高效且模块化的方式，为LLM增加外部知识。在接下来的章节中，我们将审视利用LangChain实现此架构中每个组件的实际方法。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://doi.org/10.48550/arXiv.2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）的概念和初始架构。
- [RAG (Retrieval Augmented Generation) Concepts](https://python.langchain.com/docs/concepts/#rag) — LangChain documentation contributors (2024)
  Publisher: LangChain
  在LangChain框架内，提供了RAG系统组件的实用概述。
- [Text Splitters](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG-MRXaWNhSOKsVG2vFJao1O-JcA23ily_vNveEXwuuH8EX-MPvCwlKEqgERid35purnimGykhH1s0MDbbWHuO7uNPTGwUAaegWAMaa3alDUhyh7flCqDQ-P9Yu8GRikAJz3hQnIDiDEgs0h2W--RbBvZdUHQk=) — LangChain (2024)
  Publisher: LangChain
  详细介绍了为RAG将文档分割成更小、语义连贯的块的各种策略和考量。

---

[上一节](../04-%E4%BC%9A%E8%AF%9D%E5%BA%94%E7%94%A8%E4%B8%AD%E7%9A%84%E8%AE%B0%E5%BF%86%E5%8A%9F%E8%83%BD/06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%B8%A6%E8%AE%B0%E5%BF%86%E7%9A%84%E8%81%8A%E5%A4%A9%E6%9C%BA%E5%99%A8%E4%BA%BA.md) · [下一节](02-%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3%E5%8A%A0%E8%BD%BD%E5%99%A8%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE.md)
