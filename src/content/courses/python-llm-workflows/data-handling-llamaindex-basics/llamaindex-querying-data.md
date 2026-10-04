---
course: "python-llm-workflows"
chapter: "data-handling-llamaindex-basics"
lesson: "llamaindex-querying-data"
sourceId: 5178
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics/llamaindex-querying-data"
title: "查询您的已索引数据"
description: "从您的LlamaIndex索引中获取相关信息，用于大型语言模型（LLM）。"
order: 5
plots: []
sourceHash: "e9edb573b7ed524742711005a292ea9f802e5dd3f0e4c3c4793e43ac66028ec6"
sourceCorrections: []
---

要有效利用组织成LlamaIndex索引的外部数据，主要目标是从中检索信息。仅仅对数据进行索引本身并没有多大用处；目的是提出问题并获得基于特定数据的相关回答。这就是LlamaIndex查询功能变得不可或缺的原因。

### 查询引擎接口

在LlamaIndex中，与您的已索引数据交互的主要方式是通过`QueryEngine`。可以将查询引擎看作是一个组件，它负责接收您的自然语言问题，在索引中查找最相关的信息片段（节点），然后（通常使用大型语言模型LLM）综合生成一个连贯的回答。

从现有索引创建基本查询引擎很简单。如果您有一个`index`对象（如前述索引章节所示创建），您可以像这样实例化一个查询引擎：

```python
# 假设 'index' 是您之前创建的 LlamaIndex 索引对象
query_engine = index.as_query_engine()
```

这个简单的调用会设置一个默认查询引擎，其配置合理，适用于许多常见情况。

### 执行查询

一旦您有了`query_engine`对象，提问就像调用其`query`方法一样简单：

```python
# 针对已索引数据提问
response = query_engine.query("What were the main findings of the research paper?")

# 打印由LLM综合生成的文本回答
print(response.response)
```

`query`方法接收您的提问作为字符串参数 (parameter)。在内部，LlamaIndex执行以下几个步骤：

1. **检索：** 它搜索索引（通常使用向量 (vector)嵌入 (embedding)和相似度搜索）以找到包含与您查询最相关信息的节点。
2. **综合生成：** 它获取检索到的节点（即上下文 (context)）和您的原始查询，将它们输入给大型语言模型（LLM），以生成自然语言回答。

这种先检索后综合的模式是检索增强生成（RAG）的根本，我们将在下一章更详细地介绍这种技术。

### 理解回答

`query`方法返回的对象不仅仅包含最终的文本回答。它通常会提供关于查询过程的重要元数据。

```python
# 获取回答文本
print(f"回答文本:\n{response.response}\n")

# 获取用于生成回答的源节点
print("源节点:")
for node in response.source_nodes:
    print(f"  节点ID: {node.node_id}")
    print(f"  相似度得分: {node.score:.4f}")
    # 显示源文本片段
    print(f"  文本片段: {node.text[:150]}...")
    print("-" * 20)
```

两个最重要的属性通常是：

- `response.response` (或某些版本中的`response.response_txt`)：此属性包含由大型语言模型（LLM）基于检索到的上下文 (context)生成的最终综合回答字符串。
- `response.source_nodes`：这是一个`NodeWithScore`对象列表。每个对象代表从您的索引中检索到的一个数据块，该数据块被用作生成回答的上下文。检查这些节点对于以下方面非常有用：
  - **透明度：** 了解LLM给出特定回答的*原因*。
  - **调试：** 辨别检索步骤是否获取了不相关的信息或遗漏了重要上下文。
  - **验证：** 让用户或开发人员能够将回答追溯到原始源文档。

`source_nodes`中的每个`NodeWithScore`对象通常包含：

- `node`：实际的`TextNode`（或其他节点类型）对象，包括其文本内容（`node.text`）和元数据。
- `score`：一个数值分数（通常是来自向量 (vector)搜索的相似度分数），表示在检索阶段该节点被认为与查询的相关程度。分数越高通常表示相关性越大。

以下是一个图表，它展示了基本的查询流程：

> 查询过程涉及查询引擎搜索索引、检索相关节点，并使用LLM根据查询和检索到的上下文来综合生成回答。

### 配置与定制

尽管`index.as_query_engine()`提供了一个方便的起点，LlamaIndex为查询引擎提供了丰富的定制选项。您可以配置以下方面：

- 要检索的节点数量（`similarity_top_k`）。
- 用于综合生成的具体大型语言模型（LLM）。
- 回答模式（大型语言模型LLM如何使用检索到的上下文 (context)）。
- 基于节点元数据的筛选。

这些高级配置使您能够微调 (fine-tuning)检索和综合生成过程，以在特定任务上获得更好的性能和相关性，我们将在讨论RAG系统时提及这些。

目前，创建默认查询引擎并检查综合回答及源节点的能力，提供了一种在大型语言模型（LLM）应用中使用您的已索引外部数据的有效机制。下一步是将此功能集成到更复杂的工作流程中，并构建完善的RAG管道。

## 参考资料

- [LlamaIndex Documentation: Querying](https://docs.llamaindex.ai/en/stable/module_guides/querying/index.html) — LlamaIndex Developers (2024)
  提供 LlamaIndex 查询引擎的官方信息和使用示例，包括自定义配置。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 9459-9471; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍检索增强生成 (RAG) 的核心概念，该概念是查询机制的基础。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  提供 RAG 系统的全面综述，讨论进展、组成部分和评估方法。
