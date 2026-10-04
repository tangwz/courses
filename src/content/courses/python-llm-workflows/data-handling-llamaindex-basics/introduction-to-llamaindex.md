---
course: "python-llm-workflows"
chapter: "data-handling-llamaindex-basics"
lesson: "introduction-to-llamaindex"
sourceId: 5171
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics/introduction-to-llamaindex"
title: "LlamaIndex 简介"
description: "LlamaIndex 概述以及它在连接 LLM 与外部数据方面的作用。"
order: 1
plots: []
sourceHash: "af3a5dd3dc265a82355cd959b999af9d5e7e48352cdf7cf957a77f55a1455456"
sourceCorrections: []
---

大型语言模型（LLMs）拥有令人印象深刻的通用知识，但在处理需要访问私人、特定行业或最新信息的任务时，它们通常表现不足。由于上下文 (context)窗口的限制，以及重复查询的低效率，直接将大量文本输入到提示中通常不切实际。这正是 LlamaIndex 旨在解决的问题。

LlamaIndex 是一个数据框架，专门用于摄取、组织和访问私有或外部数据，以供 LLM 应用程序使用。可以将其看作一个专门的工具集，用于管理将上下文提供给您的 LLM 的数据管道。LangChain 等框架擅长协调整体 LLM 工作流程（链、代理、提示），而 LlamaIndex 则着重于数据连接方面，提供精密的工具来处理各种数据源并优化检索。

LlamaIndex 的核心理念围绕着一个简单而有效的模式展开：

1. **加载：** 从各种来源（例如 `.txt`、`.pdf`、`.csv` 文件、数据库、API、网页）摄取数据，将其转换为 LlamaIndex 可识别的格式。
2. **索引：** 将这些已加载数据组织成专用索引。这些索引经过优化，可根据用户查询实现快速、相关的检索。这通常涉及创建向量 (vector)嵌入 (embedding)等技术，以便进行语义搜索。
3. **查询：** 提供接口（查询引擎），允许用户或应用程序查询索引数据。查询引擎从索引中检索最相关的上下文，并通常将其与原始查询结合，以提示 LLM 生成最终的、具备上下文意识的回答。

这种加载-索引-查询流程构成了检索增强生成（RAG）的原理，我们将在后面详细介绍这项技术。LlamaIndex 提供了实现 RAG 系统所需的基本组件。

> LlamaIndex 提供的工作流程概述：数据被加载并组织成索引，然后被查询以检索用于 LLM 的上下文，LLM 再生成响应。

LlamaIndex 使用 Python 编写，使其能自然融入丰富的 Python 数据科学和机器学习 (machine learning)生态系统。其模块化设计允许您轻松替换组件，例如使用不同的 LLM、嵌入模型或向量数据库。

在接下来的章节中，我们将审视 LlamaIndex 中的具体组件和流程，从如何从各种来源加载数据开始，并理解像 `Nodes` 和 `Indexes` 这样的基本数据结构。

## 参考资料

- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) — LlamaIndex Contributors (2024)
  了解 LlamaIndex 组件、用法和示例的权威资源。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 9459-9474; DOI: [10.55917/cb9f407b](https://doi.org/10.55917/cb9f407b)
  介绍检索增强生成 (RAG) 的基础概念，LlamaIndex 旨在实现这一技术。
