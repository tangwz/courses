# 什么是LLM工作流？

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-1-intro-llm-workflows-python/what-are-llm-workflows)

[返回章节目录](README.md) · [返回课程目录](../README.md)

“虽然与大型语言模型（LLM）的交互通常始于简单的提示与响应对话，但构建有用的应用需要更多结构。不要只考虑一次查询。任务通常涉及多个步骤、数据转换，并可能与LLM或其他服务进行多次互动。”

**LLM工作流**指的是这种结构化的操作序列，旨在以一个或多个LLM为组成部分，达成特定目标。它是有序安排输入、LLM调用、数据处理以及可能的外部工具或数据源，以产生期望结果的过程。

请考虑以下几点：

- \*\*不止是单次调用：\*\*虽然基本交互涉及发送提示和接收响应，但一个工作流可能涉及预处理用户输入、从数据库获取相关信息、使用这些信息构建详细提示、调用LLM、解析结构化输出，然后根据该输出采取进一步行动。
- \*\*编排：\*\*工作流管理不同组件之间的数据和控制流动。这可以像将一步的输出作为下一步的输入一样简单，也可能涉及复杂的条件逻辑和决策，有时甚至直接使用LLM来决定下一步。
- \*\*应用的构成单元：\*\*工作流是将LLM能力整合到功能性软件中的基本结构。无论您是构建聊天机器人、文档摘要工具、代码生成器还是数据分析工具，您实际上都在设计和实现一个LLM工作流。

### 简单工作流与复杂工作流

工作流的复杂程度差异很大：

- \*\*简单工作流：\*\*一个直接的例子可能包括获取用户文本、使用模板将其格式化为特定提示（例如，“Summarize the following text: {user\_text}”）、将其发送给LLM，并显示原始摘要。
- \*\*复杂工作流：\*\*一个更复杂的工作流，例如用于检索增强生成（RAG）的工作流，可能如下所示：
  1. 接收用户问题。
  2. 使用问题在向量 (vector)数据库中查找相关文档片段。
  3. 将问题和检索到的上下文 (context)组合成一个提示。
  4. 将组合后的提示发送给LLM。
  5. 接收LLM的响应，该响应以所提供的上下文为依据。
  6. 为用户格式化响应。

此图展示了一个通用结构：

> 一个典型流程涉及处理输入、准备提示、与LLM互动以及处理输出。一些工作流可能涉及反馈循环或多次LLM调用。

理解工作流很重要，因为它将视角从把LLM视为一个独立的魔法盒子转变为将其看作一个更大、经过设计的系统中的一个强大组件。本课程侧重于使用Python以及LangChain和LlamaIndex等特定库来有效设计、实现、测试和部署这些工作流。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  描述了基础的检索增强生成（RAG）架构，这是一个结合信息检索和语言模型的关键复杂LLM工作流示例。
- [LangChain Expression Language (LCEL)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlCgdSJmPyVAlJE70zEyy5KZxng4D_w_egjm6Z4We0riCtV8qn1b4IFtvXK596qYeRT9Vv5FR41OheXuFKEBL6OiCeKfx5lBovWJ71zKrI6UhMmSWUpgtEncBKBvBpnxHzYVg6VW21) — LangChain (2024)
  提供了一种编程方式来组合复杂的LLM工作流、链和代理，展示了如何使用流行框架编排多个步骤和LLM调用。
- [Core Modules: Data Connectors, Indexes, Query Engines, and Agents](https://docs.llamaindex.ai/en/stable/module_guides/) — LlamaIndex (2024)
  详细介绍了将外部数据源与LLM集成的构建模块，这对于开发数据感知的LLM应用和RAG等复杂工作流至关重要。
- [Prompt engineering](https://platform.openai.com/docs/guides/llm-engineering/prompt-engineering/strategies) — OpenAI (2024)
  Publisher: OpenAI
  提供了关于构建提示、管理对话以及与LLM有效交互的通用指南，这些都是设计稳健工作流的基础。

---

[下一节](02-LLM%20%E5%BA%94%E7%94%A8%E7%9A%84%E5%B8%B8%E8%A7%81%E7%BB%84%E6%88%90%E9%83%A8%E5%88%86.md)
