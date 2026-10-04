---
course: "python-llm-workflows"
chapter: "intro-llm-workflows-python"
lesson: "python-llm-ecosystem-overview"
sourceId: 5092
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-1-intro-llm-workflows-python/python-llm-ecosystem-overview"
title: "Python LLM 生态系统概览"
description: "熟悉 Python 中用于 LLM 任务的主要库和工具。"
order: 4
plots: []
sourceHash: "f9af26eccd34256b54da6da581712565b9f5f3b25ed7e7bce37c35e60008343f"
sourceCorrections: []
---

Python 适用于 LLM 开发，这不仅因为它本身的语言特性，也因为它周围构建的各种丰富库和框架。这个生态系统提供专用工具，简化常见工作，让开发者能专注于构建复杂的应用程序，而不是重复造轮子般地实现基本功能。

了解主要参与者及其作用，对顺利进行 LLM 开发很有必要。Python LLM 生态系统大致可以分为以下几个方面：

### 1. LLM 交互库

在最基础的层面，你需要一种与 LLM 提供商通信的方式。虽然你总是可以使用像 `requests` 这样的通用 HTTP 库，但大多数主要 LLM 提供商都提供专用 Python 客户端库。

- **官方 SDK：** 像 `openai`、`anthropic`、`google-generativeai` 和 `cohere` 这样的库，为各自的 API 提供了便捷的 Python 包装器。它们处理认证、请求格式化和响应解析等细节，使交互比原始 HTTP 调用顺畅得多。我们将在第 3 章中介绍如何使用它们。
- **Hugging Face `transformers` 和 `huggingface_hub`：** `transformers` 以在本地加载和运行模型而闻名，它也与 Hugging Face Hub 交互，后者托管着无数模型并提供 API。`huggingface_hub` 方便下载模型并与平台功能进行交互。

### 2. 工作流编排框架

构建复杂的 LLM 应用程序通常需要将对 LLM 的多次调用串联起来、与不同工具（如搜索引擎或数据库）进行交互，以及管理状态。编排框架为这些工作流提供结构。

- **LangChain：** 这可以说是这一领域最受欢迎的框架。LangChain 提供组件，用于管理提示、连接到各种 LLM、解析输出、与外部数据源（向量 (vector)存储、API）进行交互，以及构建序列（Chains）和决策循环（Agents）。本课程的大部分内容将侧重于应用 LangChain（第 4 章和第 5 章）。
- **其他：** 尽管 LangChain 很突出，但 Haystack（通常侧重于 RAG 和搜索）和 Semantic Kernel（微软的产品）等其他框架也存在，提供构建 LLM 应用程序的替代方法。

### 3. 数据处理和索引框架

LLM 通常需要访问并基于不在其训练集中的私有或特定外部数据进行推理 (inference)。这是检索增强生成（RAG）的范围，为此专门设计的框架非常重要。

- **LlamaIndex：** 这个框架擅长将 LLM 连接到外部数据。它为各种格式（PDF、网页、数据库）提供精细的数据加载器，提供高效索引这些数据的方法（通常涉及嵌入 (embedding)和向量 (vector)存储），以及查询引擎，用于检索相关信息以增强 LLM 提示。我们将在第 6 章中详细介绍 LlamaIndex，并在第 7 章中将其整合到 RAG 系统中。

### 4. 向量 (vector)数据库和库

RAG 系统高度依赖语义搜索，这通常由向量嵌入 (embedding)和向量数据库支持。Python 提供与这些交互的接口。

- **向量存储库：** 像 `chromadb`、`faiss-python`（用于 Facebook AI 相似性搜索）以及 Pinecone (`pinecone-client`)、Weaviate (`weaviate-client`) 和 Qdrant (`qdrant-client`) 等托管服务的客户端，允许你高效地存储、管理和查询高维向量嵌入。它们经常与 LlamaIndex 或 LangChain 结合使用（第 7 章）。

### 5. 评估和监控工具

测试和评估 LLM 应用程序因其非确定性特性而带来独特的挑战。正在出现专用工具来提供帮助。

- **评估框架：** 像 LangSmith（来自 LangChain）、TruLens 和提供特定指标（例如 RAGAs）的库，有助于评估 LLM 输出和工作流的质量、正确性和安全性。我们将在第 9 章中涉及评估策略。
- **监控平台：** 通用应用程序性能监控（APM）工具可以使用，但专用 LLM 可观测性平台也正在出现，侧重于跟踪 token 使用、延迟、成本以及随时间可能出现的提示注入或幻觉 (hallucination)漂移等问题（第 10 章）。

### 它们如何协同工作

这些不同类型的库并非孤立运行；它们设计为协同工作。一个典型的 LLM 应用程序可能使用官方客户端库与 LLM API 通信，由 LangChain 编排，后者又通过 LlamaIndex 从 ChromaDB 向量 (vector)存储中获取相关数据，然后生成最终响应。

> 显示 Python LLM 生态系统中不同组件在典型 RAG 应用中的协作关系的图示。

本课程将为你提供实用技能，以便你应用该生态系统中的核心组件，侧重于 LangChain 用于编排和 LlamaIndex 用于数据整合，使你能够构建强大的、具备上下文 (context)感知能力的 LLM 应用程序。我们将在下一章中从设置开发环境开始。

## 参考资料

- [Hugging Face \`transformers\` Documentation](https://huggingface.co/docs/transformers/index) — Hugging Face (2024)
  《transformers》库的官方文档，提供了使用预训练模型、微调以及与Hugging Face Hub交互的详细指南。
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) — LangChain (2024)
  Publisher: LangChain
  LangChain 框架的官方文档，提供了使用模块化组件构建LLM应用的指南和API参考。
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) — LlamaIndex (2024)
  LlamaIndex 的官方文档，涵盖了将LLM与外部数据源连接的数据加载、索引和查询策略。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成 (RAG) 方法，这份论文有助于理解LLM如何结合外部知识。
