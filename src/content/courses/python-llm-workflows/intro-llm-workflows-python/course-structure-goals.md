---
course: "python-llm-workflows"
chapter: "intro-llm-workflows-python"
lesson: "course-structure-goals"
sourceId: 5095
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-1-intro-llm-workflows-python/course-structure-goals"
title: "课程结构与目标"
description: "概述本课程涵盖的主题和学习目标。"
order: 5
plots: []
sourceHash: "8b278418a090a9067ac58163f2063561cdf816a7989d8dc8d91d9169637f59ef"
sourceCorrections: []
---

这里将阐述LLM工作流中Python的应用结构以及你将掌握的具体技能。主要目的是使你具备使用大型语言模型构建、测试和部署功能应用所需的实用知识和工具熟练度。

本课程从初始设置到应用部署，循序渐进，如下描绘：

> 此图描绘了学习路径，从要点开始，逐步介绍核心工具和技术，构建如RAG等特定应用，最后涵盖测试和部署方法。

### 课程进展

1. **入门 (第1-3章)：** 我们首先确保你的开发环境已正确配置以执行LLM任务。你将学习使用Python直接与LLM API交互的要点，包括请求、响应、身份验证和错误处理。
2. **核心技能与框架 (第4-6、8章)：** 本部分介绍简化LLM应用开发的核心库。
   - **LangChain：** 你将学习如何使用`LangChain`来协调复杂的工作流、管理提示词 (prompt)、解析输出，并构建链和智能体。我们将涵盖其基本功能和更高级的功能。
   - **LlamaIndex：** 接着，我们将侧重于`LlamaIndex`，用于将LLM连接到你的私有数据源。你将学习数据加载、索引策略和查询机制。
   - **Prompt Engineering：** 本节还将专门讲解直接在Python中实现的提示工程 (prompt engineering)技术，并过渡到结构化和动态提示词的生成。
3. **应用构建 (第7章)：** 在掌握了核心工具后，我们将重点构建检索增强生成（RAG）系统。这包括将数据索引（LlamaIndex）和工作流管理（LangChain）与向量 (vector)数据库结合，使LLM的响应基于特定信息。
4. **投入生产 (第9-10章)：** 最后几章讨论将LLM应用实际投入使用的细节。
   - **测试与评估：** 你将学习专门应对LLM输出测试挑战的方法，包括单元/集成测试、评估指标和监控。
   - **部署：** 我们将介绍应用打包（例如，使用Docker）、创建API端点（例如，使用FastAPI）、选择部署策略，以及采纳如CI/CD等操作优良做法。

### 学习目标

完成本课程后，你将能够：

- 设置一个专为LLM开发定制的Python环境，包括安全的API密钥管理。
- 以编程方式与各种LLM提供商的API进行交互。
- 使用`LangChain`来设计、构建和管理LLM工作流，包括链和智能体。
- 运用`LlamaIndex`为LLM应用加载、索引和查询外部数据源。
- 构建并评估检索增强生成（RAG）系统。
- 在Python代码中应用有效的提示工程 (prompt engineering)技术。
- 为基于LLM的系统实施合适的测试和评估方法。
- 理解并运用Python LLM应用的打包、部署和监控优良做法。

在整个课程中，实践环节将巩固所学内容，让你能够立即运用所学。本课程专为已具备Python编程知识并希望专攻大型语言模型应用构建的学习者设计。我们假设你熟悉Python语法、数据结构和标准库的使用，但不要求具备LLM或`LangChain`或`LlamaIndex`等特定框架的先验经验。

## 参考资料

- [OpenAI Platform Documentation](https://platform.openai.com/docs/) — OpenAI (2024)
  提供与OpenAI大型语言模型交互的全面信息，包括API使用、提示工程指南和应用开发考量。
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) — LangChain (2024)
  学习和使用LangChain的官方资源，LangChain是一个用于开发由语言模型驱动的应用程序的框架。它详细介绍了工作流编排和智能体构建。
- [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/) — LlamaIndex (2024)
  LlamaIndex的官方文档，该库将大型语言模型与外部数据连接起来。它涵盖了数据摄取、索引和查询策略。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  一篇介绍检索增强生成（RAG）的基础研究论文，RAG是一种将信息检索与语言生成相结合以完成知识密集型任务的方法。
