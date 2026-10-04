---
course: "building-llm-apps-with-langchain"
chapter: "intro-to-langchain-framework"
lesson: "langchain-core-architecture"
sourceId: 7404
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-1-intro-to-langchain-framework/langchain-core-architecture"
title: "LangChain 的核心架构"
description: "LangChain 主要组成部分的概览：模型、提示、链、检索、记忆和代理。"
order: 2
plots: []
sourceHash: "bf41beffae362ac9c53572c939103adbb12ebbf32988c821ad6fbae63a17321f"
sourceCorrections: []
---

LangChain 围绕一组模块化、可组合的组件进行构建。它不是一个单一、固定的系统，而是一系列可以组合起来构建复杂应用的构建单元。这种架构通过为常见任务提供标准接口和实现，简化了开发工作。您将用到的六个主要组件类别是：模型、提示、链、检索、记忆和代理。

> LangChain 各主要组件间的关联。链和代理作为应用逻辑，协调对模型的调用，这些模型由提示引导。记忆提供状态，而检索连接外部数据，工具则为代理提供外部能力。

### 模型

任何 LLM 应用的核心都是模型本身。LangChain 提供了一个标准化接口，用于与来自 OpenAI、Cohere 和 Hugging Face 等提供商的众多不同语言模型进行交互。这种抽象使得在模型之间切换变得简单，只需最少的代码改动。该框架区分两种模型类型：

- **大型语言模型（LLMs）：** 这是对那些接收单个字符串作为输入并返回单个字符串作为输出的模型的封装。它适合简单的文本补全任务。
- **聊天模型：** 这个接口结构更规整，接收聊天消息列表作为输入，并返回由 AI 生成的聊天消息。这种格式是为对话交互而设，通常比更简单的 LLM 接口能处理更多样化的任务。

### 提示

模型需要指令才能生成所需输出。提示是给模型的指令集。虽然您可以硬编码这些指令，但应用通常需要根据用户输入或其他变量动态生成提示。

LangChain 的 `PromptTemplate` 类促成此事。它让您定义一个带有变量占位符的模板字符串。然后，您可以使用输入值格式化该模板，为模型生成一个完整的提示。此组件是创建可复用、灵活和易于维护提示的必要部分。

### 链

链用于将多个组件组合成一个单一、连贯的应用。它们让您可以按序调用模型、工具或数据处理步骤。构建这些序列的标准方式是使用 LangChain 表达式语言 (LCEL)。这种声明式方法让您可以将 `PromptTemplate` 连接到 `Model`，再连接到输出解析器。

对于更复杂的工作流程，链可以根据逻辑将数据路由到不同步骤，或并行运行操作。这取代了传统的基于类的链，代之以更透明、更可定制的管道语法，实现构建可靠的多步骤工作流程。

### 检索

语言模型通常在固定数据集上训练，因此不了解您的私有数据。检索组件构造您的文档，以便模型能够有效处理它们。这是检索增强生成 (RAG) 的根基，应用首先从外部源获取相关数据，然后将其与用户查询一同传递给模型。

检索的工作流程包含四个主要步骤：

1. **文档加载器** 从源头获取数据，例如文本文件、PDF 或网站。
2. **文本分块器** 将大型文档切分为更小、有语义意义的片段。
3. **嵌入 (embedding)模型** 将这些片段转换为数值向量 (vector)。
4. **向量存储** 存储这些向量，并提供高效方法来查找与用户查询相似的向量。

### 记忆

默认情况下，链和代理是无状态的，这意味着它们独立处理每个传入请求。对于许多应用，例如聊天机器人，记住以前的交互不可或缺。记忆是使应用能够在不同调用之间保持状态的想法。

LangChain 提供管理此历史的实用程序。例如，您可以使用可运行的历史类来保持对话的运行日志，确保聊天机器人可以引用交流的早期部分。此上下文 (context)随后被注入到下一次交互的提示中，使模型能回溯以前的交流。

### 代理

链执行预定步骤序列，而代理则使用模型作为推理 (inference)引擎，决定采取哪些行动以及按什么顺序。代理可访问一套工具，并根据用户输入决定使用哪些工具。

代理架构包含几个部分：

- **工具：** 这些是代理可以调用的函数，例如 Google 搜索工具、数据库查询或 Python 解释器。
- **代理：** 这是使用模型决定下一步行动的核心逻辑。它在循环中运行：接收输入，考虑下一步，选择一个工具，并观察工具的输出，重复直到任务完成。
- **代理运行时：** 这是执行代理选定行动并管理循环的环境。虽然 `AgentExecutor` 是最初的运行时，但较新的方法通常使用基于图的执行，以对代理的状态转换有更大的控制权。

这六个组件构成一套全面的工具包，用于构建超越简单文本生成的应用。通过了解它们的作用，您可以组合它们来创建数据感知、有状态和自主的系统。

## 参考资料

- [LangChain Documentation](https://www.langchain.com/docs/) — LangChain Team (2024)
  这是了解 LangChain 模块化架构、模型、提示、链、索引、内存和代理等组件及其实际应用的最官方和最全面的资源。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://papers.neurips.cc/paper/2020/file/6b4932302831206badd6f296d115456e-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Narsimha Chatti, Mike Lewis, Kyunghyun Cho, and Douwe van der Plas (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS; Volume: 33; Pages: 9459-9474; DOI: [10.55919/neurips.2020.01017](https://doi.org/10.55919/neurips.2020.01017)
  介绍了检索增强生成 (RAG) 范式，它是 LangChain “索引”组件的概念基础，使模型能够与外部数据交互。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  描述了 ReAct 框架，该框架为 LLM 代理提供了一种通用的推理、规划和与工具交互的方法，与 LangChain 的“代理”组件直接相关。
