---
course: "getting-started-rag"
chapter: "building-basic-rag-pipeline"
lesson: "overview-rag-frameworks"
sourceId: 3658
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/overview-rag-frameworks"
title: "RAG 框架概述 (例如 LangChain、LlamaIndex)"
description: "简要介绍简化 RAG 应用程序构建的热门框架。"
order: 1
plots: []
sourceHash: "ad5234a58a93ec8288311ae871266bbd7f36f2409ed0705568a11111c82825d7"
sourceCorrections: []
---

构建一个检索增强生成（RAG）系统需要协调多个不同部分：加载和处理数据、文本嵌入 (embedding)、存储和检索向量 (vector)、与大型语言模型（LLM）结合，以及管理它们之间的信息流。虽然使用各种库（用于向量数据库、嵌入模型和 LLM 交互）单独编写每个步骤的代码当然可行，但这种做法可能很快变得复杂且耗时，特别是随着需求的变化。

这就是专门框架的作用所在。LangChain 和 LlamaIndex 等框架提供了更高层次的抽象和预置组件，它们专门设计用于构建由 LLM 驱动的应用程序，包括 RAG 系统。它们旨在简化开发，促进模块化，并加快原型制作，通过处理大部分底层样板代码。

使用框架有以下几点好处：

- **模块化：** 它们鼓励将应用程序分解为可重用组件（数据加载器、文本分割器、检索器、LLM 包装器、输出解析器）。
- **集成性：** 它们提供各种数据源、向量数据库、嵌入模型和 LLM 的现成连接器，大大节省了集成工作。
- **标准化：** 它们提供通用接口和结构（例如 LangChain 中的“链”或 LlamaIndex 中的“查询引擎”）来组合组件，使整体架构更易于理解和管理。
- **预置逻辑：** 它们通常包含常见模式的实现，例如基本的 RAG 循环、处理聊天记录，甚至更复杂的代理行为。
- **社区与支持：** 热门框架得益于活跃的社区，提供示例、文档和支持。

下面我们简要介绍两个常用于 RAG 开发的知名框架：

### LangChain

LangChain 是一个综合性框架，旨在开发由语言模型驱动的应用程序。其理念围绕着将不同组件“链接”起来，以创建复杂的流程。对于 RAG，LangChain 提供以下模块：

- **文档加载器：** 摄取来自各种来源（文件、网页、数据库）的数据。
- **文本分割器：** 实现不同的文档分块策略。
- **文本嵌入 (embedding)模型：** 用于各种嵌入提供商的接口。
- **向量 (vector)存储：** 与众多向量数据库集成，用于存储和搜索嵌入。
- **检索器：** 用于根据查询获取相关文档的抽象。
- **LLMs / 聊天模型：** 用于与不同语言模型交互的包装器。
- **链：** 用于常见任务的预置序列，例如 `RetrievalQA` 链，它封装了一个基本的 RAG 流程（检索文档、将其填充到提示中、调用 LLM）。

LangChain 的优势在于其灵活性和广泛的集成能力，使开发者可以轻松组合搭配组件。它可作为一个通用的 LLM 应用程序开发工具集。

### LlamaIndex

LlamaIndex 定位为一个数据框架，专门用于 LLM 应用程序。LangChain 侧重于应用程序，而 LlamaIndex 则注重将 LLM 连接到自定义数据源。它提供精细的工具，用于数据摄取、索引和检索，旨在优化 RAG 流程。核心要点包括：

- **数据连接器 / 读取器：** 类似于 LangChain 的文档加载器，用于摄取数据。
- **节点解析器：** 负责将文档分块为“节点”（可包含文本和元数据）。
- **索引：** 从节点构建的数据结构，针对查询进行优化。RAG 最常用的是 `VectorStoreIndex`。
- **检索器：** 根据查询从索引中获取相关节点的组件。
- **查询引擎：** 高级接口，接收用户查询，使用索引和检索器获取上下文 (context)，并使用 LLM 合成答案。
- **聊天引擎：** 类似于查询引擎，但专为对话交互设计，维护聊天记录。

LlamaIndex 在 RAG 的数据摄取和检索方面表现突出，提供对索引的细致控制以及各种高级检索策略，旨在提升相关性并处理复杂数据结构。

### 框架选择

LangChain 和 LlamaIndex 都是构建 RAG 系统的强大工具，随着它们的发展和相互集成，两者之间的界限常常变得模糊。选择可能取决于：

- **主要侧重：** 如果您的主要难题是复杂数据摄取和优化检索，LlamaIndex 可能更具优势。如果您需要一个通用的工具集来构建具有许多集成的更广泛的 LLM 应用程序，LangChain 可能更合适。
- **所需抽象：** 审视每个框架提供的具体组件和抽象，看哪个更符合您的思维模型或项目要求。
- **生态系统：** 考虑您正在使用的任何现有工具或平台，因为一个框架可能提供更好的集成。

许多开发者在同一个项目中同时使用这两个框架的组件。重要的是，理解 RAG 的基本组件（检索器、生成器、数据准备），如前几章所述，能够让您有效使用任一框架，甚至在必要时手动构建系统部分。

虽然这些框架大大简化了开发，但本章将首先侧重于使用基本库交互来组装核心 RAG 流水线。这种做法有助于巩固您对底层机制的理解，然后在更高级的应用程序中可能使用框架对其进行抽象。您在此处学到的内容可以直接应用，无论您以后是否选择使用框架。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Narsimha Chatti, Mike Lewis, Kyunghyun Cho, and Douwe van der Plas (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇原创论文介绍了检索增强生成 (RAG) 框架，提供了对其架构和优势的基础理解。
- [LangChain Documentation](https://www.langchain.com/) — LangChain (2024)
  Publisher: LangChain
  LangChain 框架的官方文档，详细介绍了其模块、集成以及构建 RAG 系统等大型语言模型应用的模式。
- [LlamaIndex Documentation](https://www.llamaindex.ai/) — LlamaIndex (2024)
  Publisher: LlamaIndex
  LlamaIndex 框架的官方文档，重点介绍如何将大型语言模型与自定义数据源连接，并通过高级索引和检索来优化 RAG 流程。
