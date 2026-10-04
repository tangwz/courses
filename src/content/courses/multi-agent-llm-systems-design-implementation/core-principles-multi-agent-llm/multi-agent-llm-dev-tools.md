---
course: "multi-agent-llm-systems-design-implementation"
chapter: "core-principles-multi-agent-llm"
lesson: "multi-agent-llm-dev-tools"
sourceId: 6638
sourceUrl: "https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-1-core-principles-multi-agent-llm/multi-agent-llm-dev-tools"
title: "多智能体大型语言模型开发工具概览"
description: "简要介绍当前可用于构建多智能体大型语言模型应用的工具和库。"
order: 6
plots: []
sourceHash: "2b3f6dc31e80b696fcc014c01583f3ca9f292f046a6ff30b4f3609ee8b3d007f"
sourceCorrections: []
---

构建结合大型语言模型（LLMs）的多智能体系统（MAS），不仅仅需要单个智能体编程。智能体间通信、工作流程编排以及管理涌现 (emergence)行为的复杂性，使得专门的开发工具和框架成为必要。提供工程师当前可用工具的概览，侧重于那些有助于设计和实现复杂多智能体大型语言模型应用的工具。这个方面变化很快，新工具和新功能迅速出现，但掌握主要的选项将使您能够为项目选择合适的方案。

现有工具通常分为不同类别：有些是全面的大型语言模型应用开发框架，它们已扩展其能力以支持多智能体架构；另一些则是专为编排智能体合作而构建的。

### 主要框架和库

多个框架因其在构建多智能体大型语言模型系统中的实用性而在开发者群体中获得了关注。每个框架都提供了一组不同的抽象，并侧重于多智能体系统开发的特定方面。

#### LangChain 和 LangGraph

LangChain 已成为一个开发由语言模型驱动的多功能框架。虽然其最初的优势在于创建链和单智能体应用，但其演进，特别是随着 **LangGraph** 的引入，大大增强了其对多智能体系统的支持。

LangGraph 允许您通过将工作流程构建为图来创建有状态、多参与者的应用。图中的节点表示函数或大型语言模型调用（智能体或工具），边定义了控制和数据流。这个模型对多智能体系统很有用，原因在于：

- 它明确管理智能体交互中的状态。
- 它支持循环，从而实现迭代过程和智能体之间的对话循环。
- 条件边允许基于智能体输出进行动态路由，有助于智能体集体内部的复杂决策。
- 人类参与的交互可以自然地作为节点纳入图中。

结合使用 LangChain 的智能体原语（如 `AgentExecutor`）和 LangGraph，您可以定义具有各自工具和推理 (inference)能力的单个智能体，然后在有状态的图结构中编排它们的交互。

#### AutoGen (微软研究院)

AutoGen 是一个旨在简化复杂大型语言模型工作流程的编排、优化和自动化的框架。它提供了一个多智能体对话框架，智能体可以在其中相互交互以解决任务。主要功能包括：

- **可对话智能体**：AutoGen 的核心抽象是 `ConversableAgent`，它能够发送和接收消息，并选择性地执行代码。您可以创建专门的智能体，如 `AssistantAgent`（一个由大型语言模型支持的智能体）和 `UserProxyAgent`（可以获取人类输入或执行代码）。
- **自动化智能体聊天**：智能体可以设置为在各种拓扑结构中对话（例如，群聊、分层讨论），以共同执行代码生成、问答等任务。
- **工具运用和函数调用**：智能体可以通过大型语言模型函数调用能力运用工具，增强它们与外部环境交互或执行特定计算的能力。

AutoGen 在问题可以分解为专门智能体之间的一系列对话步骤的场景中表现出色。它侧重于实现更自主和灵活的智能体合作。

#### CrewAI

CrewAI 是另一个专门侧重于编排扮演角色的自主人工智能智能体的框架。它旨在培养合作式智能，让一“组”智能体，每个都有明确的角色、工具和目标，共同完成复杂任务。其设计理念围绕：

- **智能体**：由其角色、目标、背景故事（上下文 (context)）、大型语言模型配置和工具定义。
- **任务**：智能体的具体分配，描述了预期输出。任务可以是顺序的或并行的，并且可以依赖于其他任务的输出。
- **工具**：智能体可用于与外部环境交互或执行特定动作的函数。
- **团队**：智能体的集合以及它们需要执行的任务集，由一个明确的过程（例如，顺序执行）管理。

CrewAI 提供了一种更高层、声明式的方法来定义智能体合作，使得为流程自动化设置和管理智能体团队相对直接。

#### LlamaIndex 智能体

LlamaIndex 因其将大型语言模型连接到自定义数据源的能力而受到好评，主要用于构建复杂的检索增强生成（RAG）应用。虽然它并非完全与 AutoGen 或 CrewAI 同类别的多智能体编排框架，但 LlamaIndex 提供了可以构成多智能体系统构建单元的智能体抽象。

- **数据中心智能体**：LlamaIndex 智能体（例如 `OpenAIAgent`、`ReActAgent`）擅长对数据进行推理，无论是结构化还是非结构化数据。它们可以配备各种工具，包括 LlamaIndex 强大的查询引擎和数据加载器。
- **大型系统中的组成部分**：您可以构建多个 LlamaIndex 智能体，每个专门处理不同的数据集或推理类型，然后使用自定义 Python 逻辑或像 LangGraph 这样的框架来编排它们的交互。例如，一个智能体可能总结文档，另一个可能查询向量 (vector)数据库，第三个可能综合它们的发现。

LlamaIndex 智能体的优势在于它们与数据源的深度集成，当智能体任务高度依赖信息检索和分析时，这使得它们具有很高价值。

### 比较概览

为了有助于在多智能体系统开发背景下区分这些工具，下表突出显示了它们的一些特点：

| 特性 | LangChain (与 LangGraph 结合) | AutoGen | CrewAI | LlamaIndex 智能体 |
| --- | --- | --- | --- | --- |
| **主要侧重** | 通用大型语言模型应用开发，基于图的状态机 | 多智能体对话，研究 | 合作式人工智能智能体团队，流程自动化 | 数据中心检索增强生成 (RAG)，单个智能体任务 |
| **智能体定义** | 高度灵活，自定义代码，`Runnable` | 预定义（`AssistantAgent`，`UserProxyAgent`）并可自定义 | 基于角色，面向目标，大型语言模型配置 | 使用工具的智能体，侧重检索增强生成 |
| **通信模型** | 图中的消息传递，共享状态 | 回合制对话，消息传递 | 管理的任务移交，共享上下文 (context) | 主要通过工具输出和函数调用 |
| **编排方式** | `LangGraph`（有状态图） | 群聊管理器，顺序/自动回复 | 流程驱动，任务依赖 | 自定义逻辑或其他框架 |
| **工具集成** | 广泛的 LangChain 工具生态系统 | 函数调用，代码执行 | 每个智能体的自定义工具 | 广泛的 LlamaIndex 工具生态系统 |
| **复杂性管理** | 明确的状态和边定义 | 通过智能体交互进行抽象 | 任务委托和团队结构 | 侧重于数据交互复杂性 |
| **对多智能体系统的适用性** | 适用于复杂、有状态的交互 | 适用于对话式、研究导向的多智能体系统 | 适用于基于角色、流程自动化的多智能体系统 | 中等（作为数据任务的组成部分） |

> 该表提供了多智能体系统开发框架的简要比较，侧重于智能体定义、通信和编排等方面。

### 选择合适的工具

选择合适的开发工具很大程度上取决于您的多智能体系统的具体要求：

- **交互性质**：如果您的系统依赖复杂、有状态且可能循环的交互，LangGraph 提供精细的控制。对于更多对话式或回合制合作，AutoGen 可能是更自然的选择。对于角色明确且流程顺序的系统，CrewAI 提供精简的方法。
- **数据依赖**：如果智能体需要对大型、多样的数据集执行复杂操作，LlamaIndex 智能体是很有竞争力的选择，并可能集成到更广泛的编排框架中。
- **开发成本**：一些框架提供更高层次的抽象，可以加快常见模式的开发（例如，CrewAI 用于流程自动化），而另一些框架则提供更大的灵活性，但代价是增加了设置复杂性（例如，LangGraph 用于高度自定义的图）。
- **生态系统和社区**：框架的成熟度、预构建工具和集成的可用性，以及其社区的规模和活跃度，也是实际考量因素。

在您学习本课程的过程中，特别是在实践环节，您将获得这些工具的一些直接经验。记住，这个方面在发展。新工具会涌现 (emergence)，现有工具会成熟，增加更复杂的功能，用于管理智能体团队、确保可观察性和优化性能。了解这些发展对任何从事高级多智能体大型语言系统工作的工程师都会有益。

## 参考资料

- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGidaSZ4QRFiOa2NRD-0GgOGWMSMk1O4YtYZcl-slbpkUlL0gpAlkK1K-euWmcj6786t-DOXLH4MI8cEQ8btbKtzVg71qBHPhfVXYfdkzcW-bvqw5NL7J5QENGT8Xa_-ATh2g==) — Qingyun Wu, G. K. Mohan, S. K. Narra, L. Xia, J. Ma, S. Dong, T. Li, Y. Liu, C. M. Singh, C. Zhang, H. Xia, M. S. Khan, A. K. Mallya, J. G. Y. Huang, B. Van Durme, J. J. Li, Y. Liu (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2308.08155](https://doi.org/10.48550/arXiv.2308.08155)
  介绍了AutoGen框架，用于多智能体LLM系统，详细阐述了其为灵活可扩展的对话式智能体交互而设计。
- [LangGraph Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgZSHDBKeJdzzos9k4d7Bxw4uH2K1MhPFRzJrXpQ-YSJCkfqllp2qgaOA8auaVDzKetSlTjQfkgIWnOjnkVxmgagA4dL2ieOq5kBOWHqHH_Tz3_VGW7e_2CRxScs7ugKuBOQ7Yp8JodkomeLUwye78LqU=) — LangChain (2024)
  Publisher: LangChain
  提供了使用LangGraph构建有状态、多角色应用的官方指南，涵盖了其基于图的智能体工作流编排。
- [CrewAI Documentation](https://docs.crewai.com/) — João Moura (2024)
  Publisher: CrewAI
  CrewAI的官方资料，解释了其用于协作式AI系统和流程自动化的角色扮演智能体编排框架。
- [LlamaIndex Agents Documentation](https://docs.llamaindex.ai/en/stable/module_guides/agents/agents.html) — LlamaIndex (2024)
  Publisher: LlamaIndex
  官方文档详细介绍了如何在LlamaIndex框架内构建和使用以数据为中心的智能体，重点关注其与RAG和工具的集成。
