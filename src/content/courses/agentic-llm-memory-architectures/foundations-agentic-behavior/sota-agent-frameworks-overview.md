---
course: "agentic-llm-memory-architectures"
chapter: "foundations-agentic-behavior"
lesson: "sota-agent-frameworks-overview"
sourceId: 4439
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-1-foundations-agentic-behavior/sota-agent-frameworks-overview"
title: "先进智能体框架概览"
description: "简要介绍 LangChain、AutoGen 和 CrewAI 等主要框架的架构角度。"
order: 4
plots: []
sourceHash: "27ac46af67e10ef0d6363d63f18eb31d34cac4be24b7c703cf53f9a0c7f8cb46"
sourceCorrections: []
---

在智能体组件的基础理念及相关设计难点之上，若干框架应运而生，以简化复杂智能体系统的开发。这些框架提供预制组件、抽象层和架构模式，使开发者能够专注于智能体行为的更上层逻辑，而非从零开始重新实现核心功能。理解它们的架构理念对于选择合适的工具并在复杂项目中高效使用它们很有助益。

在此，我们从架构角度概括性地介绍主要的框架，这对于旨在构建高级智能体系统的专家级从业者有所帮助。

### LangChain

LangChain 可以称为是开发由语言模型驱动的应用最全面、最广为采纳的框架之一。其核心优势在于其模块化和可组合性。它提供一系列丰富的组件，这些组件可以被链接起来以创建复杂应用，包括智能体。

从架构上说，LangChain 围绕若干基础理念来组织智能体开发：

1. **组件**：为模型（大型语言模型、聊天模型、嵌入 (embedding)）、提示词 (prompt)（模板管理、优化）、索引（文档加载器、文本分割器、向量 (vector)存储、检索器）、记忆（短期、长期）、链（调用序列）和智能体（大型语言模型进行行动决策）提供标准化接口。
2. **链**：表示操作序列，通常涉及大型语言模型调用。它们提供了一种简单的方式来连接组件，例如格式化提示词、将其发送给大型语言模型以及解析输出。
3. **智能体**：使用大型语言模型作为推理 (inference)引擎，以决定采取哪些行动以及按什么顺序。LangChain 提供多种智能体架构的实现，包括 ReAct 和 Self-Ask（我们将在第2章详细分析）。智能体执行器负责管理思考、行动和观察的循环。

对于专业开发者而言，LangChain 提供很大的灵活性，以定制提示词、智能体决策逻辑、工具集成和记忆系统。然而，其全面的特性意味着，掌握其细致之处、调试复杂链或智能体循环以及优化性能，都需要对框架的抽象和底层大型语言模型行为有透彻的理解。它在需要对单个组件及其在单个智能体或预定义序列中的交互进行精细控制的场景中表现优异。

### AutoGen

AutoGen 采用一种独特的架构方法，主要侧重于构建多智能体应用，这些智能体通过对话和协作来完成任务。由微软研究院开发，它简化了复杂大型语言模型工作流的编排，这些工作流涉及不同的智能体角色和交互模式。

架构元素包括：

1. **可对话智能体**：AutoGen 中的基本构建模块。智能体可以发送和接收消息以相互交互。提供多种专用智能体类型（例如，`AssistantAgent`、`UserProxyAgent`）。
2. **多智能体对话**：AutoGen 管理智能体之间的交互流程。开发者定义智能体、它们的能力（包括工具使用），以及管理其对话的规则或模式（例如，顺序聊天、群组讨论）。
3. **自动化聊天编排**：该框架处理消息传递、轮流和终止条件的机制，基于定义的对话流程或智能体能力。`UserProxyAgent` 通常充当人类输入的代理或执行其他智能体建议的代码。

AutoGen 特别适合专家级任务，这些任务受益于分布式推理 (inference)或专业角色，例如带测试的自动化代码生成、涉及不同专家智能体的多步数据分析，或复杂交互的模拟。其优势在于自动化多个智能体之间的协作，使开发者能够定义精巧的协作或分层工作流。在使用 AutoGen 时，管理通信协议的复杂性以及确保复杂多智能体设置中系统行为的可预测性，都是需要考虑的方面。

> AutoGen 工作流的一个简化表示，用户代理与编码和审查智能体进行交互。

### CrewAI

CrewAI 将自身定位为一个专门用于编排角色扮演式自主智能体团队的框架，这些团队旨在通过协作完成复杂任务。它提供更高级别的抽象，侧重于定义具有特定职责的智能体，并管理它们如何通过结构化流程协同工作。

CrewAI 的核心架构理念如下：

1. **智能体**：定义了特定的 `role`（角色）、`goal`（目标）、`backstory`（背景故事）、工具，以及可能详细的操作模式。这种详细的定义指导大型语言模型在智能体指定功能内的行为。
2. **任务**：分配给智能体的具体任务。任务有描述、预期输出，并被分配给特定的智能体。
3. **团队**：编排智能体和任务。一个团队定义智能体集合、任务集合以及智能体执行任务的流程（例如，顺序、分层）。流程决定信息流和执行顺序。
4. **流程管理**：CrewAI 管理执行流程，根据定义的流程将任务输出从一个智能体传递给下一个，从而促成一种结构化的协作形式。

CrewAI 擅长构建模拟人类团队工作流的系统，在这些工作流中，明确的角色和清晰的流程是有利的。与 AutoGen 更通用的对话方法相比，CrewAI 侧重于结构化、流程驱动的协作。对于专家来说，它提供一种简化实现精巧智能体团队的方式，用于研究报告生成（研究员智能体、作家智能体、编辑智能体）或软件开发规划（规划者智能体、高级工程师智能体、质量保证智能体）等任务。设计上的关注点转向了定义有效的角色、清晰的任务描述和最优的协作流程。

### 框架选择考量

这些框架代表了构建智能体系统的不同理念和抽象层次。

- **LangChain**：提供可组合组件的广泛工具集，非常适合对单智能体逻辑或特定操作链进行精细控制。其灵活性伴随着潜在的复杂性。
- **AutoGen**：侧重于实现多智能体之间复杂的对话和交互，自动化编排层。适用于受益于多样化、相互作用的专业智能体的任务。
- **CrewAI**：专门为具有定义角色和流程的协作智能体团队提供更高级别的抽象。简化了模拟结构化人类团队协作系统的创建。

选择框架很大程度上取决于应用程序特定的架构需求。通常，高级系统甚至可能集成来自多个框架的组件或理念。无论选择如何，扎实掌握智能体行为、推理 (inference)、规划和记忆的核心原则，如本课程中反复提及，对于设计、实现和调试高效的智能体系统来说仍然极为重要。这些框架是强大的工具，但它们都建立在我们正在研究的基础理念之上。这一技术方向正在快速发展，及时了解这些以及新出现的框架所提供的架构模式和能力，对从事大型语言模型应用前沿开发的从业者来说很有意义。

## 参考资料

- [AutoGen: Enabling Next-Gen LLM Applications with Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) — Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang (2023)
  Journal: arXiv preprint arXiv:2308.08155; Publisher: arXiv; DOI: [10.48550/arXiv.2308.08155](https://doi.org/10.48550/arXiv.2308.08155)
  介绍了AutoGen框架，用于通过可对话代理构建多代理LLM应用程序，提供了核心研究和架构细节。
- [LangChain Documentation](https://www.langchain.com/docs/) — LangChain Contributors (2024)
  官方文档，全面指导LangChain的模块化组件、链和代理开发模式。
- [CrewAI Documentation](https://docs.crewai.com/) — CrewAI Developers (2024)
  CrewAI的官方文档，详细介绍了其编排协作型、角色扮演代理团队以执行复杂任务的方法。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，结合推理和行动，通过生成推理过程和特定任务动作来增强语言模型执行复杂任务的能力。
