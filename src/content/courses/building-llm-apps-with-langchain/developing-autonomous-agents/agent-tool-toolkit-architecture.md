---
course: "building-llm-apps-with-langchain"
chapter: "developing-autonomous-agents"
lesson: "agent-tool-toolkit-architecture"
sourceId: 7436
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-6-developing-autonomous-agents/agent-tool-toolkit-architecture"
title: "代理、工具和工具集架构"
description: "了解代理、它们可用的工具（例如搜索引擎）以及工具集（工具的集合）之间的关系。"
order: 2
plots: []
sourceHash: "29fea6fe023a8ca7192d6e2ff795d5f99caaf2f791cf49adbde0a2454ecedb3a"
sourceCorrections: []
---

链式流程执行预设的步骤序列，而代理则在一个推理 (inference)和行动的动态循环中运行。这种行为并非由单一的、整体式对象管理，而是由一个包含三个主要组成部分的协作架构管理：**代理**、它能使用的**工具**以及协调整个过程的**运行时**（通常使用 **LangGraph** 构建）。理解这些部分如何相互配合，对于构建高效的代理系统来说是基本的。

### 代理：推理 (inference)引擎

代理的核心是决策组件。然而，它并非像Python类那样是一个独立的程序代码。相反，LangChain中的“代理”是一种抽象，它将语言模型与提示词 (prompt)和定义的工具模式结合起来。这种结合将一个通用LLM转变为一个能够推动任务前进的推理引擎。

提示词指导LLM遵循一个特定的思考过程，这个过程常被称为“推理循环”。ReAct（推理与行动）等框架使用的一种常见模式，要求模型将其思考分解为明确的步骤：

- **思考**：代理的内心独白，它在此分析当前情况，评估实现目标的进展，并决定下一步做什么。
- **行动**：代理决定用来取得进展的特定工具。具有**工具调用**能力的现代LLM将其生成为结构化对象，而非原始文本。
- **行动输入**：要传递给所选工具的参数 (parameter)或查询。
- **观察**：执行工具返回的结果，代理将在其下一个“思考”步骤中使用此结果。

LLM生成遵循这种结构的输出。代理的唯一职责是根据先前的行动和观察历史，产生下一个思考和行动。

### 工具：代理的能力

如果代理是大脑，那么工具就是它的双手。工具是代理可以调用以在LLM外部进行交互的函数或服务。代理需要做的任何事情，从网页搜索到查询数据库或计算数字，都作为工具公开。

每个工具都由两个主要属性定义：

1. **功能**：实际执行的代码。这可以是一个公共API的包装器、一个Python函数，甚至是另一个LangChain链。
2. **描述**：对工具作用、输入应是什么以及它返回什么的清晰、自然语言描述。

描述非常必要。代理的LLM没有工具代码的固有知识；它*完全*依赖于描述来判断哪个工具适合给定任务。一个准确反映工具用途和输入格式的良好描述，决定了一个代理是高效工作还是持续失败。

例如，一个`SearchTool`可能有一个这样的描述：“当您需要回答有关时事的问题或在互联网上查找信息时很有用。”

### 工具集：组织能力

工具集简单来说是旨在协同工作以完成特定场景中任务的工具集合。您可以加载一个工具集，它提供预配置的能力集合，而不是逐个加载工具。

例如，LangChain提供一个`SQLDatabaseToolkit`，其中包含用于以下目的的工具：

- 列出数据库中的表。
- 检查特定表的模式。
- 执行SQL查询。
- 检查SQL查询是否存在语法错误。

使用工具集简化了设置，并确保代理具有一组一致的相关能力，以便与特定系统交互。

### 运行时循环

运行时环境是将所有组件聚合在一起并驱动代理运行的系统。它负责执行使代理实现自主的循环。尽管`AgentExecutor`类是管理此循环的传统方式，但现代LangChain应用通常使用**LangGraph**来定义这种协调。这使得可以更好地控制状态和执行流。

以下是运行时管理的逐步执行流程：

1. 它接收用户的初始输入或目标（例如，“上届超级碗的比分是多少？谁是MVP？”）。
2. 它调用代理（带有提示词 (prompt)/工具的LLM），并传递输入和所有先前的步骤。
3. 代理进行推理 (inference)并输出下一个要执行的`行动`（例如，对“搜索”的结构化工具调用）以及`行动输入`（例如，“第57届超级碗比分和MVP”）。
4. 运行时解析此输出以识别工具及其输入。
5. 它调用指定的工具并提供输入。
6. 工具执行并返回一个结果，该结果被格式化为`观察`（例如，“堪萨斯城酋长队以38-35击败费城老鹰队。帕特里克·马霍姆斯是MVP。”）。
7. 运行时获取此`观察`并更新交互的状态（历史）。
8. 它重复此过程，将更新后的历史传递回代理进行下一步推理。代理可能会看到结果并决定其任务已完成。
9. 一旦代理回应“最终答案”或文本消息而非工具调用，运行时就会停止循环并将此最终回应返回给用户。

下图说明了组件之间的这种交互流程。

> 代理运行时循环。它协调用户、推理代理（LLM）和可用工具之间的交互，直到产生最终答案。

这种模块化架构带来了极大的灵活性。您可以更换LLM、修改代理的提示词，或添加新的自定义工具，而无需改变底层运行时逻辑。在接下来的章节中，我们将通过为代理配备预构建和自定义工具来解决问题，从而将这种架构付诸实践。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: ICLR; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，结合推理和行动来增强智能体能力。
- [Agents](https://api.python.langchain.com/en/latest/agents/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain智能体模块的官方文档，包含其组件和使用方法。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint arXiv:2201.11903; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  提出了一种提示方法，使大型语言模型能够执行复杂的推理步骤。
