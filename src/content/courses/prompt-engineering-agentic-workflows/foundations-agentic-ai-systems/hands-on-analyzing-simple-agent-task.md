---
course: "prompt-engineering-agentic-workflows"
chapter: "foundations-agentic-ai-systems"
lesson: "hands-on-analyzing-simple-agent-task"
sourceId: 6716
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-1-foundations-agentic-ai-systems/hands-on-analyzing-simple-agent-task"
title: "动手实践：分析一个简单代理的任务执行过程"
description: "参与一个实践环节，剖析并分析一个基本代理任务的执行流程。"
order: 7
plots: []
sourceHash: "695bf93f7869594433454ee475c16241293d025993a5627e04eab7df388dcb31"
sourceCorrections: []
---

为了让代理系统的抽象想法更具体，这里分析一个AI代理执行任务的简化示例。此示例说明了包括LLM、工具、内存（即使是基础的）和提示在内的组成部分如何协同工作。目标是观察并了解代理的决策过程，即使是在模拟环境中。

### 场景：一个简单信息获取任务

假设用户问我们的AI代理：*“法国的首都是哪里，它目前的居住人口是多少？”*

这是一个直接的问题，但它可能要求代理执行多个步骤：识别所需的两部分信息，获取它们（可能借助工具），然后综合答案。

### 我们的代理工具包（简化版）

为进行此次演示，我们的代理配备了：

- **一个大型语言模型（LLM）：** 这是代理的“大脑”，负责理解请求、规划步骤、解释工具输出，并生成最终回应。我们假设它是一个能遵循指令的模型。
- **一个 `web_search(query: str)` 工具：** 这个工具允许代理在互联网上查找信息。它接受一个字符串形式的搜索查询，并返回包含搜索结果的文本片段。
- **短期记忆：** 对于这个简单任务，代理的短期记忆通过LLM的上下文 (context)窗口隐式管理。思想、行动和观察的历史被保留，以指导后续步骤。

### 指导代理的主提示

为了指导代理的行为并确保其遵循结构化方法，我们为其提供了一个“主提示”或一组系统级指令。这个提示概述了代理应如何思考、行动和使用其工具。它是代理行为提示工程 (prompt engineering)的一个非常基本的要素。

以下是这样一个主提示的示例，它促使代理形成类似ReAct（推理 (inference)与行动）的循环：

```
你是一个AI助手。你的任务是准确回答用户的问题。
你可以使用以下工具：
- web_search(query: string): 根据查询在互联网上搜索信息并返回摘要。

要回答问题，请重复以下步骤：
1. 思维：简要说明你的计划或下一步将采取的行动。
2. 行动：指定要使用的工具及其输入。如果你有了答案，请声明 'Final Answer:'。
3. 观察：记录工具的结果。
根据需要重复步骤1-3，直到你能提供最终答案。

用户问题：法国的首都是哪里，它目前的居住人口是多少？
```

这个提示明确告诉了代理：

- 它的总体目标（回答用户问题）。
- 可用的工具以及如何调用它们。
- 其内部推理（`思维`）、工具使用（`行动`）和处理工具输出（`观察`）的结构化格式。
- 如何表明它已完成任务（`Final Answer:`）。

### 模拟代理的执行轨迹

现在，让我们模拟代理在主提示的指导下，处理用户问题时的内部思考和行动。

**第一次迭代：查找首都**

- **思维：** 用户的问题包含两部分：法国的首都及其居住人口。我应该首先查找法国的首都。
- **行动：** `web_search("capital of France")`
- *（模拟工具执行：`web_search` 工具被调用。它查询搜索引擎并处理结果。）*
- **观察结果：** “法国的首都是巴黎。”

此时，代理已成功获取第一部分信息。“观察结果”现在是其工作上下文 (context)的一部分。

**第二次迭代：查找居住人口**

- **思维：** 我已查到巴黎是首都。现在我需要查找它的当前居住人口。
- **行动：** `web_search("current population of Paris")`
- *（模拟工具执行：`web_search` 工具再次被调用。）*
- **观察结果：** “根据近期估算，巴黎目前的居住人口约210万。”

代理现在已获取第二部分信息。

**第三次迭代：综合最终答案**

- **思维：** 我已拥有两部分信息：法国首都是巴黎，其居住人口约210万。我现在可以构建最终答案了。
- **行动：** `Final Answer: 法国首都是巴黎，其目前居住人口约210万。`

代理发出信号表示它已得到完整答案，整个过程结束。

### 分析代理的流程

这个简单的执行轨迹展现了代理系统的几个重要方面：

- **主提示的作用：** 主提示起到了重要作用。它不仅告诉代理*做什么*，还告诉*如何*处理问题：通过循环地思考、行动和观察。这种结构化方法是提示工程 (prompt engineering)的直接成果。
- **任务分解：** LLM在提示和问题性质的指导下，隐式地将主要任务（“回答关于法国首都和居住人口的问题”）分解为子任务（查找首都，查找居住人口）。
- **工具选择与使用：** 代理正确识别何时需要工具，并为 `web_search` 工具制定了适当的查询。主提示对工具的描述对此很重要。
- **信息整合：** 代理在不同迭代中保持了上下文 (context)。第一次搜索的结果（巴黎）为第二次搜索（巴黎的居住人口）的查询提供了信息。然后它结合两部分信息来构成最终答案。
- **LLM作为推理 (inference)引擎：** 每个“思维”步骤都代表着LLM的推理过程。它评估当前状态，规划下一步行动，并最终决定何时完成任务。

### 工作流程可视化

代理的操作可以被可视化为一系列状态和转换，由其内部推理 (inference)和与工具的互动驱动。

> 图示代理在信息获取任务中的迭代思维-行动-观察循环。

此图显示了代理如何循环地进行思考、行动（通常涉及工具）和观察结果，更新其内部状态或上下文 (context)，直到它能生成最终答案。

### 联系章节起始主题

本次动手实践分析虽然简化，但直接关联到本章的起始主题：

- **AI代理的核心组成部分：** 我们看到了LLM作为推理 (inference)核心，`web_search` 工具作为外部能力，以及隐式使用短期记忆（上下文 (context)窗口）来连接步骤。
- **提示工程 (prompt engineering)的作用：** 主提示是提示工程如何指导和构建代理行为的一个典范示例，使其能够有条不紊地执行多步骤任务。特定的`思维:`、`行动:`、`观察:` 格式是一种基于提示的机制。
- **代理提示与标准提示的对比：** 主提示比LLM的简单问题更复杂。它包含关于流程、工具使用和输出格式的指令，这是代理提示的特点。
- **代理架构概述：** `思维 -> 行动 -> 观察` 循环镜像了ReAct等代理架构的核心循环，我们对此进行了简要介绍。这个示例提供了一个实用的视角，展示了此类架构如何运作。
- **评估代理性能的方法：** 在本例中，评估是直接的：代理是否提供了正确的首都和合理的居住人口数字？对于更复杂的任务，前面讨论的评估方法变得更有意义。

通过剖析一个简单的代理任务，我们开始认识到不同组成部分如何协同工作，以及仔细的提示设计所扮演的重要作用。随着课程的进行，我们将在这些基本原理上继续，构建和管理更复杂的代理工作流。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct范式，该范式将语言模型的推理和行动能力结合在一个迭代的“思考-行动-观察”循环中，本节中直接演示了该范式。
- [Toolformer: Language Models That Can Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint arXiv:2302.04761; DOI: [10.48550/arXiv.2302.04761](https://doi.org/10.48550/arXiv.2302.04761)
  描述了大型语言模型如何通过自监督训练学习使用外部工具，构成了智能体与环境交互的基础机制。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint arXiv:2201.11903; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  提出了一种提示技术，使大型语言模型能够展示中间推理步骤，这是智能体工作流中“思考”组件的基础。
- [The Prompt Engineering Guide](https://www.promptingguide.ai/) — DAIR.AI (2024)
  Publisher: DAIR.AI
  一个全面的社区驱动资源，提供设计有效提示的实用技术和示例，包括适用于构建智能体行为的方法。
