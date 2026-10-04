---
course: "prompt-engineering-agentic-workflows"
chapter: "foundations-agentic-ai-systems"
lesson: "overview-agent-architectures"
sourceId: 6714
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-1-foundations-agentic-ai-systems/overview-agent-architectures"
title: "智能体架构概览"
description: "了解ReAct等成熟智能体架构及其运行原理。"
order: 5
plots: []
sourceHash: "3f4af8fc193c3d5669e5bc23d0eff43fce06f02ffad5b11744383214638218c4"
sourceCorrections: []
---

为有效处理复杂、多步骤的任务，AI智能体常依赖结构化的操作框架，这些框架被称为智能体架构。可以将架构视为一个蓝图，它定义了智能体的核心组成部分，例如其大型语言模型（LLM）大脑、记忆和工具，如何相互作用以达成目标。这些架构为智能体规划、执行动作和适应新信息提供了一种系统化的方式。尽管存在多种架构，但理解一两种常见模式能为智能体系统中的提示工程 (prompt engineering)提供良好基础。

### ReAct框架：推理 (inference)与行动

一种有影响力且被广泛讨论的智能体架构是**ReAct**，其全称为“推理与行动”（Reasoning and Acting）。由Google Brain和普林斯顿大学的研究人员开发，ReAct帮助LLM通过明确地将推理步骤与特定任务的动作交错进行，从而解决复杂任务。这种方法让智能体能够生成关于下一步做什么的内部思考，采取一个动作，观察结果，然后利用该观察结果来指导其下一步思考和随后的动作。这个迭代循环一直持续到任务完成。

ReAct智能体的核心循环通常包含三个主要组成部分：

1. **思考 (Thought)**：这是智能体的LLM组件进行内部推理的地方。它分析当前任务、之前动作和观察的历史，以及总体目标。在此基础上，它为下一步动作制定计划或理由。你的提示在引导此思考过程中扮演重要角色，指导LLM如何分解问题、考虑哪些策略以及如何评估进展。例如，一个思考可能是：“我需要找到法国的首都。我应该为此使用搜索工具。”
2. **行动 (Action)**：基于“思考”，智能体选择并执行一个动作。动作通常是对智能体可用的预定义工具或函数的调用。例子包括查询搜索引擎、访问数据库、执行计算，甚至向用户请求澄清。提示定义了可用的工具以及LLM应如何指定其选择的动作和参数 (parameter)的格式。例如：`Action: search("capital of France")`。
3. **观察 (Observation)**：动作执行后，智能体接收到一个观察结果。这是执行动作后产生的报告或反馈，例如搜索结果、从API检索到的数据、计算输出，或者动作失败时的错误消息。这个观察结果随后被反馈到智能体的上下文 (context)，供下一个“思考”步骤使用。例如：`Observation: "Paris"`。

这种思考-行动-观察的循环使智能体能够动态规划并调整其策略。如果某个动作未能产生预期结果，智能体可以在下一步思考中推理失败原因，并尝试不同的方法。

让我们用一个简化例子来说明。假设我们希望一个智能体回答问题：“埃菲尔铁塔所在城市的居民数量是多少？”

- **初始任务**：回答“埃菲尔铁塔所在城市的居民数量是多少？”
- **循环1**：

  - **思考**：我首先需要确定埃菲尔铁塔所在的城市。我将使用搜索工具。
  - **行动**：`search("city of Eiffel Tower")`
  - **观察**：“法国巴黎”
- **循环2**：

  - **思考**：好的，埃菲尔铁塔在巴黎。现在我需要找到巴黎的居民数量。我将再次使用搜索工具。
  - **行动**：`search("population of Paris")`
  - **观察**：“约210万（截至2023年）”
- **循环3**：

  - **思考**：我已找到城市（巴黎）及其居民数量（210万）。我已拥有回答原始问题所需的所有信息。
  - **行动**：`finish("埃菲尔铁塔所在城市巴黎的居民数量约为210万。")`

ReAct框架为智能体解决需要多步骤并与外部信息源交互的问题提供了清晰的结构。提示工程 (prompt engineering)在每个阶段都非常重要，从定义初始任务和可用工具，到指导LLM的推理过程并帮助它解读观察结果。

下方是描绘ReAct架构一般流程的图表：

> ReAct循环：一个由思考、行动和观察组成的迭代过程，引导智能体完成任务。

### 其他架构考量

虽然ReAct是一个突出例子，但它只是设计智能体架构的众多方法之一。其他架构可能强调不同方面，例如：

- **分层规划**：将大目标分解为子目标和子任务的层级结构，并系统地处理每个层级的智能体。
- **多智能体系统**：为多个专门化智能体之间的协作而设计的架构，以解决更大的问题。
- **状态机**：其行为被建模为一系列状态和转换的智能体，提示可以定义转换条件或在某个状态下要采取的行动。

架构的选择通常取决于任务的复杂性、可用工具的类型以及期望的自主性和适应性水平。然而，许多基本原则，如清晰的目标定义、动作选择和观察结果处理，在不同的智能体设计中是普遍存在的。

无论具体的架构如何，提示工程 (prompt engineering)仍然是引导智能体的核心机制。你的提示将在其架构框架内定义智能体的“SOP”（标准操作程序），指导其推理 (inference)、决策以及与环境和工具的交互。随着本课程的学习，你将学会如何制作能够在智能体系统中有效实现这些元素的提示。理解这些底层结构对于设计能带来可靠、高效智能体行为的提示非常重要。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍ReAct框架的原始研究论文，详细阐述了其在基于LLM的智能体中结合推理和行动的设计。
- [A Survey of Large Language Model Based Agents](https://arxiv.org/abs/2308.11432) — Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2308.11432 (cs); DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  一份广泛的调查报告，概述了大型语言模型智能体中采用的不同架构和方法。
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint arXiv:2302.04761; DOI: [10.48550/arXiv.2302.04761](https://doi.org/10.48550/arXiv.2302.04761)
  本文展示了LLM如何学习与外部工具交互和使用工具，这是ReAct等智能体架构中“行动”组件的一个基本方面。
