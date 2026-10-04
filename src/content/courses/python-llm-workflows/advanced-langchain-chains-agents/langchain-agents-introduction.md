---
course: "python-llm-workflows"
chapter: "advanced-langchain-chains-agents"
lesson: "langchain-agents-introduction"
sourceId: 5157
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/langchain-agents-introduction"
title: "代理介绍：LLM作为推理引擎"
description: "理解代理如何使用LLM决定行动并使用工具。"
order: 3
plots: []
sourceHash: "be6b9995293a0874df7055152d001792b583021b72d763e185c9ae745125f5eb"
sourceCorrections: []
---

许多任务在LLM工作流中需要动态决策，其中确切步骤通常事先未知，并取决于先前行动的结果。虽然某些LLM应用程序使用结构化的、预定的LLM或其他工具调用序列，但LangChain代理为这些更复杂的场景提供了所需的灵活性。

代理不遵循链那样的固定路径，而是使用LLM作为其核心推理 (inference)引擎。可以将LLM视为不仅仅是一个文本生成器，而是一个决策者。针对用户目标，代理利用LLM决定*下一步采取什么行动*、*为此行动使用哪个工具*以及*向该工具提供什么输入*。

### 代理循环：思考、行动、观察

代理运作的核心是一个迭代循环：

1. **思考：** 基于初始目标和任何先前步骤，LLM对当前情况进行推理 (inference)。它会考虑自己拥有哪些信息、还需要什么，以及采取何种行动最能接近目标。这种内部“独白”在正确提示时通常由LLM明确生成。
2. **行动：** 基于其思考过程，LLM决定执行某个具体行动。一个行动通常涉及选择一个**工具**并指定该工具的输入。工具是允许代理与外部环境交互、执行计算、获取信息或运行代码的功能或服务。例子包括网络搜索、Python REPL、数据库查询界面或您定义的自定义函数。
3. **观察：** 所选工具根据指定输入执行。此执行的结果被记录为观察结果。此观察可能来自网页的文本内容、计算结果、API调用数据，或者工具失败时的错误消息。
4. **重复：** 观察结果连同原始目标和之前思考-行动-观察步骤的历史一起反馈给LLM。LLM随后开始下一个推理循环（思考），可能根据从观察中获得的新信息选择不同的行动。

此循环持续进行，直到LLM确定原始目标已完全达成，或直到达到预设的停止条件（例如最大步骤数）。

> 代理遵循的迭代过程，通过LLM根据工具执行的观察结果来决定行动。

### 为什么使用代理？

与简单的链相比，代理在处理复杂性和不确定性时提供了显著的优势：

- **适应性：** 它们能根据中间结果动态调整策略。如果某种方法失败或产生了意想不到的信息，代理可以对此进行推理 (inference)并尝试不同的工具或行动。
- **工具集成：** 代理提供了一个自然的框架，用于赋予LLM访问外部能力，从而克服LLM的固有局限性（例如缺乏实时信息或无法进行精确计算）。
- **复杂问题解决：** 它们能将复杂的任务目标分解成一系列更小、可管理并通过工具执行的任务，这模仿了人类解决问题的方式。

本质上，代理帮助LLM从简单的输入-输出转换转变为自主的问题解决者，能够与环境交互以达成目标。链擅长处理可预测的线性工作流，而代理在需要推理、规划和与外部资源交互的场景中表现出色。后续部分将详细介绍如何为代理配备工具并构建它们。

## 参考资料

- [Agents](https://python.langchain.com/v0.2/docs/concepts/#agents) — LangChain (2024)
  Publisher: LangChain
  LangChain 框架中代理的概述和实现细节官方文档。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍 ReAct 模式，该模式将推理与行动结合，构成了迭代代理循环的基础。
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint arXiv:2302.04761; DOI: [10.48550/arXiv.2302.04761](https://doi.org/10.48550/arXiv.2302.04761)
  一篇论文，展示了语言模型如何通过自监督学习来使用外部工具。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  探究 LLM 作为自主代理，能够模拟人类行为、推理和互动。
