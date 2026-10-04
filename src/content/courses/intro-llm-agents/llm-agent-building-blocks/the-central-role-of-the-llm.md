---
course: "intro-llm-agents"
chapter: "llm-agent-building-blocks"
lesson: "the-central-role-of-the-llm"
sourceId: 6553
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-2-llm-agent-building-blocks/the-central-role-of-the-llm"
title: "LLM 的核心作用"
description: "理解大型语言模型如何作为 LLM 代理的认知引擎。"
order: 1
plots: []
sourceHash: "b5e2454c81e301a1d0964d472fd42dc7b6a30cfc593893b213b40daf917298cc"
sourceCorrections: []
---

每个 LLM 代理的中心是其认知引擎：大型语言模型本身。如果一个代理被设计为智能地执行任务，那么 LLM 就是负责“智能”部分的组件。它是大脑，是解释信息、做出决策和生成响应的中央处理器。

你可能已经知道，LLM 是经过大量文本数据训练的复杂人工智能模型。这种训练使它们具有理解、生成和处理人类语言的显著能力。在代理的背景下，LLM 使用这些能力不仅用于对话，还用于推动行动。这使代理能够摆脱预设响应，表现出更灵活、以目标为导向的行为。

那么，LLM 究竟如何作为这个中心核心发挥作用呢？它的作用可以分解为几个主要职责：

1. **理解输入**：LLM 在代理中的首要任务是理解当前的任务和上下文 (context)。输入可以来自不同来源：

   - 来自您的直接指令（例如，“为下周末查找飞往巴黎的廉价航班”）
   - 工具获取的信息（例如，“天气 API 报告：巴黎，周六，22°C，晴”）
   - 来自代理记忆的数据（例如，“用户之前询问过巴黎的酒店”）
     LLM 处理所有这些传入数据，以形成对情况和需要完成事项的清晰认识。
2. **推理 (inference)和决策**：一旦理解输入，LLM 就会进行某种形式的推理。它“思考”实现目标需要哪些步骤。这可能包括：

   - 将复杂请求分解为更小、更易管理的子任务。
   - 决定是否拥有足够的信息，或者是否需要提出澄清问题。
   - 确定是否需要特定工具来收集更多数据或执行操作（例如使用计算器或搜索引擎）。
   - 制定一系列行动，形成基本计划。
3. **生成用于行动和沟通的输出**：根据其推理，LLM 然后生成一个输出。这个输出不总是对您的直接回答。通常，它是给代理其他部分的指令，或是要传递的响应：

   - **工具调用**：它可能会决定使用搜索引擎工具并生成准确的搜索查询，或者制定命令来与日历 API 交互。
   - **内部独白/思考**：一些代理设计允许 LLM 生成内部“思考”或推理步骤，这些可以被记录或用于指导其下一步行动。这是更高级计划（我们稍后会提到）运作方式的一部分。
   - **面向用户的响应**：LLM 也负责制作代理反馈给您的消息。这可能包括进度更新、请求更多详情（“您正在寻找哪些日期的航班？”）或对您查询的最终答案。

可以将 LLM 想象成一位非常熟练的人类私人助理。您给助理一个任务。他们会听取并理解（输入）。他们会思考完成任务的最佳方法，也许会查找一些信息，使用应用程序或打电话（推理和工具使用）。他们会采取必要的行动，然后向您报告结果或询问更多信息（输出和沟通）。LLM 扮演着助理的角色，但其‘思考’和‘行动’是通过软件指令以及与其他数字组件的交互来完成的。

为代理选择的特定大型语言模型会显著影响其能力。有些 LLM 更擅长编码，另一些擅长创意写作，还有一些则优化用于精确地遵循复杂指令。尽管 LLM 的详细选择是一个更高级的话题，但在当前阶段，了解您的代理的“大脑”可以有不同的类型，每种都有其优点和缺点，这是有帮助的。这种选择将直接影响您的代理理解请求、对问题进行推理以及选择适当行动的程度。

下图说明了 LLM 如何作为代理操作的中心，处理各种输入并产生推动代理行为的输出。

> LLM 充当中央处理中心，接收信息、进行推理并确定代理的下一步行动或响应。

如果没有 LLM 的理解、推理和生成能力，代理将仅仅是一个预定义的脚本。正是这个中心组件使代理能够以更动态、更智能的方式处理任务。当我们了解工具和记忆等其他组成部分时，请记住，LLM 通常是决定如何以及何时使用这些其他部分的组件。

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇论文介绍了Transformer架构，它是大多数现代大型语言模型（LLM）的基础。理解它有助于深入了解LLM如何实现其理解、生成和处理语言的能力，而这些能力正是LLM作为代理认知引擎的核心。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 35; Pages: 24824–24837; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  这篇论文展示了思维链（CoT）提示如何通过生成一系列中间步骤，使大型语言模型能够执行复杂的推理。这项技术对于LLM在代理中扮演“推理和决策”角色至关重要，使其能够在采取行动之前分解问题并制定计划。
