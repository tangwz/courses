---
course: "prompt-engineering-llm-application-development"
chapter: "building-applications-llm-frameworks"
lesson: "introduction-to-agents-frameworks"
sourceId: 3050
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks/introduction-to-agents-frameworks"
title: "智能体简介"
description: "构建利用LLM决定行动并与工具交互的智能体。"
order: 5
plots: []
sourceHash: "66e30941eed8af04a388003da4f34ad0588e3b9eff05660fc4fbe69fe2d2ce05"
sourceCorrections: []
---

LLM框架通常使用“链式调用”来执行预设的LLM调用序列及其他操作。然而，许多任务需要比这些固定序列更具动态性的行为。设想一个场景，应用需要回答一个问题，例如“伦敦昨天的天气如何，以及上个月那里报告的下雨天数的平方根是多少？”一个直接的链式调用可能难以处理，因为具体步骤依赖于中间结果（例如，查询天气报告，提取下雨天数，然后进行计算）。对于这类情况，智能体非常有效。

### 智能体究竟是什么？

在LangChain这类LLM框架中，一个**智能体**不只是使用大型语言模型生成文本，更是将其作为推理 (inference)引擎，来决定要执行的一系列动作。与其说它像遵循食谱（链式调用），不如说它更像一个能够利用一系列可用**工具**来找出如何完成目标的助手。

其核心思想是智能体：

1. 接收一个输入或目标。
2. 利用LLM“思考”下一步该怎么做。
3. 这步可能涉及使用特定工具（如网页搜索、计算器或访问你数据库的自定义函数），或者判断自己已有足够信息来给出最终回复。
4. 如果使用了工具，智能体将执行它并观察结果。
5. 结果（或“观察”）会反馈给LLM，以便进行下一轮“思考”。
6. 这个循环会一直重复，直到达成目标。

### 智能体循环

这种决策过程在一个循环中运行，这个循环通常被称为“智能体执行器”或“运行时”。在每一步中，LLM不仅会收到原始目标，还会收到迄今为止已执行动作的历史记录和收到的观察结果，以及它*可以*使用的工具的描述。

以下是智能体执行流程的简化视图：

> 一张图表，说明了智能体执行循环，其中LLM根据输入和之前的步骤决定是使用工具还是生成最终回复。

用于引导LLM推理 (inference)的提示语非常重要。它通常会指导LLM如何一步步思考，列出可用工具及其功能和预期输入，并指定LLM应使用的格式来指示其选择的动作及其输入。ReAct（思考+行动）等技术常作为这些提示语的基础，鼓励LLM在选择动作前明确说明其推理过程。

### 为什么要使用智能体？

智能体在处理某些类型的问题时，比简单的链式调用具有显著优势：

- **灵活性：** 它们可以根据中间结果调整方法，按需选择不同的工具或动作序列。
- **复杂问题解决：** 它们能将复杂目标分解为更小、更易于管理的小步，并在此过程中使用工具收集信息或执行计算。
- **交互性：** 它们使LLM能够以结构化的方式与外部系统（数据库、API、文件系统）交互，运用其预训练 (pre-training)知识来发挥作用。

然而，这种动态特性也意味着智能体可能不如链式调用那样可预测。它们的行为很大程度上取决于LLM的质量、提示语的清晰度以及工具的可靠性。调试有时需要追溯智能体在多个步骤中的思考过程。此外，如果推理 (inference)引导不当或工具产生意外错误，智能体也有可能陷入循环或未能完成任务。另外，由于智能体在其执行循环中经常对LLM进行多次调用，因此与单次链式调用相比，它们可能产生更高的运营成本。

在下一节中，我们将详细了解如何定义和整合这些赋予智能体行动能力的“工具”。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: International Conference on Learning Representations; Publisher: International Conference on Learning Representations; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct提示策略，该策略结合了推理和行动步骤，对于LLM代理选择行动和与工具交互很重要。
- [LangChain Agents Documentation](https://python.langchain.com/docs/) — LangChain (2024)
  Publisher: LangChain
  LangChain框架中实现代理的官方文档，提供了代理类型、工具和执行的实践示例和详细信息。
