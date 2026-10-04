---
course: "building-llm-apps-with-langchain"
chapter: "developing-autonomous-agents"
lesson: "introduction-to-agents-and-tools"
sourceId: 7435
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-6-developing-autonomous-agents/introduction-to-agents-and-tools"
title: "代理和工具介绍"
description: "了解代理的主要思想，它们使用大型语言模型作为推理引擎来决定采取哪些行动。"
order: 1
plots: []
sourceHash: "c2ff9ff70134ed78d6af6fa340474953ecfbc46fdbec857b94f9435e05c74faa"
sourceCorrections: []
---

工作流可以预先确定执行路径来构建。例如，标准的链或`RunnableSequence`遵循固定的、开发者定义的步骤序列。这种方法对于结构化任务是有效的，但它缺乏处理事先不知道解决方案路径的问题的灵活性。为了解决这个问题，引入了**代理**。

代理不仅使用大型语言模型来处理文本，还将其作为决策的推理 (inference)引擎。代理不会遵循僵化的序列，而是根据用户目标和可用资源动态选择一系列要采取的行动。它在一个循环中运行，不断观察其环境，思考最佳的下一步行动，并执行它，直到最初的目标达成。

### 代理循环：思考、行动、观察

每个代理的核心都是一个推理 (inference)循环。在现代LangChain开发中，这通常通过状态图（经由LangGraph）来实现，它协调了流程。这个循环使代理能够规划和执行可能需要获取外部信息或进行计算的多步任务。

该过程通常遵循以下步骤：

1. **目标**：代理从用户那里收到一个高层次的目标（例如，“微软现任CEO是谁？他们上任时的股价是多少？”）。
2. **思考**：大型语言模型作为代理的大脑，分析目标。它考虑可用的工具并制定计划。它可能会思考：“要回答这个问题，我首先需要找到CEO是谁以及他们何时上任。一个网页搜索工具会很合适。”
3. **行动**：大型语言模型输出要采取的具体行动，指明使用哪个工具以及提供什么输入。例如，它可能决定使用`Search`工具，输入“微软现任CEO”。
4. **观察**：代理运行时调用带有给定输入的指定工具。工具运行（例如，执行网页搜索）并返回一个结果。这个结果称为观察。
5. **重复**：观察结果被传回大型语言模型。大型语言模型现在有了新信息并重新评估其计划。它可能会思考：“我发现萨蒂亚·纳德拉于2014年2月4日成为CEO。现在我需要找到那天的数据。” 然后它选择另一个行动，也许是使用一个金融数据工具。

这个思考、行动和观察的循环会重复，直到大型语言模型确定它已收集到足够的信息来充分回答用户的原始目标。届时，它会中断循环并生成最终回应。

> 代理遵循的推理循环。大型语言模型反复选择一个工具，执行它，并使用结果观察来指导其下一步决策，直到目标达成。

### 工具：代理的能力

代理的能力受限于它能使用的工具。一个**工具**是一个接口，它允许代理与外部环境进行互动。在LangChain中，工具本质上是一个具有特定用途的函数，它附带一个大型语言模型可以理解的名称和描述。

工具的例子包括：

- **网页搜索**：一个查询如Google或DuckDuckGo等搜索引擎的工具。
- **Python REPL**：一个可以执行任意Python代码的工具。
- **数据库查询**：一个对数据库运行SQL查询的工具。
- **API调用器**：一个用于向外部API发出请求的通用工具。

工具的**核心所在**是其**描述**。大型语言模型不知道工具的代码如何工作；它完全依赖工具的描述来决定何时以及如何使用它。一个写得好的描述是具体的，并清楚地解释了工具做什么、预期的输入是什么以及它返回什么。

例如，考虑一个公司内部API的自定义工具：

- **好的描述**：`当你需要查找客户订单的当前发货状态时有用。输入应为一个有效的订单ID。`
- **差的描述**：`运行查询。`

第一个描述为大型语言模型提供了关于工具目的和输入要求的明确指示，使其能够做出明智的决定。第二个描述过于模糊，很可能会导致代理要么忽略该工具，要么错误地使用它。通过为代理配备一套描述良好的工具，你赋予它执行超出大型语言模型自身内置知识的复杂、多步任务的能力。

## 参考资料

- [Agents](https://www.langchain.com/agents) — LangChain Team (2024)
  Publisher: LangChain
  LangChain框架中开发和使用智能体及工具的官方文档，为实现智能体循环提供了实用指导和示例。
- [Tool use](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQExbCw8PYfkDlckFTnG4_PzJJ1kcYl-WdTERZbDg-6MyWLt17Z44L7_KDQoQrVlxqbZ1WMsQOjeQ5p_2ZcXtYKKU8c8Xxo6j1ty1Ot8g1aDFPWXfqaSNgTCDD2-LZ_8kjwc11JcUKO29vu3mNCVvsvi2ay2) — OpenAI (2024)
  Publisher: OpenAI
  解释了OpenAI模型如何通过生成结构化JSON调用来使用外部工具，这是LLM智能体与环境交互的基本机制。
