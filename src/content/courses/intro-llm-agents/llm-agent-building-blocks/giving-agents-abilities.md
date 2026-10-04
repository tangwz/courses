---
course: "intro-llm-agents"
chapter: "llm-agent-building-blocks"
lesson: "giving-agents-abilities"
sourceId: 6556
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-2-llm-agent-building-blocks/giving-agents-abilities"
title: "为Agent赋予功能"
description: "了解工具如何让LLM Agent与外部系统沟通并执行多种任务。"
order: 3
plots: []
sourceHash: "8987860a68bf5a24b88dcae0470f7d19db4e7bf7351237026ee0e1cd49c86fc5"
sourceCorrections: []
---

尽管大型语言模型（LLM）作为Agent令人印象深刻的认知引擎，擅长理解和生成文本，但在与其文本知识外部的交互方面存在固有限制。LLM本身无法查询当前股价，无法可靠地进行精确数学计算，也无法访问您的个人日历。它是一个强大的大脑，但它需要一种方法在其训练数据之外进行行动和感知。

这就是工具发挥作用的地方。可以将工具视为专用扩展，赋予您的Agent“功能”，使其能够执行LLM自身无法完成的操作或获取信息。这些工具是程序或与外部服务的连接，Agent可以根据需要调用它们。

如果LLM是Agent的大脑，那么工具就像它的手、眼睛和专用仪器。大脑可以决定它需要知道当前温度，但它需要一个温度计（一个工具）来实际测量。同样，Agent可能会决定它需要计算抵押贷款支付；它将使用计算器工具来确保准确性，而不是仅仅依赖LLM有时近似的数学技能。

### Agent如何使用工具

Agent使用工具的过程通常遵循明确的模式，构成了其操作循环的重要部分：

1. **判定**：根据当前任务和指令，LLM判定它需要一种其内部知识无法提供的外部能力。它识别出某个特定工具会有帮助。
2. **选择与输入**：LLM从已配置的预设可用工具列表中选择最合适的工具。然后，它准备该工具所需的输入（或参数 (parameter)）。例如，如果它需要搜索网页，输入将是搜索请求。
3. **执行**：Agent使用准备好的输入调用选定的工具。该工具，本质上是一段代码，然后执行其特定功能。这可能涉及进行API调用、运行计算或查询数据库。
4. **观察**：工具将结果（一个观察值）返回给Agent。这可能是一段数据、动作确认或错误消息。
5. **整合**：LLM处理这个结果。新信息融入Agent对当前情况的理解中，LLM使用它来决定下一步或形成最终回复。

这种交互可以可视化为以下流程：

> Agent的LLM核心分析任务，决定使用工具，准备输入，调用工具，然后整合工具的输出以继续其工作。

### Agent工具的常见例子

Agent可以使用的工具范围很宽，主要受限于可通过编程或API访问的内容。以下是一些常见类别：

- **计算器工具**：用于精确的数学运算。虽然LLM可以执行简单算术，但在更复杂的计算或需要高精度时可能会出错。`Calculator`工具可确保准确性。例如，如果任务是“计算半径为7.5单位的圆的面积”，Agent可以将“radius = 7.5”传递给设计用于计算圆面积的计算器工具。
- **搜索工具**：从互联网获取最新信息。LLM基于其训练数据有知识截止日期。`Search`工具，通常通过连接到网页搜索引擎的API（如Google搜索或Bing搜索），允许Agent查找实时信息、新闻或LLM训练后发布的数据。如果用户询问“伦敦现在天气如何？”，Agent将使用搜索工具或专门的天气API工具。
- **代码执行工具**：运行小段代码，通常在像Python解释器这样的安全环境中。这对于需要特定逻辑、数据处理或复杂算法的任务很有用，这些任务比自然语言提示更容易用代码表达。例如，Agent可以使用`Code Interpreter`来对列表中的项目进行排序、解析复杂数据结构或生成图表。
- **API交互工具**：许多应用程序和服务提供应用程序编程接口（API），允许程序化交互。Agent可以使用充当这些API客户端的工具。

  - `Calendar API`工具可以允许Agent检查您的日程或创建新事件。
  - `Email API`工具可以使Agent代表您起草或发送电子邮件。
  - `Database Query`工具可以连接到特定数据库，并根据Agent的需求获取或更新记录。
  - `File System`工具可以允许Agent读写本地文件（需具备适当权限和安全措施）。
- **自定义工具**：对于专门任务，开发人员可以创建自定义工具。例如，一个旨在帮助电子商务的Agent可能有一个工具来检查特定公司数据库中的产品库存，或者一个客户服务Agent可能有一个工具来查询客户订单历史。

通过为Agent配备此类工具，我们大大扩展了它们的功能，使其超出单纯的文本生成。工具允许Agent收集最新信息、执行精确计算、与其他软件系统沟通，并最终采取更有意义的行动来完成被赋予的目标。理解如何定义、集成和管理工具是构建有效LLM Agent的一个重要方面，我们将在第4章“为Agent配备工具”中更详细地讨论。目前，需要理解的是，工具是Agent通向更广泛功能集的桥梁。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHwZp3f4rvcdfnsntRrfgs1hj_4LjoginuS7EXueSzfwvgQkT-94oB--VyPlpmkzOwNaETjjJfzSpMiKhlWCbommPwNsRk5wkGN-R7azeuLMFPbp9vKk88L5MY/) — Yao, Shunyu; Zhao, Jeffrey; Yu, Dian; Du, Nan; Shafran, Izhak; Narasimhan, Karthik; Cao, Yuan (2023)
  Journal: International Conference on Learning Representations (ICLR)
  这项研究介绍了ReAct范式，它将语言模型中的推理（思考）与行动（行动、观察）相结合，使其能够以类似于本文描述的交互过程有效使用工具。
- [A Survey on Large Language Model Based Autonomous Agents](https://arxiv.org/abs/2309.17242) — Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhi-Yuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (2023)
  Journal: arXiv preprint
  本综述对基于LLM的自主智能体进行了广泛概述，其中一个章节讨论了整合外部工具以增强智能体能力的各种方法和挑战。
- [Tools](https://python.langchain.com/docs/modules/agents/tools/) — LangChain (2024)
  Publisher: LangChain
  LangChain工具模块的官方文档，说明了如何为LLM智能体定义和使用外部工具，并提供了各种常见工具类型的实际示例。
