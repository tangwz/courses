# LLM Agent 到底是什么？

来源：[原文](https://apxml.com/zh/courses/intro-llm-agents/chapter-1-what-are-llm-agents/what-exactly-is-an-llm-agent)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大型语言模型（LLM）不仅仅是复杂的文本预测器；它们正在成为能够*行动*的系统的中心。虽然标准的LLM或简单的聊天机器人主要通过根据您的输入生成文本来进行对话，但LLM代理则更进一步。但这个“更进一步”到底意味着什么？LLM代理究竟是什么？

就其本质而言，一个**LLM代理**是一个旨在达成特定目标的系统。它使用LLM作为其主要的推理 (inference)引擎，就像大脑一样，用于理解指令、做出决策和规划行动。与仅仅向LLM提问并获得文本回复不同，代理的设计是为了与周围环境互动以完成任务。

可以这样理解：

- **标准的LLM互动**就像与一个知识渊博但只能口头回应的人交谈。你问，他答。
- **LLM代理**则像那个知识渊博的人，同时拥有手、工具和一项任务。他们不仅仅是说话；他们会*行动*。

### LLM作为代理的核心处理器

大型语言模型是赋予代理智能的主要组成部分。当代理获得一项任务时，例如“找到我附近现在营业且意大利面评价不错的前三家意大利餐厅”，LLM不会仅仅尝试从其训练数据（可能已过时）中回忆这些信息。相反，它会*推理 (inference)*如何达成这个目标。

它可能会将任务分解为：

1. 需要知道当前位置（如果未提供）。
2. 需要搜索意大利餐厅。
3. 需要按“现在营业”进行筛选。
4. 需要专门查看意大利面的评价。
5. 需要选择前三家。

### 使用工具采取行动

这种推理 (inference)会促使行动。代理通常配备有**工具**，这实际是函数或与其他服务的连接，让它们能够与环境互动。以我们的餐厅为例，工具可能包括：

- 一个用于获取您设备当前位置的工具（需经许可）。
- 一个用于执行网络搜索的工具（例如，“[位置]附近的意大利餐厅”）。
- 一个用于解析搜索结果并提取营业时间、评论摘要等信息的工具。

LLM决定计划当前步骤中*哪个*工具是合适的，为该工具制定正确的输入（例如，搜索查询），然后解释工具的输出以决定下一步行动。如果某个工具失败或返回意外信息，LLM可以推理如何进行，也许是尝试不同的工具或调整其方法。

### 观察-思考-行动循环

许多代理都遵循一个被称为“观察、思考、行动”的基本循环：

1. **观察**：代理收集有关其当前状态及其环境的信息。这可以是用户的初始请求、传感器数据，或先前使用的工具的输出。
2. **思考**：LLM处理这些观察结果，考虑整体目标，并决定下一个最佳行动。这是LLM推理 (inference)能力展现之处。
3. **行动**：代理执行所选的行动，通常是使用其工具之一。此行动会改变环境状态或代理的内部状态。

这个循环会重复，直到目标达成，或代理确定无法达成。

下面是一个描绘此一般流程的图表：

> 此图描绘了代理如何接收用户目标，使用其LLM“大脑”决定行动，利用工具与环境互动，然后观察结果以指导其下一步。

### LLM代理的特点

所以，总结来说，LLM代理的特点是：

- **目标驱动**：它有一个明确的目标，并努力达成。
- **LLM驱动推理 (inference)**：它使用LLM来理解、规划和决策。
- **工具使用**：它可以运用各种工具与外部系统互动或完成专门任务。
- **互动循环**：它经常在感知环境、思考和行动的循环中运作。
- **自主程度**：一旦给定目标，它就可以采取多个步骤来达成，而无需每一步都进行人工干预。这不意味着它有意识或完全独立，而是指它可以根据其程序和LLM的指导执行一系列操作。

正是LLM推理能力与采取行动并与环境互动的能力相结合，才真正定义了LLM代理。它是一个从简单的文本生成转变为积极参与完成任务的系统。

## 参考资料

- [Toolformer: Language Models That Can Use Tools](https://arxiv.org/abs/2302.04761) — Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
  Journal: arXiv preprint
  提出了一种训练语言模型使用外部工具（如搜索引擎、计算器）的方法，通过学习对API调用进行自监督，这是大型语言模型代理执行行动的基础。
- [Function calling and the OpenAI API](https://openai.com/blog/function-calling-and-the-openai-api) — OpenAI (2023)
  Publisher: OpenAI Blog
  OpenAI官方博客文章，详细介绍了其模型如何可靠地调用外部函数，这是大型语言模型代理利用工具与环境交互的实际实现。
- [A Survey on Large Language Model Based Autonomous Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH2iaN60vu-7GObMVjAxiniXOPNqYPP962TYG9Auho-eIVOkyyjbFrM-0h9Y5IbnrNt7aZmpepVuHfi9BHiVovCG2XGzVeNuXmAnUV4PXDeeZiQc_URcFRf-g==) — Lei Wang, Chen Ma, Xueyang Feng, Zhiyuan Liu, Maosong Sun, Lei Hou (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  对基于大型语言模型的自主代理领域进行了全面概述，涵盖了其架构、组成部分和应用，是很好的入门资料。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint
  介绍了思维链（CoT）提示技术，该技术使大型语言模型能够通过生成中间步骤来执行复杂推理，这对于代理规划和思考任务的能力至关重要。

---

[上一节](02-%E4%B8%8D%E5%86%8D%E5%B1%80%E9%99%90%E4%BA%8E%E7%AE%80%E5%8D%95%E8%81%8A%E5%A4%A9%E6%9C%BA%E5%99%A8%E4%BA%BA.md) · [下一节](04-LLM%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%9A%84%E4%BD%9C%E7%94%A8.md)
