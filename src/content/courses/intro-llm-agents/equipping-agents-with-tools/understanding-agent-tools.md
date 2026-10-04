---
course: "intro-llm-agents"
chapter: "equipping-agents-with-tools"
lesson: "understanding-agent-tools"
sourceId: 6581
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-4-equipping-agents-with-tools/understanding-agent-tools"
title: "理解代理工具"
description: "对LLM代理中工具的定义及其功能的说明。"
order: 1
plots: []
sourceHash: "539b54993a5e7dfe7a56cb33cd7b68c31038a778acd80127401e32809c3d50c5"
sourceCorrections: []
---

大型语言模型（LLM）是代理的核心，能够理解、推理 (inference)并生成文本。然而，LLM本身就像一个空房间里聪明的头脑；它知道很多，但不能直接与外部系统互动，或执行复杂的计算、查询今日股价等专门任务。这时，工具就派上用场了。

那么，在LLM代理的背景下，工具究竟是什么呢？
可以将工具看作是LLM代理可以调用的专门助手或扩展，用于执行特定操作或获取特定类型的信息。就像你用计算器做数学题，用搜索引擎找资料，或用特定应用查天气一样，LLM代理使用工具来增强其能力。这些工具本质上是代理可以调用的函数或API（应用程序编程接口）。

每种工具都设计用于特定目的。例如：

- 一个**计算器工具**可以执行数学运算。LLM可能理解“257乘以389”是什么意思，但它会将实际计算委托给计算器工具，以获得精确结果。
- 一个**搜索工具**可以访问搜索引擎（如Google或Bing），在互联网上查找当前信息。这很有用，因为LLM的知识通常仅限于其训练数据，不包含实时事件。
- 一个**数据库工具**可以查询数据库以获取或存储特定记录。
- 一个**代码执行工具**可以运行一段代码（如Python）并返回输出。这使得代理能够执行复杂的数据操作或使用现有软件库。

当LLM代理遇到任务中需要自身能力之外的部分时，其推理部分（即LLM本身）会判断需要一个工具。代理随后向合适的工具发出请求，发送出去，并等待工具执行并返回结果。代理会将此结果纳入其持续的思考过程，以完成主要任务。

> 此图显示了LLM代理如何识别出对外部能力的需求，然后调用一个特定工具，该工具再与外部系统或信息源进行互动。工具处理此互动并将观察结果返回给代理。

本质上，工具使得LLM代理能够：

1. **获取最新信息：** 克服LLM的知识时效性限制。
2. **执行精确计算：** 确保数学或数据处理等任务的准确性。
3. **与外部系统互动：** 连接到数据库、API或其他软件。
4. **在外部环境中执行操作：** 代理可以使用工具发送电子邮件或更新日历。

没有工具，LLM代理主要局限于其已有的知识和文本生成能力。有了工具，它就成为了一个更通用和实用的系统，能够执行更广泛的有用任务。在接下来的章节中，我们将了解这项能力为何如此重要，以及如何实际构建和集成这些工具。

## 参考资料

- [ReAct: Synergistic Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint
  介绍了一种通用框架，使大型语言模型能够交错进行推理和行动步骤，从而通过利用搜索引擎等外部工具解决复杂任务。
- [LangChain Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEeYmYkfNsmqN16Mevdx5-RkZ3Hi6HIMqScp8CP-w1vdolFC_QKP517B2eJtAXcDyzg7qUUn_5pgsIcR81JbXS5GcoPzDsRbJUq7NEuEWvGTcaQTJzJmzThlHoEmb0MMIADoVn9XCpbpnQgvb-NdH49bHbRZg==) — LangChain Team (2024)
  Publisher: LangChain
  官方文档，解释了大型语言模型代理的概念和实际实现、其组件（大型语言模型、工具、记忆）以及如何使用LangChain框架构建它们。
