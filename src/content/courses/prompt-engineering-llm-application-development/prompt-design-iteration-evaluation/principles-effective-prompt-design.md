---
course: "prompt-engineering-llm-application-development"
chapter: "prompt-design-iteration-evaluation"
lesson: "principles-effective-prompt-design"
sourceId: 2998
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-3-prompt-design-iteration-evaluation/principles-effective-prompt-design"
title: "有效提示词设计的方法"
description: "创建清晰、简洁、有效提示词的指南。"
order: 1
plots: []
sourceHash: "2804f63653efc5bbe5251c57ab312e937396a67675dd5c0cc4e48f71b0b275cb"
sourceCorrections: []
---

有效提示词 (prompt)设计需要理解提示词机制和技巧，并遵循指导原则。构建提示词是一项工程任务；它需要周密的构造，以从大型语言模型（LLM）获得可预测且有用的结果。持续应用这些原则将优化你的开发流程，减少试错，并使LLM在你的应用中表现更稳定。

将提示词看作与LLM功能的主要交互方式。就像设计一个好的用户界面（UI）或应用程序接口（API）一样，设计一个好的提示词需要考虑清晰度、上下文 (context)以及预期结果。以下是指导你创建提示词的基本方法：

### 清晰具体

模糊性是有效提示词 (prompt)的障碍。LLM尽管复杂，但无法理解你的意图。模糊的指令会导致模糊或不可预测的输出。

- **直接阐明任务：** 清晰说明你希望LLM做什么。与其说“谈论这份文档”，不如尝试说“总结这份附带研究论文的重要发现。”
- **使用动作动词：** 用明确的动作动词开始指令，如“总结”、“翻译”、“提取”、“生成”、“分类”、“重写”、“解释”等。
- **分解复杂任务：** 如果任务包含多个步骤或组成部分，请在提示词中进行分解。例如，与其说“分析这份客户反馈并提出改进建议”，不如尝试：“1. 识别以下客户反馈中表达的主要情绪（积极、消极、中立）。 2. 提取提到的具体产品特点。 3. 根据反馈提出一项具体的改进建议。”
- **尽可能避免否定表达：** 尽管LLM可以处理否定，但正面表达通常更清晰。与其说“不要包含技术术语”，不如考虑“用非专业人士能理解的简单术语解释此内容。”

考虑以下差异：

- **效果不佳：** `告诉我关于机器学习。` (过于宽泛，范围不清)
- **更有效：** `解释监督式机器学习的理念，包括其目标和两种常见算法，并使用一个适合数据科学初学者的类比。` (任务明确，范围确定，指定目标受众)

### 提供足够的上下文 (context)

LLM仅根据提示词 (prompt)中提供的信息（及其预先存在的训练数据）运行。它们缺乏对过去交互的记忆（除非明确管理，参见第5章），并且除非特别提供，否则无法获取外部信息（参见第6章）。因此，包含所有必要的上下文是不可或缺的。

> 一个图示，显示了构成有效提示词结构的主要组成部分。

### 定义预期输出

明确说明你希望输出如何组织是应用开发中最有影响力的原则之一。这使得LLM的回复更容易解析并整合到下游流程中。

- **指定格式：** 请求特定格式，如JSON、Markdown、HTML、项目符号列表、编号列表、逗号分隔值等。
- **提供示例（少样本原则）：** 如第2章所示，在提示词 (prompt)中提供所需输出格式的示例（少样本学习 (few-shot learning)）可以大幅提升一致性。
- **定义结构元素：** 如果请求JSON等结构化数据，请指定键、值类型和嵌套结构。“生成一个JSON对象，包含键 'product\_name' (字符串)、'feature\_summary' (字符串，最多50字) 和 'tags' (字符串列表)。”
- **引导语气和风格：** 指定所需的写作风格（例如，“正式”、“随意”、“技术性”、“有说服力”）或语气（例如，“乐观”、“担忧”、“中立”）。
- **效果不佳：** `从报告中提取重要细节。`
- **更有效：** `从以下报告中提取项目名称、完成日期和主要成果。以JSON对象形式呈现输出，键分别为 "projectName"、"completionDate" (YYYY-MM-DD格式) 和 "primaryOutcome"。`

### 分配角色（角色设定）

通过为LLM分配角色或人设来引导它，可以有效塑造其回复的风格、语气和知识范围。这在第2章中作为一种技巧引入，但它是一种强大的设计方法。

- **示例：** `扮演一位资深软件架构师。审查以下系统设计方案，并找出潜在的可扩展性瓶颈。` 这鼓励LLM采纳特定视角并运用相关专业知识。

### 力求简洁（但不能牺牲清晰度）

虽然提供足够的细节很重要，但过长或冗余的提示词 (prompt)有时会使LLM混淆或超出上下文 (context)窗口限制（本章后续会讨论）。

- **消除冗余：** 避免不必要地重复指令或上下文。
- **直接明了：** 直奔主题。虽然礼貌无害，但过多的寒暄通常没有必要。
- **平衡细节与简洁：** 目标是使用尽可能少的词语，在不牺牲必要信息或指令的前提下，做到所需程度的清晰和具体。

系统地应用这些方法是设计不仅功能完善，而且稳定高效的提示词的依据。请记住，提示词设计通常是一个迭代过程。这些方法提供了一个良好的起点，并指导你的优化工作，我们将在后续章节中详细说明。

## 参考资料

- [Prompt Engineering Guide](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFDdypUOWbeJUsnFXLuzm78ea2nemOdjTsOVR6KHaGj-dnNWFTmB3piCl5GJhAz4Cz8g0rSLE2D7eAW8dvR_UI4NqXY7jRh32MiVG0MvJDHNzT8kCqenZXO) — OpenAI (2024)
  Publisher: OpenAI
  提供设计有效提示的指导，包括清晰性、上下文、少样本示例和结构化输出。
- [Prompting Large Language Models: A Comprehensive Review](https://openreview.net/forum?id=VfTf1p-sI9) — Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Ohbayashi, Graham Neubig (2023)
  Journal: Transactions on Machine Learning Research; Publisher: OpenReview.net; DOI: [10.48550/arXiv.2307.11796](https://doi.org/10.48550/arXiv.2307.11796)
  对提示工程技术和原则进行了系统回顾，说明如何为各种任务构建提示。
- [Practical Prompt Engineering: Transform your everyday communication and interaction with Large Language Models](https://www.packtpub.com/product/practical-prompt-engineering/) — Sumit Pal (2023)
  Publisher: Packt Publishing
  提供制作有效提示的实用指导和示例，详细说明了各种提示设计原则。
- [Prompt Engineering Guide](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFojqnkTZhWb8D_2A3eLQn9v8fE_hICS-ax7qyLYV7lFMsrgoOKQpMb9BXrjYjARrkVYisHB2zB3VMDj2lq2T26hrBoORcgpXEc6rcwJGXSyYcZz1FuIQhNRKAThPsyQlDgckyRHPKroNCLnjZKYSv4oXmeymB6) — Google Cloud (2024)
  Publisher: Google Cloud
  提供构建提示的最佳实践建议，以确保生成式AI模型输出的清晰性、具体性和符合预期。
