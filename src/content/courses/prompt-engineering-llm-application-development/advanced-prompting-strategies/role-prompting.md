---
course: "prompt-engineering-llm-application-development"
chapter: "advanced-prompting-strategies"
lesson: "role-prompting"
sourceId: 2982
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-2-advanced-prompting-strategies/role-prompting"
title: "角色设定提示"
description: "在提示中为大型语言模型设定身份或角色。"
order: 4
plots: []
sourceHash: "92e3e19dcfff6685b864c5f62796c43672f00d9499b36ad385b44a94b13b7fd2"
sourceCorrections: []
---

**角色设定提示**侧重于指导大型语言模型（LLM）在执行任务时*扮演何种角色*，这与主要告诉模型*做什么*任务的通用指令有所不同。通过指定特定身份、职业或人物，这项技术可以显著影响生成输出的语气、风格、细节程度和视角。它利用了LLM训练时所用的丰富文本数据，包括对话、角色刻画和专家撰写内容，使其能够有效模拟不同视角。

你可以将其看作是为角色选角。不仅仅是提供台词（指令），你还在告诉模型：“你是一位经验丰富的侦探”，或“你是一位充满热情的科学传播者。”这种背景信息有助于模型生成与预期角色更一致的回答。

### 为何使用角色设定提示？

设定角色相比简单指令有多种益处：

1. **语气与风格控制：** 你可以指导模型采用特定的语气，如正式、非正式、技术性、幽默、富有同情心或批判性，使输出符合你的应用需求。
2. **领域专业模拟：** 要求模型扮演某个领域的专家（例如：“你是一名专注于分布式系统的高级软件工程师”）可以帮助集中其知识，并在该领域内生成更贴切、更详细的解释或方案。
3. **视角采纳：** 从独特的视角生成内容，这对于创意写作、模拟用户角色或探讨论点不同方面非常有用。
4. **提高任务遵循度：** 定义明确的角色通常暗示了关于格式、细节水平或预期知识的某些约定，隐含地指导模型生成更合适的回答。

### 编写有效的角色设定提示

好的角色设定提示通常包含：

- **明确角色定义：** 清楚说明角色身份。提示的开头可以使用“扮演…”、“你是一个…”、“承担…的角色”等短语。务求具体。“扮演一位专攻18世纪法国历史的历史学家”优于“扮演一位历史学家”。
- **背景信息（可选但建议）：** 简要描述情境或角色互动对象。例如，“…向高中生解释量子物理”或“…为新产品发布撰写营销邮件。”
- **任务指令：** 清楚说明指定角色需要完成什么。

这是一个简单的比较：

**情境：** 解释 API 速率限制的理念。

**提示1：基本指令**

```
解释 API 速率限制。
```

**提示2：角色设定提示**

```
You are a technical writer creating documentation for developers. Explain API rate limiting clearly and concisely, outlining why it's necessary and common ways it's implemented.
```

第二个提示通过定义角色（“技术文档撰写者”）和受众（“开发者”），更有可能促使模型生成结构更严谨、技术上更准确且使用恰当术语的回答，而第一个提示则更为泛化。

### 角色设定提示的实例

让我们看看不同角色如何影响相同核心任务（“总结容器化的好处”）的输出：

**示例1：角色 - DevOps 工程师**

```prompt
Act as a DevOps Engineer explaining the benefits of containerization to a development team during a technical briefing. Focus on deployment speed, consistency, and resource utilization.

好的，各位。我们来谈谈容器。主要优点是速度和一致性。我们将应用程序及其依赖项一起打包，这样在你的机器上能运行的，在预生产和生产环境也能完全一样地运行。不再有“在我的机器上可以运行”的问题。这大大加快了部署周期。此外，容器很轻量。与传统虚拟机相比，我们可以在相同的硬件上运行更多的应用程序，从而提高资源使用率并节省成本。它使我们的环境标准化，让一切都更可预测、更易于管理。
```

**示例2：角色 - 营销经理**

```prompt
You are a Marketing Manager writing a short blog post snippet highlighting the business advantages of containerization for non-technical executives. Emphasize efficiency, cost savings, and agility.

通过容器化，获取更高水平的效率和敏捷性！想象一下，前所未有地更快部署应用程序，并确保它们在任何地方稳定运行。容器可以简化操作，通过优化服务器使用来降低基础设施成本。这项技术帮助您的业务更快创新，迅速响应市场变化，并最终获得竞争优势。这关乎以更少资源、更快速度完成更多工作。
```

请注意，即使核心主题相同，指定角色如何塑造回答的语言、侧重点和技术细节水平。

### 注意事项和最佳实践

- **具体性很重要：** 模糊的角色导致模糊的输出。“专家”不如“理论物理专家”有效。
- **与其他技术结合使用：** 角色设定提示与指令遵循、少样本示例以及结构化输出请求搭配良好。你可以告诉角色*如何*格式化其回答。
- **避免刻板印象：** 定义角色时要谨慎。确保它们不会无意中延续有害的刻板印象。基于技能、知识或功能来定义角色，而非人口统计特征。
- **测试与迭代：** 角色设定提示的效力可能因LLM而异。尝试不同的措辞和详细程度，以找到最适合你的特定模型和任务的方法。
- **角色一致性：** 在较长的互动中，模型可能偶尔偏离指定角色。定期强化角色设定或使用管理对话历史的框架有助于保持一致性。

角色设定提示是你的提示工程 (prompt engineering)工具集中的一项有效技术。通过策略性定义LLM应*扮演何种角色*，你可以对其输出获得更精细的控制，使得能够创建更具针对性、更恰当、更有效的回答，适用于广泛的应用场景。

## 参考资料

- [Large Language Models as Universal Role-Players](https://arxiv.org/abs/2307.03920) — Bo Wang, A. K. Qin, Sajjad Shafiei, Hussein Dia, Adriana-Simona Mihaita, Hanna Grzybowska (2023)
  Journal: 2023 IEEE International Joint Conference on Neural Networks (IJCNN); Publisher: IEEE; DOI: [https://doi.org/10.48550/arXiv.2307.03920](https://doi.org/10.48550/arXiv.2307.03920)
  这份研究论文探讨了大型语言模型有效扮演和维持各种角色或人物的能力，直接支持了角色提示的概念。
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI 的官方指南提供了设计有效提示的实用策略，包括关于如何为模型指定人物角色以供其采纳的建议。
- [Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) — Isa Fulford, Andrew Ng (2023)
  Publisher: DeepLearning.AI
  这门热门在线课程提供了构建提示的实践方法，其中包含专门章节，介绍如何为LLM指定角色或人物，以指导其在不同应用中的响应。
