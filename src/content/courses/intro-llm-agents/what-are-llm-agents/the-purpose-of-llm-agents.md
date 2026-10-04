---
course: "intro-llm-agents"
chapter: "what-are-llm-agents"
lesson: "the-purpose-of-llm-agents"
sourceId: 6546
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-1-what-are-llm-agents/the-purpose-of-llm-agents"
title: "LLM 智能体的作用"
description: "了解使用 LLM 智能体的原因以及它们可以自动化或辅助的任务类型。"
order: 4
plots: []
sourceHash: "050b57f1590ff6d43de0087c82ff9fd2c531fb86334596511fe1b088a6052488"
sourceCorrections: []
---

LLM 智能体代表了从标准大型语言模型和简单聊天机器人发展而来的新形态。鉴于它们是更精密的系统，它们的主要作用是什么？开发者和研究人员正在构建这些先进的系统，有必要理解他们的动机。

LLM 智能体的主要作用是**在一定程度上自主行动以达成特定目标**。虽然标准 LLM 擅长理解指令并生成类人文本，但智能体在此基础上更进一步。它将 LLM 用作其“大脑”，进行推理 (inference)、规划，然后*执行*数字环境中的操作。想象一个能力很强的助手：一个标准 LLM 就像一个助手，如果你提供所有细节，它能为你起草一封邮件。而 LLM 智能体则更像一个助手，如果给它“下周与 Pat 安排会议”的目标，它可以查看你的日历，查看 Pat 的可用时间（如果可访问），提出时间，并发送邀请，同时处理小的日程冲突。

这种将理解转化为有目的行动的能力正是智能体与众不同之处。以下图表概括性地比较了传统脚本、标准 LLM 和 LLM 智能体如何运作：

> 传统脚本、标准大型语言模型和 LLM 智能体的运作模式，说明了智能体达成目标的迭代过程。

这种行动和适应的能力带来了几个重要用途和益处：

### 复杂多步骤流程的自动化

我们执行的许多任务，尤其是在使用计算机时，不仅仅是单个问题或命令。它们通常涉及一系列步骤、从不同地方收集信息，并在此过程中做出小决定。
例如，考虑规划一次周末短途旅行。这可能包括：

1. 查看潜在目的地的天气预报。
2. 搜索预算内的可用住宿。
3. 寻找有趣的当地活动或餐馆。
4. 预订住宿，或许还有一些活动。
5. 将这些添加到您的日历。

LLM 智能体可以设计来处理此类多步骤流程。它能将总目标（“规划一次周末短途旅行”）分解成更小、更易于管理的任务。然后，它可以使用不同的“工具”（我们稍后将详细介绍），例如用于查询天气的网页搜索、用于检查酒店可用性的 API 或日历集成，来执行这些步骤。这比简单的脚本有了很大的进步，因为简单的脚本需要明确编程每一个步骤和每一种可能的变体。

### 增强适应性并处理不精确的指令

人类在传达目标时，通常不会指定所有细节。我们可能会说“给我找个好的意大利面食谱”，而不会列出所有饮食限制或偏好的烹饪时间。LLM 智能体通过运用其底层 LLM 强大的自然语言理解能力，通常能够解读这些不那么精确的指令，并做出合理推断。

此外，智能体还能表现出一定程度的适应性。如果首次达成目标的尝试失败，或者出现意外情况（例如，网站故障，偏好商品缺货），智能体可能会被编程为尝试替代方法、请求澄清或记录问题，而不是像僵硬的脚本那样简单地停止。

### 与更广的数字环境交互

标准 LLM 主要以文本形式存在。它们接收文本输入并生成文本输出。然而，LLM 智能体被设计为与更宽泛的数字环境交互。这主要通过使用工具来实现。这些工具可以是连接到以下内容的接口：

- 网页搜索引擎（用于查找最新信息）
- 数据库（用于检索或存储特定数据）
- 各种服务的应用程序接口（API）（例如，发送电子邮件、管理文件、与社交媒体交互）
- 代码解释器（用于运行小段代码以进行计算或数据处理）

这种使用工具的能力意味着智能体不仅仅是思考；它还在跨越不同的软件和服务*做*事情。例如，一个智能体可以监控您的电子邮件以获取紧急消息，提取重要信息，然后相应地更新项目管理工具。

### 提供更精细的个性化

通过记住过去的交互（使用一个称为记忆的组件，我们将在后续章节讨论）并从自然语言中理解用户偏好，智能体可以提供更个性化的体验。一个负责总结新闻的智能体可以学习您最感兴趣的话题并优先处理它们。一个帮助编程的智能体可以学习您偏好的编程风格或您常用的库。

### 为什么不直接使用传统代码或脚本？

您可能想知道，既然我们能编写复杂的软件程序和脚本，为什么还需要 LLM 智能体。传统程序要求开发者预见并明确编写每种可能的场景、每个决策点以及任务的每一步的逻辑。对于高度可变、涉及理解人类语言或需要常识性推理 (inference)的任务，这种明确的编程变得极其复杂且通常脆弱；如果发生任何意外，程序可能会崩溃。

LLM 智能体提供了不同的方法。LLM 提供核心的推理、规划和语言理解能力。开发者则专注于：

- 明确定义智能体的总目标。
- 为智能体提供它可能需要的正确工具集。
- 设置基本运作循环（它如何观察、思考和行动）。

智能体在 LLM 的指导下，在找出达成目标的中间步骤方面拥有更大的自主性。

考虑以下任务：“查明比特币当前价格，计算用 500 美元能买多少，并根据近期新闻情绪告诉我，这普遍认为是否是投资的好时机。”

对于这项任务，编写一个**传统脚本**会非常困难：

- 它需要一个硬编码的方式来获取比特币价格（哪个 API？如果它变了怎么办？）。
- 计算数量很容易，但是……
- 它如何评估“近期新闻情绪”？这需要理解文章，而不仅仅是简单的关键词匹配。
- 它会极其僵硬。

一个**LLM 智能体**，配备网页搜索工具和其固有的语言理解能力，可以：

- 使用搜索工具从可靠来源查找比特币当前价格。
- 进行计算。
- 再次使用搜索工具查找关于比特币的近期新闻文章。
- 运用其 LLM 能力分析这些文章中的情绪（积极、消极、中立）。
- 根据这些收集到的信息形成响应。

智能体更灵活，能够更有效地处理任务中模糊和依赖语言的部分。

总而言之，LLM 智能体的作用是创建更具能力、更自主、更灵活的 AI 系统。它们旨在承担不仅仅需要信息处理，还需要决策和与数字环境交互的任务。通过这样做，它们旨在自动化更复杂的流程，提供更智能的辅助，并允许人类委托更宽泛的数字任务，推动我们走向更有用、更集成的 AI 应用。

## 参考资料

- [ReAct: Synergistic Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: 11th International Conference on Learning Representations, ICLR 2023; Publisher: ICLR 2023; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了“推理与行动”（ReAct）范式，这是LLM智能体将推理与特定任务行动相结合以实现自主目标的关键方法。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://api.semanticscholar.org/CorpusID:258040990) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology; Publisher: ACM; DOI: [10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
  描述了一种由LLM驱动的智能体架构，通过结合记忆、规划和反思来模拟可信的人类行为，以实现复杂的长期目标。
- [Artificial Intelligence: A Modern Approach](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEik_QlISYzybqSlwp5WYdfXJp62G55dwoNRlUMkqMp_wFZwI8HQLRhnjyDunzURf312TwSpBvIm9_fE5UUQEmjImVvBpsj5SaHf7mzU4lyUzn-VXuN_nS5Cq6t38pk7mvesf0iBiFIDcJoGoqqDJdB73JdYdsA_cJj2pNbOcTWyiJ6WOW9xLNbGu1Ml_vkY9vn6cmsF4s9eeSHN8-rmxWJCFUHZvNFlEtecrQMm1N6OWbAaBd2Kn7Fe9BsnpEZog==) — Stuart J. Russell, Peter Norvig (2020)
  Publisher: Pearson
  一本权威的人工智能教科书，提供了智能体的基本概念、架构及其与环境交互的基础知识。
