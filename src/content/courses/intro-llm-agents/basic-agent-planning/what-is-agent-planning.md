---
course: "intro-llm-agents"
chapter: "basic-agent-planning"
lesson: "what-is-agent-planning"
sourceId: 6595
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-5-basic-agent-planning/what-is-agent-planning"
title: "什么是智能体规划？"
description: "介绍大语言模型智能体中的规划以及它对复杂任务的重要性。"
order: 1
plots: []
sourceHash: "5397fcad03f2826c473ecf9d4d6d183773d7c2d1e90b8c9328ec52085d0747ce"
sourceCorrections: []
---

设想一下，你希望你的大语言模型 (LLM)智能体完成比回答一个单独问题或执行一个独立动作更复杂的事情。比如，你需要它从多个网页收集资料，然后归纳信息，再用这些归纳结果撰写一封邮件？这并非一步到位，而是一系列步骤。智能体规划正是为此而生。

智能体规划，简言之，就是大语言模型智能体用来确定一系列行动以达成特定目标的过程。具备规划能力的智能体，能够预先考虑，找出必需的步骤，并决定执行它们的顺序，而不是仅仅对即时指令做出反应。这就像在开始一项任务前制定一份待办事项清单或路线图。

### 为什么规划对大语言模型 (LLM)智能体很重要？

没有规划，大语言模型智能体主要局限于单次交互或非常简单的预设序列。规划使其能够处理更复杂的任务：

1. **完成多步目标：** 许多任务需要按逻辑顺序执行多个动作。例如，预订度假可能涉及搜索航班、然后是酒店、再是租车，最后确认所有预订。规划让智能体能够管理这类多步流程。
2. **智能工具使用：** 正如我们在上一章看到的，工具扩展了智能体的能力。规划帮助智能体决定不仅是*是否*需要工具，还有*使用哪个*工具，*何时*使用，以及*传递什么信息*给它，甚至可能将一个工具的输出作为另一个工具的输入。
3. **任务拆解：** 复杂的目标可能让人不知所措。规划让智能体能将一个宏大、艰巨的目标分解成更小、更易于管理的小任务。这使得整个问题更容易解决。

想想你会如何处理组装平板包装家具这样的任务。你不会随意抓取部件就开始拧螺丝。你会查看说明（即规划），确定第一步，执行它，然后进入下一步，依此类推。大语言模型智能体通过规划，尝试一种类似但更简单的方法。大语言模型本身，凭借其推理 (inference)能力，在制定这项规划中起着核心作用，通常通过“思考”所需步骤来完成。

### 智能体规划的核心思想

智能体规划的核心涉及几个基本思路：

- **明确目标：** 智能体需要一个清晰界定的目标。对于这项任务来说，“成功”是怎样的？这是它路线图上的目的地。
- **可用动作：** 智能体实际能*做*什么？这些是它的工具，它调用大语言模型 (LLM)进行推理 (inference)的能力，或其他它可以执行的功能。
- **理解当前情况：** 智能体通常需要考虑当前的状况或已收集到的信息，以便对下一步做出明智的决定。这包括观察和记忆。
- **行动选择策略：** 智能体需要一种方法，从其可用选项中选择最合适的下一步行动，以更接近目标。本章将通过思维链（Chain-of-Thought）和ReAct等方法来介绍这一策略。

当需要LLM智能体完成比回答单一问题或执行单个独立操作更复杂的任务时，例如研究多个网页的主题，总结发现，然后用该摘要起草一封电子邮件，这并非一步就能完成，而是一系列步骤。这就是智能体规划发挥作用的地方。例如，如果智能体的目标是“提供今天关于可再生能源的新闻摘要”，它的规划可能包括：

1. 使用查询“今日可再生能源新闻”访问新闻搜索工具。
2. 收到文章列表。
3. 对于每篇相关文章（或几篇置顶文章）：
   a. 使用网页浏览工具获取文章内容。
   b. 使用大语言模型归纳内容。
4. 将多篇归纳内容整合为一个连贯的概述。
5. 呈现最终归纳。

这个序列就是一个规划。每一步都是一个动作，且顺序很重要。

### 可视化基本规划流程

我们可以从宏观层面描绘这个流程。智能体从一个目标开始，经历规划阶段（通常使用大语言模型 (LLM)的推理 (inference)），然后执行一系列动作，同时观察沿途的结果。

> 一个简化流程图，展示了智能体如何从一个目标出发，通过规划和动作执行，逐步实现该目标。规划过程本身可能是迭代的，会根据新的观察调整规划。

本章将侧重于*基本*的规划方法。更高级的人工智能规划可能涉及复杂的算法，用于在众多可能的动作序列中进行搜索，适应高度变化的环境，并从过去的经验中学习。然而，理解我们在此介绍的基础方法是重要的第一步。

通过掌握什么是规划以及它为何是智能体能力提升的必要组成部分，你已准备好了解这些智能体如何开始制定和执行简单规划。我们将首先研究如何通过引导智能体的推理过程本身来形成一种规划。

## 参考资料

- [Artificial Intelligence: A Modern Approach](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH6nSeco5ueUldw3dWX6V9VP2tj3msrhvO0LhQ-Fn9g6ybdlyeoTK8We3ZNIz2Sw85yTP9EgSkAM3ooifusrUc1-cqAFe-QbleZyzRaEak1cDqsfIDmW2-jZjvhP6ymr3sZBu2Dy30djpUJrMEsB0hS0Z2Be53V1DypZQkKbhKW7C2D8JaScOW0LMDx) — Stuart Russell, Peter Norvig (2020)
  Publisher: Pearson
  一本关于人工智能的综合性教科书，其中一章专门介绍自动化规划及其原理。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEReu2KaqNp9i_2JXNKhclfjPsHjaG9IFi347QzDYnimlMFrFUIys8zjJ38fVdnAm3UozrjBtIsm3S_jpOigzDFyAmdg-8MLwoGF61jVfVBxHnFBqKddgiieA==) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了思维链提示法，这种方法通过生成中间思维过程，使大型语言模型能够进行多步骤推理。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，该框架结合了大型语言模型的推理和行动能力，以解决需要动态行动序列和工具使用的复杂任务。
