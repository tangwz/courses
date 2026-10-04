---
course: "intro-llm-agents"
chapter: "llm-agent-building-blocks"
lesson: "the-agents-operational-loop"
sourceId: 6562
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-2-llm-agent-building-blocks/the-agents-operational-loop"
title: "代理的运行循环"
description: "了解代理遵循的观察、思考和行动的基本循环。"
order: 6
plots: []
sourceHash: "fd70378866f3fee5ca23d18c4832ace187118c316b5a37e1444d50113e9ea3a5"
sourceCorrections: []
---

LLM代理不会仅仅完成一项任务就停止；它以持续的循环方式工作。可以这样理解：一个代理观察其所处环境（或任务的当前状态），根据这些观察和其目标思考下一步做什么，然后采取行动。这个序列不是一次性发生的。它会重复，让代理能够取得进展，适应新信息，并努力完成更复杂的任务目标。这个基本的重复过程通常被称为代理的运行循环。它说明了代理如何从简单地处理语言，到执行一系列行动。

让我们来细分这个循环的主要阶段：

1. **观察（收集信息）**
   代理在每个循环中的第一步是收集信息。这种“观察”可以有多种形式。它可能是你给出的初始指令，例如“为我总结这份文档”。它也可能是代理之前采取行动的结果，比如“搜索查询返回了三篇文章”或“尝试访问文件时发生错误”。或者，它可能是来自外部来源的新数据。本质上，代理正在收集它做出知情决策所需的当前数据点。这个步骤通常涉及检查其短期记忆，以理解当前情况相对于过去交互的背景。
2. **思考（处理和规划）**
   一旦代理有了观察结果，它就需要“思考”。这是大型语言模型（LLM），即代理的大脑，开始工作的阶段。LLM处理从观察中获得的新信息，考虑其总体目标（由其初始指令或提示定义），并决定下一步。这可能包括：

   - 分析观察结果：“用户需要摘要，我已经有了文档内容。”
   - 制定或更新计划：“步骤1：阅读文档。步骤2：识别要点。步骤3：生成摘要。”
   - 决定具体行动：“下一步行动是使用‘文本摘要’工具。”
     LLM将即时观察与更广泛的指令以及任何进行中的计划结合起来，以确定要做什么。
3. **行动（执行和交互）**
   在“思考”阶段之后，代理“行动”。行动是代理执行的任何操作。这可能是：

   - 为用户生成文本响应（例如：“这是您请求的摘要...”）。
   - 使用工具（例如，访问计算器、进行网络搜索或查询数据库）。
   - 调用外部API（例如，获取天气数据或发送电子邮件）。
   - 修改其内部状态或记忆（例如，记录子任务已完成）。
     如果代理在其思考阶段决定使用天气工具查询“巴黎”，那么行动将是以“巴黎”作为输入执行该工具。

这种观察-思考-行动的循环随后重复。行动的结果（例如，工具检索到的天气信息，或工具失败时的错误消息）成为循环下一次迭代的新观察结果。代理观察这个新状态，思考它在其目标背景下的含义，并决定随后的行动。这个过程会一直持续，直到代理成功完成其总体任务或被指示停止。

> LLM代理的运行循环涉及重复观察情况、思考下一步并采取行动，直到达成总体目标。

这种运行循环使代理对各种任务特别有效。这种循环过程不是单一输入产生单一输出，它使代理能够：

- **管理多步骤任务**：复杂目标可以分解并逐一处理，每个循环都取得进展。
- **适应新信息**：如果行动不按计划进行（例如，工具返回错误），代理可以观察这个结果，思考替代方法，然后在下一个循环中采取不同行动。
- **保持持续交互**：通过将观察结果（这可以包括之前的行动和思考历史，通常存储在记忆中）反馈到循环中，代理可以进行更连贯和更长的对话或过程。

你可以看到，我们本章中讨论的代理核心组成部分，都这个循环中发挥各自的作用：

- **LLM** 是“思考”阶段的引擎，执行推理 (inference)。
- **指令和提示** 在每个“思考”阶段持续引导LLM的决策。
- **工具** 是在“行动”阶段使用的工具，让代理能够执行多种操作。
- **短期记忆** 非常重要。它在“观察”时被查阅，以提供先前循环的背景，并且通常在“思考”阶段期间或之后更新，包含新的学习或状态变化。
- **规划能力**，帮助代理“先思考后行动”，在“思考”阶段中使用，以策略性地规划达到目标所需的行动序列。

理解这种观察-思考-行动循环，是理解LLM代理如何运作并完成它们被设计用来执行的任务的根本所在。在下一节“简化的代理工作流程”中，我们将查看一个更具体的逐步示例，说明这个循环在实践中的应用。

## 参考资料

- [Artificial Intelligence: A Modern Approach](https://www.pearson.com/us/higher-education/program/Russell-Artificial-Intelligence-A-Modern-Approach-4th-Edition/PGM244975.html) — Stuart J. Russell and Peter Norvig (2020)
  Publisher: Pearson; Pages: Chapter 2
  一本基础教科书，定义了智能体及其操作循环，包括感知、动作和各种智能体程序。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton, Andrew G. Barto (2018)
  Publisher: The MIT Press; Pages: Chapter 3
  介绍了智能体-环境交互循环，其中智能体观察状态、采取行动并接收新状态和奖励。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHyagDsSnq6s70sXKJYLpwobh78r67mYYjELUXqdfMtVbc3twjiyZooeDsm9r5MPDhm0sIpp6kEWceBIdVU9YIwAMHESJTIMPwr0HYcqagPwhPzgJl4hkABgklgr_oLEmaV9Jgtltg=) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442
  描述了一种采用观察-规划-反思循环的智能体架构，展示了具有记忆和规划的LLM驱动智能体复杂的运行循环。
