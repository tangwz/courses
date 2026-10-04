---
course: "agentic-llm-memory-architectures"
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-2-advanced-agent-architectures-reasoning"
sourceId: 866
chapter: "advanced-agent-architectures-reasoning"
title: "进阶智能体架构与推理"
order: 2
description: "实现并分析复杂的智能体架构，侧重于推理和决策过程。"
hasQuiz: false
---

承接智能体构成要素和基本交互循环的认识，本章重点介绍使大型语言模型（LLM）智能体能够进行精密推理和决策的架构。简单的提示-行动循环在需要详细规划或考虑多种情况的复杂任务中常感吃力。在此，我们将审视旨在组织智能体思维过程的特定框架，以期更有效地解决问题。

您将学习分析并实现几种主要智能体架构：

*   **ReAct（推理 + 行动）：** 了解此框架如何根据观察（$o_i$）将推理步骤与行动（$a_i$）结合，以动态地创建和调整计划。我们将研究其特有的思维-行动-观察循环。
*   **自问（Self-Ask）：** 了解智能体如何将复杂的查询分解为更简单、相互关联的问题，旨在在生成最终答案前提升事实依据。
*   **思维树（Tree of Thoughts, ToT）：** 研究此方法如何让智能体同时考察多条推理路径，将中间“思维”作为树的节点进行评估，并在思维空间中使用搜索策略（如BFS或DFS），该思维空间通常表示为图$$G=(V, E)$$。
*   **基于图的推理：** 考察显式图结构在管理智能体内部复杂的推理状态、计划和依赖关系方面的运用。

本章提供实现细节、这些推理框架的比较分析以及实际练习，例如构建一个定制的ReAct智能体。目标是让您掌握选择和构建适用于进阶智能体系统的恰当推理机制的知识。
