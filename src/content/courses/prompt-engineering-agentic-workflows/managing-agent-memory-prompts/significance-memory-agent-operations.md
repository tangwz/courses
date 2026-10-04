---
course: "prompt-engineering-agentic-workflows"
chapter: "managing-agent-memory-prompts"
lesson: "significance-memory-agent-operations"
sourceId: 6741
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-5-managing-agent-memory-prompts/significance-memory-agent-operations"
title: "记忆在智能体运行中的重要性"
description: "理解为何记忆是实现智能体有效且连贯行为的重要组成部分。"
order: 1
plots: []
sourceHash: "3b441ccee15d3dbd74b52810cfeb24c13a48d8dbd31ac7eb78bf059fcd9ac6b8"
sourceCorrections: []
---

AI智能体的记忆能力不只是一种便利；它是智能体处理复杂、多步骤操作能力的根本所在。如果缺乏回忆的机制，智能体将永远停留在当前时刻，无法将过去的互动与当前目标或未来行动联系起来。记忆对有效的智能体系统是必不可少的。

设想一下，如果你在组装复杂的机械，却不记得已经安装了哪些部件或整体蓝图。每个行动都会是孤立的，可能重复或适得其反。AI智能体，特别是那些为复杂工作流程设计的智能体，如果缺少记忆能力，也会面临类似的困境。它们执行随时间推移而展开、涉及多次交流或需要根据先前结果进行调整的任务的能力，直接取决于它们能记住什么以及如何有效使用那些保留的信息。

记忆在智能体操作中的主要作用包含：

1. **维护上下文 (context)和连贯性**: 为了使智能体进行有意义的对话或执行一系列相关的操作，它必须保留互动的上下文。这包括记住以前的用户输入、它自己的先前响应以及沿途收集的任何信息。例如，客户支持智能体需要回忆用户的初始问题描述和已尝试的解决方案，以避免令人沮丧的重复交流。记忆确保智能体的行为在不同轮次和步骤之间是连贯且逻辑相关的。
2. **实现多步骤任务执行**: 复杂的目标很少能一步达成。智能体常需要分解任务、执行子任务并整合结果。记忆对于追踪这些阶段的进度非常重要。它使智能体能够记住整体计划、哪些子任务已完成、中间结果以及接下来需要做什么。没有这个，智能体可能会陷入循环或未能达到最终目标。
3. **促进任务内学习和适应**: 当智能体与其环境或工具彼此作用时，它会收集新信息并体验结果。有效的记忆使智能体能够在当前任务的范围内“学习”这些体验。例如，如果一个智能体尝试使用一个特定的API端点（一个工具）但失败了，记住这次失败使其能够尝试替代方法或报告问题，而不是重复尝试相同的失败操作。这种适应能力是更智能系统的一个标志。
4. **支持规划和推理 (inference)**: 在智能体行动之前，它常常需要规划。这包括考虑当前状态、期望目标和潜在的行动序列。记忆通过持有当前理解、目标本身以及任何限制或偏好来为此提供根本。当智能体对其计划进行推理时，它可能会将中间想法或假设存储在工作记忆中，很像一个草稿本。

下面的图表说明了记忆如何融入智能体的操作循环：

> 一个智能体的操作循环，展现了记忆的核心作用。新信息被感知，记忆被访问以提供上下文，进行推理和规划，生成行动（可能涉及外部工具），最后，记忆用新的学习或任务进度进行更新，为智能体的后续步骤做好准备。

智能体通常处理不同时间范围的记忆：

- **短期记忆**（常与LLM的上下文窗口类似）持有与当前操作直接相关的信息。它不稳定，但对于维持对话流程和追踪即时任务步骤很重要。
- **长期记忆**涉及在较长时间内，可能跨越多个会话或任务存储和提取信息。这可以包括习得的事实、用户偏好或广泛的知识库。

任一类型记忆的缺陷，或管理它们的机制不完善，都会严重降低智能体的表现。记忆管理不善的现象包含：

- 询问已提供的信息。
- 在冗长任务中失去对主要目标的追踪。
- 未能使用互动早期相关信息。
- 在一次会话中无法从错误中学习。
- 生成不一致或矛盾的响应。

本质上，记忆将AI智能体从一个简单的、无状态的输入-输出处理器，转变为一个更有活力和能力的系统，能够智能地处理复杂问题。随着本章后续内容的展开，我们将审视具体的提示工程 (prompt engineering)方法，这些方法能让您有效地指导智能体如何存储、提取和使用信息，从而提升它们的记忆能力和整体操作效率。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  提出了一种通过非参数记忆（检索器）来增强语言模型进行知识检索的方法，解决了仅依赖参数记忆进行长期知识存储的局限性。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  描述了一种生成式智能体的架构，该架构通过将过去的经验存储并综合为长期记忆来模拟可信的人类行为，使智能体能够记忆、反思和规划。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint; Pages: 144; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  提供了大型语言模型的全面综述，包括对其架构、能力以及将外部知识和记忆集成到智能体应用中的挑战的讨论。
