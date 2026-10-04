---
course: "intro-llm-agents"
chapter: "what-are-llm-agents"
lesson: "understanding-agent-independence"
sourceId: 6551
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-1-what-are-llm-agents/understanding-agent-independence"
title: "理解智能体自主性"
description: "掌握大型语言模型智能体如何以一定程度的自主性进行操作的基本理念。"
order: 7
plots: []
sourceHash: "83af3a82ce1f04932557dbe42ce2580f480847404357c6da1bd978d6d29e3022"
sourceCorrections: []
---

当我们讨论大型语言模型（LLM）智能体时，它们与简单程序甚至基本LLM交互的不同之处在于其自主能力。但在这里，“自主性”到底指什么？它并非指智能体拥有自己的秘密计划或愿望。相反，它是指智能体在无需人类指示每个步骤的情况下，操作并做出决策以达成指定目标的能力。

想想使用简单计算器与委托研究任务之间的区别。使用计算器时，你一步步输入数字和操作，它完全是被动的响应。而一个LLM智能体，当被赋予“查找伦敦当前天气并总结明天的预报”这样的目标时，它能够自行采取一系列行动。它可能会决定使用搜索工具、解析结果，然后形成摘要，这一切都无需你手动指导每个子任务。这种朝向目标的自主推进，正是智能体自主性的核心所在。

这种操作自由与传统自动化脚本形成鲜明对比。脚本就像一个非常严格的食谱：它遵循预设的命令序列，不能偏离。如果出现意外情况，脚本通常会失败或产生不正确的结果。LLM智能体凭借大型语言模型的推理 (inference)能力，具备更大的灵活性。尽管其行动仍受其编程和LLM训练的指引，但它能以更具动态性的方式解释情况并选择下一步行动。

需要理解的是，这种自主性并非无限。智能体在开发者设定的范围内运作。它们被赋予特定目标，能够使用一套限定的工具（我们将在第4章中进一步说明），其“思考”基于它们所用LLM中存在的模式和信息。它们不具备意识或个人意图。它们的自主性是一种功能性能力，使它们能够以一定程度的自我管理来达成目标。

下图展示了不同的任务执行模式，突出了智能体自主性与直接人工控制和固定脚本的区别：

> 该图展示了一种演进：直接人工控制的每一步都需要持续输入。固定脚本自动化了序列但无法适应。具备自主性的智能体可以接收一个目标，然后循环进行观察、思考和行动，以朝着该目标前进。

那么，为什么这种自主性有益呢？

1. **处理多步骤任务**：智能体能够管理需要多个操作的任务，例如从多个来源收集信息、处理信息，然后根据结果采取行动。
2. **适应轻微变化**：由于大型语言模型能够理解语境，智能体或许能够处理输入中的轻微变化或意料之外（但并非完全不可预见）的中间结果，而不会立即失败。例如，如果一个网站的布局略有变化，设计良好的智能体可能仍能找到它需要的信息。
3. **减少繁琐工作**：通过不仅自动化执行，还自动化部分中间决策，智能体可以使人类用户从任务的更多操作细节中解脱出来。

设想你给同事分配一项任务，比如：“请查明Alpha项目的项目经理是谁，获取他们的电子邮件地址，并替我草拟一封询问简要状态更新的介绍性邮件。”你的同事不需要你告诉他们*如何*找到项目经理（例如，查看内部目录、询问其他团队成员），也不需要你告诉他们草稿中每一句话的具体措辞。他们有一个目标，并具备自主决定中间步骤的能力。大型语言模型智能体力求实现类似的职能自主性，尽管是在一个更结构化和数字化的范畴。

这种具备一定自主性进行操作的能力，是使大型语言模型智能体变得强大的根本所在。在学习本课程的过程中，你将了解更多关于规划（第5章）、工具（第4章）和记忆（第6章）等组成部分，它们支持并塑造了这种自主性，使智能体能够执行日益复杂的任务。

## 参考资料

- [Artificial Intelligence: A Modern Approach](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCFOTStNRaa4Pg0lP3sS5iD-caOXpD3X-b6G2Zx5mRpROmFzZ76BKtqw8H9YqKvTIZjOpLAeP4zeZ24zGRHlUOOYNoy-3-nNfI4FLwd1W9g_L_mqHaBPHZuBQL3zBvXLHiqLGwEIogw_iM37l37F09uiCmOPhvxLnIiPhdA8RoLAjJHao7qghUMDW1) — Stuart Russell, Peter Norvig (2020)
  Publisher: Pearson
  这本教科书提供了智能代理的基本概念、自主性、目标以及感知-行动循环，直接为LLM代理的独立性提供了理论基础。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGBrcCyi399BkU41qEnyEL1i-A-qetXdS8r2apaYT9LUXO-IxM4QdIcQadop-7FC6iYFOCYWqcw8TvQLLjq3PE0FCVZAa_Ngud8IX198ck1XeTz0bQ2ZvoCahScCHTKI-bL_z3a) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR)
  介绍了ReAct框架，展示了大型语言模型如何将推理过程与特定任务行动交织起来，以执行动态的、目标导向的任务，这是代理独立性的核心。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEUuGOuCNN7_j-FfFRQO34ssJ_3NKZ7Ytm-0XjNinTZSabtjt3U-iX2U29ueIkv6U2Jg4gmWRSxGtsfuA5Fj5QjAgKKgZKkVsFfefbLGGuwjxgjPwchyv01DgXS7EUCPLgGB4Dzkpu-lbWxOIG0il8z2hWPRdQCzn1gpiIQfMtI0PpVsB-E6KFe7r9I3Gcwi0BfnAzswx2iaj0Xm1zyGXoK_2TmVHB4c_MRig==) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou (2022)
  Journal: NeurIPS; Pages: 24824–24837
  这篇论文详细阐述了思维链提示如何使大型语言模型能够执行多步骤推理，这是代理独立思考、规划和实现目标的核心能力。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHMtHpYduGRLgFwTAv9BN5J3f06iu_DwjcrN5aUyY-jhBSMmLeCE_3BLB0XQK_W5_dGGcCzwFBzyFFEhKLbmCKIlYlDkh8ARUDS4VlHAzzy02nyamn2bH_WhA==) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology; Pages: 1–22; DOI: [10.1145/3586183.3606763](https://doi.org/10.1145/3586183.3606763)
  提出了一个计算框架，用于模拟可信人类行为的代理，展示了复杂的规划、自我反思和互动，说明了在模拟环境中高度的独立性。
