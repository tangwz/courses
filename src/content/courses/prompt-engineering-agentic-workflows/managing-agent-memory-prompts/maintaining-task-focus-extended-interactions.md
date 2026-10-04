---
course: "prompt-engineering-agentic-workflows"
chapter: "managing-agent-memory-prompts"
lesson: "maintaining-task-focus-extended-interactions"
sourceId: 6747
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-5-managing-agent-memory-prompts/maintaining-task-focus-extended-interactions"
title: "在长时间交互中保持任务重心"
description: "使用提示帮助代理在长时间或复杂的交互中保持对主要目标的专注。"
order: 7
plots: []
sourceHash: "e85510d95a9d54432aa5599b0706eaaa4f3f89528ac5ba2ae0415405d3b6268d"
sourceCorrections: []
---

虽然有效管理代理的记忆有助于其回忆信息，但在漫长或复杂的交互中，一个相关的问题是确保代理坚定地专注于其主要目标。如果缺乏具体指导，代理可能会偏离其主要目的，特别是在处理新的数据流或处理多个子任务时。本节详细说明了提示工程 (prompt engineering)技术如何帮助代理在长时间操作中维持任务重心，确保它们持续地朝着预期的结果努力。

### 维持代理注意力的挑战

代理，特别是那些基于大型语言模型构建的，按顺序处理信息。在长时间对话或多步骤任务中，最初的指令或总体目标可能会被“挤出”即时上下文 (context)窗口，或者被更新的信息稀释。这可能导致以下几个问题：

- **目标稀释**：代理可能开始优先处理近期子任务，而非主要目标。
- **上下文偏移**：对话可能会偏离主题，导致代理忘记其原本要解决的问题。
- **处理效率低下**：代理可能在不直接有助于主要目标的活动上花费资源。

有效的记忆管理，如本章前面讨论的，为保留信息提供了根基。维持任务重心在此基础上，通过使用提示主动引导代理的注意力，使其专注于最相关的信息和最终目标，特别是在交互跨越多个回合或涉及动态输入时。

> 代理在有无重心维持提示下的路径。偏离（左侧）可能发生在支线任务模糊主目标时。维持重心（右侧）涉及在提示中定期强化目标。

### 维持任务重心的提示策略

为对抗偏离并使代理与其目标保持一致，您可以采用几种提示工程 (prompt engineering)策略：

#### 1. 明确重申目标

最直接的方法之一是持续提醒代理其总体目标。这可以通过以下几种方式实现：

- **系统提示强化**：将主要目标声明直接包含在系统提示中。这确保它成为每次交互的根基指令的一部分。

  ```text
  System: You are a research assistant. Your primary mission is to compile a detailed report on renewable energy trends. All tasks should contribute to this mission.
  ```
- **每回合提醒**：在后续回合的提示中，特别是在完成复杂子任务或处理大量信息后，重申主要目标。

  ```text
  User: ... (previous interaction summary) ...
  Based on the new data on solar panel efficiency, update the relevant section.
  REMEMBER: The overall goal is the comprehensive report on renewable energy trends.
  ```

  将此类提醒放在提示的开头或结尾可能很有效，因为大型语言模型通常会更重视这些位置的信息。

#### 2. 结构化任务表示与进度跟踪

通过提供清晰的任务结构及其在结构中的当前位置，帮助代理维持重心。

- **任务清单或提纲**：在您的提示中包含一个主要任务步骤的汇总清单或提纲，或许突出显示当前或下一步。

  ```text
  User: Goal: Plan a 3-day business trip to Tokyo.
  Status:
  [X] Day 1: Flights and Hotel Booked.
  [ ] Day 2: Meeting Schedule (Current Focus)
      - Confirm meeting with ABC Corp.
      - Schedule dinner with XYZ Ltd.
  [ ] Day 3: Return Travel Arrangements.

  Please help draft an email to ABC Corp to confirm the meeting time for Day 2.
  ```

  这使得更广泛的上下文 (context)保持可见，同时专注于即时子任务。

#### 3. 重心指令与防护措施

您可以明确指示代理优先处理主要任务，并对照此目标自行评估其行动。

- **指令性防护措施**：添加引导代理决策过程的短语，以保持重心。

  ```text
  User: Before generating a response or taking an action, briefly verify that it directly supports the primary objective of [main objective stated here]. If it seems to deviate, state your reasoning for the deviation or ask for clarification.
  ```
- **自我纠正提示**：鼓励代理在发现自己偏离轨道时暂停并反思。（这还与错误处理相关，错误处理在第2章中有所涵盖。）

  ```text
  User: If you find the current line of inquiry is not contributing to [main goal], please state "Recalibrating focus to [main goal]" and suggest a more relevant next step.
  ```

#### 4. 上下文摘要作为锚点

虽然摘要技术（在“提示的信息浓缩技术”中讨论）对于管理上下文窗口限制很有用，但它们也可以根据需要进行调整以维持重心。指示代理创建明确与主要目标关联的摘要。

- **以目标为导向的摘要**：

  ```text
  User: Summarize our conversation so far, highlighting how each completed step has moved us closer to achieving the final goal of [final goal stated here].
  ```

  这不仅浓缩了信息，还强化了过去行动与总体目标之间的关联。

#### 5. 管理子任务并返回主流程

代理通常需要执行子任务，例如使用工具或执行特定信息查询。提示应确保顺利返回主任务。

- **明确的返回指令**：

  ```text
  User: First, use the search tool to find the current population of City X.
  After you have the population figure, return to the main task of comparing City X with City Y for urban development planning.
  Your next step after getting the population should be to analyze its impact on infrastructure needs for City X.
  ```

  这为代理创建了一个清晰的“堆栈”：执行子任务，然后返回主流程。

#### 6. 处理不断变化的信息流

当代理处理持续更新的信息时，维持对稳定、长期目标的重心尤其重要。提示需要引导代理整合新数据，而不会被其完全干扰。

- **使用目标过滤器整合新数据**：

  ```text
  User: Current Objective: Optimize the supply chain for Product Z based on cost and delivery time.
  New Information Stream: Received an alert about a potential delay at Port A.
  Task: Analyze this alert. How does it impact the current objective? Propose adjustments to the supply chain plan specifically to mitigate this new risk while still aiming for optimal cost and delivery time for Product Z.
  ```

  此提示指导代理从现有主要目标的视角处理新信息。

### 迭代是过程的一部分

找到这些重心维持提示的正确措辞和频率通常需要反复试验。一种任务类型或代理架构效果好的方法可能需要为另一种进行调整。在较长时间的交互中监控代理的行为：

- 它是否倾向于忘记最初的指令？添加更明确的目标重申。
- 它是否在子任务的细节中迷失？加强管理从子任务返回的提示，或使用结构化任务表示。
- 它是否容易被新的、不相关的信息影响？实施更严格的重心指令或目标过滤的信息处理。

通过深思熟虑地设计您的提示，使其包含这些重心维持机制，您可以大幅度提升代理在长时间交互中可靠且连贯地执行任务的能力，确保其努力与您的预期目的保持一致。这一点尤其重要，因为代理系统被赋予越来越复杂和长期运行的操作任务。

## 参考资料

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023)
  Journal: Transactions of the Association for Computational Linguistics (TACL); DOI: [10.48550/arXiv.2307.03172](https://doi.org/10.48550/arXiv.2307.03172)
  探讨大型语言模型在处理和检索长输入上下文时面临的挑战，指出了注意力衰减和‘迷失中间’现象，这些都与目标稀释问题直接相关。
- [The Hitchhiker's Guide to LLaMA-based Agents: A Comprehensive Overview](https://arxiv.org/abs/2403.04509) — Shuai Wang, Junda Wu, Bin Li, Weilin Zhao, Fan Wu, Kai Zheng (2024)
  Journal: arXiv preprint arXiv:2403.04509; DOI: [10.48550/arXiv.2403.04509](https://doi.org/10.48550/arXiv.2403.04509)
  提供了基于大型语言模型的智能体系统概述，涵盖了架构、内存以及在长时间运行操作中保持任务专注和目标一致性的难点。
