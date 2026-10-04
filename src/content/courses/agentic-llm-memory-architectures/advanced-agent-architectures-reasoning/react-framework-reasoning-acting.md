---
course: "agentic-llm-memory-architectures"
chapter: "advanced-agent-architectures-reasoning"
lesson: "react-framework-reasoning-acting"
sourceId: 4442
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-2-advanced-agent-architectures-reasoning/react-framework-reasoning-acting"
title: "ReAct 框架：推理与行动的整合"
description: "ReAct（推理 + 行动）提示词和执行循环用于任务完成的细致分析。"
order: 1
plots: []
sourceHash: "24084f446f5dd686c16208b514e4490f23a0e92d01933168f6c92a4c908e66c3"
sourceCorrections: []
---

基本的智能体循环涉及接收输入和生成行动，但它们通常缺少内部思考的结构化方式，特别是在处理多步骤问题或与外部工具互动时。当行动失败或产生意外结果时，简单的反应式智能体可能难以适应。由 Yao 等人 (2022) 提出的 ReAct 框架，通过在智能体的操作周期内明确地将推理 (inference)步骤与行动结合，提供了一种更为有效的方法。

ReAct 代表着“推理 + 行动”。其基本原则是为智能体的行动空间增加一个内部推理空间。在每一步，智能体不仅决定下一步*做什么*（行动）；它首先阐明*为什么*（推理）。这种明确的阐述，常被称为“思考”，具有多重作用：它有助于分解复杂任务，追踪进度，处理异常情况，并根据中间结果动态调整计划。

## ReAct 操作循环

ReAct 过程迭代进行，通常遵循思考-行动-观察的步骤序列：

1. **观察 ($o_t$):** 智能体接收初始输入或观察其先前行动的结果。这构成了当前步骤的上下文 (context)。
2. **思考 ($t_t$):** 根据当前观察 $o_t$ 和总体目标，大型语言模型（LLM）生成内部推理 (inference)轨迹。此思考可能涉及分析情况，评估上一步的成功情况，规划下一步动作，或确认所需信息。例如：“搜索结果提供了出版年份，但没有作者的所属机构。我需要再次执行搜索查询，侧重于作者。”
3. **行动 ($a_t$):** 在思考之后，LLM 确定并生成要执行的下一步行动。这可以是与外部工具（如搜索引擎、计算器或 API）互动，查询数据库，或向用户制定回应。该行动直接来源于思考步骤中阐明的推理。
4. **执行与新观察 ($o_{t+1}$):** 环境或专门的工具执行模块执行行动 $a_t$。此执行的结果（例如，API 响应、搜索结果、计算输出、错误消息）成为循环下一次迭代的观察 $o_{t+1}$。

此循环持续进行，直到智能体通过其推理过程确定任务完成并生成最终答案或回应。

> ReAct 智能体中的迭代流程。观察触发思考，思考指导行动，行动结果成为新的观察。

## 通过提示词 (prompt)实现

ReAct 通常通过对提供给 LLM 的提示词进行结构化来实现。提示词指示模型在输出行动命令之前，明确生成与其内部思考过程相对应的文本。一种常见的格式鼓励 LLM 按照以下模板输出回应：

```
Thought: [智能体的推理过程在此]
Action: [智能体选择的行动，例如，Search[查询] 或 Final Answer: [答案]]
```

智能体系统随后解析此输出。如果存在 `Action`，则执行该行动，其输出作为下一个 `Observation` 附加到提示历史中。更新后的历史随后被反馈给 LLM，用于后续的思考-行动生成。

### 互动示例片段

设想一个智能体，其任务是查找马来西亚首都及其人口。

**第一次迭代：**

- **观察：** 马来西亚的首都及其人口是什么？
- **提示词 (prompt)包含：**

  ```
  Question: What is the capital of Malaysia and its population?
  Think step-by-step. You have access to a Search tool. Use the format:
  Thought: [你的推理]
  Action: [Search[查询] 或 Final Answer: 答案]
  ```
- **LLM 输出：**

  ```
  Thought: 问题要求提供两项信息：马来西亚的首都及其人口。我应首先查找马来西亚的首都。我可以使用搜索工具来完成此任务。
  Action: Search[马来西亚首都]
  ```

**第二次迭代：**

- **行动执行：** 系统执行 `Search[马来西亚首都]`。
- **观察：** 吉隆坡是马来西亚的首都。
- **提示词包含（先前步骤 + 新观察）：**

  ```
  ... [先前的提示历史] ...
  Observation: Kuala Lumpur is the capital of Malaysia.
  Thought: [你的推理]
  Action: [Search[查询] 或 Final Answer: 答案]
  ```
- **LLM 输出：**

  ```
  Thought: 好的，首都是吉隆坡。现在我需要查找吉隆坡的人口。我应再次使用搜索工具，专门查找吉隆坡的人口。
  Action: Search[吉隆坡人口]
  ```

**第三次迭代：**

- **行动执行：** 系统执行 `Search[吉隆坡人口]`。
- **观察：** 吉隆坡的人口在 2023 年估计为 190 万。
- **提示词包含：**

  ```
  ... [先前的提示历史] ...
  Observation: The population of Kuala Lumpur was estimated to be 1.9 million in 2023.
  Thought: [你的推理]
  Action: [Search[查询] 或 Final Answer: 答案]
  ```
- **LLM 输出：**

  ```
  Thought: 我现在已经拥有所需的两项信息：首都（吉隆坡）及其人口（190 万）。我可以制定最终答案了。
  Action: Final Answer: 马来西亚的首都为吉隆坡，其人口约为 190 万。
  ```

## 优点与考虑事项

ReAct 框架提供明显的优点：

- **可解释性提升：** 明确的 `思考` 轨迹为智能体的推理 (inference)过程提供了清晰的视角，使其更容易调试故障并理解其策略。
- **可靠性增强：** 通过对观察结果进行推理，智能体能够辨识失败的行动或意外的结果（例如，工具返回错误），并尝试纠正行动或替代策略。
- **更好地处理复杂任务：** 该框架自然鼓励将复杂问题分解为由内部推理引导的中间的、可管理步骤。
- **有效的工具结合：** ReAct 提供了一种结构化方式来决定何时以及如何使用外部工具，将工具使用建立在明确的推理之上。

然而，专家从业者应注意某些考虑事项：

- **提示词 (prompt)工程：** ReAct 智能体的性能对提示词的结构和措辞非常敏感。构建能持续引出所需思考-行动格式的有效提示词，需要仔细设计和反复试验。
- **计算成本增加：** 与直接行动生成相比，生成明确的思考会增加计算开销（token 使用量和延迟）。每个推理步骤都需要 LLM 进行额外的推理过程。
- **潜在的低效率：** 如果思考过程未得到良好引导，或者任务缺少清晰的中间检查点，智能体有时可能会陷入推理循环或采取不必要的冗长路径。
- **上下文 (context)长度限制：** 对于需要多步骤的任务，累积的观察、思考和行动历史最终可能超出 LLM 的上下文窗口限制，这需要摘要或其他上下文管理方法（在第 3 章中有更多讨论）。

尽管存在这些考虑，ReAct 代表着构建更具能力和更可靠的 LLM 智能体的重要支撑。它强调明确的推理，为处理需要规划、工具使用和适应的问题提供了一种机制。理解和实现 ReAct 是迈向开发复杂智能体系统的一步。下一部分将详细说明实际的实现方法。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR 2023); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍ReAct框架的原始论文，详细阐述了其设计与评估，以构建更强大的LLM代理。
- [A Survey of Large Language Model Based Autonomous Agents](https://arxiv.org/abs/2308.11432) — Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  全面概述了各种大型语言模型代理架构、框架和应用，为在代理领域理解ReAct提供了更广阔的背景。
