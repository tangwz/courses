# 界定LLM情境下的自主行动能力

来源：[原文](https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-1-foundations-agentic-behavior/defining-llm-agency)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管大型语言模型（LLM）擅长文本 (long context)生成、理解、甚至在单次交互中进行基础推断，但“自主行动能力”的理念将它们从被动处理器提升为能够自主追求目标的主动参与者。在LLM核心、记忆、规划和行动执行等构成要素的情境中，准确地界定自主行动能力对于设计精巧的系统来说是必不可少的。

对一个基于LLM的系统来说，自主行动能力指的是在环境中独立且持续地运行以达成特定目标的能力。它超过简单的输入-输出映射，包含一个持续的感知、决策和行动循环，通常涉及与外部工具或数据源的交互。一个具自主行动能力的系统不只是被动回应；它会*有目的性地*行动。

### LLM自主行动能力的特点

有几个特点将具自主行动能力的LLM系统与标准LLM应用区分开来：

1. **目的导向：** 系统受一个或多个目的驱动，这些目的可以是用户明确定义的（例如：“总结这些研究论文中的主要发现并将其发送给团队”），也可以是其自身结构决定的（例如：一个旨在持续追踪系统日志以发现异常的监测智能体）。从理论上讲，所有行动都可根据这些目的得到合理解释。
2. **自主性：** 系统可以在长时间内以最少的人工干预运行。它可以启动行动，决定后续步骤，并处理中间结果，而无需为每一个微小决策持续提示。自主程度可以明显不同。
3. **环境交互（感知与行动）：** 智能体感知其环境，这可以是纯数字的（文本输入、API响应、数据库状态），也可能通过适当的接口与物理交互相关联。重要地，它还可以对这个环境*采取行动*，不只是通过生成文本，还可以通过执行代码、调用API、更新数据库或控制其他系统。
4. **推断与规划：** 自主行动能力暗示着深思熟虑。系统必须具备机制来推断其当前状态、期望的最终状态以及可能弥补差距的行动序列。这涉及规划，其范围可以从简单的任务分解到复杂的、涉及回溯和适应的多步骤策略。
5. **有状态性（记忆）：** 为了长时间连贯地行动，智能体必须保持状态。这包括短期记忆（例如，对话历史、当前计划步骤）和可能的长期记忆（例如，已获知识、过往经验、用户偏好），使其能够学习、适应并避免重复错误。记忆是上下文 (context)感知和连贯性的根本。

### 自主性程度的范围

自主行动能力不是单一的特性，而是存在一个程度范围。系统表现出这些特点的不同程度。弄清这个范围有助于对不同架构及其能力进行情境化。

> 此图展示了LLM系统自主行动能力的程度区分，从基础回应生成到复杂、具交互性的智能体。

- **低自主性：** 在较低一端，我们发现用于无状态请求-回应模式的普通LLM或具有限对话记忆的简单聊天机器人。它们对即时输入作出反应，但缺乏独立的目的追求或复杂规划。检索增强生成（RAG）系统，尽管使用外部工具（向量 (vector)数据库），但通常在单次交互或有限状态下运行，表现为迈向自主行动能力的一步，但通常缺乏复杂规划或自主循环。
- **中等自主性：** 采用思维链（CoT）提示等技术的系统呈现出更结构化的推断，但通常在单次推断过程中执行预定义的推断路径。它们显示出提升的深思熟虑能力，但在该过程中基于环境反馈的独立行动或适应能力有限。
- **高自主性：** ReAct（推断+行动）、自我提问、思想树（ToT）和多智能体系统等架构具现了更高程度的自主行动能力。它们明确地对感知-推断-行动循环进行建模。它们可以动态规划、执行行动（通常涉及多次工具调用）、观察结果、更新其内部状态（记忆），并在多次交互中调整策略以达成复杂的、长期目标。这些系统是本课程的首要关注点。

弄清自主行动能力是目的导向、自主性、环境交互、推断和有状态性的结合，这为设计、实现和评估后续章节中探讨的先进LLM体系提供了必要的框架。它将视角从将LLM视为精巧的文本补全引擎转变为将它们设计为自主解决问题体系中的主要推断引擎。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，结合了大型语言模型中的推理和行动步骤，对于自主目标追求和与环境交互非常重要。
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023)
  Journal: NeurIPS 2023; DOI: [10.48550/arXiv.2305.10601](https://doi.org/10.48550/arXiv.2305.10601)
  介绍了思，使大型语言模型能够进行审慎规划并探索多条推理路径以解决复杂问题。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou (2022)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 35; Pages: 24824--24837; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了思维链提示，这是一种通过促使模型生成中间推理步骤来提高大型语言模型推理能力的技术。

---

[下一节](02-%E8%87%AA%E4%B8%BB%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%B3%BB%E7%BB%9F%E7%9A%84%E7%BB%84%E6%88%90%E9%83%A8%E5%88%86.md)
