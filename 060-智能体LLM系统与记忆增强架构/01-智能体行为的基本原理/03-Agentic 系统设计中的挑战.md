# Agentic 系统设计中的挑战

来源：[原文](https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-1-foundations-agentic-behavior/challenges-agentic-design)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管自主LLM代理的潜力可观，但构建可靠且有效的此类系统需要面对一些主要的工程难题。这些系统以一定程度的自由度运行，这放大了底层LLM局限性的影响，并引入了新的系统复杂性。了解这些挑战是设计先进架构的必要前提。

### 幻觉 (hallucination)与事实依据

众所周知，标准LLM会生成听起来合理但事实不准确的信息，通常被称为“幻觉”。在会话式聊天机器人中，这可能仅仅是不便。然而，在一个负责根据其推理 (inference)或检索到的信息执行动作的agentic系统中，幻觉可能导致有害结果。代理可能会尝试使用不存在的工具，根据有缺陷的理解，使用不正确的参数 (parameter)调用API，或根据其推理过程中产生的错误前提做出重要决策。

缓解代理中的幻觉比简单的核实事实更复杂。这需要以下机制：

- **可验证的推理：** 确保代理的推理步骤可追溯，并在可能的情况下，以从可靠来源或工具输出中检索到的可验证信息为依据。像Self-Ask（第二章会介绍）这样的技术旨在将查询分解为基于事实的子问题。
- **信息整合：** 代理通常需要从多个来源（记忆、工具输出、用户指令）整合信息。确保这个整合过程不引入事实错误是一个重要挑战。
- **置信度评估：** 理想情况下，代理应该对其生成的计划或事实声明有一定的置信度，这样当置信度较低时，它可以寻求澄清或替代策略。

对持续依据的需求贯穿代理设计，影响着从提示策略到记忆检索机制的一切。

### 长期规划与连贯性

LLM，尤其是在固定上下文 (context)窗口内运行时，难以保持长期连贯性并执行复杂的多步计划。Agentic任务通常需要记住很多步骤之前的信息，根据中间结果调整计划，并确保行动与总体目标保持一致。

困难包括：

- **状态漂移：** 随着交互的进行和上下文窗口的填满，对维持计划完整性很重要的早期信息可能会丢失，导致代理“忘记”其目标或之前的决定。有效的记忆系统（第三章）对于解决这个问题非常必要。
- **组合复杂性：** 规划涉及到对潜在的庞大行动序列空间进行考量。虽然像思维树（第二章会介绍）这样的架构试图进行结构化考察，但有效地生成和评估潜在的未来仍然计算成本高昂且容易出错，特别是在处理不确定性或不完整信息时。
- **错误传播：** 计划早期步骤中的一个错误（例如，误解工具输出，做出次优决策）可能会层层传递，使后续的整个执行序列偏离轨道。代理需要自我修正和计划完善的机制（第四章）。

从视觉上考虑其复杂性：即使是简单的工具交互序列也可能导致分支故障。

> 显示了计划中工具使用错误如何传播或需要恢复路径的流程。

有效的长期规划需要LLM的推理 (inference)能力、外部记忆存储以及可能的层级规划结构之间的紧密结合。

### 可靠的工具集成与执行

允许代理访问外部工具（API、数据库、代码解释器）显著扩展了它们的能力，但也引入了重要的故障点。使工具使用可靠涉及几个子问题：

- **工具选择：** 代理必须准确判断哪个工具（如果有的话）适合当前子任务，这要基于工具的描述和任务上下文 (context)。模糊的描述或重叠的工具功能可能导致不正确的选择。
- **参数 (parameter)构造：** 为API调用或函数生成正确且格式良好的参数，需要精确理解所需的参数、数据类型和约束。LLM可能难以严格遵守模式。
- **输出解析与解读：** 代理必须正确解析工具返回的输出（这可能是JSON等结构化数据、非结构化文本或错误消息），并将此信息整合回其推理 (inference)过程。误解可能导致后续步骤出现问题。
- **错误处理：** 工具可能因多种原因（网络问题、无效输入、API变更、速率限制）而失败。代理需要错误处理逻辑来识别故障，可能重试操作，或在工具不可用或返回错误时调整其计划。

构建有韧性的工具使用代理通常需要显式验证层、结构化的输入/输出模式以及复杂的提示技术，来引导LLM与外部系统交互，如第四章所述。

### 其他考量

设计agentic系统还涉及以下挑战：

- **状态管理：** 在潜在的长期和复杂交互中，有效地追踪代理的内部状态、信念和任务进展。
- **评估：** 为这些复杂、通常非确定性的系统定义有意义的指标并开发可靠的评估工具，比评估传统的监督学习 (supervised learning)模型要困难得多（第六章会论及）。评估推理 (inference)或规划的质量不仅仅需要衡量最终任务的成功。

应对这些多方面的挑战构成了本课程介绍的先进架构、记忆系统和设计模式的核心驱动力。简单的提示或基本的API调用不足以构建复杂的自主代理；需要一种更具深度的架构方法。

## 参考资料

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.10601](https://doi.org/10.48550/arXiv.2305.10601)
  提出一种通过探索多条推理路径和自我评估来增强LLM问题解决能力的方法，解决了长期规划的挑战。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  提出一种将LLM生成内容基于外部知识库的方法，直接解决了幻觉和事实基础的挑战。
- [A Survey on Large Language Model Agents: Architecture, Capabilities, and Challenges](https://arxiv.org/abs/2309.07864) — Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Tao Gui (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2309.07864](https://doi.org/10.48550/arXiv.2309.07864)
  涵盖了LLM智能体架构、能力，并讨论了基础、规划和评估等核心挑战。

---

[上一节](02-%E8%87%AA%E4%B8%BB%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%B3%BB%E7%BB%9F%E7%9A%84%E7%BB%84%E6%88%90%E9%83%A8%E5%88%86.md) · [下一节](04-%E5%85%88%E8%BF%9B%E6%99%BA%E8%83%BD%E4%BD%93%E6%A1%86%E6%9E%B6%E6%A6%82%E8%A7%88.md)
