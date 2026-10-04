---
course: "langchain-production-llm"
chapter: "sophisticated-agents-tools"
lesson: "agent-execution-tracing"
sourceId: 3869
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-execution-tracing"
title: "智能体执行追踪与分析"
description: "理解和分析智能体决策过程的方法。"
order: 6
plots: []
sourceHash: "1473a1f2c600705aa546dc19dc91805379b2f3cbc0aa3ecbef6eafe0e1eed834"
sourceCorrections: []
---

随着智能体执行涉及多次大型语言模型调用、工具交互和内部推理 (inference)环节的复杂步骤，弄明白智能体做出特定决定的原因或过程出错的位置变得越来越难。智能体不是一个具有清晰输入和输出的简单函数；它是在问题空间内运行的动态系统。如果无法观察其内部状态和决策过程，调试和提升智能体表现会感觉像凭空猜测。方法和工具，特别是LangSmith，被用于追踪和分析智能体执行过程，以理解智能体的行为。

### 可见性至关重要

设想一个智能体，其任务是研究一个主题、查询数据库以获取相关统计数据，并综合生成一份报告。如果最终报告不准确或不完整，可能的原因有很多：

- 大型语言模型是否误解了初始请求？
- 智能体是否选择了错误的工具（例如，使用了网页搜索而非数据库查询）？
- 传递给工具的参数 (parameter)是否不正确？
- 工具本身是否返回了错误或意外数据？
- 大型语言模型是否未能正确综合工具输出中的信息？
- 智能体是否陷入循环，重复尝试相同的失败动作？

简单的日志记录可以捕获工具的输入和输出，但它常常遗漏中间的推理 (inference)步骤或大型语言模型发出的结构化工具调用。为了高效地诊断问题并优化行为，我们需要一份详细、分步的智能体执行路径记录。

### 使用LangSmith进行执行追踪

LangSmith专为解决LangChain生态系统中的这一难题而设计。当集成到您的应用程序中时（详见第5章），它会自动捕获LangChain组件的详细追踪信息，包括智能体及其组成部分。

LangSmith中典型的智能体执行追踪提供了一个分层视图，记录了：

1. **智能体调用：** 当智能体开始处理请求时的顶层入口点。
2. **大型语言模型调用：** 智能体每次咨询大型语言模型。这包括发送的确切提示、使用的模型参数 (parameter)以及收到的原始响应（包括内容和工具调用请求）。
3. **行动步骤：** 智能体决定使用特定工具。
4. **工具输入：** 智能体提供给所选工具的参数。
5. **工具执行：** 调用工具的底层函数。
6. **工具输出（观测）：** 工具返回的结果，会反馈给智能体。
7. **推理 (inference)和工具调用：** 大型语言模型的明确推理或执行工具的结构化请求会被清晰地记录。

这种结构化、按时间顺序排列的视图让你能够直观地回放智能体的处理过程。

> 这是一个智能体执行流程的简化表示，其中包含一次大型语言模型调用、一次工具执行，以及一次最终大型语言模型调用来综合生成答案。LangSmith会记录这些步骤的详细输入和输出。

### 从追踪信息分析智能体行为

分析这些追踪信息对理解智能体行为非常有益：

- **追踪推理 (inference)过程：** 检查模型的逻辑是否将当前目标、可用工具以及前一步的观测信息关联起来。所选的工具调用是否与内部推理一致？
- **核查工具使用：** 检查传递给工具的参数 (parameter)。智能体格式化输入是否正确？它是否从内存或前一步中提取了正确的信息作为输入？
- **检查观测结果：** 查看工具的输出。返回的信息是否符合智能体的预期？如果工具出错，追踪信息会显示异常，这有助于准确定位故障。
- **追踪状态：** 观察信息（或信息缺失）如何在各个步骤中传递。智能体是否正确地将来自观测结果的新数据融入后续步骤？

### 通过追踪信息排查智能体常见问题

执行追踪信息对调试工作非常有帮助：

- **工具选择错误：** 如果智能体在有更合适的特定数据库查询工具时，总是选择`web_search`，追踪信息会显示这种模式。您可能需要调整智能体的提示、工具描述或智能体的核心架构。
- **工具执行错误：** 如果工具调用失败，追踪信息会显示导致错误的具体输入以及抛出的异常。这有助于将问题隔离为智能体提供了不良输入，或是工具本身的错误。
- **模式和验证错误：** 现代智能体通常使用结构化输出（如函数调用）。追踪信息会捕获原始的大型语言模型输出，这使得查看大型语言模型是否生成了无效的JSON或违反工具模式的参数 (parameter)、从而导致验证失败变得容易。
- **智能体循环或效率低下：** 通过审查行动序列，您可以发现智能体未能取得进展的重复循环。这可能表明逻辑缺陷、糟糕的工具设计或步骤之间传递的信息不足。
- **虚假工具输入：** 有时，大型语言模型可能会为工具生成语法上有效但事实上不正确的输入（例如，数据库查询中不存在的用户ID）。追踪信息通过显示有问题的参数使其显而易见。

### 性能和成本分析

追踪信息有助于分析效率：

- **延迟分析：** LangSmith会自动记录每个步骤（大型语言模型调用、工具执行）的持续时间。您可以快速识别智能体执行中哪些部分耗时最长。工具调用的慢速外部API会在追踪计时中清晰可见。
- **令牌消耗：** 追踪中的每次大型语言模型调用都与令牌计数（提示令牌和完成令牌）相关联。通过对典型执行中的这些计数进行求和，您可以估算运营成本并识别特别消耗令牌的步骤。也许更短的提示或不同的模型可以更经济地实现相同结果。

### 运用分析优化智能体

追踪不仅仅用于修复错误；它是一个持续改进的工具。定期审查来自真实或模拟交互的追踪信息：

- 是否存在效率低下的推理 (inference)模式？
- 工具描述能否更清晰，以更好地引导大型语言模型？
- 智能体是否总是需要多个步骤来完成可以简化的任务？

来自追踪分析的发现直接指导提示工程 (prompt engineering)、工具设计，甚至可能影响智能体架构或底层大型语言模型的选择，从而产生更高效和可靠的智能体。

## 参考资料

- [LangSmith Documentation: Tracing & Debugging](https://docs.smith.langchain.com/) — LangChain (2024)
  Publisher: LangChain
  LangSmith官方文档，详细介绍了其用于跟踪、调试和监控LangChain应用程序（包括代理执行）的功能。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，结合了语言模型的推理和行动，为许多代理提供了架构基础，追踪有助于阐明其内部步骤。
- [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432) — Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2308.11432; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  对基于大型语言模型的自主代理的全面调查，讨论了架构、能力和挑战，强调了理解其复杂行为的必要性。
- [Designing Machine Learning Systems: An Iterative Process for Developing Real-World Applications](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  为在生产环境中设计和部署机器学习系统提供了实用指导，涵盖了监控、评估和调试等重要方面，这些也适用于LLM代理应用程序。
