# Agent 工具选择逻辑

来源：[原文](https://apxml.com/zh/courses/intro-llm-agents/chapter-4-equipping-agents-with-tools/agent-tool-selection-logic)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Agent 可以使用工具来执行超出简单文本生成的动作。这与大型语言模型（LLM）独立能做到的相比，是一个很大的进步。一个主要问题出现了：Agent 如何决定 *何时* 使用工具，以及如果有多个选项，要选择 *哪一个* 特定工具？这种决策过程是使 Agent 真正有用和自主的中心所在。

Agent 的“大脑”——LLM——其核心是负责这种工具选择逻辑的。这不是魔法；相反，它是一个复杂的模式匹配和推理 (inference)过程，由你在 Agent 主提示中提供的指令以及工具本身的描述所引导。

### LLM 作为决策者

想象你有一个工具箱。当你面对一项任务，比如挂一幅画时，你不会随便抓起一把扳手。你会评估任务，看看你的工具（锤子、螺丝刀、水平仪），然后选择锤子，因为你知道它是用来敲钉子的。一个 LLM Agent 以类似的方式工作，尽管是基于文本的。

LLM 接收用户的请求或一个内部目标。然后它“查看”它被授权使用的工具列表。对于每个工具，它都有一个名称，最重要的是，一个关于该工具作用以及何时使用的**描述**。

### 提示词 (prompt)和工具描述的作用

影响 Agent 工具选择的主要方式是通过仔细的提示词工程。这涉及两个主要方面：

1. **Agent 整体指令：** Agent 主提示的一部分会指示它在遇到可能需要工具的情况时如何表现。这可能包括一般的指导方针，如“如果你需要进行计算，使用计算器工具”，或“如果你需要当前信息，使用搜索工具”。
2. **清晰的工具描述：** 真正的细节在此。提供给 Agent 的每个工具都必须附带清晰、简洁和准确的描述。LLM 在很大程度上依赖这些描述来理解工具的能力。

例如，如果你提供一个计算器工具，一个好的描述可能是：
`"calculator: 用于对数字进行数学计算。输入应为有效的数学表达式，如 '2+2' 或 '15*3/5'。"`

一个不太有用的描述会是：
`"math_tool: 执行数学运算。"`

描述越精确和信息越丰富，LLM 就能更好地判断该工具是否适合从用户查询派生出的当前子任务。

### Agent 如何“思考”工具的使用

尽管我们说 Agent “思考”有点拟人化，但 LLM 确实经历了一个类似于推理 (inference)的过程来决定工具的使用。以下是一个简化版分解：

1. **分析请求：** Agent 首先处理用户的输入或其当前目标。它试图理解意图以及所需的具体信息或动作。例如，如果用户问“伦敦的天气怎么样？5 的阶乘是多少？”，Agent 会识别出两个不同的子任务。
2. **对照自身能力检查：** LLM 拥有来自其训练数据中的知识。对于请求的某些部分（比如定义“阶乘”），它可能可以直接回答。
3. **扫描可用工具：** 对于它无法直接处理的请求部分（比如“当前天气”或“计算 5 的阶乘”），它会审查提供给它的工具描述。

   - 它在请求和工具描述之间寻找关键词和语义相似性。
   - 如果“伦敦的天气”是子任务，并且它有一个工具被描述为“获取给定城市的当前天气”，那么这是一个很强的匹配。
   - 如果“5 的阶乘”是子任务，并且它有一个被描述为“计算数学表达式”的“计算器”工具，它会发现另一个匹配。
4. **选择工具并格式化输入：** 如果识别出合适的工具，Agent 会决定使用它。这一步骤的一个重要部分是 LLM 还必须理解 *如何* 调用该工具。工具的描述（或附带的说明）应指定预期的输入格式。然后 LLM 根据用户的请求为工具格式化输入。

   - 对于天气工具，它可能会将输入格式化为 `{"city": "伦敦"}`。
   - 对于计算器，它可能会将输入格式化为 `"5*4*3*2*1"` 或在计算器工具支持此类函数的情况下传递 `"factorial(5)"`。
5. **无工具或不明确：**

   - 如果没有合适的工具，LLM 可能会尝试根据其训练数据回答，或告知用户它无法完成请求的该部分。
   - 如果多个工具看起来都合适，一个设计良好的 Agent（或更高级的 Agent）可能有一个优先级系统，或者它甚至会向用户寻求澄清。对于基本 Agent，清晰和明确的工具描述有助于最大限度地减少这种不确定性。

以下图表说明了这种决策流程：

> Agent 决定使用哪个工具、选择哪一个并准备其输入的内部逻辑。

例如，如果一个 Agent 有一个 `CalculatorTool` 和一个 `SearchTool`，并且用户问“法国的首都在哪里？12 乘以 12 是多少？”，Agent 的内部“对话”可能是：

- **用户请求：** “法国的首都在哪里？12 乘以 12 是多少？”
- **LLM 分析：** “好的，这里有两部分。第一，‘法国首都’。第二，‘12 乘以 12’。”
- **LLM 考虑‘法国首都’的工具：**
  - `CalculatorTool`： “描述说是‘数学计算’。不符合。”
  - `SearchTool`： “描述说是‘查找当前信息或一般知识’。这看起来很合适。”
  - **决定：** 使用 `SearchTool`，输入类似“法国首都”。
- **LLM 考虑‘12 乘以 12’的工具：**
  - `CalculatorTool`： “描述说是‘数学计算’。这非常合适。”
  - `SearchTool`： “描述说是‘查找当前信息’。虽然搜索*可能*找到，但计算器更直接。”
  - **决定：** 使用 `CalculatorTool`，输入“12\*12”。

然后 Agent 协调调用这些工具，获取它们的结果，并将它们综合成一个连贯的答案给用户。

这种逻辑的有效性在很大程度上取决于你如何良好地定义工具及其目的。模糊或重叠的工具描述会使 Agent 感到困惑，导致它使用错误的工具，或者在需要时根本不使用工具。随着你构建更复杂的 Agent，完善这些描述和 Agent 的指导提示会成为一个迭代的测试和改进过程。这确保你的 Agent 不仅拥有工具，而且拥有“智慧”去明智地使用它们。

## 参考资料

- [Designing Language Model Agents: A Survey](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFACnyLv-2zF3vOoJGqq49V3MlvfVtYh3OTXz7EvVfrtYoxBmz_7ebJZR-LpLVyrxfOz2pVdHmSctavAUNacdEj4W0FiaWxbeUB5ZMYqKk1V9LTYx8B0hh8B3hEBOk=) — Hongliang Ding, Jie Fu, Junpei Zhang, Kaiwen Yao, Hongyu Li, Zhiyuan Li, Shilin Quan, Haowen Yao, Tianbo Liu, Han Yang, Jiaheng Liu, Zhengyang Qiu, Runpeng Geng, Xiangru Tang, Yuyang Qian, Jielin Qiu, Zhiyang Wang, Qiaoyu Jia, Xiaomin Li, Peixuan Xu, Ruijie Xu, Shidi Li, Jiawen Shi, Wenwei Zhang, Kailong Wang, Shuo Wang, Wenxuan Zhou, Wenhao Zhang, Yifan Pu, Hongzhu Shi, Jingqi Shi, Yuqiao Song, Kaifeng Zhang, Xiaoyang Li, Xiangli Zhang, Mengting Li, Junzhe Chen, Zehao Zhang, Weiyue Ma, Xiaoyu Lei, Jingwen Li, Weifan Lu, Huajun Chen, Yunxuan Xiao, Jiaji Wen, Junyi Li, Hanyu Meng, Haozheng Li, Yicheng Wang, Yuxuan Chen, Yuanbo Lei, Qingyi Ding, Yuheng Lu, Mengdi Zhou, Juncheng Wan, Zehao Huang, Xiaoteng Ma, Yi Zeng, Jieming Zhu, Qi Zhang, Xuanjing Huang, Yue Zhang, Lei Li, Min Zhang, and Xipeng Qiu (2024)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2402.13848](https://doi.org/10.48550/arXiv.2402.13848)
  一份全面的综述，提供了大型语言模型代理的学术概览，涵盖了架构、工具集成等组件以及各种决策策略，对于理解工具选择逻辑的更广泛背景很有价值。
- [Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFE6G29PQxj6991q4C5YoB9te9mVuIBkCeds0sl0NkWvyWlDP5tnAbRMtU1M2Ve8KF-bVidah4_9C3uqwxVcQ4j-XbG7i_W8xEfN_DpVdR3rFCAELph3p1k-_PdCAmIwPa3rLB93Ls2jD93zAWfH43JXQyNDQ==) — LangChain Developers (2024)
  LangChain代理的官方文档，展示了工具定义、代理提示的实际实现，以及流行LLM代理框架中工具选择和执行的操作流程。

---

[上一节](05-%E6%8F%90%E7%A4%BAAgent%E4%BD%BF%E7%94%A8%E5%B7%A5%E5%85%B7.md) · [下一节](07-%E5%8F%AF%E7%94%A8%E5%B7%A5%E5%85%B7%E7%B1%BB%E5%88%AB%E6%A6%82%E8%A7%88.md)
