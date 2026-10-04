---
course: "python-llm-workflows"
chapter: "advanced-langchain-chains-agents"
lesson: "debugging-langchain-chains-agents"
sourceId: 5166
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/debugging-langchain-chains-agents"
title: "调试链和代理"
description: "解决复杂LangChain工作流的技巧和工具。"
order: 6
plots: []
sourceHash: "f61f7c93d618832c4521ac854fc9470757d581dcc2fee907e9fdaf801b087c74"
sourceCorrections: []
---

构建LangChain的链和代理的复杂应用程序时，常常会遇到运行不如预期的情况。调试这些系统面临特有的难题，这主要归因于底层大型语言模型（LLM）的非确定性特点以及多个组件之间的协作。代理可能会选择意外的工具，链可能会产生混乱的输出，或者整个过程可能会意外停止。因此，系统化的调试非常重要。

### 理解流程：详细输出

了解链或代理内部运行情况最直接的方法之一是启用详细输出。大多数LangChain链和代理对象在初始化时都接受 `verbose=True` 参数 (parameter)。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools

# 假设 llm 和 tools 已定义
# llm = ChatOpenAI(temperature=0)
# tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True # 启用详细输出
)

# 执行示例
# response = agent_executor.invoke({"input": "Who is the current CEO of OpenAI and what is his age raised to the power of 0.3?"})
# print(response)
```

当 `verbose=True` 时，链或代理会向控制台打印关于其执行步骤的详细信息。

**对于链：** 您通常会看到链的输入、各个组件（LLM调用、工具使用、提示格式化）的输入和输出，以及最终输出。这有助于查明数据可能在何处转换不正确，或者意外值从何处出现。

**对于代理：** 详细输出特别有启发性。它通常一步一步地显示代理的“思考过程”：

- **思考 (Thought)：** 代理根据输入和之前的步骤推理 (inference)下一步要采取的行动。
- **行动 (Action)：** 代理决定使用的具体工具。
- **行动输入 (Action Input)：** 提供给所选工具的输入。
- **观察 (Observation)：** 从工具接收到的输出。
- *(此循环重复，直到代理确定已有最终答案)*
- **最终答案 (Final Answer)：** 代理的最终回复。

查看这份详细日志可以帮助您追踪代理的决策，并精准定位可能出错的地方。是它误解了目标？选择了错误的工具？为工具格式化输入不正确？还是从工具接收到意外信息？

> 此图展示了ReAct风格代理的典型执行循环。详细输出有助于追踪此循环中的步骤。

### 隔离组件

如果详细输出未能立即显示问题，或者链非常长，请尝试将其分解。

- **单独运行组件：** 隔离测试链的每个部分（提示模板、LLM调用、输出解析器、工具）。手动提供示例输入，并检查输出是否符合预期。例如，使用模板生成的精确提示来测试LLM调用。LLM是否适当回应？输出解析器是否正确处理了LLM的原始输出？
- **检查中间输出：** 临时修改代码，打印或记录步骤之间的输出。例如，在 `SequentialChain` 中，打印每个子链执行后保存状态的字典。这有助于验证数据是否正确传递。

### 调试代理中的工具使用

工具集成是代理中常见的问题源。

- **工具输入/输出：** 仔细检查详细日志中的 `Observation`。工具是否成功运行？它是否返回了代理所需的信息？有时工具会静默失败或返回代理可能误解的错误消息。确保工具期望的输入格式与代理提供的 (`Action Input`) 相符。
- **工具描述：** 代理高度依赖工具描述来决定使用哪个工具。确保您的描述清晰、准确，并提供足够的信息，以便LLM理解工具的用途和使用方法。如果代理总是选择错误的工具，完善工具描述通常是解决方法。
- **错误处理：** 如果需要，在自定义工具代码中或代理执行器逻辑中，将工具执行调用封装在 `try...except` 块内。这可以防止单个工具故障导致整个进程崩溃，并允许代理（或您的外围代码）更优雅地处理错误。

### 处理LLM行为

有时，问题出在链或代理推理 (inference)过程中LLM的响应。

- **提示工程 (prompt engineering)：** 重新检查所使用的提示。指令是否清晰？代理是否被赋予了相互冲突的目标？少样本示例（如果使用）是否有帮助且相关？措辞上的微小改变可以显著影响LLM的行为。这通常是一个迭代的完善过程。
- **模型选择：** 不同的LLM有不同的优势和劣势。如果代理在推理或遵循指令方面遇到困难，尝试不同的模型（即使只是为了调试，也可能是更强大的模型）可能会产生更好的结果或指出问题所在。
- **温度：** 如果LLM使用了非零的 `temperature`，请记住输出的确定性较低。为了调试复杂逻辑，暂时将 `temperature` 设置为 `0` 可以使LLM的行为更可预测，从而帮助您隔离其他问题。

### 高级追踪工具

对于更复杂的情况，尤其是在生产环境或团队协作场景中，请考虑使用LangSmith等专用追踪平台。这些工具提供基于网络的界面，可以详细可视化链和代理的运行情况。您可以查看每一步的精确输入/输出、延迟信息、令牌数量和错误消息，通常比原始的 `verbose=True` 日志呈现得更清晰。它们允许您检查失败的运行、比较不同版本的提示或链，以及协作调试工作。虽然评估框架的详细内容将放在第九章，但这些工具在开发和调试阶段非常有价值。

调试链和代理需要耐心和系统化的方法。通过使用详细输出、隔离组件、仔细检查工具交互，以及通过其提示和响应分析LLM的行为，您可以有效诊断并解决高级LangChain应用程序中的问题。

## 参考资料

- [LangChain Agents Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGEB4bn5zUG_Zs0-FzP-gOBztXYy9SBlWxWsdS4n5VH696EpvM2ZaQv1ggNZYsYTh19pwihzlDPv_53mIsEgQbswSUkMkswmjsFyr2khX33MwhHQXefpxiWrdi0ttSfiA71OgCgD7ojHEnUTYaPAbou) — LangChain (2024)
  Publisher: LangChain
  LangChain代理的官方文档，涵盖其结构、用法以及详细输出等调试功能。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，该框架提供了高级LLM代理中常用的基础推理和行动循环，与图示和详细输出的解释直接相关。
- [LangSmith Documentation](https://docs.smith.langchain.com/) — LangChain (2024)
  Publisher: LangChain
  LangSmith的官方文档，这是一个用于观察、评估和调试使用LangChain构建的LLM应用的平台。
