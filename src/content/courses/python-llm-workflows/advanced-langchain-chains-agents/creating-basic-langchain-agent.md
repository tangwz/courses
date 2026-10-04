---
course: "python-llm-workflows"
chapter: "advanced-langchain-chains-agents"
lesson: "creating-basic-langchain-agent"
sourceId: 5163
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/creating-basic-langchain-agent"
title: "建立一个基本智能体"
description: "实现一个能够使用工具达成目标的简单智能体。"
order: 5
plots: []
sourceHash: "ae88d7d41f59ede1beaa464112174a4456f52573bc0be00ef76a53658dcc3211"
sourceCorrections: []
---

智能体代表了一种动态的LLM工作流方法。与链式结构不同，链式结构按预设顺序连接组件，其中开发者定义精确的操作顺序。而智能体则使用大型语言模型（LLM）作为推理 (inference)引擎，根据一组可用`工具`来决定采取*哪些*行动以及以*何种*顺序。这使得它们能够处理那些事先不明确解决方案路径的更复杂、不可预测的任务。

建立一个基本智能体需要结合几个特定组件：

1. **LLM：** 为智能体推理过程提供支持的核心语言模型。
2. **工具：** 智能体可用于与外部环境交互或执行特定任务的功能或能力（例如，搜索网页、执行计算、查询数据库）。
3. **智能体提示模板：** 一个专用提示，指导大型语言模型如何推理、哪些工具可用、如何格式化其思考和行动，以及如何处理输入查询和中间步骤。
4. **智能体执行器：** 管理智能体执行循环的运行时环境：调用大型语言模型、解析其输出以识别工具使用、运行选定的工具，并将结果反馈给大型语言模型，直到任务完成。

### 定义工具

对于一个基本智能体，我们可以从LangChain提供的简单、预构建工具开始。让我们考虑一个需要执行数学计算的智能体。LangChain提供了一个`load_tools`函数，简化了常用工具的加载。我们将加载`llm-math`工具，该工具使用大型语言模型执行计算。

```python
from langchain.agents import load_tools
from langchain_openai import OpenAI # 或者你偏好的大型语言模型提供商

# 假设LLM对象已初始化（例如，llm = OpenAI(temperature=0)）
# 工具名称是LangChain可识别的特定字符串
tool_names = ["llm-math"]
tools = load_tools(tool_names, llm=llm)

print(f"已加载工具: {tools[0].name}")
print(f"工具说明: {tools[0].description}")
```

`load_tools`函数返回`Tool`对象的列表。每个工具都有一个`name`（智能体用于识别它）和一个`description`（对大型语言模型理解工具功能和何时使用它很重要）。

### 选择智能体类型和提示

LangChain支持多种智能体类型，每种都采用不同的提示策略。对于初学者而言，一种常见且有效的类型是\*\*ReAct（推理 (inference)与行动）\*\*智能体。ReAct提示指导大型语言模型遵循特定的思考过程：

1. **思考（Thought）：** 思考当前状态并决定下一步。
2. **行动（Action）：** 指定要使用的工具。
3. **行动输入（Action Input）：** 为选定的工具提供输入。
4. **观察（Observation）：** （这是智能体执行器在运行工具后提供的）记录工具的结果。
5. 重复步骤1-4，直到获得最终答案，然后输出最终答案。

LangChain为不同的智能体类型提供了标准化提示。对于基本智能体，我们通常不需要手动制作这些复杂的提示。

### 初始化智能体

大型语言模型和工具准备好后，我们就可以初始化智能体了。我们使用`create_react_agent`函数（或其他智能体类型的类似函数），传入大型语言模型、工具列表以及一个适合ReAct智能体的预定义提示结构。LangChain通常会根据你正在创建的智能体类型自动拉取正确的提示。

```python
from langchain import hub
from langchain.agents import create_react_agent

# 拉取标准ReAct提示模板
# 此提示旨在与ReAct智能体逻辑配合使用
prompt = hub.pull("hwchase17/react")

# 创建ReAct智能体
agent = create_react_agent(llm, tools, prompt)

# 注意：'agent' 对象本身包含逻辑蓝图，
# 但它需要AgentExecutor才能实际运行。
```

现在，这个`agent`对象捆绑了大型语言模型、可用工具以及关于大型语言模型应如何使用这些工具进行推理 (inference)和行动的特定指令（即提示）。

### 智能体执行器

最后一部分是`AgentExecutor`。这是实际执行智能体推理 (inference)循环的运行时。它以`agent`对象和`tools`作为输入。

```python
from langchain.agents import AgentExecutor

# 创建智能体执行器
# verbose=True 允许我们查看智能体的思考过程
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# max_iterations 防止潜在的无限循环
```

在开发期间强烈建议使用`verbose=True`参数 (parameter)，因为它会打印智能体的内部独白（思考、行动、观察），这使得理解其行为和调试问题变得容易得多。`max_iterations`是一个安全措施，用于防止智能体陷入循环。

智能体执行循环可以如下所示：

> 该图表说明了智能体执行器内部的流程。大型语言模型进行推理，提出行动，执行器解析此内容，如果指定则运行工具，获取结果（观察），将其添加到草稿本，并反馈给大型语言模型，直到确定最终答案。

### 运行智能体

现在，我们可以使用`AgentExecutor`的`invoke`方法，通过特定任务来调用智能体。

```python
# 我们的数学智能体的示例查询
query = "What is 25 percent of 300?"

# 运行智能体执行器
response = agent_executor.invoke({"input": query})

print("\n最终回应:")
print(response)

# 一个更复杂的计算
query_complex = "If a train travels at 60 km/h for 2.5 hours, how far does it travel?"
# 注意：基础的'llm-math'工具可能在单位转换或隐式物理计算方面表现不佳。
# 它主要用于算术/代数计算。
# 一个更完善的智能体可能需要特定的物理工具或更强的推理能力。
response_complex = agent_executor.invoke({"input": query_complex})
print("\n最终回应（复杂查询）:")
print(response_complex)
```

如果设置了`verbose=True`，你将在执行期间看到类似以下输出：

```text
> 进入新的AgentExecutor链...
思考：用户想计算300的25%。我可以使用计算器工具。计算式是0.25 * 300。
行动：计算器
行动输入：0.25 * 300
观察：答案：75.0
思考：我已从计算器获得结果。答案是75.0。我现在可以向用户提供最终答案。
最终答案：300的25%是75.0。

> 链已完成。

最终回应:
{'input': 'What is 25 percent of 300?', 'output': '25 percent of 300 is 75.0.'}

# ... （第二个查询的类似输出，可能涉及 60 * 2.5）
```

此输出清晰地显示了ReAct流程：智能体的思考、选定的行动及其输入、从工具接收到的观察结果，以及最终导致答案的思考。

这个例子演示了创建和运行一个基本智能体的核心过程。你初始化一个大型语言模型，加载必要的工具，使用创建函数（如`create_react_agent`）将它们与适当的智能体提示结合，将所有内容封装在`AgentExecutor`中，然后用你的查询来调用它。执行器处理交互循环，运用大型语言模型的推理 (inference)能力来决定如何有效使用工具。在下一节中，我们将了解当事情不按计划进行时的调试方法。

## 参考资料

- [Agents](https://python.langchain.com/docs/modules/agents/) — LangChain Contributors (2024)
  Publisher: LangChain
  LangChain框架构建LLM代理的官方文档，涵盖各种代理类型、工具和执行器。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: ICLR; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，一种指导语言模型进行推理和行动的提示策略，与本节讨论的代理类型直接相关。
- [A Survey on Large Language Model based Autonomous Agents](https://arxiv.org/abs/2308.11432) — Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  对基于LLM的自主代理进行了全面综述，包括其架构、能力和挑战，为代理设计提供了更广阔的背景。
