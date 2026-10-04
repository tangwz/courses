---
course: "building-llm-apps-with-langchain"
chapter: "developing-autonomous-agents"
lesson: "exploring-agent-types-executors"
sourceId: 7439
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-6-developing-autonomous-agents/exploring-agent-types-executors"
title: "代理类型和执行器介绍"
description: "考察ReAct和Self-Ask with Search等不同的代理算法，并理解Agent Executor如何运行推理循环。"
order: 5
plots: []
sourceHash: "79f425dfd22244feb818245d9aea1aef536540a199d2a0c7157063d076409e5d"
sourceCorrections: []
---

虽然代理的核心是其大型语言模型（LLM），但其用于推理 (inference)和解决问题的策略可能差异很大。LangChain提供了多种代理类型，每种都体现了不同的推理方法。这些不同的方法并非仅仅是理论上的；它们对代理的性能、可靠性以及特定任务的适用性都有实际影响。驱动这些推理策略的组件是`AgentExecutor`，它充当代理的运行时，管理LLM、工具和用户输入之间的交互。

### AgentExecutor：代理的引擎

在查看不同的代理类型之前，理解`AgentExecutor`很重要。它是促成代理运作的机制。它接收一个代理和一组工具作为输入，并运行一个循环，直到满足停止条件。这个循环通常被称为“推理 (inference)循环”。

这个循环的典型流程如下：

1. `AgentExecutor` 将用户输入和任何之前的步骤传递给代理。
2. 代理的LLM生成一个响应，其中包括一个 `Thought`（思考）和一个提议的 `Action`（行动）。
3. `AgentExecutor` 解析此响应。如果指定了 `Action`，它会识别要使用的工具及该工具的输入。
4. 执行器使用给定输入运行指定的工具。
5. 工具的输出，现在是一个 `Observation`（观察），被发送回代理。
6. 这个 `Thought -> Action -> Observation` 循环重复，直到代理认为它有了最终答案并提供。

这个迭代过程允许代理建立语境，从错误中恢复，并将复杂问题分解为更小、更易管理的步骤。

> AgentExecutor 管理着LLM推理（思考）、选择工具（行动）并接收新信息（观察）以继续其推理过程的循环。

### 代理类型：ReAct

**ReAct** 框架代表“Reason and Act”（推理 (inference)与行动），是代理最常见且有效的通用策略之一。其主要思想是，LLM在决定 `Action` 之前明确地将其推理过程（`Thought`）口述出来。这种思维链过程使代理的行为更易于理解，并常导致更可靠的工具使用。

ReAct 代理的提示模板指导LLM生成特定格式的文本：

```text
Question: The user's query goes here.
问题：用户的查询在此。
Thought: The model's reasoning about what it needs to do next.
思考：模型关于下一步需要做什么的推理。
Action: The tool to use and the input for that tool.
行动：要使用的工具及该工具的输入。
Observation: The result from executing the tool.
观察：执行工具得到的结果。
... (this Thought/Action/Observation cycle repeats)
...（这个思考/行动/观察循环重复）
Thought: I now have enough information to answer the user's question.
思考：我现在有足够的信息来回答用户的问题。
Final Answer: The final response to the user.
最终答案：对用户的最终响应。
```

这种结构迫使LLM规划其步骤，执行它们，然后在继续之前反思结果。对于需要多次工具调用或从多个来源收集信息以形成答案的任务，它是一种强大模式。

你可以使用 `create_react_agent` 构造函数初始化一个基于ReAct的代理，该函数会组装所需的提示和逻辑。

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub

# 加载ReAct提示模板
prompt = hub.pull("hwchase17/react")

# 初始化LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建代理
# 工具在此处定义并传入
# tools = [...] 
agent = create_react_agent(llm, tools, prompt)

# 创建AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 代理类型：Self-Ask with Search

**Self-Ask with Search** 代理类型专为问答任务设计，其中一个复杂问题需要分解为更简单、基于事实的子问题。它迭代地“自问”这些更简单的问题，并使用 `Search` 工具查找答案。一旦所有中间答案被收集，它将它们合成一个最终响应。

此代理的提示鼓励一种不同的推理 (inference)模式：

```text
Question: Who was the maternal grandfather of the 14th Dalai Lama?
问题：第14世达赖喇嘛的外祖父是谁？
Are follow up questions needed here: Yes.
这里需要后续问题吗：是的。
Follow up: Who was the 14th Dalai Lama's mother?
后续：第14世达赖喇嘛的母亲是谁？
Intermediate answer: The 14th Dalai Lama's mother was Diki Tsering.
中间答案：第14世达赖喇嘛的母亲是迪吉·才仁。
Follow up: Who was Diki Tsering's father?
后续：迪吉·才仁的父亲是谁？
Intermediate answer: Diki Tsering's father was Choekyong Tsering.
中间答案：迪吉·才仁的父亲是秋吉·才仁。
So the final answer is: Choekyong Tsering.
所以最终答案是：秋吉·才仁。
```

这种方法对于涉及多跳信息检索的问题特别有效。通过明确声明中间问题和答案，代理的推理路径透明，更易于验证。顾名思义，这种代理类型与搜索工具紧密关联。

### 代理类型：对话式代理

标准代理通常是无状态的；它们独立处理每个查询。**对话式代理** 旨在通过纳入记忆来克服这一限制。这使它们能够记住对话中的先前交互，并使用该语境来指导其响应。

这些代理通常使用ReAct风格的逻辑，但其构建的提示专门设计用于处理聊天历史。提示模板包含一个对话历史的占位符，该占位符由内存对象管理，如第四章所述。

设置对话式代理涉及创建内存对象并将其传递给 `AgentExecutor`。

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 定义一个内存对象
memory = ConversationBufferWindowMemory(
    k=2, # 记住最近的2次交互
    memory_key="chat_history",
    return_messages=True
)

# 加载支持聊天历史的提示
prompt = hub.pull("hwchase17/react-chat")

# 创建一个对话式代理
agent = create_react_agent(llm, tools, prompt)

# AgentExecutor现在包含内存对象
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

agent_executor.invoke({"input": "Hi, my name is Bob."})
agent_executor.invoke({"input": "What's my name?"})
```

代理现在可以通过访问存储在内存中的对话历史来正确回答第二个问题。

### 选择合适的代理和执行器配置

代理类型的选择依据你的应用需求：

- **ReAct：** 适用于需要灵活工具使用和清晰推理 (inference)路径的通用、复杂任务。它是一个强大的默认选择。
- **Self-Ask with Search：** 适用于基于事实、多跳的问答，其中推理过程的透明度有助理解。
- **对话式：** 构建聊天机器人、助手或任何需要在对话的多个轮次中保持语境的应用所必需。

配置 `AgentExecutor` 时，你可以使用多个参数 (parameter)来帮助控制其行为并使其更可靠。两个值得注意的参数是：

- `max_iterations`：通过设置 `Thought -> Action -> Observation` 循环次数的硬性限制，防止代理陷入长循环或无限循环。
- `handle_parsing_errors`：LLM有时会生成代理无法解析成有效 `Action` 的输出。将此参数设置为 `True` 或提供自定义函数，允许执行器优雅地处理此类错误，例如，通过将错误消息发送回LLM进行自我纠正。

通过了解这些不同的推理策略以及驱动它们的 `AgentExecutor`，你可以构建更精密和专业的代理，根据你应用的独特挑战量身定制。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  解释了ReAct框架，该框架结合推理（Thought）和行动（Action）以增强LLM能力，与讨论的核心代理循环直接相关。
- [Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF3VxwpPO3veW6gE5yru9M2QT4IaxvydZgIq1yMfcmDCvDNQPsx2QCZCEcTK1ffLyX42HFi6m61TzAc2i13xpIKFkneoOKM1HTp6Xv2rHbYdwUoX5QvG81jzsEZ41rCVYkUq8Haj3nC5UCK5x8UA0BNaw==) — LangChain (2024)
  Publisher: LangChain
  LangChain代理模块的官方文档，提供了AgentExecutor、代理类型和配置的实践细节，内容中直接引用。
- [Agent Executors](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html) — LangChain (2024)
  Publisher: LangChain
  LangChain官方文档，详细介绍了AgentExecutor组件、其功能以及如`max_iterations`和`handle_parsing_errors`等配置参数。
