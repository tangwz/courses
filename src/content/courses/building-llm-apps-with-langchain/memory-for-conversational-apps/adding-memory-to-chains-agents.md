---
course: "building-llm-apps-with-langchain"
chapter: "memory-for-conversational-apps"
lesson: "adding-memory-to-chains-agents"
sourceId: 7426
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-4-memory-for-conversational-apps/adding-memory-to-chains-agents"
title: "为链和代理增加记忆功能"
description: "一份关于将记忆对象整合到大型语言模型链（LLMChain）或对话式代理中的实用步骤指南。"
order: 5
plots: []
sourceHash: "7a52afd2369eef30b447662d335e539386cf3b9e3ad85069a4ad4dc3b7c85325"
sourceCorrections: []
---

### 结合记忆与链

使应用程序保持状态的现代方法是，通过封装链的执行过程并加入历史管理能力。使用LangChain表达式语言（LCEL），我们可以定义一个链，然后通过`RunnableWithMessageHistory`来提升它的功能。这个组件管理对话历史，在提示执行前自动读取之前的消息，并用新的交互更新历史记录。

为使其运行，需要如下设置：

1. 包含一个用于对话历史的`MessagesPlaceholder`的`ChatPromptTemplate`。
2. 一个`get_session_history`函数，用于获取或创建针对给定会话ID的消息历史对象。

我们来看一个实际实现方案。我们将使用`ChatOpenAI`创建一个简单的链，并为其加入记忆功能。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 设置语言模型
# 请确保您的OPENAI_API_KEY已在环境变量中设置
llm = ChatOpenAI(temperature=0.7)

# 1. 定义ChatPromptTemplate
# 我们使用MessagesPlaceholder来动态插入聊天历史。
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的聊天机器人，正在与人类对话。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 2. 使用LCEL创建链
chain = prompt | llm

# 3. 定义记忆管理
# 我们需要一个函数来获取特定会话的历史记录。
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. 用消息历史包装链
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 第一次交互
# 我们必须提供一个包含session_id的配置，以保留历史记录
response_1 = conversation_chain.invoke(
    {"input": "Hi there! My name is Alex."},
    config={"configurable": {"session_id": "session_1"}}
)
print(response_1.content)
# > Hi Alex! It's nice to meet you. My name is AI. How can I help you today?

# 第二次交互
response_2 = conversation_chain.invoke(
    {"input": "What was the name I just told you?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(response_2.content)
# > You told me your name is Alex.
```

以下图表说明此数据流。`RunnableWithMessageHistory`包装器充当一个有状态管理器，在每次执行期间从记忆存储中读取数据并写入数据。

> 有状态链执行的内部运作流程。每次运行时，包装器为提示词 (prompt)提供之前的上下文 (context)，在大型语言模型生成响应后，记忆会用最新一轮对话进行更新。

### 结合记忆与代理

代理使用大型语言模型来做决定，关于使用哪些工具，它们也需要记忆功能来处理多轮交互。例如，用户可能会让代理在线查找信息，然后问一个关于查找结果的后续问题。没有记忆功能，代理将没有上下文 (context)来回答第二个问题。

在现代LangChain中，我们定义代理，然后使用`AgentExecutor`来执行它。记忆通常通过将一个`memory`对象传递给执行器来管理。这使代理能够维持跨越不同步骤和交互的上下文。

这里是一个对话式代理的示例，它使用一个搜索工具并记住之前的交互。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain import hub

# 初始化模型
llm = ChatOpenAI(temperature=0)

# 加载工具
# 使用'ddg-search'需要duckduckgo-search包
tools = load_tools(["ddg-search"], llm=llm)

# 获取用于工具调用代理的标准提示词
# 这个提示词包含一个用于'chat_history'的占位符
prompt = hub.pull("hwchase17/openai-tools-agent")

# 初始化代理
agent = create_tool_calling_agent(llm, tools, prompt)

# 初始化记忆功能
# 对于聊天模型，'return_messages=True'是很重要的
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 初始化AgentExecutor
# 我们将记忆传递给执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 第一次交互
agent_executor.invoke({"input": "Who is the current CEO of NVIDIA?"})

# 第二次交互，引用第一次交互的内容
agent_executor.invoke({"input": "What year was that company founded?"})
```

在第二次调用`agent_executor.invoke()`时，代理明白“那家公司”指的是NVIDIA。这是因为`AgentExecutor`自动加载来自`ConversationBufferMemory`的上下文，并将其包含在发送给大型语言模型的提示词 (prompt)中。大型语言模型随后看到之前的问题和答案，使其能够正确解析这个指代，并使用搜索工具找到NVIDIA的成立年份。

此机制依赖于提示词具有一个特定的占位符（通常是`chat_history`），`AgentExecutor`在调用代理之前会从记忆对象中填充数据。这种简单的配置将一个无状态的工具使用系统转换成一个有能力的对话助手。

## 参考资料

- [Memory](https://python.langchain.com/docs/modules/memory/) — LangChain Contributors (2023)
  Publisher: LangChain Inc.
  LangChain官方文档，解释各种内存类型、其配置以及它们如何在LLM应用中管理对话历史。
- [Agents](https://python.langchain.com/docs/modules/agents/) — LangChain Contributors (2024)
  Publisher: LangChain
  LangChain官方文档，详细介绍了智能体、其设计原则、支持的类型以及它们如何集成工具和内存以进行复杂的多轮交互。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct范式的基础论文，该范式将语言模型的推理和行动步骤结合起来。这种方法对于理解LangChain等复杂智能体如何使用LLM进行决策和利用带内存的工具很重要。
