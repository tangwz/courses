---
course: "langchain-production-llm"
chapter: "advanced-memory-management"
lesson: "integrating-memory-agents-chains"
sourceId: 3882
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-3-advanced-memory-management/integrating-memory-agents-chains"
title: "将记忆功能与代理和链条结合"
description: "有效融入高级记忆功能到代理和复杂链条的最佳实践。"
order: 5
plots: []
sourceHash: "a1a0b0e72b1aad034061884c97987fe19b15f43b5df375f8ee9e489a6fbdd2e7"
sourceCorrections: []
---

### 将记忆功能连接到链条

对于链条管理的顺序操作，将记忆功能结合进去的最佳处理方式是使用 LangChain 表达式语言 (LCEL)。标准做法是使用历史管理可运行对象来封装您的链条逻辑。

#### LCEL 的标准链条结合

不再将记忆对象传入旧版链条类，您可以使用 `RunnableWithMessageHistory`。这将您的链条核心逻辑与对话历史的持久化功能分离。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 初始化 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建一个接受历史记录占位符的提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 创建 LCEL 链条
chain = prompt | llm

# 定义一个函数来管理聊天历史状态
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用消息历史功能封装链条
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 链条现在根据 session_id 管理历史记录
response = conversation.invoke(
    {"input": "Hi there! My name is Alex."},
    config={"configurable": {"session_id": "user_123"}}
)
print(response.content)
# 输出: 你好，Alex！很高兴认识你。我今天能帮你什么？

response = conversation.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "user_123"}}
)
print(response.content)
# 输出: 你的名字是 Alex。
```

#### 与检索 (RAG) 结合

在构建既需要长期存储（如向量 (vector)数据库）又需要短期对话记忆的链条时，您可以将 `RunnablePassthrough` 用于检索，并结合 `RunnableWithMessageHistory` 用于对话上下文 (context)。

```python
import faiss
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

# 设置向量存储
embedding_model = OpenAIEmbeddings()
index = faiss.IndexFlatL2(1536)
# 初始化空存储
vectorstore = FAISS(embedding_model, index, InMemoryDocstore({}), {})
vectorstore.add_texts(["Bob lives in California.", "Bob enjoys hiking."])
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 设置 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建包含上下文和历史记录的提示
template = """仅根据以下上下文回答问题：
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# RAG 链条
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"]))
    )
    | prompt
    | llm
)

# 状态管理
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversation_with_retrieval = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 交互
response = conversation_with_retrieval.invoke(
    {"input": "Where does Bob live?"},
    config={"configurable": {"session_id": "bob_session"}}
)
print(response.content)
# 输出: Bob 住在加利福尼亚。
```

### 将记忆功能与代理结合

对于生产级别应用，LangGraph 是构建代理的标准环境。LangGraph 将代理视为状态机。在此背景下，记忆功能由“检查点”处理，它们在交互之间持久化图的状态（包括聊天历史）。

#### LangGraph 代理结合

初始化 LangGraph 代理时，您提供一个 `checkpointer`。该组件在每次节点执行后保存状态，使代理能够有效恢复或参考过去的交互。

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 定义工具
@tool
def get_weather(city: str):
    """获取一个城市的天气。"""
    return "sunny"

tools = [get_weather]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 初始化一个 Checkpointer 用于记忆持久化
# 在生产环境中，使用 PostgresSaver 或类似工具
memory = MemorySaver()

# 创建代理
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# 运行带有包含 thread_id 配置的代理
config = {"configurable": {"thread_id": "thread_1"}}

# 第一次交互
response = agent_executor.invoke(
    {"messages": [("user", "My name is Clara.")]},
    config=config
)

# 第二次交互 - 通过 thread_id 访问记忆
response = agent_executor.invoke(
    {"messages": [("user", "What's my name?")]},
    config=config
)

# 响应对象包含完整状态，包括答案
print(response["messages"][-1].content)
# 输出: Clara
```

在这种架构中，您无需手动将记忆对象传递给提示占位符。相反，`checkpointer` 会在执行前自动加载与 `thread_id` 相关的图状态（包括消息列表），并在执行后保存更新后的状态。

#### 自定义图控制

如果您使用 `StateGraph` 构建自定义图，而不是使用预构建代理，记忆功能的工作方式类似。您需要定义状态模式（通常包括消息列表），并将检查点器传递给编译后的图。

```python
# 运行带有记忆的自定义图的示例（概念性）
# graph = StateGraph(StateSchema)
# ... 定义节点和边 ...
# app = graph.compile(checkpointer=memory)
# app.invoke(inputs, config={"configurable": {"thread_id": "123"}})
```

这种做法可以细致控制所保存的内容。例如，您可以定义状态来存储特定变量以及消息历史记录。

### 结合流程可视化

下图展示了记忆功能（检查点）如何融入 LangGraph 执行过程。

> 该图展示了图代理内部的循环。状态在开始时从检查点加载，在每次节点执行后更新并保存，即使进程中断也能确保持久化。

### 高级结合

- **会话管理：** 结合 `RunnableWithMessageHistory` 和 LangGraph 检查点，`session_id` 或 `thread_id` 很重要。这个键决定了加载哪个历史上下文 (context)。在 Web 应用中，此 ID 应与用户的会话或特定对话主题绑定。
- **状态细粒度：** 在 LangGraph 中，您定义状态的模式。这使得您能够存储结构化数据（例如，“用户偏好”、“当前任务状态”）以及原始消息列表。这比旧版链条的基于字符串或列表的记忆功能强大得多。
- **并发：** 使用持久化检查点（如 `PostgresSaver` 或 `SqliteSaver`）时，并发性在数据库层面处理。确保您的数据库配置支持来自不同线程的并发写入负载。

## 参考资料

- [LangChain Documentation: Memory](https://python.langchain.com/docs/modules/memory/) — LangChain Developers (2025)
  Publisher: LangChain Inc.
  提供了LangChain框架内各种记忆类型及其集成的详细解释和示例。
- [LangChain Documentation: Agents](https://python.langchain.com/docs/modules/agents/) — LangChain Developers (2024)
  Publisher: LangChain
  LangChain官方指南，涵盖了代理的架构、实现和集成，包括记忆管理。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）架构，该架构是`VectorStoreRetrieverMemory`从外部知识库检索相关信息的基础。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  提出了一个生成代理的计算框架，展示了记忆、规划和反思对于代理长期一致行为的重要性。
