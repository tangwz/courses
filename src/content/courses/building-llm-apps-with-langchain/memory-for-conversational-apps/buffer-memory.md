---
course: "building-llm-apps-with-langchain"
chapter: "memory-for-conversational-apps"
lesson: "buffer-memory"
sourceId: 7423
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-4-memory-for-conversational-apps/buffer-memory"
title: "用于短期回顾的缓冲区记忆"
description: "学习如何实现ConversationBufferMemory来维护对话历史的运行日志。"
order: 2
plots: []
sourceHash: "61e060fef111cf83b97591311eeae5c68bc54531d7115d82cefec68fe1569ea8"
sourceCorrections: []
---

让应用程序拥有记忆最直接的方式是存储整个对话历史，并将其包含在每一个后续请求中。这种做法常被称为**缓冲区记忆**。它会逐字记录所有用户输入和模型输出，确保大型语言模型（LLM）在对话的下一轮拥有完整的上下文 (context)信息。

这种方式简单且对需要回顾具体细节的短对话很有用。在现代LangChain应用中，这通过管理一个聊天历史存储并为每次新交互将完整的消息列表注入到提示模板中来实现。

### 缓冲区记忆的工作原理

其机制很直接。消息被收集并存储在一个历史对象中。当链执行时，这个过去消息列表会被取回并插入到发送给大型语言模型的提示中，通常是使用一个占位符变量。这为模型提供了完整的对话线索。

下图说明了这一循环过程。每次用户发送消息时，系统都会更新消息历史，然后该历史用于构建下一次大型语言模型调用的提示。

> 带有缓冲区记忆的对话链的流程。历史存储会用最新的交流更新，然后用于填充下一个提示的上下文 (context)。

### 实现缓冲区记忆

我们来实践一下。要实现这种模式，我们使用 `RunnableWithMessageHistory` 来包装我们的链。这个组件会自动处理从历史存储读取和写入的逻辑。

首先，请确保您已设置好环境变量，例如您的 `OPENAI_API_KEY`。

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# 设置您的 API 密钥
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 1. 初始化 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 2. 创建提示模板
# 我们使用占位符来注入对话历史
prompt = ChatPromptTemplate.from_messages([
    ("system", "以下是人类与人工智能之间的一次友好对话。AI健谈，并会从其上下文中提供大量具体细节。如果AI不知道某个问题的答案，它会如实说自己不知道。"),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])

# 3. 创建链
chain = prompt | llm | StrOutputParser()

# 4. 设置记忆管理
# 我们需要一个字典来存储不同会话的历史记录
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 用消息历史功能包装链
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 开始对话
# 我们必须提供一个 session_id 配置来跟踪独立的对话
config = {"configurable": {"session_id": "alex_session"}}

response1 = conversation.invoke(
    {"input": "Hi, my name is Alex. What's the biggest planet in our solar system?"},
    config=config
)
print(response1)

# 继续对话
response2 = conversation.invoke(
    {"input": "Great, and what's its most famous feature?"},
    config=config
)
print(response2)

# 模型记住了上下文（“木星”）
response3 = conversation.invoke(
    {"input": "What was the name I gave you earlier?"},
    config=config
)
print(response3)
```

当您运行此代码时，`RunnableWithMessageHistory` 会查找会话历史（如果不存在则创建），并将存储的消息注入到 `{history}` 占位符中。对于第二轮，系统会构建一个类似于这样的提示：

```text
以下是人类与人工智能之间的一次友好对话...

[系统消息]
[人类消息]: 你好，我叫Alex。我们太阳系中最大的行星是什么？
[AI 消息]: 你好，Alex！我们太阳系中最大的行星是木星...

[人类消息]: 太棒了，它最著名的特征是什么？
```

历史对象已追加了第一次人机交流，为模型理解新问题指的是木星提供了必要的上下文 (context)。

### 优点与局限

这种缓冲区记忆做法的主要优点是它的完美回顾能力。模型收到完整、未经更改的历史记录，这最大限度地减少了在中短对话中误解上下文 (context)的风险。

然而，主要局限在于令牌消耗。随着对话的继续，存储的历史会线性增长。这会带来两个直接后果：

1. **成本增加：** 历史中的每个令牌都会随每一次新的 API 调用发送，这会大幅增加长时间对话的成本。
2. **上下文窗口限制：** 最终，对话历史加上新的输入将超出大型语言模型（LLM）的最大上下文窗口（例如，4,096 或 16,384 个令牌）。当这种情况发生时，API 调用将失败。

由于这一局限，这种全缓冲区方法最适用于预期对话相对简短的应用程序，例如处理单个问题的客户支持聊天机器人或简单的任务型助手。对于需要长期记忆的应用程序，您将需要更进阶的做法，我们将在下文介绍。

## 参考资料

- [Memory](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF0N-SdGr6wszWv2YUyslL6LTgKdoT9Qd-mjLV7DvpgbiQYJ0ocstlgA0zFbeECg9sb86sYsqkYayTQZ7rpV4z8e2FY6jRy846h0TWuIF--z97EVy30GTMjPAw5zcE839LojiHNAOWgDKO6X49cAv7woCZHyz7J4c8XFSs=) — LangChain (2025)
  Publisher: LangChain Inc.
  LangChain 内存模块的官方文档，详细介绍了 `ConversationBufferMemory` 及其他内存类型的实现和使用方法。
- [Prompt engineering](https://platform.openai.com/docs/guides/llms/best-practices-for-prompt-engineering-with-the-openai-api) — OpenAI (2024)
  Publisher: OpenAI
  解释了 OpenAI 语言模型提示工程和上下文管理的基本原理，这与文中讨论的上下文窗口限制直接相关。
- [Building LLM-Powered Applications: Learn to use Large Language Models for tasks like chatbots, summarization, and content creation](https://www.packtpub.com/product/building-llm-powered-applications/9781837637845) — Josh Starmer (2023)
  Publisher: Packt Publishing
  为 LLM 应用的架构和实现提供了实用见解，包括管理对话状态和上下文的策略，这些都是 `ConversationBufferMemory` 等内存概念的基础。
