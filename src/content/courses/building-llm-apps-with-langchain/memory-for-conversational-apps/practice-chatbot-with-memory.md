---
course: "building-llm-apps-with-langchain"
chapter: "memory-for-conversational-apps"
lesson: "practice-chatbot-with-memory"
sourceId: 7427
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-4-memory-for-conversational-apps/practice-chatbot-with-memory"
title: "实践：构建一个带记忆的聊天机器人"
description: "一个实操练习，用于创建一个能记住之前对话内容的聊天机器人，它使用ConversationBufferMemory。"
order: 6
plots: []
sourceHash: "af54556dfebb37f7a5f9c8c5ed33498d03b3306fe0d14b4caf0c8ac83282e0fa"
sourceCorrections: []
---

构建一个能记住之前对话的简单聊天机器人，需要在实践中应用记忆组件。这个实操例子展示了如何将记忆对象整合到链中，以便在对话的多个轮次中保持状态。

### 环境设置

首先，让我们准备环境。我们需要安装必要的库并配置API密钥。本例将使用OpenAI API，但其原理适用于LangChain支持的任何LLM。

```python
# 确保已安装所需的库
# pip install langchain langchain-openai langchain-community python-dotenv

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 从.env文件加载环境变量
load_dotenv()

# 建议将您的API密钥设置为环境变量
# os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

### 无状态性问题：快速回顾

在添加记忆之前，让我们快速说明我们正在解决的问题。如果没有记忆组件，每次对LLM的调用都是独立的。如果您提出一个后续问题，模型将无法了解您之前陈述的背景。

考虑以下对话：

1. **用户：** “我最喜欢的动物是墨西哥钝口螈。”
2. **LLM：** “这是一个有趣的偏好！墨西哥钝口螈是一种迷人的两栖动物。”
3. **用户：** “我最喜欢的动物是什么？”
4. **LLM（无记忆）：** “抱歉，我不知道您最喜欢的动物是什么。您没有告诉我。”

模型在后续问题上失败了，因为第三轮对话的处理完全独立于第一轮。

### 构建对话链

为了解决这个问题，我们将使用LangChain的`RunnableWithMessageHistory`。这个工具将标准链（模型+提示）包装起来，并根据会话ID自动管理从历史存储读写数据。

#### 1. 初始化LLM和提示

我们首先定义模型和提示模板。提示必须包含一个消息历史的占位符，这样模型才能看到之前的轮次。

```python
# 初始化LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# 定义提示模板
# 我们包含一个系统消息来设定行为，一个历史记录的占位符，
# 以及一个新用户输入的模板。
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，会记住对话中的细节。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 创建基本链
chain = prompt | llm
```

`variable_name="chat_history"`告知系统将过去的消息注入何处。

#### 2. 定义历史存储

接下来，我们需要一种方式来存储对话数据。本例中，我们将使用一个简单的字典来存放`InMemoryChatMessageHistory`对象，以会话ID为键。在生产应用中，您可能会在这里使用Redis等数据库。

```python
# 存储不同会话对话历史的字典
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

#### 3. 用记忆包装链

现在我们使用`RunnableWithMessageHistory`来包装我们的基本链。这个对象连接我们的链、历史工厂函数以及提示中使用的特定键。

```python
# 创建对话可运行对象
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

我们指定`input_messages_key="input"`以匹配我们提示中的`{input}`变量，并指定`history_messages_key="chat_history"`以匹配`MessagesPlaceholder`。

### 与有状态聊天机器人交互

现在，让我们测试聊天机器人。我们使用`.invoke()`方法，传入输入和一个包含`session_id`的`config`字典。这个ID让系统获取正确的历史记录。

**首次交互：**
我们提供一些初始信息。

```python
# 带有特定会话ID的第一个用户输入
config = {"configurable": {"session_id": "user_123"}}

response = conversation.invoke(
    {"input": "你好，我叫克拉拉。我正在构建一个聊天机器人。"},
    config=config
)

print(response.content)
```

聊天机器人会提供友好的问候。`InMemoryChatMessageHistory`现在已将我们的输入和模型的响应存储在键`user_123`下。

**第二次交互：**
现在进行真正的测试。让我们问一个依赖于第一条消息上下文 (context)的后续问题。

```python
# 使用相同会话ID的第二个用户输入
response = conversation.invoke(
    {"input": "我叫什么名字？"},
    config=config
)

print(response.content)
```

预期输出将类似于：

```
你的名字是克拉拉。
```

成功了。模型正确识别了名称，因为可运行对象获取了`user_123`的历史记录，将其插入到提示中，并将完整的背景信息发送给LLM。

### 带记忆的对话流程

每次交互的流程遵循清晰的循环模式，这确保了背景信息的保存和利用。

> 此图展示了单个对话轮次的生命周期。可运行对象从历史存储中读取数据，格式化提示，从LLM获取响应，然后将新的交互写回存储，为下一轮对话做好准备。

这个实践展示了构建有状态应用的基本机制。虽然将所有历史保存在内存中很简单，但对于短会话来说它很强大。对于需要非常长对话的应用，您可以采用策略来总结旧消息或截断历史记录，这可以通过修改历史记录获取或链步骤中的逻辑来处理。

## 参考资料

- [LangChain Documentation: Memory](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFqyvncGEdXeL47cu4U9WK_3JSKANf1TNua9EAQZsb4aiCj8MrhD0_1Bn0cfqPSoOq9Fs03NmnskNM638oKjoxlEYVufMLchTZJpDUnnVWsnvnf56tP2TE5S4n9WlIhr5-T194HOrKcfySXk7Yh4B5dYJ7UXyx0Hycyiik=) — LangChain Developers (2023)
  Publisher: LangChain Inc.
  LangChain官方内存模块文档，解释了不同内存类型及其在对话应用链中的使用方法。
- [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) — Harrison Chase, Andrew Ng (2023)
  Publisher: DeepLearning.AI
  一个在线短课程，涵盖了LangChain的核心概念，包括用于构建对话式AI代理的内存实现的实践演示。
