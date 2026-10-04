---
course: "prompt-engineering-llm-application-development"
chapter: "building-applications-llm-frameworks"
lesson: "managing-memory-llm-applications"
sourceId: 3048
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks/managing-memory-llm-applications"
title: "LLM应用中的记忆管理"
description: "在应用中实现对话历史和记忆管理。"
order: 4
plots: []
sourceHash: "e04d6f921e62c5c5179d1dc4d7d52a38e506254bae509525deea8f35d66062fa"
sourceCorrections: []
---

大多数通过API与大型语言模型（LLM）的交互本质上是无状态的。每次API调用都被独立处理，不会记住之前的任何交流。这对于独立任务来说是高效的，但当构建需要记住正在进行的交互上下文 (context)的对话式应用（例如聊天机器人或助手）时，这会成为一个很大的局限。如果没有记忆功能，LLM无法回顾对话的早期部分，导致回应支离破碎且重复。

像LangChain这样的LLM框架提供了专门的组件，通常称为“记忆”模块，以应对这一难题。这些模块存储有关过往交互的信息，并在后续调用时将其作为上下文的一部分提供给LLM。这使得模型能够保持对话连贯性。

### 记忆类型

框架通常提供几种记忆机制，每种都有不同的策略来存储和检索对话历史。让我们来看一下LangChain中常见的几种类型：

#### 1. ConversationBufferMemory

这是最简单的记忆形式。它逐字逐句地保留对话消息的历史记录，并在每次新查询时将所有这些消息包含在发送给LLM的上下文 (context)中。

- **工作方式：** 它存储完整的用户输入和AI回应列表。
- **优点：** 易于理解和实现。确保LLM拥有完整的最新上下文。
- **缺点：** 对于长对话，可能很快超出LLM的上下文窗口限制，导致错误或截断。每次对话轮次，Token用量（和成本）都会增加。

这里是一个示例，说明如何将其集成到LangChain链中（假设`llm`和`prompt_template`已定义）：

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI # 或您偏好的LLM提供商

# 初始化LLM（替换为您的实际模型设置）
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# 初始化记忆
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True # 设置为True可查看发送的完整提示
)

# 第一次交互
response1 = conversation.predict(input="Hi there! My name is Bob.")
print(response1)
# 示例输出: Hello Bob! It's nice to meet you. How can I help you today?

# 第二次交互 - 记忆提供上下文
response2 = conversation.predict(input="What is my name?")
print(response2)
# 示例输出: Your name is Bob.
```

如果`verbose=True`，您会看到记忆如何将前一轮对话（“Human: Hi there! My name is Bob.\nAI: Hello Bob! ...”）注入到第二次调用的提示中。

#### 2. ConversationBufferWindowMemory

为了减轻`ConversationBufferMemory`的上下文长度限制，此变体只保留最后$k$次交互。

- **工作方式：** 存储固定数量的最近对话轮次。当达到限制时，最旧的交互将被丢弃。
- **优点：** 防止上下文无限增长，有助于保持在Token限制内。
- **缺点：** 较旧的上下文会丢失，这对于需要回忆早期细节的长时间对话可能很重要。

您通过指定`k`来配置它：

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最后3次交互（用户消息 + AI回应 = 1次交互）
window_memory = ConversationBufferWindowMemory(k=3)

# 在创建ConversationChain时使用此记忆实例
conversation_window = ConversationChain(
    llm=llm,
    memory=window_memory,
    verbose=True
)

# 示例交互...
# conversation_window.predict(input="Turn 1: User message")
# conversation_window.predict(input="Turn 2: User message")
# conversation_window.predict(input="Turn 3: User message")
# conversation_window.predict(input="Turn 4: User message") # 第一轮将被丢弃
```

#### 3. ConversationSummaryMemory 与 ConversationSummaryBufferMemory

对于非常长的对话，即使是窗口记忆也不足时，可以采用摘要技术。

- **ConversationSummaryMemory：** 它不是存储原始交互，而是使用LLM逐步创建对话历史的摘要。然后将此摘要作为上下文传递。
  - **优点：** 即使对于极长的对话，也能使上下文长度保持可控。
  - **缺点：** 需要额外的LLM调用来生成摘要，从而增加延迟和成本。摘要可能会丢失具体细节。
- **ConversationSummaryBufferMemory：** 一种混合方法。它保留了近期交互的缓冲区（类似于`ConversationBufferMemory`），但也维护了较旧交互的摘要。一旦缓冲区超过某个Token限制，缓冲区中最旧的交互将被摘要并添加到主摘要中。
  - **优点：** 平衡了细节（来自缓冲区）与简洁性（来自摘要）。比每轮都进行摘要更高效。
  - **缺点：** 配置更复杂。仍然涉及摘要成本和潜在的细节丢失。

选择合适的记忆类型取决于应用的需求：对话的预期长度、保留旧细节的重要性，以及对成本和延迟的容忍度。

### 记忆集成

在像LangChain这样的框架中，记忆模块设计为可以顺畅地集成到`Chains`或`Agents`中。正如示例所示，您通常会初始化一个记忆对象并将其传递给链的构造函数。然后，链会在调用LLM之前自动处理从记忆中加载相关上下文 (context)，并在调用后将最新的交互保存回记忆中。

> 此图示了记忆模块在对话轮次期间如何在应用链中进行交互的流程。

### 考量因素

- **上下文 (context)窗口限制：** 始终注意LLM的最大上下文窗口大小。选择一种记忆类型（如窗口记忆或摘要记忆），以防止超出此限制。
- **成本：** 记忆机制，特别是摘要类型，可能会增加处理的Token数量或需要额外的LLM调用，从而影响API成本。
- **信息丢失：** 摘要或窗口记忆本质上涉及信息丢弃。评估这种丢失对您的使用场景是否可接受。
- **复杂性：** 尽管框架简化了记忆管理，但理解不同记忆类型的工作方式及其影响对于构建高效应用来说很重要。

通过纳入适当的记忆模块，您可以将无状态的LLM交互转换为有状态、连贯的对话，大幅提升聊天机器人、助手以及其他使用框架构建的多轮应用的用户体验。

## 参考资料

- [Memory](https://python.langchain.com/docs/modules/memory/) — LangChain (2025)
  Publisher: LangChain Inc.
  LangChain内存模块的官方文档，提供了本节讨论的各种内存类型的实现细节、示例和描述。
- [Building Applications with GPT-4 and ChatGPT](https://www.oreilly.com/library/view/building-applications-with/9781098144675/) — Ben Forta (2023)
  Publisher: O'Reilly Media
  提供了开发基于LLM的应用程序的实用指导，包括管理会话上下文和状态的策略，其中涵盖了内存管理。
- [A Survey on Dialogue Systems: Recent Advances and New Challenges](https://ieeexplore.ieee.org/document/8863674) — Ruihong Zhao, Kai Zheng, Yan Liu, Yuchuan Luo, Yanchun Zhao, and Philip S. Yu (2021)
  Journal: IEEE Transactions on Knowledge and Data Engineering; Publisher: IEEE; Volume: 33; Pages: 1475-1493; DOI: [10.1109/TKDE.2019.2945722](https://doi.org/10.1109/TKDE.2019.2945722)
  提供了对话系统的学术概述、其组件以及管理对话状态和上下文的历史和持续挑战，这构成了LLM应用程序中内存的理论基础。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  综述了扩展LLM上下文和知识的技术，解决了直接上下文窗口的局限性以及通过摘要或外部内存检索等机制管理对话历史的需求。
