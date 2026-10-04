---
course: "getting-started-with-llm-toolkit"
chapter: "conversational-applications-with-memory"
lesson: "challenge-of-stateful-conversations"
sourceId: 7778
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-7-conversational-applications-with-memory/challenge-of-stateful-conversations"
title: "有状态对话的难题"
description: "理解大型语言模型为何无状态，以及构建持续对话为何需要记忆。"
order: 1
plots: []
sourceHash: "3df1f32c3002e24d942a222f441aca0de076593567abec93c8fe8ccd66ed97ec"
sourceCorrections: []
---

语言模型在设计上是按每次请求运行的。每次你向大型语言模型 (LLM) 发送提示时，它都会独立处理该输入，没有关于你之前交流的固有记忆。可以将其想象成一位才华横溢但健忘的专家；你可以提出任何问题，但每次都必须提供所有必要的背景信息。

这种无状态特性在构建需要持续对话的应用时，是一个重要的阻碍。对话不仅仅是一系列不相关的问题和回答；它是一种累积的交流，上下文 (context)会随时间累加。如果没有一个机制来保存此上下文，对话流程就会中断。

例如，请看这个简单的交流：

**用户:** 我叫Alex。我对机器学习 (machine learning)很感兴趣。

**助手:** 你好！机器学习是一个引人入胜的话题。你想具体了解什么？

**用户:** 主要类型有哪些？

**助手:** 主要类型有监督学习 (supervised learning)、无监督学习 (unsupervised learning)和强化学习 (reinforcement learning)。今天我能帮你什么？

在最后一轮中，助手已经忘记了用户的名字和已确定的对话主题。这迫使用户重复信息，并使交流感觉不连贯、不自然。

### 无状态API模型

这种行为是大型语言模型API通常设计方式的直接结果。每次调用都是一个独立的、无状态的事务。模型接收输入，处理它，生成输出，然后丢弃与该请求相关的状态。这种架构确保了可扩展性和可预测性，但将管理对话上下文 (context)的负担完全放在了开发者的应用上。

> 每次API调用都是一个独立的事务。交互2中的模型没有关于交互1中发生过什么的记忆。

为了构建连贯的对话，我们的应用必须充当模型的记忆。标准做法是收集交流历史，并将其包含在每个新的用户消息中。通过发送对话记录，我们为大型语言模型提供了必要的上下文，以生成相关的、有状态的回复。

尽管你可以用一个简单的Python列表来管理这个，但这种方法会变得很麻烦。最好使用一个专门的结构来管理对话历史。`ConversationBuffer` 类正是为此目的设计的。

```python
from kerb.memory import ConversationBuffer

# 初始化一个缓冲区来存储对话
buffer = ConversationBuffer()

# 第一轮
buffer.add_message("user", "My name is Alex. I'm learning about machine learning.")
buffer.add_message("assistant", "Hello Alex! Machine learning is a great topic. Where should we start?")

# 第二轮
buffer.add_message("user", "What are the main types?")
# 为了获得有状态的回复，我们现在会将整个缓冲区历史发送给LLM。
# 在这里，我们只将预期回复添加到缓冲区。
buffer.add_message("assistant", "The main types are supervised, unsupervised, and reinforcement learning.")

# 你可以查看存储的消息
print(f"Messages stored: {len(buffer.messages)}")
for msg in buffer.messages:
    print(f"- {msg.role}: {msg.content}")
```

这种方法有效地解决了无状态问题，但它引入了一个新的、重要的限制：上下文窗口。语言模型一次只能处理有限量的文本，这个限制以 token 衡量。随着对话的进行，我们随每个请求发送的历史记录也随之增长。最终，历史记录中的 token 总数加上新的用户查询将超过模型的上下文窗口，导致错误。此外，每次请求都发送长的历史记录会增加API成本和延迟。

有效管理在提供足够上下文和保持在 token 限制内之间的权衡，是构建对话应用的核心问题。接下来的部分将讨论处理此问题的不同记忆策略，从最直接的方法——对话缓冲区——开始。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇基础论文介绍了Transformer架构，它是现代大型语言模型（LLM）的基础。它阐明了这些模型如何以无状态、每次请求独立的方式处理输入。
- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这是一本关于自然语言处理和计算语言学的综合性教材，其中有专门章节讨论对话系统以及管理对话状态和上下文的关键方面。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Volume: 33; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  本文介绍了检索增强生成（RAG），这是一种通过从外部知识库检索信息来扩展语言模型的方法。它提供了一种超越模型固有上下文窗口限制来管理和扩展上下文的策略，与对话历史问题相关。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  这篇综合性综述回顾了大型语言模型（LLM）的最新进展和现有挑战。它涵盖了多个方面，包括导致LLM无状态的架构设计，以及用于构建对话应用程序的技术。
