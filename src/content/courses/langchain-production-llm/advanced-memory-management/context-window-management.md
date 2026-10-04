---
course: "langchain-production-llm"
chapter: "advanced-memory-management"
lesson: "context-window-management"
sourceId: 3878
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-3-advanced-memory-management/context-window-management"
title: "上下文窗口管理策略"
description: "用于总结、压缩或选择性检索信息以适应LLM上下文限制的方法。"
order: 3
plots: []
sourceHash: "db701b40360ec0f0892dbea2024df2f791796f5c7767257cd70f2bf882a8199f"
sourceCorrections: []
---

在LLM上下文 (context)窗口的限制内有效管理对话历史，是构建复杂、有状态应用的一个基本难题。虽然简单的缓冲区足以应对简短的交流，但处理长时间交互或大型背景文档的生产系统需要更高级的方案。未能有效管理上下文窗口会导致回答不连贯、必要信息丢失，并最终导致糟糕的用户体验。

核心问题是容量有限。LLM在单个推理 (inference)请求中只能处理有限数量的令牌（即其上下文窗口）。随着对话或相关数据的增加，您必须决定哪些信息最重要并保留在该窗口内。本节介绍如何明智地做出这些决定的方法。

### 总结技术

总结是指将交互历史中较旧的部分浓缩成更短的形式，从而为新消息腾出上下文 (context)窗口空间。LangChain为此提供了内置机制：

- **`ConversationSummaryMemory`：** 此方案维护整个对话的运行总结。每次交流后，历史记录（包括之前的总结和新消息）会发送给LLM，LLM生成一个更新的、合并的总结。虽然这确保了对话的精髓得以保留，但它会在每一步都进行一次LLM总结调用，增加了延迟和成本。上下文的质量完全由LLM生成良好总结的能力来决定。
- **`ConversationSummaryBufferMemory`：** 一种实用的混合方法，这种内存会逐字保留最近交互的缓冲区，同时维护较旧交流的总结。它使用`max_token_limit`参数 (parameter)。交互会添加到缓冲区，直到超出令牌限制。此时，缓冲区中最旧的消息会被总结（通过调用LLM），并合并到现有的总结部分，从而在缓冲区中腾出空间。这平衡了对近期细节的需求和浓缩历史的必要性，通常能在成本、延迟和信息保留之间提供一个不错的折中方案。

请仔细考虑权衡。总结保留了长期上下文，但会引入计算开销，并且可能依据总结质量造成信息损失。对于对延迟敏感的应用，可以考虑后台总结任务等方法。

### 滑动窗口缓冲区

最简单的方法是只保留对话的最新部分。

- **`ConversationBufferWindowMemory`：** 这种内存类型会跟踪最后`k`次交互（用户输入和AI响应对）。当发生新的交互并且历史记录超出`k`轮时，最旧的交互将被丢弃。

```python
# 示例：配置一个用于保留最近3轮交互的滑动窗口内存
from langchain.memory import ConversationBufferWindowMemory

# 保留最近k=3次交互（输入/输出对）
window_memory = ConversationBufferWindowMemory(k=3)
```

这种方法计算成本低廉且易于实现。其主要缺点是上下文 (context)的突然丢失。早于`k`次交互的信息将完全被遗忘，无论其与当前轮次的潜在关联性如何。这适用于上下文主要局限于本地、长期依赖性较小的应用。

### 令牌限制约束

许多内存模块，特别是基于缓冲区的模块，允许您指定`max_token_limit`。或者，可以在链中直接应用显式令牌裁剪工具（如`langchain_core`中的`trim_messages`）来强制执行限制。这相当于对发送给LLM的上下文 (context)中包含的令牌数量设置了一个硬性上限。LangChain的内存类或裁剪工具使用此限制来决定何时修剪消息、触发总结（如`ConversationSummaryBufferMemory`中所示）或仅仅截断历史记录。设置适当的令牌限制很有必要：

1. **防止错误：** 避免超出LLM的最大上下文大小。
2. **成本控制：** 直接管理每次API调用处理的令牌数量。
3. **性能：** 较短的上下文通常会带来更快的LLM响应时间。

此限制通常与总结或窗口化等其他方法配合使用。

### 从内存存储中选择性检索

向量 (vector)存储支持的内存改变了线性存储对话历史的方式。它不是试图将可能庞大的历史记录塞入一个小的窗口，而是将整个历史记录（或相关文档）外部存储，通常在一个向量数据库中。

- **`VectorStoreRetrieverMemory`：** 当应用需要上下文 (context)时，这种内存使用检索器（通常基于语义相似性搜索）根据当前输入或查询，从向量存储中查找最相关的片段。只有这些相关的片段，可能与最新消息结合，才会被注入LLM提示中。

这种方法有效地将总历史记录大小与LLM的上下文窗口解耦。如果与当前话题相关，应用可以从很长的对话的早期"记住"信息。其效果受检索机制的质量影响——在需要时呈现*正确*的过往信息的能力。这与第4章中讨论的RAG技术密切相关。

### 对话压缩

与总结不同，压缩旨在减少历史记录的令牌数量，同时保留最重要的信息和关系，通常使用LLM来完成这项任务。

- **基于实体的内存：** 像`ConversationKGMemory`这样的系统试图从对话中构建知识图谱，提取重要的实体及其关系。提供给LLM的上下文 (context)可能包含最新消息和从该图谱合成的相关事实。
- **自定义压缩逻辑：** 更高级的实现可能涉及一个专门的LangChain序列（通过LCEL构建），其目的是获取当前历史记录并输出一个经过压缩、提炼的版本，专门为下游任务设计，可能侧重于特定类型的信息（例如，用户偏好、已确定的目标）。

压缩比基本总结寻求更高信息保真度，但通常会带来更高的计算成本和实现难度。

### 策略组合

生产应用经常从这些策略的组合中获益。一个设置可能包括：

1. 一个`ConversationBufferWindowMemory`，用于保存最近几轮的即时上下文 (context)。
2. 一个`ConversationSummaryMemory`或类似机制，用于浓缩早于窗口的交互。
3. 一个`VectorStoreRetrieverMemory`，通过语义搜索提供对完整、长期历史或相关文档的访问。
4. 在将最终上下文化合物传递给主LLM之前，对从这些来源组装的最终上下文应用严格的令牌限制。

> 不同的策略处理完整历史记录，以将相关信息放入有限的LLM提示上下文中。

### 选择方法

选择合适的策略组合需要分析您的应用需求：

- **交互方式：** 是简短的问答、长时间运行的聊天机器人，还是执行任务的代理？长时间交互更适合总结或检索。
- **信息需求：** LLM需要精确地逐字回顾最近的消息，还是过去交互的要点就足够了？
- **延迟要求：** 简单的窗口化最快；总结/压缩会显著增加LLM调用延迟。检索延迟受向量 (vector)存储性能影响。
- **成本预算：** 用于总结/压缩的LLM调用会增加直接成本。向量存储有基础设施成本。窗口化通常最便宜。
- **保真度与成本：** 通过总结或窗口化可能造成的信息损失多少是可接受的，与更复杂方法带来的成本相比如何？

没有单一的“最佳”方法。有效的上下文 (context)窗口管理通常需要基于观察到的性能和成本指标进行实验和调整，使用LangSmith（第5章会介绍）等评估工具来衡量不同内存配置对应用质量的影响。监控每次交互的令牌数量以及内存操作引入的延迟对生产系统来说是必不可少的。

## 参考资料

- [LangChain Memory Documentation](https://python.langchain.com/docs/modules/memory/) — LangChain Team (2025)
  Publisher: LangChain Inc.
  提供LangChain中各种内存类的详细解释和使用示例，与本节讨论的方法直接相关。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  提出了一个创建生成式智能体的框架，该智能体具有记忆流，使其能够保留和回忆与其交互相关的长期信息，为基于LLM的应用程序的高级内存架构提供了见解。
