---
course: "langchain-production-llm"
chapter: "advanced-memory-management"
lesson: "comparing-memory-types"
sourceId: 3874
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-3-advanced-memory-management/comparing-memory-types"
title: "高级记忆类型比较"
description: "分析基于向量存储、实体及其他高级记忆模块，适用于不同情况。"
order: 1
plots: ["plots/3874-0.json"]
sourceHash: "899c5e409d0222282462a8251d08ccf76928b86ad9209f4e889dc88e91b5f97e"
sourceCorrections: []
---

简单的缓冲区等基本记忆机制在处理生产级LLM应用中常见的长时间交互时通常不足。当对话变得很长或需要回顾早期交流中的特定细节时，仅仅存储完整的原始历史记录会变得效率低下，并最终超出LLM的上下文 (context)窗口限制。高级记忆类型提供更复杂的策略，用于存储、检索和总结对话上下文，从而支持更连贯、更具知识的应用。

选择合适的高级记忆类型是一个重大的架构决定。它很大程度上依据应用的性质、交互的预期长度和复杂程度，以及需要保留的特定上下文类型。让我们看看LangChain中现有或可适配的一些主要先进记忆方法。

### 基于向量 (vector)存储的记忆

这种方法处理对话历史的方式，与检索增强生成（RAG）中处理文档类似。并非按顺序存储原始文本，而是将对话轮次（或其摘要）嵌入 (embedding)并存储在向量数据库中。

**工作原理：**

1. **存储：** 每条消息或近期消息的摘要使用嵌入模型（例如，OpenAI嵌入、Sentence Transformers）转换为数值向量。这个向量捕捉文本的语义含义。这些向量，连同原始文本和元数据，被存储在向量存储中（如Chroma、FAISS、Pinecone、Weaviate）。
2. **检索：** 当新输入到来时，它也会被嵌入。会根据存储中的向量执行相似性搜索（例如，余弦相似度、点积），以根据语义含义（而非仅仅是新近度）找到 $k$ 个最相关的过往交互。
3. **上下文 (context)注入：** 检索到的历史交互被格式化并注入到提示上下文中，如果需要，与最近的消息一起。

**LangChain实现：** 这种逻辑实际上是一个检索增强生成（RAG）流程，其中文档是过去的对话轮次。开发者通常使用LCEL (LangChain表达式语言) 链中的`VectorStoreRetriever`来选择上下文。`VectorStoreRetrieverMemory` 类可作为一个简化封装器使用，但与明确构建链相比，它对检索过程的控制较少。

**优点：**

- **可扩展性：** 有效处理极长的对话历史记录，因为检索时间依据向量存储的效率，而非历史记录的线性长度。
- **相关性：** 根据语义相似性检索上下文，允许回顾对话中很早之前的相关信息，即使在时间上不相邻。
- **灵活性：** 可以存储完整消息、摘要，甚至提取的事实。

**缺点：**

- **严格时间顺序的丢失：** 检索基于相关性，因此对话的严格顺序可能会在检索到的上下文中部分丢失，除非明确管理（例如，通过元数据）。
- **计算开销：** 存储和检索需要嵌入计算，增加了延迟和成本。
- **调整：** 检索效果依据嵌入的质量以及搜索参数 (parameter)（例如要检索的文档数量 $k$）的调整。
- **召回不相关内容的可能性：** 语义搜索有时可能会检索到表面相似但上下文不相关的过往交流。

**用例：** 适用于需要回顾特定信息或主题的应用，这些信息或主题源自可能非常长的交互，例如长期聊天机器人、处理大量对话的知识助手，或需要从以前工单中获取上下文的客户支持机器人。

### 实体记忆

实体记忆侧重于识别和跟踪在整个对话中提及的特定实体（例如人、地点、组织、想法）。它为每个识别出的实体维护一份摘要或主要事实。

**工作原理：**

1. **提取：** LLM（或专门的NLP流程）处理对话以识别主要实体。
2. **总结/存储：** 对于每个实体，记忆模块维护一个关于迄今为止从对话中收集到的相关信息的摘要。当新的相关信息出现时，此摘要会更新。
3. **检索：** 当实体在当前输入或上下文 (context)中被提及，其相关摘要会从记忆存储中检索。
4. **上下文注入：** 检索到的实体摘要被添加到提示上下文中，为LLM提供正在讨论的主要主题背景信息。

**LangChain实现：** 现代应用通常使用**结构化输出**或**工具调用**来提取实体并更新持久状态（例如图数据库或JSON存储）。这种方法提供更好的准确性和模式遵循，相比于旧版的`ConversationEntityMemory`封装器，后者依赖于较不确定的提示策略。

**优点：**

- **简洁性：** 提供主要主题的紧凑摘要，对上下文窗口很高效。
- **聚焦上下文：** 提供关于当前讨论的特定实体的高度相关信息。
- **状态跟踪：** 适用于跟踪对话中特定项目随时间变化的状态或属性。

**缺点：**

- **对提取的依赖：** 很大程度上依赖LLM准确识别实体和总结相关信息的能力。提取或总结中的错误会影响记忆质量。
- **潜在的信息丢失：** 未直接与识别出的实体相关联的上下文可能会被遗漏。
- **复杂程度：** 设置和管理可能比缓冲区记忆更复杂，通常需要额外的LLM调用进行提取/总结，增加了延迟和成本。

**用例：** 适用于跟踪特定命名实体有较高要求或作用的应用，例如CRM聊天机器人记住客户详情、虚拟助手回顾与特定项目相关的用户偏好，或技术支持代理跟踪有关特定设备或软件组件的信息。

### 向量 (vector)存储记忆和实体记忆的比较

在这些高级类型之间做出选择通常涉及权衡。以下是一个比较概览：



![记忆类型特点比较（分数越高表示越好/越复杂）](plots/3874-0.json)



> 基于向量存储的记忆和实体记忆在主要方面的比较。请注意，“时间顺序保留”表示默认机制保留严格顺序的程度；相关性侧重于语义相似性。复杂程度包括设置和操作开销。

**选择考量：**

- **上下文 (context)性质：** 如果回顾与主题相关的过往信息（无论何时发生）是首要的，**向量存储记忆**通常更受欢迎。如果目标是跟踪特定的人、地点或事物及其相关细节，**实体记忆**是一个有力的选择。
- **对话长度：** 对于完整的历史记录不切实际的极长对话，**向量存储记忆**提供更好的可扩展性。**实体记忆**根据独特实体的数量进行扩展，这也可能变得很大，但比原始历史记录提供更压缩的表示。
- **成本和延迟：** **实体记忆**通常需要额外的LLM调用进行提取和总结，与向量存储操作相比，可能增加成本和延迟，尽管嵌入 (embedding)也有成本。**向量存储记忆**的性能依据向量数据库的效率。
- **实现复杂程度：** 两者都比基本缓冲区更复杂。**向量存储记忆**需要设置和管理向量存储。**实体记忆**依赖于配置LLM以实现可靠的提取和总结。

组合策略也很常见。例如，一种架构可能会使用滑动窗口缓冲区来处理最近的消息，以确保即时连贯性，同时并行查询向量存储或实体记忆以获取主要的长期细节。这使得应用能够保持最近的上下文，同时回顾主要的历史信息。

最终，选择涉及了解应用的具体需求，关于上下文持续时间、信息回顾类型、性能要求和可接受的复杂程度。实验和评估，可能使用LangSmith（第五章介绍）等工具，通常是确定生产系统最佳记忆策略所必需的。

## 参考资料

- [LangChain Memory](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6TowNa64_CHep6ZpbfnhBD3r0BBY3iK1pX-DkQWdipgtcMdlXkxSn9KLI1ZbjQ2FkrYPPhCI_WFQsCkMGnqSvYhtQvfN9FZ6sCNbzAcLU3Q7kPwR77Dfx4Nxj088F1amQvTDBp8ZG9_5otmRdkY08Szs9Kr2M-J4Hfw==) — LangChain (2024)
  LangChain官方文档，介绍LangChain中可用的各种内存类型，包括向量存储支持和实体内存等高级实现。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; Volume: 33; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  一篇介绍检索增强生成（RAG）的奠基性论文，通过将检索与语言模型生成相结合，为向量存储支持的内存提供了架构基础。
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih (2020)
  Journal: Empirical Methods in Natural Language Processing (EMNLP); DOI: [10.48550/arXiv.2004.04906](https://doi.org/10.48550/arXiv.2004.04906)
  本文介绍了密集段落检索（DPR），详细说明了一种为文本段落创建和检索密集向量表示的高效方法，直接适用于向量存储支持的内存。
