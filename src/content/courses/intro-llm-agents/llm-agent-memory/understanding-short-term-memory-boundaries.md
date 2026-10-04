---
course: "intro-llm-agents"
chapter: "llm-agent-memory"
lesson: "understanding-short-term-memory-boundaries"
sourceId: 6620
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-6-llm-agent-memory/understanding-short-term-memory-boundaries"
title: "了解短期记忆的界限"
description: "讨论与简单短期记忆实现相关联的局限性和问题。"
order: 6
plots: []
sourceHash: "859d0507dc9e2efd8a9eb482c63e0e0b7464db58fd47a3fc9d6745fee45ab1fa"
sourceCorrections: []
---

虽然增加一个简单的短期记忆，比如最近对话轮次的列表，可以大幅提升智能体进行连贯对话的能力，但这种方法并非没有自身的问题。了解这些界限很重要，以便为你的智能体能力设定实际的预期，并在情况不如预期时排查问题。让我们来看看基本短期记忆实现方式的一些固有局限。

### 有限的上下文 (context)窗口

LLM智能体的核心部分是大型语言模型本身。这些模型虽然强大，但有一个根本限制：**上下文窗口**。可以将上下文窗口看作LLM在任何给定时刻可以“查看”或处理的文本量（包括指令、当前查询以及任何提供的历史信息）。如果对话历史变得过长，它就根本无法适应这个窗口。

当使用追加最近交互的简单短期记忆时，对话中较早的部分最终会被推出，为新的交互腾出空间。这就像试图往一个已经装满的玻璃杯里倒更多的水；有些水不可避免地会溢出。

> 上图说明了固定大小的上下文窗口可能只能看到较长对话的最新部分。如果总历史记录加上当前查询超出窗口大小，早期交流（例如“用户：你好！请告诉我关于LLM的信息。”）可能会从LLM的视图中被截断。

**影响：**

- **遗忘早期指令：** 如果你在长时间交互的开始就给智能体特定指令或上下文，它之后可能会“遗忘”它们。
- **失去对长期目标的追踪：** 对于需要多轮展开的多步骤任务，最初的目标可能会超出上下文窗口。
- **无法回答关于久远过去的问题：** 如图所示，如果第一个问题不再上下文中，询问“我的第一个问题是什么？”可能无法实现。

上下文窗口的大小在不同的LLM之间有所不同（例如，4,096个令牌、8,192个令牌、32,768个令牌，或者对于较新的模型甚至更大，其中一个令牌大致相当于一个词或词的一部分）。你需要了解你所使用的LLM的限制。

### 近因效应：“最响亮的声音”效应

简单的短期记忆机制通常会在每次新轮次时向LLM呈现整个记忆的历史。在这种情况下，最新信息往往对LLM的响应产生更大的影响。这有时被称为近因效应。

想象你正在阅读一份建议列表。你最后阅读的那些可能比开头的更容易留在你的记忆中。类似地，如果智能体的短期记忆只是一个按时间顺序的日志，最新的用户输入或智能体动作可能会盖过较早的、可能更重要的信息。

**影响：**

- 智能体可能很容易被最后说的话影响，即使它与对话中较早、更确定的事实相矛盾。
- 如果对话出现一些偏离，智能体可能更难保持对最初目标的专注。

### 简单检索：全有或全无

基本短期记忆，例如存储过去消息的列表，通常采用非常简单的检索策略：它将所有存储的历史记录（直到上下文 (context)窗口限制）包含在给LLM的提示中。没有智能选择哪些过去交互与当前查询最相关。

LLM本身随后必须筛选整个历史记录以找到它需要的信息片段。虽然LLM在这方面做得不错，但并非总是高效。

**影响：**

- **增加处理负担：** LLM会耗费资源处理可能不相关的历史部分。
- **嘈杂的上下文：** 一段长且未分类的历史记录可能会引入“噪音”，使LLM更难精准定位真正重要的信息。
- 这与更高级的记忆系统不同，后者可能会使用语义搜索等技术来只检索最相关的记忆，但这超出了*简单*短期记忆的范围。

### 成本和性能障碍

不断将不断增长的历史记录输入LLM的上下文 (context)窗口会带来直接的实际后果：

- **API成本：** 如果你通过API使用商业LLM，通常会根据处理的令牌数量（输入和输出）收费。更长的历史记录意味着更多的输入令牌，导致每次交互的成本更高。
  例如，如果你的记忆存储了最近10条消息，每条平均100个令牌，那么历史记录就是1000个令牌。再加上一个新的用户查询50个令牌，以及LLM指令（提示）的另外50个令牌，你作为输入发送的总计是1100个令牌。
- **延迟：** 处理更多的令牌需要更多时间。随着短期记忆的增长，智能体的响应时间会增加，导致用户体验变慢。

这里存在一个直接的权衡：更长的记忆提供更多上下文，但代价是更高的运营成本和可能更慢的性能。

### 处理真正的长距离依赖的困难

如果一个任务要求智能体将交互中很早期的信息与很久以后发生的事情联系起来，并且早期信息已经被固定大小的短期记忆推出，那么智能体很可能会失败。

例如，想象一个智能体的任务是：

1. “记住这段代码：`X = 10`。”
2. （在又进行了20轮对话并填满记忆窗口后）
3. “那么，X的值是多少？”

如果在第21轮中提供给LLM的短期记忆中不再包含最初的声明`X = 10`，智能体将无法回答。简单短期记忆的性质决定了它不太适合处理超出其容量的长距离依赖任务。

### 存储而非整合

重要的是要记住，大多数简单的短期记忆系统仅仅充当存储，一个所说或所做内容的日志。它们通常不涉及智能体主动“理解”、总结或将信息整合为更抽象或压缩的形式。

记忆内容通常是原始记录。这意味着LLM每次都必须重新处理这些原始信息。相比之下，人类会整合记忆，提取要点，并形成抽象概念。基本的LLM智能体记忆通常不会这样做。

**影响：**

- **效率低下：** 智能体没有从更深刻的意义上学习LLM可以从其当前上下文 (context)窗口中的原始历史记录推断出的内容。
- **易受无关细节影响：** 如果历史记录包含大量闲聊或无关信息，这些都会被传递，可能会稀释重要的信号。

了解这些界限的目的不是让你气馁，而是让你拥有实际的视角。简单的短期记忆是一个基本组成部分，了解其局限性是设计更有效智能体的第一步，并且在必要时，了解更高级的记忆技术，这些是更高级学习的主题。对于许多直接的任务，管理良好的短期记忆完全足够。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  描述了Transformer架构，它确立了固定大小的注意力机制，从而奠定了LLM中基本的上下文窗口限制。
- [Lost in the Middle: How Language Models Use Long Contexts](https://aclanthology.org/2024.tacl-1.9.pdf) — Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023)
  Journal: Transactions of the Association for Computational Linguistics; Publisher: MIT Press; Volume: 12; Pages: 157-173; DOI: [10.1162/tacl_a_00638](https://doi.org/10.1162/tacl_a_00638)
  研究了LLM如何处理长上下文，发现上下文窗口中间的信息往往检索效率较低，这说明了近因偏见和记忆挑战。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: CoRR; Volume: abs; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  提供了LLM的广泛概述，包括对上下文窗口限制等架构约束及其对记忆和智能体能力影响的讨论。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQENcT3RH-ueTXr2d7eb6Kl4Wg179H75hlzhAQypN-g4DvpuxqJkrKB7WwnWP53S1_QgKgrgL7xDBD96bBHfXWWlILopi_ad2sldZcXJQTgLJZloXX4D_wkIhfo=) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems 33; Publisher: Curran Associates, Inc.; Pages: 9459–9474
  介绍了RAG，这是一种通过检索相关外部知识来解决固定上下文窗口和短期记忆限制的方法，从而提高了长程事实回忆能力。
