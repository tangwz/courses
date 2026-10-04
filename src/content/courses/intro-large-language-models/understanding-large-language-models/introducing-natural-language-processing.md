---
course: "intro-large-language-models"
chapter: "understanding-large-language-models"
lesson: "introducing-natural-language-processing"
sourceId: 3673
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-1-understanding-large-language-models/introducing-natural-language-processing"
title: "自然语言处理 (NLP) 简介"
description: "了解自然语言处理 (NLP) 的基础知识以及计算机如何处理人类语言。"
order: 2
plots: []
sourceHash: "5869e6427bba152d1136a5826106886621193ec533eb2574cba41e56289ea298"
sourceCorrections: []
---

人工智能 (AI) 是指让计算机完成通常需要人类智能的任务。人类智能的一个重要组成部分是理解和使用语言。想想你阅读这段文字、理解其含义、提出问题或撰写电子邮件是多么的轻松自如。对于计算机来说，这非常具有挑战性。人类语言复杂，充满了细致入微的表达、语境和歧义。

这正是自然语言处理 (NLP) 的用武之地。

### 什么是自然语言处理？

自然语言处理 (NLP) 是人工智能的一个专业分支，专门致力于让计算机理解、解析、处理和生成人类语言（如英语、西班牙语或普通话），并使其具有实际价值。它处于计算机科学、人工智能和语言学的交汇点。

自然语言处理的目标是弥合人类交流与计算机理解之间的鸿沟。它不再要求人类通过僵硬的代码或指令来“与计算机对话”，而是旨在让计算机“理解人类语言”。

### 自然语言处理为何重要？

计算机处理语言的能力带来了许多可能性：

- **信息获取：** 搜索引擎使用自然语言处理来理解你的查询并找到相关网页。
- **交流沟通：** 机器翻译工具消除语言隔阂。聊天机器人提供客户支持。
- **自动化：** 自然语言处理分析大量文本数据（如客户评论或报告）的速度远超人类。
- **人机互动：** 语音助手如 Siri 或 Alexa 主要依赖自然语言处理来理解口头指令。

### 常见的自然语言处理任务

自然语言处理涵盖了多种旨在解析和处理语言的任务。以下是一些主要例子：

- **文本分类：** 将预设类别分配给一段文本。例如，自动将邮件标记 (token)为`“垃圾邮件”`或`“非垃圾邮件”`，或判断电影评论表达的是`“正面”`还是`“负面”`情绪。
- **命名实体识别 (NER)：** 识别并归类文本中的特定实体，例如人名、组织机构、地点、日期或货币值。在一个新闻文章中找到`“苹果公司”`并将其归类为`“组织机构”`就是一个例子。
- **机器翻译：** 自动将文本从一种语言翻译成另一种语言，例如将英文句子翻译成法文。
- **问答系统：** 对以自然语言提出的问题提供答案，通常通过搜索给定的文本文档或知识库来完成。
- **文本生成：** 创建新的、连贯的文本。其范围可以是从补全句子到撰写整篇文章或摘要。这项任务是我们即将学习的大型语言模型的核心能力。

### 人类语言的复杂性

让计算机理解语言很困难，因为人类语言天生复杂：

- **多义性：** 词语可以有多种含义（例如，“bank”可以是金融机构或河岸）。需要根据周围文本（语境）来确定其确切含义。
- **语境依赖性：** 句子的含义通常在很大程度上依赖于前面的句子或整体情况。
- **多样性：** 我们以多种不同方式表达相同的想法（同义词、句子结构）。
- **非正式语言：** 俚语、习语、错别字和语法错误在文本中很常见。

### 自然语言处理与大型语言模型

历史上，自然语言处理系统通常依赖于复杂的手工规则集或应用于较小数据集的统计方法。尽管这些方法对于特定任务有效，但它们在处理语言的复杂特性时常常遇到困难。

大型语言模型 (LLMs) 代表了自然语言处理的重大进步。它们使用深度学习 (deep learning)技术，并在海量文本数据上进行训练。这使它们能够更全面地理解语言模式、语法、语境甚至知识，从而以出色的表现执行各种自然语言处理任务，特别是文本生成和理解。

可以把自然语言处理看作是更广的学科，而大型语言模型则是其中一组强大的工具和技术，它们极大地提升了该学科的能力。

> 人工智能、机器学习 (machine learning)、自然语言处理和大型语言模型之间的关系。大型语言模型是自然语言处理中的一种特定模型，而自然语言处理本身是机器学习和人工智能的一部分。

理解自然语言处理的基础知识有助于我们更好地认识大型语言模型的目标以及它们的结构原因。接下来，我们将更仔细地研究“大型语言模型”的定义。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  一本综合性教材，提供自然语言处理的基础知识，涵盖从语言学基础到计算模型的各个方面。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Christopher Manning and Richard Socher and Abolfazl Asudeh and John Hewitt and Chenhao Tan (2023)
  Publisher: Stanford University
  一门优秀的大学课程，涵盖应用于自然语言处理的深度学习方法，对于理解现代语言模型至关重要。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇开创性论文介绍了Transformer架构，该架构构成了大多数大型语言模型及其进展的基础。
