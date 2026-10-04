---
course: "intro-large-language-models"
chapter: "understanding-large-language-models"
lesson: "llm-task-examples"
sourceId: 3680
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-1-understanding-large-language-models/llm-task-examples"
title: "LLM能完成的任务示例"
description: "查看大型语言模型可以执行的常见任务示例，如文本生成和摘要。"
order: 5
plots: []
sourceHash: "a2aebc0197c258efd8f4216bf7300779a0d6f56563e91cefb65cf587947ab3ef"
sourceCorrections: []
---

大型语言模型（LLM）能够执行各种涉及阅读、写作和处理文本的任务。由于LLM经过大量文本的训练，它们对语言结构、含义和语境有了细致的理解。这种理解使它们能够完成多种实际任务。

### 文本生成

这可能是许多LLM最核心的能力。它们可以生成连贯且与上下文 (context)相关的新文本。这可以从简单的句子补全到撰写整篇文章、故事甚至代码片段。

- **示例：** 您可以要求LLM“写一段关于回收好处的短文”，它将根据在训练期间学到的模式生成解释这些好处的文本。
- **工作原理（简化）：** 简单来说，模型根据它目前已看到或生成的词序列，预测下一个最有可能的词（或标记 (token)），并重复此过程来构建句子和段落。

### 问答

LLM通常可以根据其训练数据中包含的信息或根据提示中提供的上下文 (context)来回答问题。这可以从事实回忆到回答有关您提供的特定文档的问题。

- **示例：** 询问“最高的山是什么？”很可能会得到“珠穆朗玛峰”的答案。如果您提供一篇新闻文章并询问“根据文章，会议的主要结果是什么？”，LLM会尝试查找并提取该特定信息。
- **注意：** 尽管功能强大，但LLM不像人类那样“知道”事物。它们的回答是基于训练数据中的模式和信息，这些信息可能并非总是最新或完全准确的。

### 摘要

给定一段长文本 (long context)，例如文章、报告或研究论文，LLM可以将其浓缩成一个更短的摘要，突出主要内容。

- **示例：** 您可以提供一篇冗长新闻报道的文本，并要求LLM“将这篇文章总结为三点。”
- **用途：** 这有助于快速把握大量信息的要点。

### 翻译

许多LLM是多语言的，可以将文本从一种语言翻译成另一种语言。

- **示例：** 您可以询问“将‘Hello, world’翻译成德语”，模型很可能会回答“Hallo, Welt。”
- **范围：** 支持的语言质量和范围在很大程度上取决于具体的模型及其训练数据。

### 分类和情感分析

LLM可以将文本分类到预设类别中，或确定其潜在的情绪（积极、消极、中性）。

- **示例（分类）：** 您可以提供一封电子邮件并询问“这封邮件是垃圾邮件还是非垃圾邮件？”
- **示例（情感分析）：** 假设有一条产品评论如“这款相机拍出的照片太棒了！”，LLM可以识别出情绪是积极的。
- **应用：** 企业使用此功能来分析客户反馈、分类支持工单或审核内容。

### 对话代理（聊天机器人）

LLM是许多现代聊天机器人和虚拟助手背后的驱动力。它们可以理解用户自然语言输入并生成类似人类的回复，并在多轮对话中保持上下文 (context)。

- **示例：** 与网站上的客户支持聊天机器人交互或使用语音助手通常涉及LLM技术。
- **机制：** 这通常结合了文本生成、问答和上下文管理能力。

这些例子展示了LLM的多功能性。虽然它们看起来像是魔法，但它们的能力源于从大量数据集中学习到的复杂模式识别。随着您学习本课程，您将更清楚地了解这些任务背后的原理，并学习如何使用提示词 (prompt)有效引导LLM。

## 参考资料

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; Pages: 144 pages; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  提供了大型语言模型的广泛概述，涵盖了其架构、训练方法和在自然语言处理中的多种应用。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本备受推崇的自然语言处理教材，提供了语言模型的基础知识和应用，包括对各种任务的讨论。
