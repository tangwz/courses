---
course: "getting-started-rag"
chapter: "rag-generation-augmentation"
lesson: "generating-final-response"
sourceId: 3656
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-4-rag-generation-augmentation/generating-final-response"
title: "生成最终回复"
description: "描述LLM如何将增强型提示中的信息合成为一个连贯答案的过程。"
order: 5
plots: []
sourceHash: "80a2d8adfcf2ede64b960957a0d8b857e57fb25c6f49f8f709f516f1d933a032"
sourceCorrections: []
---

大型语言模型（LLM）在检索增强生成（RAG）中负责生成最终响应。在此阶段，LLM接收一个增强型提示，该提示结合了您的原始查询和已检索到的相关上下文 (context)，从而产生最终输出。这个过程正是RAG中“生成”功能的真正体现。

### 合成任务

到目前为止，我们已经检索到了相关信息，并将其精心组织到提示中。现在，这个增强型提示被传递给LLM。需要理解的是，LLM在这里的任务并非简单地复制粘贴检索到的上下文 (context)片段。相反，它执行一项复杂的**合成**任务。

LLM整合多种信息：

1. **原始用户查询：** 用户最初提出的问题。
2. **检索到的上下文：** 检索器提供的相关文本段落。
3. **其自身的内部知识：** 在预训练 (pre-training)期间学习到的大量信息和语言模式。

目标是生成一个直接回应用户查询的回复，该回复以提供的上下文为事实依据，并以连贯、自然的方式呈现。可以将检索到的上下文视为在LLM回答问题前提供给它的具体证据或补充阅读材料。

> LLM处理结合了查询和上下文的信息，以生成最终答案。

### 增强如何影响生成

您组织提示的方式（如“RAG的提示结构”中讨论的）很大程度上影响这种合成。通过清楚地指示LLM将其答案基于提供的上下文 (context)，您引导它优先考虑这些外部信息，而不是可能过时或不够具体的训练数据中的知识。

例如，考虑一个查询：“上个月发布的X产品的主要特点是什么？”

- **没有RAG时：** LLM可能会提供有关类似产品的通用信息，或说明它没有其知识截止日期之后的发布信息。
- **有RAG时：** 检索器找到详细说明最新发布的文档。增强型提示包含这些详情。LLM随后将这些*新*信息合成到其回复中，提供检索到的文档中提到的具体特点。

LLM运用其语言能力，将检索到的事实编织成一个结构良好的答案。它可能会总结多个片段的要点，为清晰起见重新措辞技术细节，或者将上下文中的信息与其一般理解结合，以提供一个全面的回复。

### 保持连贯性和流畅性

一个重要的挑战是确保最终输出听起来自然，而不仅仅是上下文 (context)事实的零散集合。这正是LLM生成能力得以体现之处。训练有素的LLM擅长生成流畅的文本。当由结构良好的增强型提示引导时，它们通常能够顺畅地整合检索到的信息。

然而，生成的质量取决于几个因素：

- **检索到的上下文质量：** 如果检索到的片段不相关或书写不佳，LLM可能难以合成一个好的答案，甚至可能忽略上下文。
- **提示清晰度：** 模糊的指令可能导致LLM误解如何使用上下文。
- **LLM能力：** 不同的LLM在遵循指令和合成方面具有不同的优势。

### 处理潜在冲突

有时，检索到的上下文 (context)可能与LLM的内部知识或在其他检索到的片段中找到的信息相矛盾。虽然高级RAG系统采用策略来处理这种情况，但基本方法通常依赖于提示来指示LLM优先考虑提供的上下文。例如，一个提示可能包含这样的措辞：“请*仅*根据以下文档回答问题...”这指示LLM将其答案牢牢地基于检索到的数据。

生成步骤结束了RAG的核心流程，将查询和一组相关文档转化为一个有上下文依据、内容丰富的答案。下一个自然的考量是理解*哪些*具体的上下文片段对最终答案有所贡献，这将我们引向来源归属的话题。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG）框架，详细阐述了生成器组件如何从检索到的文档和输入查询中综合信息以生成最终响应。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  全面概述了RAG，详细讨论了生成组件、提示技术以及处理冲突信息和保持响应质量的策略。
