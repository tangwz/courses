---
course: "getting-started-rag"
chapter: "introduction-to-rag"
lesson: "rag-core-architecture"
sourceId: 3635
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-1-introduction-to-rag/rag-core-architecture"
title: "RAG系统的核心架构"
description: "概述典型RAG系统的主要部分：检索器、生成器和数据来源。"
order: 3
plots: []
sourceHash: "58d283950b3db0b7ff1c2a7a015298c957aa3d6c2732b70294b87bcc45240aaa"
sourceCorrections: []
---

检索增强生成（RAG）有效处理标准大型语言模型（LLM）的限制，例如知识截止和幻觉 (hallucination)问题。RAG系统旨在利用外部知识库，其基本结构由两个主要的职能部分组成，它们协同工作。

可以把它想象成一个两步过程：首先，找到相关信息，然后使用这些信息生成答案。这种职责分离使得每个部分都能专注于自己的功能，并有效地为最终输出做出贡献。

1. **检索器：** 这是RAG系统的信息检索引擎。它的主要目的是接收用户的输入查询，并从预定义的外部知识来源中查找最相关的信息片段（常被称为“文档”或“块”）。这个来源可以是文本文件、PDF、数据库条目、网页或其他结构化或非结构化数据的集合。检索器不理解语言生成的细节；它只专注于基于与查询的语义相似性进行高效准确的信息查找。我们将在第2章详细研究检索技术，特别是涉及向量 (vector)嵌入 (embedding)和向量数据库的技术。
2. **生成器：** 这个部分通常是一个标准的大型语言模型（LLM）。它的职责是接收原始用户查询*以及*由第一个部分检索到的相关上下文 (context)，然后合成一个连贯的、类似人类的答案。通过接收与查询一同获取的上下文，大型语言模型被“增强”了，它能够获取到在其原始训练数据中可能不存在的特定、相关且可能最新的信息。这使得它能够根据提供的文档生成更具事实依据并针对特定查询定制的回复。整合此上下文并生成最终输出的过程将在第4章介绍。

这两个部分通过一个明确定义的工作流程彼此作用，通常由一个编排器或框架来协调（我们将在后续章节中提及）。

> 典型RAG架构中的数据流。用户查询启动从知识来源的检索，检索到的上下文增强了生成器大型语言模型的输入。

这种模块化架构有其重要性。它使得开发者能够：

- **轻松更新知识：** 无需重新训练大型且计算成本高的生成器LLM，即可修改或添加外部数据源。这相对于通过微调 (fine-tuning)进行知识更新来说是一个主要优点。
- **提升可追溯性：** 由于生成器使用特定的检索文档，通常可以引用用于创建回复的来源，从而提高透明度和信任度。
- **独立优化组件：** 检索器和生成器可以根据特定的任务要求和可用资源进行独立选择、配置和优化。例如，您可以使用一个高度专业化的检索器处理法律文档，并将其与一个通用会话型大型语言模型作为生成器结合使用。

理解这个核心架构，以及专业检索器与强大生成器之间基于外部知识来源的彼此作用，对于构建和理解RAG系统来说是根本的。接下来的章节将更详细地分析这个结构的每个部分。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  这篇基础论文介绍了原始的检索增强生成（RAG）模型，详细阐述了它如何结合预训练的参数记忆（一个seq2seq模型）和非参数记忆（一个预训练的神经检索器），以改进知识密集型自然语言处理任务。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://doi.org/10.48550/arXiv.2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  一篇近期综述，提供了RAG技术进展的现代视角，涵盖了其架构、不同的设计选择、评估方法以及将检索与大型语言模型集成时面临的挑战。
