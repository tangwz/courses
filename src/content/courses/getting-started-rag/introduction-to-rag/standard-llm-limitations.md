---
course: "getting-started-rag"
chapter: "introduction-to-rag"
lesson: "standard-llm-limitations"
sourceId: 3633
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-1-introduction-to-rag/standard-llm-limitations"
title: "标准大型语言模型的局限性"
description: "概述LLM的常见问题，如知识截止和幻觉，为RAG解决方案做铺垫。"
order: 1
plots: []
sourceHash: "087a42a59273a533ff2f0a84be7e13209abf713444ebdf47a0a1d922671a4d3c"
sourceCorrections: []
---

大型语言模型（LLM）代表着人工智能的重大进展，在处理和生成模仿人类交流的文本方面展现出杰出能力。这些模型从其训练所用的庞大数据集中学习模式、语法和大量常识。然而，尽管有其优点，标准LLM仍存在固有局限，可能会妨碍其在某些应用中的效用，尤其那些需要最新准确性或专业信息的场景。了解这些制约有助于理解为何检索增强生成（RAG）等技术已具实用价值。

### 知识截止

最主要的局限之一是**知识截止**。LLM的知识基本上固定在其训练数据收集和处理的时间点。它没有内在机制来获取或学习训练完成后出现的信息。

试想询问一个在2023年初训练的标准LLM关于2023年末某项重大体育赛事冠军或上个月发布的某个软件库的功能。模型根本“不知道”答案，因为该信息不在其训练集中。它可能会根据旧的、相关数据中的模式尝试给出答案，但无法获取实时或最新信息。

> 一条时间线，显示LLM的知识固定在“训练结束”点，造成与当前事件相关的信息鸿沟。

这种时间上的局限意味着标准LLM会迅速过时，使其对于需要当前知识的任务而言不可靠。

### 幻觉 (hallucination)与事实不准确性

LLM本质上是复杂的模式匹配系统。它们学习词语和意义之间的统计关系。虽然这使它们能够生成流畅且常有条理的文本，但这不保证事实准确性。LLM有时会产生**幻觉**：听起来合理且自信，但事实不准确、毫无意义或完全捏造的回应。

这发生的原因是，模型在训练期间的目标通常是预测序列中的下一个词（或标记 (token)），从而最大化基于输入提示和学习到的数据模式的生成文本的统计可能性。在*生成过程中*，它本身不具备一种机制来对照外部现实或可信知识来源验证事实。

例如：

- 编造某人生活或成就的细节。
- 引用不存在的学术论文或法律先例。
- 信心满满地提供错误的 F技术规范或历史日期。

在医学、金融或法律等对准确性要求极高的专业中，产生幻觉的倾向构成重大风险。

### 普遍知识与专业性不足

尽管LLM通过涵盖许多主题的庞大数据集进行训练，但它们的知识常常是宽泛而非深入的，尤其是在高度专业化或小众方向。训练数据可能缺乏对特定科学专业、复杂工程学科或某一组织独有的专有信息的足够覆盖。

一个标准LLM不太可能详细了解：

- 您公司的内部文档、编码标准或项目历史。
- 最近发表的、高度专业化研究论文的具体内容。
- 网上未广泛讨论的某个地方性法规的具体情况。

试图将通用型LLM用于需要此类专业知识的任务，往往会得到通用、模糊或不正确的答案。它无法访问包含相关信息的私有数据库或内部知识库。

### 溯源困难

当标准LLM提供答案时，它通常不引用来源或解释它如何从训练数据中整合信息。生成过程不透明，使得验证输出中的主张变得困难，甚至不可能。用户会疑惑模型为何给出特定答案，以及它是否基于可靠信息。

这种缺乏透明度和可追溯性对于建立信任以及那些信息来源清晰度很重要的应用场景来说是个问题。例如，如果LLM提供法律或医疗信息，用户需要了解这些信息的依据以评估其可信度。

这些局限性共同指出需要一种方法，能够将LLM的回答建立在特定、最新和可验证的信息来源之上。RAG正是为了弥补这一鸿沟而设计的，它通过在LLM生成最终回答*之前*明确检索相关的外部信息，从而提高准确性、相关性和可信度。接下来的章节将阐述RAG是如何做到这一点的。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了RAG，通过检索信息增强LLM来直接解决其局限性。
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) — Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, et al. (2021)
  Journal: arXiv; Publisher: Stanford University; DOI: [10.48550/arXiv.2108.07258](https://doi.org/10.48550/arXiv.2108.07258)
  一份广泛的报告，讨论了大型预训练模型的能力、局限性及社会影响，包括事实不准确和偏见等问题。
