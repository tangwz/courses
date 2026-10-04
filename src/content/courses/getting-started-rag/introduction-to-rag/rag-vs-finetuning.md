---
course: "getting-started-rag"
chapter: "introduction-to-rag"
lesson: "rag-vs-finetuning"
sourceId: 3636
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-1-introduction-to-rag/rag-vs-finetuning"
title: "RAG 对比微调：理解其差异"
description: "比较并对照 RAG 与大型语言模型微调，指出它们各自的优势和应用。"
order: 4
plots: []
sourceHash: "698ec286d5695c9edf029e22e85a686013f43206f2e1b084400290c4d6b34128"
sourceCorrections: []
---

为了更好地定制大型语言模型 (LLM)，将检索增强生成 (RAG) 与另一种广泛使用的方法：**微调 (fine-tuning)**进行对比是很有帮助的。这两种方法都旨在使通用大型语言模型适应特定任务或知识范围，但它们的工作方式大不相同。

### 什么是微调 (fine-tuning)？

微调是指将一个预训练 (pre-training)的大型语言模型（它已经从海量数据集中学习了通用语言模式），并在针对特定任务或范围的较小、精选数据集上继续其训练过程。此过程会调整模型本身的内部参数 (parameter)或权重 (weight)。

可以将其类比为让一个受过广泛教育的人（预训练大型语言模型）去一所专业学校学习特定学科，例如医学或法律（微调数据集）。目标是将这种专业知识或行为直接植入模型的内部表示中。

微调常用于：

1. **调整模型的风格或语气：** 使大型语言模型以特定的品牌声音或角色特征进行交流。
2. **教授特定的响应格式：** 训练模型生成具有特定结构（如 JSON 或特定报告格式）的输出。
3. **灌输特定范围的知识：** 提升模型在特定小众区域的理解和术语使用，假设该知识不会快速变化。

### RAG 如何不同

检索增强生成 (RAG)，正如我们所讨论的，不会修改大型语言模型的基础权重 (weight)。相反，它为现有预训练 (pre-training)大型语言模型配备了一个外部知识源和一种机制，以便在*进行查询时*（推理 (inference)时）从中检索相关信息。

用我们的类比来说，RAG 就像给那个受过广泛教育的人提供一个全面的、可搜索的图书馆或数据库，专门用于手头的任务。对于提出的每个问题，他们首先会在图书馆中查找相关信息，*然后*利用其通用技能根据所查到的内容来组织答案。这个人的核心知识没有改变，但他们使用图书馆信息回答特定问题的能力得到了显著提升。

### 差异总结

让我们从几个重要方面来剖析 RAG 与微调 (fine-tuning)之间的主要区别：

#### 1. 知识整合

- **微调：** 通过修改大型语言模型的内部参数 (parameter)（权重 (weight)）来整合知识。这种知识变为*参数化*的，隐式存储在模型结构内部。
- **RAG：** 在推理 (inference)时通过检索外部数据并将其作为上下文 (context)提供给提示，从而动态整合知识。这种知识是*非参数化*的，存储在模型外部。

#### 2. 知识更新

- **微调：** 要更新模型的知识，通常需要整理一个反映最新信息的新数据集，并重复微调过程。这可能在计算上成本高昂且耗时。
- **RAG：** 更新知识只需更新外部数据源（例如，向向量 (vector)数据库添加新文档并对其进行索引）。这通常比重新训练快得多，资源消耗也更少，使 RAG 系统能更轻松地保持最新。

#### 3. 计算资源

- **微调：** 训练阶段需要大量的计算资源（GPU、TPU）和时间。推理成本通常是基础大型语言模型的成本（尽管如果模型变大，有时会略高）。
- **RAG：** 设置时计算量极小（使用预训练 (pre-training)模型）。主要的计算成本发生在推理期间，包括检索步骤（查询向量数据库）和生成步骤（处理包含检索上下文的较长提示）。索引外部数据源有前期成本，但通常低于微调。

#### 4. 事实性和幻觉 (hallucination)

- **微调：** 尽管微调可以提高特定任务的性能，但它本身并不能解决幻觉问题。模型仍可能根据预训练或微调期间学到的模式生成听起来合理但不正确的信息。很难追溯输出中特定信息的来源。
- **RAG：** 直接将大型语言模型的响应建立在检索到的事实文档上。通过提供相关上下文，RAG 显著降低了幻觉的可能性，并提高了事实准确性，前提是检索到的信息是正确的。此外，它允许进行**来源归属**，因为系统可以引用用于生成答案的文档，从而增强透明度和可验证性。

#### 5. 用例适用性

- **微调：** 非常适合调整大型语言模型输出的*行为*、*风格*或*格式*。它也适用于那些核心知识相对稳定且需要与模型推理过程紧密结合的范围。
- **RAG：** 在知识密集型任务中表现出色，这些任务中获取特定、最新或专有信息非常重要。例子包括对公司内部文档进行问答，根据最新的产品手册提供客户支持，或从最新新闻文章中合成信息。

以下图表描绘了基本工作流程的差异：

> 此图对比了微调（顶部）的一次性权重更新过程与 RAG（底部）的动态推理时检索和增强过程。

值得一提的是，RAG 和微调并非相互排斥。一个大型语言模型可以为特定范围的风格和术语进行微调，*然后*在 RAG 系统中用作生成器组件，以访问该范围内的最新文档。然而，对于本入门课程，我们将主要侧重于在 RAG 架构中使用标准预训练模型。

"理解这些差异对于决定哪种方法或哪种组合最适合您在使用大型语言模型时的具体目标非常重要。RAG 提供了一种有效途径，可以在不改变底层模型的情况下，用外部动态知识增强大型语言模型，为许多应用带来了准确性、时效性和可验证性方面的重要优势。"

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230237300c7ee6416dd3438a531-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 33; Publisher: NeurIPS; Volume: 33; DOI: [10.55917/ccf.JMDJDS.871](https://doi.org/10.55917/ccf.JMDJDS.871)
  介绍了RAG框架，概述了其架构以及结合检索与生成的优势。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; Pages: 144; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  提供了大型语言模型的概述，包括对微调和各种适应技术的讨论。
