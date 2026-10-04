---
course: "getting-started-rag"
chapter: "data-preparation-for-rag"
lesson: "need-for-chunking"
sourceId: 3646
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-3-data-preparation-for-rag/need-for-chunking"
title: "文档分块的必要性"
description: "解释为何大型文档需要被拆分成更小的分块，以实现有效检索并符合 LLM 的上下文限制。"
order: 2
plots: []
sourceHash: "e1b7ac299ac96db1c58c691891f1a7b4c255e27cc6d6d4b502a9477c2c47689b"
sourceCorrections: []
---

处理文档，例如详述技术规范的PDF、包含新闻文章的网页，或包含公司内部报告的文本文件，通常涉及将它们加载到系统中。然而，简单地将这些整个大型文档直接传递给检索系统或大型语言模型 (LLM) 会带来严重问题。这时，**文档分块**的过程就变得十分重要。

想一想您如何与 LLM 交互。您提供一个提示，模型生成一个响应。大多数 LLM 都有一个**上下文 (context)窗口**，这是对文本量（以标记 (token)衡量，大致对应单词或单词的一部分）的严格限制，模型在处理输入和生成输出时一次性可以考虑的文本量。此限制包括您的原始查询、您提供的任何指令，*以及*RAG 系统检索到的上下文信息。

常见的上下文窗口大小可能从几千个标记（例如 4,096 或 8,192）到新模型中的数十万个。然而，即使是大型上下文窗口也并非无限。如果您的检索系统检索出几个长文档，甚至只有一个非常大的文档，试图将所有信息塞入 LLM 的提示中可能会超出其上下文限制。这会导致错误，更糟的是，LLM 会直接忽略不符合的信息，这让检索失去了意义。分块将大文档分解成小片段，确保检索到的上下文片段易于管理，并符合 LLM 的操作限制。

除了上下文窗口的限制之外，分块还很大程度上影响检索过程本身的*相关性*和*准确性*。想象一下在一本 500 页的手册中寻找一个简要提及的特定技术细节。如果您的检索系统将整本手册作为一个单元来操作，其向量 (vector)嵌入 (embedding)将代表整本书的平均含义。一个特定的查询可能会淹没在杂乱信息中。

然而，如果手册被分成逻辑段落甚至段落，每个分块都会有自己的向量嵌入，代表其更集中的内容。当您查询那个特定的技术细节时，检索系统更可能识别出包含相关信息的精确分块。对更小、语义连贯的分块进行搜索，使检索器能够准确找到最相关的段落，而不是返回过于宽泛或离题的结果。

此外，为非常大的文本片段生成单个向量嵌入有时会稀释其中的特定语义细节。通过为较小的分块创建嵌入，您可以获得文档内容更细粒度的表示，这通常会在相似性搜索中带来更好的表现。

因此，分块不仅仅是由于模型限制而产生的技术必要性；它是一种提高 RAG 系统质量和准确性的基础方法。通过将文档分解成易于消化的片段，您可以确保上下文符合 LLM，并很大程度上增加检索到的信息正是有效回答用户查询所需内容的可能性。接下来的部分将讨论实施这一重要数据准备步骤的不同方法。

## 参考资料

- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗 Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098113575/) — Lewis Tunstall, Leandro von Neumann, Lucile Saulnier (2022)
  Publisher: O'Reilly Media
  一本关于Transformer模型的实用指南，涵盖分词以及输入序列长度（上下文窗口）对语言模型操作的影响。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  对检索增强生成（RAG）系统进行全面调查，其中包含专门讨论数据预处理、检索策略以及文档分块对RAG性能影响的部分。
