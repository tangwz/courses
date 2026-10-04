---
course: "getting-started-rag"
chapter: "rag-retrieval-component"
lesson: "retriever-role"
sourceId: 3638
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-2-rag-retrieval-component/retriever-role"
title: "检索器在 RAG 中的作用"
description: "说明检索器的主要功能：从知识源中查找相关文档或文本段落。"
order: 1
plots: []
sourceHash: "919a41786d46e0c56d775b16791f128a3eb2a85e15ff0dfe17725a817f2f8136"
sourceCorrections: []
---

检索器是 RAG 系统的专门信息查找器。可以把标准的大型语言模型（LLM）想象成一个知识渊博的人，但其知识更新停留在某个时间点，并且可能偶尔会记错细节或缺乏特定领域的专业知识。检索器的任务是为这个知识组件（LLM）提供特定、及时且相关的文件或文本段落，以便在需要时准确回答查询。

其主要职能很简单：给定用户查询后，检索器会在预定义的知识源中搜索，例如文档、文章、网站数据或数据库条目集合。然后它会识别并提取最有可能有助于形成准确且基于上下文 (context)的回答的信息片段。

请看工作流程：

1. **输入：** 检索器接收用户的输入查询。通常，此查询会被转换为数值表示（一种嵌入 (embedding)，我们稍后会介绍），以捕获其语义含义而非仅是关键词。
2. **搜索：** 它对已索引的知识库执行搜索操作。这通常涉及复杂的技术，如向量 (vector)相似度搜索，旨在找到与查询意图语义相关的内容，而不仅仅是匹配单词。
3. **输出：** 检索器返回一个相关文本“块”或文档片段的排序列表。这些片段是被认为与根据知识源内容回答原始查询最相关的信息。

这种检索到的上下文是检索增强生成 (RAG)（Retrieve-Augmented Generation）的“增强”部分。它被传递到下一个阶段，即生成器（LLM），构成输入提示的一个重要部分，指导最终的答案生成。

> 检索器根据用户的输入查询已索引的知识源，并为生成器（LLM）提供相关上下文，然后由生成器生成最终回答。

检索器输出的质量直接影响 RAG 系统的整体效率。如果检索器未能找到必要信息或提取不相关的段落，生成器无论能力多强，都会遇到困难。如果提供的上下文缺失、误导或不正确，它就无法合成准确、有充分依据的答案。“垃圾进，垃圾出”的原则在这里非常适用；检索器的表现决定了 RAG 系统最终输出质量的上限。

因此，了解检索器如何工作以及如何有效地配置它，对于构建可靠的 RAG 应用来说很重要。接下来的部分将讨论实现这一检索的核心技术：用于表示文本含义的向量嵌入、用于查找相关内容的相似度搜索算法，以及用于大规模高效管理和查询这些表示的向量数据库。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍检索增强生成（RAG）框架的开创性论文，展示了其架构以及结合检索器和生成器的有效性。
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih (2020)
  Journal: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 8644-8657; DOI: [10.48550/arXiv.2004.04906](https://doi.org/10.48550/arXiv.2004.04906)
  关于使用密集向量嵌入进行高效准确的段落检索的开创性工作，这对于RAG检索器的“搜索”阶段至关重要。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2024)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  检索增强生成（RAG）的综合综述，涵盖了多种检索器设计、优化方法及其对RAG系统整体性能的影响。
