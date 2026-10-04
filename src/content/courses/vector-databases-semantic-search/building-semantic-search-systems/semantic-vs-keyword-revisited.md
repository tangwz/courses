---
course: "vector-databases-semantic-search"
chapter: "building-semantic-search-systems"
lesson: "semantic-vs-keyword-revisited"
sourceId: 3560
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-4-building-semantic-search-systems/semantic-vs-keyword-revisited"
title: "语义搜索与关键词搜索再比较"
description: "对比理解含义与匹配词语的方法和益处。"
order: 1
plots: []
sourceHash: "27b332fcb28dbc4ce4f87cafef1f6bc0e6f9d9c3a93ee96e859463fc0343319d"
sourceCorrections: []
---

当我们开始构建端到端的语义搜索流程时，重新审视语义搜索与其前身关键词搜索之间的基本区别是很有价值的。虽然您可能已经从之前的讨论中了解了其不同，但在设计系统内数据和查询的实际流程时，理解它们的核心机制和局限性变得尤为重要。选择合适的方法，甚至将它们结合起来，在很大程度上取决于您应用程序的具体需求。

### 关键词搜索：匹配词语

传统的关键词搜索系统主要基于词汇匹配。其核心在于识别包含用户查询中特定词语或短语的文档。

- **索引：** 这些系统通常构建倒排索引。这种结构将文档库中找到的每个独有的词（或*术语*）映射到包含该术语的文档列表，通常还包含位置信息和频率计数。诸如词干提取（将词语还原为词根形式，例如，“running” -> “run”）和词形还原（将词语还原为词典形式，例如，“ran” -> “run”）等技术常用于在索引之前规范化术语。停用词（常见的词语如“the”、“a”、“is”）通常会被移除。
- **查询：** 当用户提交查询时，系统会进行类似处理（分词 (tokenization)、词干提取/词形还原、移除停用词）。然后，系统会在倒排索引中查找已处理的查询术语，以检索匹配的文档。
- **排序：** 相关性通常使用评分算法来确定，例如TF-IDF（词频-逆文档频率）或更现代的变体如BM25（最佳匹配25）。这些算法优先考虑查询术语在文档中出现频率较高（TF）但在整个语料库中相对较少（IDF）的文档，这表示更高的相关性。

**局限性：** 关键词搜索难以处理歧义和上下文 (context)。

1. **同义词：** 它无法检索使用不同但含义相似的词语的文档。搜索“fast laptop”可能会错过描述“quick notebook”或“high-speed portable computer”的文档。
2. **多义词：** 当查询术语具有多个含义时，它可能会返回不相关的结果。搜索“python”可能会返回关于蛇而不是编程语言的结果，这取决于语料库。
3. **理解：** 它无法把握查询背后的潜在意图或内容。搜索“how to fix a dripping sink”可能会错过一篇标题为“Stopping Leaks in Kitchen Plumbing”的优秀指南，如果其中不包含完全匹配的关键词。

### 语义搜索：理解含义

语义搜索，由向量 (vector)嵌入 (embedding)和向量数据库驱动，通过操作查询和文档之间的*含义*或*语义相似性*来解决这些局限性。

- **索引：** 语义搜索系统不索引术语，而是首先使用预训练 (pre-training)的嵌入模型（如Sentence-BERT或OpenAI的模型）将文本片段（或其他数据）转换为高维数值向量（嵌入）。这些向量捕捉内容的语义核心。向量数据库随后对这些嵌入进行索引，通常使用近似最近邻（ANN）算法（如第3章所述）进行高效检索。相关的元数据通常与向量一同存储。
- **查询：** 当用户提交查询时，相同的嵌入模型会将查询文本转换为向量。系统随后在向量索引中搜索在高维空间 (high-dimensional space)中最接近查询向量的文档向量，使用余弦相似度或欧氏距离等距离度量。
- **排序：** 主要的排序机制是查询向量与文档向量之间的相似度得分（或距离）。向量“更接近”查询向量的文档被认为在语义上更相关。基于元数据或其他信号，可能会应用进一步的重新排序步骤。

**优势：** 语义搜索在关键词搜索不足的地方表现出色。

1. **同义词/上下文 (context)：** 它能自然处理同义词和相关内容。“Fast laptop”和“quick notebook”可能具有相似的向量表示，因此可以为同一个查询被检索到。
2. **意图识别：** 它能更好地理解用户的潜在需求，即使措辞差异很大也能检索到相关信息。“dripping sink”的查询更有可能找到“Stopping Leaks”的指南。

### 为什么这种区别现在很重要

理解这些差异，在您设计搜索流程（本章的主要内容）时非常重要。

- **组件选择：** 您选择的索引机制（倒排索引与向量 (vector)索引）、查询处理（术语规范化与向量生成）以及排序策略（BM25与向量相似度）直接来源于您是优先考虑词汇匹配还是语义理解。
- **混合方法：** 许多现代系统采用*混合搜索*，结合两者的优势。关键词搜索对于特定实体查找（例如，产品ID、确切名称）可能有效，而语义搜索则处理更详细的查询。识别两者的不同之处对于设计有效的结合策略（我们将在本章稍后讨论）是必不可少的。
- **权衡：** 关键词搜索通常计算成本较低且更易于解释（您可以根据术语的存在看出文档*为何*匹配）。语义搜索需要强大的嵌入 (embedding)模型和专门的向量数据库，虽然非常有效，但解释两个向量*为何*被认为是相似的可能不太直观。

> 关键词和语义搜索流程的比较，针对查询“fastest laptop”。关键词搜索侧重于术语存在，而语义搜索则使用向量相似性来查找相关文档。

在设计语义搜索流程的架构时，请记住这些基本差异。它们将指导您如何处理数据准备、嵌入生成、索引、查询和结果排序，以构建一个真正理解用户意图的系统。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  本书是信息检索领域的标准教材，全面涵盖了关键词搜索机制、倒排索引、TF-IDF 和 BM25 等传统搜索的基础知识。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  这篇论文介绍了Sentence-BERT，一种广泛采用的方法，用于从类BERT模型生成具有语义意义的句子嵌入，直接展示了语义搜索系统如何生成查询和文档向量。
