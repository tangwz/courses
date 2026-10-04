---
course: "optimizing-rag-for-production"
chapter: "advanced-retrieval-optimization"
lesson: "advanced-document-representations-rag"
sourceId: 6181
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-2-advanced-retrieval-optimization/advanced-document-representations-rag"
title: "高级文档表示方法：多向量与 ColBERT"
description: "采用多向量嵌入和 ColBERT 等增强型文档表示方法，实现更精细的 RAG 检索。"
order: 6
plots: []
sourceHash: "1bfe5cc1dcef3683a8970c3c8e4018050754c1660c5c42414c8886d0b59fb30b"
sourceCorrections: []
---

标准的单向量 (vector)嵌入 (embedding)在把握语义方面很有效，但在处理冗长、多面文档时有时不够充分。单个向量表示整个文档可能模糊具体细节，或无法呈现不同章节讨论的独立主题。这会降低检索的准确性，特别是当用户查询指向一个较大文档中的某个特定方面时。为解决此问题，已发展出改进的文档表示方法，以提供更细致、更注重上下文 (context)的匹配。我们将探讨两种主要的方法：多向量嵌入和 ColBERT。

### 多向量 (vector)嵌入 (embedding)

多向量嵌入用多个独立向量表示一个文档。这种方式不将所有信息压缩到一个密集表示中，而是能呈现文档内不同侧面、章节或细小的部分信息。此方法在提升从长篇或复杂文本中检索的准确性方面效果显著。

#### 多向量嵌入的工作方式

核心理念是将文档划分成较小、连贯的单元，然后为每个单元生成一个嵌入。这些独立的嵌入都与父文档关联。创建多向量表示的常见方法包括：

1. **基于分块的分割**：文档被分成较小的块，例如段落、固定大小的文本窗口（通常有重叠），或通过标题识别的逻辑部分。然后，每个块独立进行嵌入。
2. **命题分割**：文档被分解为独立的命题或主张。每个命题都进行嵌入，旨在获得非常细致的检索单元。由于需要准确的命题提取，这通常实施起来更为复杂。
3. **摘要和细节向量**：一个向量可能表示文档的摘要，而其他向量表示具体的细节部分，甚至阐述不同要点的独立句子。

检索时，查询会进行嵌入，然后与知识库中文档相关的所有独立向量进行比较。如果查询向量与文档的一个或多个子向量显示出高度相似性，则该父文档被认为是相关的。父文档的相关性分数可以通过汇总其组成向量的分数来确定，例如，通过取最大相似度分数、前k个最高分数的子向量的平均值，或使用更精密的聚合函数。

> 此图展示了文档的单向量表示与多向量表示之间的比较。在多向量方法中，文档被分割，每个片段都有自己的嵌入。

#### 多向量嵌入的优势

- **提高粒度**：能够更准确地匹配文档内的特定部分或陈述，而非依赖平均表示。
- **改善长文档处理**：特别有利于处理冗长的文档，因为在这些文档中，单个向量可能会稀释重要的局部信息。
- **更好地分离主题**：如果一个文档涵盖多个不同主题，独立的向量可以更高效地表示这些主题。

#### 多向量嵌入的考虑因素

- **存储量增加**：每个文档存储多个向量自然会增加嵌入索引的总大小。
- **分割策略**：检索质量很大程度上取决于所选的分割策略。糟糕的分割可能导致碎片化或脱离上下文 (context)的匹配。这与优化分块策略（如他处讨论）紧密相关。
- **检索复杂度**：检索逻辑变得更为复杂，需要对更大的向量池进行相似性搜索，并增加一个聚合步骤来对父文档进行评分。
- **计算成本**：查询时需要更多的向量比较，如果未通过高效的索引和搜索算法进行管理，这可能会增加延迟。

正确实施多向量嵌入能显著提高检索准确性，保证传递给生成器的上下文更集中且与查询相关。

### ColBERT: 上下文 (context)化延迟交互

ColBERT，意为基于 BERT 的上下文化延迟交互，是一种改进的架构，专为密集检索设计，侧重于查询和文档之间细粒度的、词元 (token)级的交互。与为查询和文档计算单个嵌入 (embedding)，然后简单得出相似性分数的传统双编码器模型不同，ColBERT 推迟并优化了交互过程。

#### ColBERT 的工作方式

ColBERT 通过为查询和文档中的每个词元创建上下文化嵌入来处理它们。其名称中的“延迟交互”指其计算相关性的方式：

1. **查询编码**：输入查询中的每个词元都使用类 BERT 模型（查询编码器）编码为上下文化嵌入向量 (vector)。因此，查询 $Q$ 变为一组词元嵌入 $\{\mathbf{q}_1, \mathbf{q}_2, ..., \mathbf{q}_m\}$。
2. **文档编码**：类似地，并且通常离线进行，文档 $D$ 中的每个词元都使用类 BERT 模型（文档编码器）编码为上下文化嵌入向量。这些向量被存储，从而为每个文档生成一组词元嵌入 $\{\mathbf{d}_1, \mathbf{d}_2, ..., \mathbf{d}_n\}$。
3. **延迟交互 - 最大相似度**：在查询时，对于每个查询词元嵌入 $\mathbf{q}_i$，ColBERT 计算其与候选文档 $D$ 中所有词元嵌入 $\{\mathbf{d}_j\}$ 的最大相似度 (MaxSim)。这能找出该查询词元在文档中的最强匹配。
   
   $$
   MaxSim(\mathbf{q}_i, D) = \max_{j=1 \dots n} (\mathbf{q}_i^T \mathbf{d}_j)
   $$
   
4. **评分**：文档 $D$ 相对于查询 $Q$ 的最终相关性分数是所有查询词元嵌入的 MaxSim 分数之和。
   
   $$
   Score(Q, D) = \sum_{i=1}^{m} MaxSim(\mathbf{q}_i, D)
   $$
   

这种方式使 ColBERT 能够识别出那些包含大量查询词项，且这些词项可能不连续但强烈地出现于文档中的文本。

> 此图展示了 ColBERT 的延迟交互机制。查询词元嵌入与所有文档词元嵌入进行比较，以找出最大相似度，然后将其求和。

#### ColBERT 的优势

- **细粒度相关性**：通过在词元级别操作，ColBERT 能够发现单向量方法可能忽略的细微词汇和语义匹配。
- **BERT 的上下文能力**：利用 BERT 为查询和文档词元生成的丰富上下文嵌入。
- **提高排序准确性**：在重排序任务中常表现出优越的性能，从而在顶部获得更相关的结果。
- **重排序效率**：尽管交互过程详细，但文档词元嵌入是预先计算的。MaxSim 操作可以高效实现，特别是在对早期、更快阶段检索到的一小组候选文档进行重排序时。

#### ColBERT 的考虑因素

- **初次检索的计算成本**：由于词元级比较的数量，将 ColBERT 作为独立的首轮检索器应用于非常大的语料库可能计算密集。它通常更适合用作重排序器。
- **存储需求**：存储每个文档中每个词元的上下文化嵌入会导致索引大小显著增加，相比文档级甚至分块级嵌入更大。
- **实施复杂性**：搭建和优化 ColBERT 流程比标准密集检索系统更复杂。这包括管理词元嵌入的存储和实现高效的 MaxSim 计算。
- **参数 (parameter)敏感性**：性能可能对词元嵌入的维度和所使用的特定 BERT 模型等参数敏感。

### 选择合适的表示方法

选择单向量 (vector)、多向量或像 ColBERT 这样的词元 (token)级表示方法，取决于您的特定应用需求、数据特点以及可用的计算资源。

- **单向量嵌入 (embedding)**可能足以应对短小、主题集中的文档集合，或当计算效率和存储是主要限制时。
- **多向量嵌入**为更长或更复杂的文档提供了良好的平衡，当在同一文档内呈现不同方面很重要时。它们在许多用例中对相关性提供了明显的改进，同时避免了词元级模型的全部开销。
- **ColBERT**（以及类似的延迟交互模型）在追求最高检索或重排序准确性时是一个有力的选择，特别是对于需要细致理解的查询，并且计算预算允许其进行更密集的处理时。它通常最适合用作第二阶段重排序器。

实验和评估是不可或缺的。您可能会发现，采用混合方式，例如使用多向量表示进行初步候选检索，然后使用 ColBERT 对前 k 个结果进行重排序，能为您的生产 RAG 系统带来最佳的性能成本权衡。这些优化的表示方法，通过实现更准确和细致的匹配，直接帮助获取更高质量的信息用于生成阶段，最终得到更准确、更相关的 RAG 输出。

## 参考资料

- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://dl.acm.org/doi/10.1145/3397271.3401075) — Omar Khattab, Matei Zaharia (2020)
  Journal: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20); Publisher: ACM; Pages: 39-48; DOI: [10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)
  介绍ColBERT的开创性论文，详细阐述了其架构和用于细粒度检索的后期交互机制。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Curran Associates, Inc.; Volume: 33; Pages: 9459-9474
  介绍检索增强生成（RAG）框架的开创性论文，展示了将信息检索与生成模型结合的优势，为高级检索优化提供了背景。
