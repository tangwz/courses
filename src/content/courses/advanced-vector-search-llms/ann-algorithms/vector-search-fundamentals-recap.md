---
course: "advanced-vector-search-llms"
chapter: "ann-algorithms"
lesson: "vector-search-fundamentals-recap"
sourceId: 3843
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-1-ann-algorithms/vector-search-fundamentals-recap"
title: "重温向量嵌入与搜索基础"
description: "简要回顾向量嵌入、相似性度量标准和基本的最近邻搜索挑战。"
order: 1
plots: ["plots/3843-0.json"]
sourceHash: "55ccae4bc4acae7857a8fff7f8aea1aeff983c8ccaa3536c7ba8db15cb2b7a07"
sourceCorrections: []
---

理解近似最近邻（ANN）算法的复杂机制，需要对向量 (vector)嵌入 (embedding)以及高维空间 (high-dimensional space)中相似性搜索的基本挑战有清晰的理解。这种理解阐明了*为什么* HNSW 和 IVF 等先进技术对于现代大型语言模型（LLM）应用必不可少。

### 向量 (vector)嵌入 (embedding)：以数值方式捕获含义

大型语言模型（LLM）和专门的嵌入模型（如 Sentence-BERT、OpenAI 的 Ada 嵌入或 Cohere 的模型）擅长将文本、图像或其他数据类型转换成稠密的数值向量，这些向量常被称为嵌入。每个嵌入都是高维向量空间中的一个点，通常具有数百甚至数千个维度。

这些嵌入的强大之处在于它们能够捕获语义关系。含义或上下文 (context)相似的项会被映射到此向量空间中彼此靠近的点。例如，"king"（国王）和 "queen"（王后）的嵌入可能比 "king"（国王）和 "cabbage"（卷心菜）的嵌入更接近。此属性使我们能够进行有意义的相似性比较。

### 目的：相似性搜索

给定一个表示为查询向量 (vector) $q$ 的查询（例如，用户的提问、一张图片、一个产品描述），向量搜索的目的是在大型数据集 $X = \{x_1, x_2, ..., x_N\}$ 中找到与 $q$ 最相似的 $k$ 个向量。这个过程常被称为 k 最近邻（k-NN）搜索。

### 衡量相似性：度量标准

我们如何量化 (quantization)这个高维空间 (high-dimensional space)中的“相似性”或“接近度”？可以使用多种距离或相似性度量标准。以下是两种常用的：

1. **欧氏距离（$L_2$ 距离）：** 衡量空间中两点（向量 (vector)）$a$ 和 $b$ 之间的直线距离。

   
   $$
   d_{L2}(a, b) = \sqrt{\sum_{i=1}^{D} (a_i - b_i)^2}
   $$
   

   其中 $D$ 是向量的维度。距离越小表示相似性越高。
2. **余弦相似性：** 衡量两个向量 $a$ 和 $b$ 之间夹角的余弦值。它侧重于向量的朝向（方向），而不是它们的大小。其取值范围从 -1（完全相反）到 1（完全相同方向），0 表示正交性（不相关的方向）。

   
   $$
   similarity_{cosine}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} = \frac{\sum_{i=1}^{D} a_i b_i}{\sqrt{\sum_{i=1}^{D} a_i^2} \sqrt{\sum_{i=1}^{D} b_i^2}}
   $$
   

   值越高表示相似性越高。对于语义搜索中归一化 (normalization)为单位长度的嵌入 (embedding)，余弦相似性常受到青睐，因为其方向比向量的大小更能有效捕获语义内容。请注意，对于归一化向量，最大化余弦相似性等同于最小化欧氏距离。

### 暴力方法：精确 k-NN 搜索

找到精确 $k$ 最近邻的最直接方法是暴力法：

1. 获取查询向量 (vector) $q$。
2. 计算 $q$ 与数据集 $X$ 中*每个*向量 $x_i$ 之间的距离（例如，欧氏距离或余弦距离，即 $1 - \text{余弦相似性}$）。
3. 对这些距离进行排序。
4. 返回对应于最小距离（或最大相似性）的 $k$ 个向量。

### 可扩展性问题：为什么精确搜索会失败

尽管暴力方法简单且保证能找到真正的最近邻，但它有一个显著缺点：计算成本高。对于一个包含 $N$ 个向量 (vector)，每个向量维度为 $D$ 的数据集，计算查询向量与所有数据集向量之间的距离大约需要 $N \times D$ 次浮点运算。排序还会增加额外开销，通常为 $O(N \log N)$ 或 $O(N \log k)$。

考虑一个典型的大型语言模型应用场景：

- **数据集大小 (N)：** 数百万甚至数十亿文档/项。
- **维度 (D)：** 数百或数千（例如，许多基于 BERT 的模型为 768，OpenAI Ada v2 为 1536）。

使用暴力法搜索十亿个 768 维向量，每次查询涉及数万亿次计算。这使得精确 k-NN 对于需要低延迟的实时应用（如检索增强生成（RAG）或交互式语义搜索）来说，在计算上是不可行的。



![精确 k-NN 查询时间与数据集大小对比（示意）](plots/3843-0.json)



> 示意性地显示了精确 k-NN 的查询时间如何随数据集大小 (N) 和维度 (D) 大致呈线性增长，图表以对数刻度绘制。实际时间很大程度上取决于硬件和实现。

### 对近似的需求

这个可扩展性瓶颈是近似最近邻（ANN）搜索的主要推动因素。ANN 算法牺牲了找到绝对精确最近邻的保证。它们旨在非常快速地找到*极有可能的*近邻，通常比精确搜索快几个数量级。它们通过使用巧妙的索引结构和搜索策略来避免将查询向量 (vector)与数据集中的每个向量进行比较，从而实现这一点。

核心权衡在于准确性（通常用召回率衡量，即找到的真实最近邻的比例）与性能（搜索延迟、吞吐量 (throughput)、内存使用）之间。不同的 ANN 算法在此权衡曲线上提供不同的点。

理解精确搜索的这个基本限制以及对高效近似的需求，为我们现在研究后续章节中 HNSW、IVF 和 PQ 等复杂 ANN 算法的内部工作原理提供了必要的背景。这些技术是驱动要求严格的 LLM 应用中快速且可扩展向量搜索的引擎。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) — Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  这本教材提供了向量空间模型、相似度度量以及信息检索和语义搜索原理的基础知识，解释了文本如何被表示为向量以及如何衡量相似度。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  本文介绍了Sentence-BERT，一种生成高质量句子嵌入的模型，可直接应用于现代大型语言模型（LLM）应用程序中的语义相似度任务，并解释了如何推导上下文嵌入。
- [Approximate Nearest Neighbors: Towards Optimal Solutions](https://dl.acm.org/doi/10.1145/276698.276876) — Piotr Indyk, Rajeev Motwani (1998)
  Journal: Proceedings of the thirtieth annual ACM symposium on Theory of computing; Publisher: Association for Computing Machinery (ACM); Pages: 604-613; DOI: [10.1145/276698.276876](https://doi.org/10.1145/276698.276876)
  一篇开创性的理论论文，为近似最近邻（ANN）搜索奠定了基础，阐述了高维空间中精确搜索的计算挑战，并推动了近似方法的发展。
