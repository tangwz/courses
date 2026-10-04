---
course: "advanced-vector-search-llms"
chapter: "hybrid-search-approaches"
lesson: "limitations-pure-vector-search"
sourceId: 3872
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-3-hybrid-search-approaches/limitations-pure-vector-search"
title: "纯向量搜索的局限性"
description: "讨论纯向量搜索可能不足之处，例如精确关键词匹配或冷启动。"
order: 1
plots: []
sourceHash: "86c42105c8632520b1b510d18cbd2f30e7ddb2232dfbba91c51a196b85f2500d"
sourceCorrections: []
---

向量 (vector)搜索依靠密集向量 (dense vector)表示，善于捕获语义关系，即使关键词不完全匹配也能找到相似项。这种能力是现代搜索和检索增强生成（RAG）系统的核心。然而，仅在向量空间中依赖最近邻搜索有其固有的局限性，尤其在要求精确性、特指性或处理超出嵌入 (embedding)模型熟知词汇的术语时。了解这些局限对构建真正高效的搜索系统很重要，并促使混合方法的开发。

### 精确匹配和关键词敏感性的难题

向量 (vector)嵌入 (embedding)旨在将语义相似的理念映射到向量空间中相近的位置。尽管功能强大，但此过程固然会弱化词汇差异。对 `configure_logging(level="DEBUG")` 的查询，可能在语义上接近关于通用日志设置，甚至不同日志级别如 `INFO` 或 `WARNING` 的代码片段。然而，用户可能特别需要精确的函数调用签名，或解释 `DEBUG` 级别的文档。

纯向量搜索在处理*精确*文本形式很重要的查询时存在不足：

1. **标识符和代码：** 产品ID（`SKU-A4B1-XYZ`）、错误代码（`ERR_CONN_TIMEOUT`）、特定函数名（`calculate_iou_score`）、独特的专有名词或保留关键词，通常需要精确匹配。向量搜索可能找到语义上相关的项，但漏掉包含精确标识符的那个。
2. **技术术语和缩写：** 尽管嵌入通常能很好地处理常见缩写，但高度专业化或新创造的术语可能无法清晰地表示。搜索 `HNSW`（分层可导航小世界）理想情况下应优先显示明确定义或讨论HNSW的文档，而不仅仅是关于近似最近邻（ANN）的通用文章。
3. **有意关键词限制：** 用户有时会加入特定的关键词来有意缩小搜索范围。例如，查询“机器学习 (machine learning)教程 *不包括* 深度学习 (deep learning)”意味着一个特定的负面约束，纯向量相似性可能会忽略，从而可能会返回被认为语义上接近通用机器学习的深度学习教程。

在这些情况下，使向量搜索功能强大的语义“模糊性”变为一个劣势。系统优先考虑相似性而非词汇精确性，未能检索到以精确术语匹配为主要相关性信号的文档。

### 处理新词、罕见词或词汇外（OOV）术语

嵌入 (embedding)模型在大量数据集上训练，但它们不可避免地遇到训练期间未见的术语（词汇外或OOV术语），或出现频率过低以至于无法形成高质量向量 (vector)表示的术语。这通常被称为术语的“冷启动”问题。

- **子词 (subword)模型：** 像BERT或Sentence-BERT这样的模型使用子词分词 (tokenization)（例如，WordPiece、BPE）将OOV词汇分解为已知的较小单元来处理它们。尽管这使得表示任何文本成为可能，但由通用子词组成的新词或罕见词的生成嵌入可能缺乏特指性，甚至具有误导性。例如，一个新发布的软件库名称可能被分解为常见的子词，导致其嵌入更接近通用软件理念而非其特定功能。
- **低频词：** 在训练语料库中很少出现的词汇，可能具有不稳定或区分度差的嵌入。向量搜索可能难以可靠地检索包含这些特定低频词的文档。

这种局限在快速发展的领域（例如，技术、研究、时事）中尤为明显，这些领域新术语不断涌现 (emergence)。纯向量搜索系统可能无法显示包含这些新术语的最相关、最新的文档，直到嵌入模型被重新训练或微调 (fine-tuning)。

### 当语义相似性不够具体时

有时，语义相似性过于宽泛。用户查询可能旨在更广泛主题中的特定*方面*或*关系*。向量 (vector)搜索优化整体语义接近度，可能会返回与通用主题相关的文档，但错过所要求的特定视角。

设想一个查询，例如“GDPR对用户同意书的影响”。纯向量搜索可能返回关于以下内容的文档：

- 通用GDPR规定。
- 关于用户同意书的一般信息。
- 隐私政策（语义相关）。

尽管相关，这些可能无法直接针对GDPR和同意书之间特定的*影响*关系。最接近的向量可能代表主要主题（GDPR、同意），而非查询结构和措辞所暗示的特定交集。

### 行业专业性和嵌入 (embedding)模型偏差

在海量网络语料库（如维基百科、Common Crawl）上训练的通用嵌入模型提供广泛的语义理解。然而，它们可能缺乏高度专业化行业（例如，法律判例、医学研究、复杂金融工具）所需的理解。

- **歧义术语：** 一个术语可能有一个常见含义，并在某个行业内有非常特定的含义。通用模型可能默认为其常见含义的嵌入。
- **行业术语：** 专业术语可能被视为罕见词（如上所述），或被映射到语义相关但来自通用范围的不正确理念。

如果不对特定行业语料库进行微调 (fine-tuning)，使用通用模型的向量 (vector)搜索可能在专业化应用中导致次优的相关性。

### 转向混合解决方案

这些局限说明，尽管向量 (vector)搜索提供了语义理解的重要能力，但它并非所有搜索相关性难题的普遍解决方案。无法保证精确匹配、对术语新颖性和频率的敏感性、过度泛化的可能以及行业特异性问题，都需要补充性方法。通过将向量搜索与关键词检索（例如使用BM25）等方法相结合，后者擅长词汇匹配，我们可以构建混合搜索系统，它们结合两种方法的优点，从而在更广泛的查询范围内获得更高效的结果。接下来的部分将阐述如何有效地实施这些混合策略。

## 参考资料

- [The Probabilistic Relevance Model: BM25 and Beyond](http://dx.doi.org/10.1561/1500000019) — Stephen Robertson, Hugo Zaragoza (2009)
  Journal: Foundations and Trends in Information Retrieval; Publisher: Now Publishers; Volume: 3: No. 4; Pages: 333-389; DOI: [10.1561/1500000019](https://doi.org/10.1561/1500000019)
  解释了BM25检索函数，这是一种词汇搜索方法，通过擅长精确的术语匹配来补充语义搜索。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Volume: 1; Pages: 4171–4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  描述了BERT模型及其子词分词方法，与密集嵌入的形成方式及其OOV处理有关，并间接指出了其具体局限性。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982-3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了Sentence-BERT，一种广泛使用的生成密集句子嵌入的方法，对于理解其创建方式及其在捕捉特定语义细微之处的优势和局限性至关重要。
