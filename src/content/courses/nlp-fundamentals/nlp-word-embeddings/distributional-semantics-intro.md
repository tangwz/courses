---
course: "nlp-fundamentals"
chapter: "nlp-word-embeddings"
lesson: "distributional-semantics-intro"
sourceId: 2496
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/distributional-semantics-intro"
title: "分布式语义学简介"
description: "了解词语在相似语境中出现时，它们具有相似含义的观点。"
order: 2
plots: []
sourceHash: "c2cc852fa826e83de14cfb747357ac3d96d2603113dd63870b0432e8be853bd0"
sourceCorrections: []
---

词袋模型 (Bag-of-Words, BoW) 和词频-逆文档频率 (Term Frequency-Inverse Document Frequency, TF-IDF) 等方法是根据词语的计数或频率来表示文本的。虽然这些方法对于词语出现是强烈信号的任务有用，但它们有一个很大的局限：未能捕捉到词语之间的**含义**或**语义关联 (semantic relationship)**。例如，对于一个TF-IDF模型来说，`cat` 和 `feline` 仅仅是独立的标记 (token)，可能具有不同的频率分数；模型本身并没有固有的理解，即它们指代的是非常相似的**想法**。同样，`run` 和 `ran` 被视为完全独立的实体，尽管它们有很强的语法关联。

这就是**分布式语义学**派上用场的地方。其核心想法，常被语言学家 J.R. Firth 的论断 "知其词，看其伴" (You shall know a word by the company it keeps) 所概括，即经常在相似语境中出现的词语往往具有相似的含义。我们不再仅仅统计文档中的词语，而是分析那些通常出现在给定词语**周围**的词语模式。

### 什么是“语境”？

在这种情况下，一个词的“语境”通常指的是在一定窗口大小内围绕它的词语。例如，请看这个句子：

"这只**敏捷的棕色狐狸**跃过懒狗。"

如果我们选择一个大小为2的语境窗口（意味着它前面有两个词，后面有两个词），那么词语 `fox` 的语境将是 (`quick`, `brown`, `jumps`, `over`)。

现在，想象一下处理**海量**文本（一个语料库）。我们可以观察到不同词语出现的语境。分布式假说表明，`cat` 和 `feline` 这样的词语将经常出现在相似词语的包围中（例如，`meow`、`purr`、`pet`、`claws`），而像 `car` 这样的词则会出现在不同的语境中（例如，`drive`、`engine`、`road`、`wheel`）。

> 像‘银行业’和‘金融’这样的词语，在大量文本中倾向于共享相似的相邻词（语境），这与‘河流’和‘小溪’这类词语的典型相邻词不同。

### 从语境到含义表示

分布式语义学的基本理解是，我们可以使用这些语境模式来创建词语的数值表示，这些表示通常被称为**词嵌入 (embedding)**或**词向量 (vector)**。目标是为每个词学习一个向量，使得在相似语境中出现的词语，其向量在向量空间中彼此接近（例如，它们的余弦相似度很高）。

试想一下：如果 `cat` 和 `feline` 持续共享像 `pet`、`purr` 和 `meow` 这样的语境词，那么一个旨在捕捉这些共现模式的算法应该自然地将它们学习到的向量彼此靠近。这与基于频率的方法形成鲜明对比，在那些方法中，这种相似性纯粹是基于文档重叠的巧合，而非相邻词语。

本质上，我们正在从基于计数的稀疏高维表示（例如BoW或TF-IDF向量，其中大多数条目为零）转向密集的低维向量（例如50到300维），其中每个维度都捕捉了词语含义的某种潜在方面，这些方面源自其用法。

这种方法不仅能将同义词分组。它还可以捕捉更复杂的关联，比如类比。例如，一组训练良好的词嵌入可能会展现出这种关联：$\text{向量}('king') - \text{向量}('man') + \text{向量}('woman') \approx \text{向量}('queen')$，仅仅因为在整个训练语料库中，`king` 和 `man` 之间的语境关联映射了 `queen` 和 `woman` 之间的语境关联。

使用共现统计的想法并非全新；基于构建大型词语-语境矩阵的方法已经存在了一段时间。然而，这些矩阵通常**非常庞大**且稀疏。重要的进展伴随着像 Word2Vec 和 GloVe 这样的算法出现，这些算法提供了有效的方法，可以直接从文本数据中学习这些密集且有意义的向量表示。这些算法构成了许多现代自然语言处理应用的根本，并且是本章接下来部分的重点。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  提供关于NLP中分布语义、词嵌入及其历史背景的全面教材。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  介绍GloVe，一种用于获取词向量表示的无监督算法，结合了全局矩阵分解和局部上下文窗口方法。
- [CS224n: Natural Language Processing with Deep Learning (Winter 2023 Lecture Notes)](https://web.stanford.edu/class/cs224n/2023/) — Christopher D. Manning, Abigail See (2023)
  顶尖自然语言处理课程的教材和讲义，提供了对分布语义和词嵌入模型的易懂且严谨的解释。
