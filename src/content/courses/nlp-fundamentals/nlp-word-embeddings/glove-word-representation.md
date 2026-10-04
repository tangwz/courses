---
course: "nlp-fundamentals"
chapter: "nlp-word-embeddings"
lesson: "glove-word-representation"
sourceId: 2499
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/glove-word-representation"
title: "GloVe：词语的全局向量表示"
description: "了解 GloVe 模型及其如何使用全局共现统计信息。"
order: 5
plots: []
sourceHash: "5b7993d1162e71875545fe51e81a5ad1c11c745fdc0291dae62fd3f50efaafcf"
sourceCorrections: []
---

Word2Vec 模型（如 Skip-gram 和 CBOW）通过在滑动窗口内预测局部上下文 (context)词语来学习词嵌入 (embedding)，但它们不直接使用整个语料库中存在的大量统计信息。而像潜在语义分析（LSA）这样的方法确实使用了全局统计信息（对词-文档或词-词矩阵进行矩阵分解），但它们在词语类比等衡量精确语义关系的任务上表现相对较差。

GloVe，即词语的全局向量 (vector)表示（Global Vectors for Word Representation），由斯坦福大学的 Jeffrey Pennington、Richard Socher 和 Christopher D. Manning 研发，旨在弥补这一不足。它被设计为兼顾两方面的优点：既能实现局部上下文预测方法（如 Word2Vec）所展现的含义捕获能力，又能获得全局矩阵分解技术带来的统计优势。

### 核心思想：共现概率的比率

GloVe 背后的核心理念是，词语之间共现概率的比率包含有意义的信息。请思考与“ice”（冰）和“steam”（蒸汽）相关的词语的共现概率。

令 $P(k | w) = P_{wk}$ 为词语 $k$ 出现在词语 $w$ 上下文 (context)中的概率。现在，让我们看看这些概率对于不同的考察词 $k$ 的*比率*。

- 考虑 $w_i = \text{ice}$ 和 $w_j = \text{steam}$。
- 考虑一个考察词 $k = \text{solid}$。我们预计比率 $P(\text{solid} | \text{ice}) / P(\text{solid} | \text{steam})$ 会很大，因为“solid”与“ice”的关系更紧密。
- 考虑 $k = \text{gas}$。我们预计比率 $P(\text{gas} | \text{ice}) / P(\text{gas} | \text{steam})$ 会很小，因为“gas”与“steam”的关系更紧密。
- 考虑 $k = \text{water}$（与两者都相关）或 $k = \text{fashion}$（与两者都不相关）。我们预计比率 $P(k | \text{ice}) / P(k | \text{steam})$ 会接近 1。

GloVe 假设这些比率编码了词语之间的关系，而目标是学习能够捕获这些比率的词向量 (vector)。

### 共现矩阵

GloVe 首先构建一个大型的词语共现矩阵，表示为 $X$。此矩阵中的一个条目 $X_{ij}$ 表示在整个语料库中，词语 $j$（上下文 (context)词）在词语 $i$（目标词）的特定上下文窗口内出现的次数。

“上下文窗口”的定义类似于 Word2Vec 中使用的定义。通常，共现词语对的贡献会根据词语间的距离进行加权，距离更近的词语贡献更多。例如，可以使用 $1/d$（其中 $d$ 是距离）这样的加权函数。

令 $X_i = \sum_k X_{ik}$ 为任意词语在词语 $i$ 上下文中出现的总次数。那么，词语 $j$ 在词语 $i$ 上下文中出现的概率为 $P_{ij} = P(j | i) = X_{ij} / X_i$。

> 该图表示了文本语料库中的词语共现如何影响共现矩阵 $X$ 中的条目。

### GloVe 模型和目标函数

GloVe 旨在为每个词语学习向量 (vector)，使其点积直接与其共现概率相关联。该模型从一个一般思想开始，即三个词语 $i$、$j$ 和 $k$ 之间的关系可以使用作用于其词向量的函数 $F$ 来建模：

$F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}$

这里，$w_i$ 和 $w_j$ 是主要词语的向量，而 $\tilde{w}_k$ 是考察词 $k$ 的一个单独的*上下文 (context)词向量*。使用两组向量（$w$ 和 $\tilde{w}$）使得模型更有效且更易于训练，同时捕获共现中的不对称性（例如，$P(\text{solid} | \text{ice})$ 不一定与 $P(\text{ice} | \text{solid})$ 相同）。

通过涉及向量差和同态性质的数学推导，GloVe 的作者得到一个特定形式，将词向量的点积与其共现计数的对数关联起来：

$w_i^T \tilde{w}_j + b_i + \tilde{b}_j \approx \log(X_{ij})$

这里，$b_i$ 和 $\tilde{b}_j$ 分别是目标词 $i$ 和上下文词 $j$ 的标量偏置 (bias)项。这些偏置有助于捕获与向量交互无关的频率效应。

这种关系构成了 GloVe 目标函数的基础，它是一个加权最小二乘回归模型：

$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij}))^2$

- $V$ 是词汇表 (vocabulary)大小。
- $w_i$ 和 $b_i$ 是目标词 $i$ 的向量和偏置。
- $\tilde{w}_j$ 和 $\tilde{b}_j$ 是上下文词 $j$ 的向量和偏置。
- $X_{ij}$ 是共现计数。
- $f(X_{ij})$ 是一个加权函数。

加权函数 $f(X_{ij})$ 很重要。它主要有两个作用：

1. 它避免了当 $X_{ij} = 0$ 时 $\log(X_{ij})$ 未定义的情况。实际上，$f(0) = 0$。
2. 它降低了非常频繁共现的权重 (weight)（例如“the”与“a”同时出现），防止它们主导训练目标。它也降低了极其罕见共现的权重，这些共现可能有噪声或不可靠。

一个常用的加权函数是：

$f(x) = \begin{cases} (x / x_{\text{max}})^\alpha & \text{如果 } x < x_{\text{max}} \\ 1 & \text{否则} \end{cases}$

其中 $x_{\text{max}}$ 是一个阈值（例如 100），$\alpha$ 通常设置为 $3/4$。

该模型使用诸如 AdaGrad 的技术进行训练，以最小化目标函数 $J$，从而学习词向量 $w_i$、上下文向量 $\tilde{w}_j$ 以及偏置 $b_i, \tilde{b}_j$ 的最优值。通常，词语 $i$ 的最终表示取作 $w_i + \tilde{w}_i$ 的总和。

### GloVe 对比 Word2Vec

- **训练数据**：Word2Vec 从语料库中流式获取的局部上下文 (context)窗口中学习。GloVe 首先计算整个语料库的全局共现统计数据，然后根据这些统计数据进行训练。
- **目标**：Word2Vec 使用基于预测的目标（预测上下文词语或中心词）。GloVe 使用加权最小二乘目标来直接捕获共现概率。
- **表现**：两种方法都能生成高质量的词嵌入 (embedding)。GloVe 在大型语料库上训练通常更快，因为耗时的共现计数只需预先完成一次。在下游任务上的表现有所不同；有时 GloVe 表现更优，有时 Word2Vec 更优。GloVe 明确使用全局统计信息对于捕获某些类型的关系可能具有优势。

### 使用 GloVe 词嵌入 (embedding)

与 Word2Vec 类似，预训练 (pre-training)的 GloVe 向量 (vector)广泛可用，它们在维基百科或 Common Crawl 等大型数据集上进行过训练。这些预训练向量通常为各种自然语言处理任务提供了一个很好的起点，相比从头开始训练，节省了大量计算精力。您可以加载这些向量，并将其用作处理文本分类、情感分析或命名实体识别等任务的模型输入特征，这与您使用预训练 Word2Vec 词嵌入的方式类似。

在接下来的章节中，我们将了解如何可视化这些高维词嵌入，并讨论加载和使用预训练模型的实用方法。

## 参考资料

- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  介绍GloVe模型及其理论基础的原始出版物。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University (online draft)
  一本标准的自然语言处理教材，提供了词嵌入（包括GloVe、Word2Vec和潜在语义分析）的详细说明。
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  Journal: ICLR Workshop Paper; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了最初的Word2Vec模型（Skip-gram和CBOW），是理解局部上下文词表示方法的基础。
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 26; Pages: 3111-3119; DOI: [10.48550/arXiv.1310.4546](https://doi.org/10.48550/arXiv.1310.4546)
  详细说明了Word2Vec框架的改进，对理解词向量优化技术很重要。
