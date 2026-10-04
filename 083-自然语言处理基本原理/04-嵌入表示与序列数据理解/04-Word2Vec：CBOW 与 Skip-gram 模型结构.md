# Word2Vec：CBOW 与 Skip-gram 模型结构

来源：[原文](https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/word2vec-cbow-skipgram)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Word2Vec是一种用于学习词嵌入 (embedding)的根本方法。这种方法解决了TF-IDF等基于频率方法的局限性，并建立在分布式语义的直观理解之上。Word2Vec由Tomas Mikolov及其谷歌同事于2013年开发，它并非单一算法，而是一系列模型结构和优化方法的集合，用于从大规模文本语料库中学习词的向量 (vector)表示。

其核心思想简明而优雅：Word2Vec不是简单地统计词频，而是通过一个涉及词及其上下文 (context)的代理预测任务来训练一个浅层神经网络 (neural network)。显著成果在于，该网络隐藏层学到的*权重 (weight)*本身就充当了词嵌入。这些学到的向量能捕捉到令人惊叹的丰富语义关联 (semantic relationship)。

Word2Vec主要有两种模型结构形式：即连续词袋模型（CBOW）和Skip-gram模型。我们分别介绍。

### 连续词袋模型（CBOW）

CBOW模型结构旨在根据周围的上下文 (context)词来预测当前的中心词。假设你有一个句子：“the quick brown fox jumps over”。如果你的中心词是“fox”，并且你定义了一个大小为2的上下文窗口（即中心词前2个词和后2个词），那么上下文词将是“quick”、“brown”、“jumps”和“over”。CBOW使用这些上下文词作为输入来预测中心词“fox”。

**工作原理：**

1. **输入：** 表示在预设窗口大小内的上下文词的独热编码向量 (vector)。
2. **嵌入 (embedding)查找：** 这些独热向量用于从嵌入矩阵 $W$ 中查找其对应的稠密嵌入向量。假设我们的词汇量大小为 $V$，所需嵌入维度为 $N$。嵌入矩阵 $W$ 的维度将是 $V \times N$。
3. **上下文聚合：** 上下文词的嵌入向量通常被平均（或有时求和），以形成一个单一的上下文向量 $v_{context}$。
4. **预测：** 这个聚合的上下文向量通过第二个权重 (weight)矩阵 $W'$（维度为 $N \times V$），接着是softmax激活函数 (activation function)。输出是整个词汇表 (vocabulary)上的概率分布，表示在给定上下文的情况下，每个词作为中心词的可能性。
   
   $$
   P(\text{中心词} | \text{上下文}) = \text{softmax}(v_{context}^T W')
   $$
   
5. **训练：** 通过最小化预测概率分布与中心词的实际独热向量之间的差异（例如，使用交叉熵损失）来训练网络。主要的一点是，在此过程中，嵌入矩阵 $W$ 中的权重通过反向传播 (backpropagation)进行调整。这些调整后的权重*就是*学到的词嵌入。

> CBOW模型结构流程图。上下文词向量被聚合以预测中心词。

CBOW在计算上通常比Skip-gram更快，并且对高频词提供略好的表示。它主要对来自上下文的分布信息进行平滑处理。

### Skip-gram 模型

Skip-gram模型结构颠倒了预测任务：它不是从上下文 (context)预测中心词，而是尝试在给定中心词的情况下预测周围的上下文词。沿用之前的例子，如果输入中心词是“fox”，模型的目标是预测“quick”、“brown”、“jumps”和“over”等词。

**工作原理：**

1. **输入：** 一个表示单一中心词的独热编码向量 (vector)。
2. **嵌入 (embedding)查找：** 中心词对应的稠密嵌入向量 $v_{target}$ 从嵌入矩阵 $W$ 中获取（与CBOW中使用的$V \times N$矩阵相同）。
3. **预测：** 这个中心向量 $v_{target}$ 通过第二个权重 (weight)矩阵 $W'$（同样是 $N \times V$），接着是softmax激活。然而，目标不同。对于上下文窗口中的*每个*位置，模型都会输出一个词汇表 (vocabulary)上的概率分布。目标是使实际上下文词出现在这些位置的概率最大化。
   
   $$
   P(\text{上下文}_i | \text{中心词}) = \text{softmax}(v_{target}^T W')
   $$
   
   整体目标通常包括对窗口内所有实际上下文词的对数概率求和。
4. **训练：** 类似于CBOW，网络通过最小化预测误差来训练，并通过反向传播 (backpropagation)调整嵌入矩阵 $W$ 中的权重。这些权重就成为了词嵌入。

> Skip-gram模型结构流程图。中心词向量用于预测周围的上下文词。

Skip-gram的训练时间通常比CBOW长，因为从单一中心词预测多个上下文词是一项更复杂的任务，需要更多的更新。然而，在训练数据量较少时，它通常表现更好，并且在学习稀有词或短语的良好表示方面被认为特别有效。每个（中心词，上下文）对都提供一个学习信号，而CBOW则首先对上下文进行平均。

### 训练优化方法

在大规模语料库上高效训练这些模型需要优化技巧。为每次预测计算整个词汇表 (vocabulary)（可能包含数百万词）上的softmax在计算上是不可行的。解决这个问题有两种常见方法：

1. **负采样：** 与其更新输出层中所有不正确词的权重 (weight)，不如只更新一小部分不正确词的*样本*（称为“负样本”），以及正确的中心词/上下文 (context)词。这显著减少了每个训练步骤所需的计算量。
2. **层次Softmax：** 这种技术使用二叉树结构（如霍夫曼树）来表示词汇表。预测一个词的概率是通过从根节点到表示该词的叶节点遍历树来计算的。这用大约 $\log_2(V)$ 次计算代替了昂贵的 $V$ 维softmax计算。

虽然了解这些优化方法的细节很有帮助，但像Python中的`gensim`这样的库会处理它们的实现，让你能够专注于Word2Vec的应用。

归根结底，CBOW和Skip-gram都采用了简单神经网络 (neural network)结构，并针对巧妙设计的预测任务进行训练。真正的产出并非网络本身的预测能力，而是在此过程中学到的内部表示——即词嵌入 (embedding)。这些嵌入将词映射到一个向量 (vector)空间中，其中距离和方向与语义和句法关系相对应。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; Pages: 1-12; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了用于学习词嵌入的原始连续词袋（CBOW）和Skip-gram架构。
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf) — Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, Jeffrey Dean (2013)
  Journal: Advances in Neural Information Processing Systems 26; Publisher: Curran Associates; Pages: 3111-3119; DOI: [10.48550/arXiv.1310.4546](https://doi.org/10.48550/arXiv.1310.4546)
  介绍了负采样和层次Softmax，它们是高效Word2Vec训练的优化技术。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  在更广泛的自然语言处理背景下，提供了对词嵌入（包括Word2Vec）的全面解释（第6章）。
- [gensim.models.word2vec.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) — Gensim Development Team (2024)
  Gensim库中Word2Vec实现的官方API文档和使用示例。

---

[上一节](03-%E8%AF%8D%E5%B5%8C%E5%85%A5%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md) · [下一节](05-GloVe%EF%BC%9A%E8%AF%8D%E8%AF%AD%E7%9A%84%E5%85%A8%E5%B1%80%E5%90%91%E9%87%8F%E8%A1%A8%E7%A4%BA.md)
