---
course: "intro-to-multimodal-ai"
chapter: "data-foundations-multimodal-systems"
lesson: "text-data-representation"
sourceId: 6507
sourceUrl: "https://apxml.com/zh/courses/intro-to-multimodal-ai/chapter-2-data-foundations-multimodal-systems/text-data-representation"
title: "文本数据表示：从字符到意义"
description: "理解文本数据如何转换为AI模型可处理的格式，从词语到数值向量。"
order: 1
plots: ["plots/6507-0.json"]
sourceHash: "036a84b8c6c84a0152049e35640515493749addaeb073b6ef780bd472c88bd83"
sourceCorrections: []
---

AI系统需要数据以数值形式存在才能执行任务。图像可以表示为像素值的网格，而文本，本质是字符序列，也需要类似的转换。原始文本（由字母、标点符号和空格组成）如何转换为AI模型能够理解和学习的数值结构，此处将详细阐述。这个过程是让机器从书面语言中获取意义的第一步。

### 从文本到标记 (token)：拆分细化

处理文本的第一步通常是**标记化**。可以把标记化看作是将一个长句或段落分解成更小、易于处理的部分，称为“标记”。最常见的是，这些标记是单词。例如，句子“AI is fun!”可以被标记化为：

- “AI”
- “is”
- “fun”
- “!”

像“!”这样的标点符号通常被视为单独的标记，因为它们可以带有意义（例如，表示兴奋或句子的结束）。有时，文本会根据AI任务的具体要求被标记化为子词 (subword)单元甚至单个字符。对于初学者来说，词级别的标记化是一个很好的起始点。

> 句子标记化为单词和标点符号的简单图示。

### 构建词汇表 (vocabulary)：AI的字典

有了一系列标记 (token)后，下一步是构建一个**词汇表**。词汇表本质上是我们文本数据中所有唯一标记的列表。词汇表中的每个唯一标记都会被分配一个唯一的数值标识符（整数ID）。这就像创建一本字典，其中每个单词（标记）都有一个特定的页码（ID）。

例如，如果我们的整个数据集只包含句子“AI is fun”和“AI is great”，那么我们的标记将是“AI”、“is”、“fun”、“great”。
我们的词汇表可能如下所示：

- “AI”: 0
- “is”: 1
- “fun”: 2
- “great”: 3

所以，句子“AI is fun”将以数值序列 `[0, 1, 2]` 表示。

> 示例展示了句子中的标记如何根据定义的词汇表映射到数值ID。

这种数值序列是表示文本的一种基本方式，但AI模型通常从更结构化的向量 (vector)表示中受益。

### 标记 (token)向量 (vector)化：赋予数值更多结构

虽然ID序列是一个开始，但我们经常将这些标记或标记序列转换为向量。向量就是一串数字。有几种常用的方法可以做到这一点。

#### 独热编码

独热编码（OHE）是一种直接的方法。对于每个标记，我们创建一个与整个词汇表 (vocabulary)长度相同的向量。这个向量除对应词汇表中标记ID的索引位置为“1”外，其余位置都填充“0”。

假设我们的词汇表是 `{"the": 0, "cat": 1, "sat": 2}`。

- 标记“cat”（ID: 1）将被表示为向量 `[0, 1, 0]`。
- 标记“the”（ID: 0）将是 `[1, 0, 0]`。

> 标记“cat”在小型词汇表基础上表示为独热编码向量。数值“1”出现在分配给“cat”的索引位置。

虽然简单，但独热编码存在缺点：

1. **高维度：** 如果你的词汇表有20,000个词，每个词的向量将有20,000个维度，其中大部分是零。这计算成本较高。
2. **稀疏性：** 向量是“稀疏的”（大部分为零）。
3. **无语义关系：** “cat”和“dog”的独热编码向量之间的差异与“cat”和“car”之间的差异相同。这种表示未能体现“cat”和“dog”在语义上更相似（两者都是动物）。

#### 词袋模型 (BoW)

词袋模型（BoW）是另一种流行的将整段文本（如句子或文档）表示为单个向量的方式。这个向量的每个词汇表中的词都有一个对应项。每个对应项可以是：

- 一个二进制值（0或1），表示词语的存在与否。
- 每个词语在文本中出现的次数。

例如，对于词汇表 `{"the", "cat", "sat", "on", "mat"}`：

- 句子“the cat sat”将是 `[1, 1, 1, 0, 0]`（如果使用计数）。
- 句子“the cat the cat”将是 `[2, 2, 0, 0, 0]`。

> 使用词语计数表示两个句子的词袋模型。每行代表一个句子的向量。

词袋模型的主要局限在于它丢失了所有关于词序的信息。如果句子“The cat sat on the mat”和“The mat sat on the cat”使用了相同的词语，它们的词袋表示会非常相似，即使它们的意义不同。

词袋模型的一个常见改进是 **TF-IDF（词频-逆文档频率）**。TF-IDF 不仅仅计算词频；它还考虑了一个词在文档集合中对于某个文档的重要性。它会给在特定文档中出现频繁但在所有文档中不常见的词赋予更高的权重 (weight)。对于初学者课程，理解TF-IDF是词袋模型中一种更高级的词语评分方式就足够了。

### 迈向意义：词嵌入 (embedding)简介

目前讨论的方法（独热编码、词袋模型）很有用，但它们本身不能捕获词语间的*意义*或语义关系。这就是**词嵌入**发挥作用的地方。

词嵌入将词语表示为密集的低维向量 (vector)（与独热编码的稀疏高维向量不同）。词嵌入的独特之处在于，意义相似的词语倾向于具有相似的向量表示。这些向量通常从大量文本数据中学习得到，该过程旨在将词语置于一个“向量空间”中，使其几何关系反映其语义关系。

例如，在一个理想的嵌入空间中：

- “king”的向量可能与“queen”的向量接近。
- “apple”的向量可能与“orange”接近，但远离“car”。
- 一个著名的（尽管简化）例子是，向量运算 `vector("king") - vector("man") + vector("woman")` 可能会得到一个非常接近 `vector("queen")` 的向量。

这些向量是“密集的”，因为它们的大多数值都不是零；而“低维”通常意味着它们有几百个维度，而不是像独热编码那样有数万个维度。



![简化词嵌入空间](plots/6507-0.json)



> 简化2D视图，其中相关词语（如水果或王室成员）在向量空间中彼此靠近。“Car”和“truck”形成另一个聚类。

Word2Vec、GloVe和FastText等词嵌入技术功能强大，因为它们提供了一种让AI模型理解意义和上下文 (context)的方式，这是从纯粹的字符到实际意义的一大进步。目前，重要的是要知道它们的存在，并且它们代表了一种将词语转化为带有语义信息的数值的更先进方式。

通过将文本转换为这些不同的数值格式，我们为AI模型处理和“理解”语言打下了基础。每种表示方法都有其优点和缺点，选择通常取决于具体的任务。这些数值表示是任何处理文本数据的AI系统的基本组成部分。

## 参考资料

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本权威且全面的自然语言处理教材，涵盖了文本预处理、分词和各种词表示方法。
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean (2013)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec的开创性论文，这是一个学习捕获语义相似性的密集词嵌入模型，影响深远。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  介绍了GloVe，一种基于全局共现统计学习词嵌入的无监督模型，为Word2Vec提供了另一种选择。
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  一本关于信息检索的经典教材，详细介绍了词袋模型和TF-IDF等基本文本表示模型。
