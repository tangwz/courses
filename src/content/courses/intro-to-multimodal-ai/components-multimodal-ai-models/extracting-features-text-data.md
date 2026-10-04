---
course: "intro-to-multimodal-ai"
chapter: "components-multimodal-ai-models"
lesson: "extracting-features-text-data"
sourceId: 6524
sourceUrl: "https://apxml.com/zh/courses/intro-to-multimodal-ai/chapter-4-components-multimodal-ai-models/extracting-features-text-data"
title: "从文本数据中提取特征"
description: "学习为AI模型从文本数据中提取有意义特征的基本方法。"
order: 1
plots: []
sourceHash: "bdd44e1b0b8d7895c901704021f7cf6c3fe3645a39dfc35365bd2f6ea953575f"
sourceCorrections: []
---

为了有效构建多模态 (multimodal)AI模型，将每种模态的原始数据转换为算法能够理解和处理的格式是必不可少的。对于文本，这意味着将字符、词语和句子的字符串转换为数值表示，常被称为特征或嵌入 (embedding)。原始文本本身不能被大多数机器学习 (machine learning)模型直接使用，因为它们期望数值输入。从文本数据中获取这些有意义特征的常用方法将被论述。

### 数值表示的必要性

设想一下，如果想教计算机理解一个句子，例如“A fluffy cat sleeps on a warm rug.”（一只毛茸茸的猫睡在温暖的地毯上）。计算机本身并不知道“fluffy”、“cat”或“sleeps”是什么意思。它只看到一串字符。特征提取是将此文本转换为一组数字（一个向量 (vector)）的过程，该向量捕捉其意义或结构的某些方面。然后，这些数值向量可以输入到机器学习 (machine learning)模型中，模型会从中学习模式。在多模态 (multimodal)系统中，这些文本特征稍后将与从图像、音频或其他数据类型中提取的特征结合。

### 一个简单的起点：词袋模型（BoW）

以数值方式表示文本最直接的方式之一是词袋（BoW）模型。可以这样理解：你把文档中的所有词都扔进一个“袋子”里，然后计算每个词出现的次数。词语的顺序和语法都被忽略；只有出现和频率才重要。

它通常是这样工作的：

1. **创建词汇表 (vocabulary)：** 首先，你收集所有文本文档（你的语料库），并整理所有出现的唯一词列表。这个列表就是你的词汇表。
2. **文档向量 (vector)化：** 对于每个文档，你创建一个数值向量。这个向量的长度等于你的词汇表大小。向量中的每个位置对应词汇表中的一个唯一词。
3. **计算频率：** 向量中每个位置的值通常是对应词在特定文档中出现的次数。

来看一个简单例子：

- 文档1：“The cat sat on the mat.”
- 文档2：“The dog chased the cat.”

我们的词汇表（为简单起见，忽略大小写和标点，并使用唯一词）可能是：{"the", "cat", "sat", "on", "mat", "dog", "chased"}。

- 文档1的BoW向量：

  - “the”：2，“cat”：1，“sat”：1，“on”：1，“mat”：1，“dog”：0，“chased”：0
  - 因此，向量可以是`[2, 1, 1, 1, 1, 0, 0]`
- 文档2的BoW向量：

  - “the”：2，“cat”：1，“sat”：0，“on”：0，“mat”：0，“dog”：1，“chased”：1
  - 因此，向量可以是`[2, 1, 0, 0, 0, 1, 1]`

**BoW的优点：**

- **简单：** 易于理解和实现。
- **对某些任务有效：** 对于文档分类等任务，某些关键词的出现是一个强有力的标志，此时BoW可以发挥出乎意料的效果。

**BoW的缺点：**

- **丢失词序：** “Man bites dog”和“Dog bites man”会有非常相似的BoW表示（如果不是完全相同，取决于词汇表的具体构建方式），尽管它们的意义是相反的。
- **丢失语境：** 词语的意义会根据周围的词而变化（例如，河流的“bank”与银行的“bank”）。BoW无法捕捉到这一点。
- **稀疏性：** 对于大型词汇表，大多数文档向量将填充零，使其变得稀疏。这会降低计算效率。
- **词汇表大小：** 词汇表可能变得非常大，导致生成维度非常高的向量。

### 增加重要性：词频-逆文档频率（TF-IDF）

词袋模型根据原始计数平等对待所有词。然而，有些词本身信息量比其他词大。例如，“the”、“a”或“is”等词在几乎所有英语文本中都出现得非常频繁，但对文档内容没有多少特定意义。TF-IDF试图通过赋予在特定文档中出现频繁但在整个文档集合（语料库）中出现较少的词更高的权重 (weight)来解决这个问题。

TF-IDF是两个统计量的乘积：词频和逆文档频率。

1. **词频 (TF)：** 这衡量了一个词项 $t$ 在文档 $d$ 中出现的频率。有几种计算TF的方法。一种简单的方法是原始计数，但通常会进行归一化 (normalization)以避免对较长文档的偏向：

   
   $$
   TF(t, d) = \frac{\text{词项 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词项数}}
   $$
   
2. **逆文档频率 (IDF)：** 这衡量了一个词项在整个语料库 $D$ 中的重要程度。它会降低在许多文档中出现的词项的权重，并增加在少数文档中出现的词项的权重。计算IDF的常用方法是：

   
   $$
   IDF(t, D) = \log \left( \frac{\text{文档总数 } |D|}{\text{包含词项 } t \text{ 的文档数}} \right)
   $$
   

   为避免除以零（如果词项未在任何文档中出现，这种情况在词汇表 (vocabulary)根据语料库构建时应该不会发生）或词项在所有文档中出现（导致 $\log(1)=0$），通常会应用平滑处理，例如，通过在分母中加1： $IDF(t, D) = \log \left( \frac{|D|}{1 + \text{包含词项 } t \text{ 的文档数}} \right) + 1$。对于我们初级课程而言，理解基本原理是重要的。

词项 $t$ 在语料库 $D$ 中的文档 $d$ 里的 **TF-IDF 分数** 则为：


$$
TFIDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$


高TF-IDF分数由高词频（该词在特定文档中常见）和该词在整个文档集合中低文档频率（该词整体上罕见）共同达成。这使得TF-IDF能有效突出特定文档的特征词。

例如，在新闻文章集合中，词“election”在一篇专门关于选举的文章中可能具有较高的TF-IDF分数，而词“today”则可能具有较低的IDF分数（因此TF-IDF分数也较低），因为它出现在许多文章中。

**TF-IDF的优点：**

- **减少常见词的影响：** 比BoW更能识别有意义的词。
- **计算简单：** 仍然相对直接。

**TF-IDF的缺点：**

- **仍丢失词序和语境：** 像BoW一样，它不理解语法或词语间的语义关联 (semantic relationship)。
- **稀疏性：** 向量 (vector)仍然可以是稀疏的，尽管现在值已被加权。

### 捕捉意义：词嵌入 (embedding)简介

虽然BoW和TF-IDF提供了有用的数值表示，但它们不捕捉词语之间的*意义*或*语义关联 (semantic relationship)*。例如，“happy”和“joyful”是同义词，但在BoW或TF-IDF模型中，如果它们被视为不同的词汇项，它们的向量 (vector)可能完全不同且不相关。

词嵌入旨在解决这个问题。它们将词表示为密集、低维的向量（例如，50到300维，相比之下，BoW/TF-IDF可能达到数万维）。这些嵌入的主要特点是，意义相似的词在向量空间中由彼此接近的向量表示。

设想一个空间，其中“king”、“queen”、“prince”和“princess”等词位于其中。词嵌入学习到的表示形式使得“king”和“queen”之间的向量关系可能类似于“man”和“woman”之间的向量关系。这常用一个著名例子来说明：vector(“king”) - vector(“man”) + vector(“woman”) \u2248 vector(“queen”)。

**它们与BoW/TF-IDF有何不同？**

- **密集与稀疏：** 嵌入是密集向量 (dense vector)（大多数值非零），而BoW/TF-IDF向量是稀疏的（大多为零）。
- **低维与高维：** 嵌入通常有几百个维度，而BoW/TF-IDF的维度可以等于词汇表 (vocabulary)大小，可能非常大。
- **意义与计数：** 嵌入旨在捕捉语义关联，而BoW/TF-IDF则基于文档层面的词语共现次数。

用于创建词嵌入的流行算法包括：

- **Word2Vec**（由Google开发）
- **GloVe (Global Vectors for Word Representation)**（由斯坦福大学开发）
- **FastText**（由Facebook开发）

一个重要优势是存在**预训练 (pre-training)词嵌入**。研究人员在大量文本数据（如维基百科的所有内容或大型新闻语料库）上训练了这些模型。这意味着你通常可以下载这些预训练嵌入并直接在你的模型中使用它们，而无需从零开始在自己（可能较小）的数据集上进行训练。这很有用，尤其当你的数据集不足以自行学习高质量嵌入时。

对于多模态 (multimodal)系统，这些密集的词（或句子）嵌入向量作为丰富、语义上有依据的输入，送入神经网络 (neural network)的某些部分，这些部分稍后将这些文本信息与图像或音频的特征结合。

### 从词到句子和文档

到目前为止，我们主要谈论了表示单个词。但通常，我们需要表示整个句子或文档。我们如何做到这一点呢？

- 对于BoW或TF-IDF，表示形式本身就已在文档级别。
- 对于词嵌入 (embedding)，你拥有单个词的向量 (vector)。要为句子或文档获取单个向量，常见方法包括：
  - **平均：** 简单地取句子/文档中所有词的嵌入向量的平均值。这是一种简单但通常有效的基线方法。
  - **加权平均：** 对词嵌入进行平均，但给予重要词语更大的权重 (weight)（例如，使用它们的TF-IDF分数作为权重）。
  - **更高级的方法：** 使用神经网络 (neural network)架构，如循环神经网络 (RNN)（RNN，特别是LSTM或GRU）或Transformer。这些模型可以处理词嵌入序列，并学习为整个序列生成单个向量表示，更有效地捕捉词序和语境。对于初学者课程，只需了解它们存在就是一个不错的起点。

### 目标：准备集成

无论选择哪种技术，BoW、TF-IDF还是词嵌入 (embedding)，文本特征提取的目的是一致的：将原始文本转换为一组数值特征。这些特征是AI模型能够理解的构成要素。一旦我们为文本获得了这些数值表示，并且类似地为图像和音频等其他模态（我们接下来会讲到）也获得了数值表示，我们就向在多模态 (multimodal)AI系统中结合它们更近一步。这些提取的特征将作为输入，送入我们在前一章讨论的集成技术和模型架构中。

## 参考资料

- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  详细介绍了词袋模型和TF-IDF等文本表示方法，是文本特征提取的基础。
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec，这是学习高效高质量分布式词表示的开创性工作。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  提出了GloVe，一种无监督的学习算法，用于获取捕获全局语料库统计信息的词向量表示。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本全面的教科书，涵盖了NLP的各个方面，包括经典的文本特征提取和现代的词句嵌入技术（第四版草稿）。
