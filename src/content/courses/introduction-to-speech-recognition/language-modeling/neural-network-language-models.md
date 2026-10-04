---
course: "introduction-to-speech-recognition"
chapter: "language-modeling"
lesson: "neural-network-language-models"
sourceId: 7123
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-4-language-modeling/neural-network-language-models"
title: "神经网络语言模型简介"
description: "简要介绍基于神经网络的现代语言模型，这类模型能识别更复杂的语言模式。"
order: 7
plots: []
sourceHash: "30291b828fa094c4eb6b1ff61bdf8a86cb92bae547ac6c06d869df3e707c518e"
sourceCorrections: []
---

N-gram模型是理解语言概率的常用方法。尽管它们在理解语言概率中扮演着基础角色，但这些模型有一个重要限制：它们难以处理在训练中未曾出现的词语序列。例如，如果三元组“wreck a nice”从未在文本数据中出现过，N-gram模型会把“beach”这个词紧随其后的概率设为零。这个问题，称为数据稀疏，意味着N-gram模型的泛化能力有限。

而神经网络 (neural network)为语言建模提供了更有效、更灵活的方式。神经网络不再仅仅统计词语同时出现的情况，而是学会用它们的语境和含义来表示词语。

### 从词到向量 (vector)

神经网络 (neural network)语言模型的第一步是，不再把词语看作是简单、独立的文本标签。相反，每个词被映射成一串密集的数字，称为**词嵌入 (embedding)**或**词向量**。

这些向量的一个重要特点是，含义相近或在类似语境中使用的词，会有相似的向量。例如，“nice”、“good”和“lovely”的向量在向量空间中会彼此数值上接近。仅此一项变化就让模型比N-gram模型更好地泛化。如果模型学会了短语“a good day”，它能推断出“a nice day”也是一个很可能出现的短语，因为“good”和“nice”的向量很相似。

> N-gram模型和神经网络方法的对比。N-gram模型依赖直接查找，而神经网络处理词语的数值表示以理解语境。

### 处理更长的依赖关系

N-gram模型的另一个限制是其固定的短时记忆。例如，一个三元模型仅考虑前两个词。它无法获取句中更早出现的词语信息。

为处理序列设计的神经网络 (neural network)结构，例如**循环神经网络 (RNN)（RNNs）**，应对了这个问题。RNN一次处理一个词，并保持一个内部状态，或者说“记忆”，它随着每个新词而更新。这个状态允许模型保存句子开头的信息，并用它来做出更好的预测。例如，在句子“My friends from Germany, who I haven't seen in years, are finally coming to visit. I can't wait to speak... ”中，RNN比N-gram模型更有可能预测“German”这个词，因为它能记住很早之前“Germany”的语境。

### 语音识别的当前标准

总而言之，神经网络 (neural network)语言模型与传统N-gram模型相比有两大优点：

1. **更强的泛化能力：** 通过使用词嵌入 (embedding)，它们能有效处理新的或不常见的词语组合。
2. **更长的语境：** 像RNN这样的架构能处理整个句子的依赖关系，而不仅仅是固定的一个小范围。

由于这些出色的性能，基于神经网络的语言模型现在几乎是所有现代语音识别系统的标准配置。LSTMs（长短期记忆网络 (LSTM)）和Transformers等模型的具体细节是更高级课程的内容，但你的N-gram知识为理解这些更复杂的模型为何如此高效提供了很好的铺垫。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University (Online Draft)
  一本综合性教科书，涵盖传统N-gram模型、神经语言模型、词嵌入以及RNN和Transformer等序列模型。
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin (2003)
  Journal: Journal of Machine Learning Research; Volume: 3; Pages: 1137-1155
  一篇基础论文，介绍了最早的学习词嵌入和基于神经网络的语言模型之一。
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781; DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec模型（Skip-gram和CBOW），用于高效创建高质量词嵌入。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008
  介绍了Transformer架构，该架构彻底改变了序列建模，并成为现代大型语言模型的基础。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Christopher Manning and Abigail See (2023)
  Publisher: Stanford University
  一门高级大学课程，提供涵盖词嵌入、循环神经网络和现代Transformer模型的讲座视频和笔记。
