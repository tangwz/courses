# 第 4 章：嵌入表示与序列数据理解

来源：[原章节](https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings)

[返回课程目录](../README.md)

前几章介绍了基于词频的文本预处理和特征表示方法，例如词袋模型（Bag-of-Words）和TF-IDF。这些技术在某些任务中有效，但难以捕捉词语的语义或语境。例如，如果“cat”和“feline”在训练数据中不经常出现在相同的语境下，这些方法会把它们视为完全不相关的词。

本章将重点转向基于词语周围语境来表示词语的方法，从而得到能够编码语义相似性的表示。我们将首先讨论基于词频方法的局限性，并介绍分布语义的理念，即出现在相似语境中的词语往往具有相似的含义。

接下来，我们将研究词嵌入：词语学习到的密集向量表示，例如 $w \in \mathbb{R}^n$。我们将了解流行的算法，如Word2Vec（包括其CBOW和Skip-gram变体）和GloVe（全局词向量表示）。最后，我们将介绍这些嵌入的可视化技术，并学习如何使用现成的预训练嵌入模型，以便将其用于其他自然语言处理任务。

## 小节

- 1. [基于频率的模型的局限性](01-%E5%9F%BA%E4%BA%8E%E9%A2%91%E7%8E%87%E7%9A%84%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [分布式语义学简介](02-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AF%AD%E4%B9%89%E5%AD%A6%E7%AE%80%E4%BB%8B.md)
- 3. [词嵌入基本原理](03-%E8%AF%8D%E5%B5%8C%E5%85%A5%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 4. [Word2Vec：CBOW 与 Skip-gram 模型结构](04-Word2Vec%EF%BC%9ACBOW%20%E4%B8%8E%20Skip-gram%20%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.md)
- 5. [GloVe：词语的全局向量表示](05-GloVe%EF%BC%9A%E8%AF%8D%E8%AF%AD%E7%9A%84%E5%85%A8%E5%B1%80%E5%90%91%E9%87%8F%E8%A1%A8%E7%A4%BA.md)
- 6. [词嵌入的可视化](06-%E8%AF%8D%E5%B5%8C%E5%85%A5%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 7. [使用预训练词嵌入模型](07-%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E8%AF%8D%E5%B5%8C%E5%85%A5%E6%A8%A1%E5%9E%8B.md)
- 8. [动手实践：使用词嵌入](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%E8%AF%8D%E5%B5%8C%E5%85%A5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/nlp-fundamentals/chapter-4-nlp-word-embeddings/quiz)
