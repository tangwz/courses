---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "introduction-embedding-layers"
sourceId: 2616
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/introduction-embedding-layers"
title: "嵌入层介绍"
description: "说明嵌入层将整数编码词元表示为密集向量的用途。"
order: 4
plots: []
sourceHash: "79221166f878cd5e3b7190b15a7797aabb82dbf79017b0f17cb93964af1d6095"
sourceCorrections: []
---

将文本表示为任意整数序列，例如将“cat”表示为5、“dog”表示为12、“house”表示为87，给机器学习 (machine learning)模型带来了挑战。虽然这提供了一种数字格式，但这些整数本身缺乏固有的含义或语义关联 (semantic relationship)。例如，就含义而言，整数12本质上并不比87“更接近”5，尽管“cat”和“dog”是语义上相关的动物。此外，将大而随意的整数表示直接输入神经网络 (neural network)通常效果不佳。

另一种方法是独热编码（将“cat”表示为 `[0,0,0,0,1,0,...]`，向量 (vector)大小与词汇表 (vocabulary)相同），对于实际规模的词汇表（数万个词），这种方法很快就会变得计算上不可行且数据稀疏。我们需要一种更高效、更有意义的数字方式来表示词语。

这正是**嵌入 (embedding)层**的作用所在。其核心思想是，将词汇表中的每个独立词元 (token)（词语、字符等）表示为浮点数的*密集*、相对*低维向量*，而非单个整数或稀疏向量。这些被称为**词嵌入**或**词元嵌入**。

可以把嵌入层看作一个可训练的查找表。

1. **初始化**：定义嵌入层时，您需要指定词汇表的大小（例如，10000个独立词语）以及嵌入向量的所需维度（例如，128个维度）。然后，该层会初始化一个形状为 `(词汇表大小, 嵌入维度)` 的权重 (weight)矩阵。该矩阵中的每行 $i$ 都对应于整数索引为 $i$ 的词元的初始嵌入向量。
2. **查找**：在训练或推断期间，您向层输入整数索引序列。对于输入序列中的每个整数，该层会简单地在其内部权重矩阵中查找对应的行（向量）。
3. **输出**：该层用其对应的密集向量 (dense vector)替换每个整数索引。如果输入是形状为 `(批次大小, 序列长度)` 的序列批次，则输出将是形状为 `(批次大小, 序列长度, 嵌入维度)` 的张量。

> 整数索引用于在嵌入层的内部权重矩阵中查找对应的密集向量表示。

### 为何使用嵌入 (embedding)？

嵌入层真正的作用在于，嵌入向量 (vector)（查找表中的行）是**可训练参数 (parameter)**。它们通常随机初始化，然后通过反向传播 (backpropagation)在网络训练过程中进行调整，就像任何其他权重 (weight)一样。

随着网络学习执行其任务（例如，情感分类、语言建模），它会隐式学习在 `嵌入维度` 空间中排列嵌入向量，从而：

1. **语义相似性**：含义相似或出现在相似上下文 (context)中的词元 (token)，它们的向量会倾向于在嵌入空间中靠得更近。例如，“cat”和“dog”的向量可能最终比“cat”和“airplane”的向量更接近。
2. **关系**：向量之间的空间关系可以捕获类比。经典的例子是，向量关系 `vector("king") - vector("man") + vector("woman")` 通常会产生一个非常接近 `vector("queen")` 的向量。
3. **降维**：我们使用维度低得多的密集向量 (dense vector)（通常为50到300），而非维度等于词汇表 (vocabulary)大小（通常 > 10,000）的稀疏独热向量，这使得后续计算更高效。

这种学习到的密集表示，与原始整数索引或独热向量相比，为网络的后续层（如RNN、LSTM或GRU）提供了更丰富、更有用的输入信号。输出形状 `(批次大小, 序列长度, 嵌入维度)` 正是循环层所期望的输入，其中最后一个维度表示序列中每个时间步的每个元素的特征。

实际中，您很少需要自己实现查找机制。像TensorFlow和PyTorch这样的深度学习 (deep learning)框架提供了内置的 `Embedding` 层，它们高效地处理初始化、查找和训练。您只需将此层作为模型中的第一步添加，置于接收整数编码序列的输入层之后。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  Journal: arXiv preprint arXiv:1301.3781
  这篇论文介绍了Word2Vec模型，该模型在学习捕获语义和句法关系的密集词嵌入方面取得了重大进展。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532–1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  介绍了GloVe，这是通过分解全局词-词共现矩阵生成词嵌入的另一项有影响力的模型，为Word2Vec提供了另一种视角。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖了深度学习的基本概念，包括神经网络架构、表示学习和词嵌入的详细解释。
- [tf.keras.layers.Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) — TensorFlow Developers (2024)
  TensorFlow Keras Embedding层的官方文档，描述了其使用方法、参数以及作为可训练的token嵌入查找表的工作原理。
