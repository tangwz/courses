---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "integer-encoding-sequences"
sourceId: 2615
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/integer-encoding-sequences"
title: "整数编码序列"
description: "根据词汇表将分词序列转换为整数序列。"
order: 3
plots: []
sourceHash: "f176fb27093ab187c62a380a604d8bfe9e584264264ae34a12dc97e52e6b1571"
sourceCorrections: []
---

包括RNN在内的机器学习 (machine learning)模型需要处理数值数据，而非字符串。为了将文本转换为这些模型可用的格式，首先需要将文本分解为分词 (tokenization)并建立一个一致的词汇表 (vocabulary)。整数编码是将分词序列转换为数字序列的过程，它通过用词汇表中对应的唯一整数ID替换序列中的每个分词来实现。

### 映射过程

您可以将上一步中建立的词汇表 (vocabulary)视为查找表或字典。每个独特的分词 (tokenization)（单词、子词 (subword)或字符）都是一个键，其分配的整数是对应的值。

例如，假设我们处理了句子“the quick brown fox”并构建了以下简单词汇表：

```python
vocabulary = {'<UNK>': 0, '<PAD>': 1, 'the': 2, 'quick': 3, 'brown': 4, 'fox': 5, 'jumps': 6}
```

这里，`<UNK>`代表未知分词（在词汇表创建过程中未出现的分词），而`<PAD>`代表填充分词，我们稍后将讨论它们如何处理可变序列长度。

分词后的序列是：

```
['the', 'quick', 'brown', 'fox']
```

为了执行整数编码，我们遍历这个分词序列，并将每个分词替换为它在`vocabulary`中的整数ID：

1. 'the' 映射到 `2`
2. 'quick' 映射到 `3`
3. 'brown' 映射到 `4`
4. 'fox' 映射到 `5`

得到的整数编码序列是：

```
[2, 3, 4, 5]
```

这个数字序列保留了分词的原始顺序，这对于序列模型来说是根本。

> 使用词汇表将分词映射到整数ID。

这种转换系统地应用于数据集中的每个序列，确保所有文本输入都被转换为模型可以处理的数值格式。

### 实现示例

在Python中，这种映射通常通过字典查找来完成。给定一个分词 (tokenization)序列列表和一个词汇表 (vocabulary)字典，您可以像这样进行转换：

```python
# 示例词汇表（已构建）
vocabulary = {'<UNK>': 0, '<PAD>': 1, 'the': 2, 'quick': 3, 'brown': 4, 'fox': 5, 'jumps': 6, 'over': 7, 'lazy': 8, 'dog': 9}

# 示例分词句子
token_sequences = [
    ['the', 'quick', 'brown', 'fox'],
    ['the', 'lazy', 'dog'],
    ['fox', 'jumps', 'over', 'dog']
]

# 执行整数编码
encoded_sequences = []
for seq in token_sequences:
    encoded_seq = [vocabulary.get(token, vocabulary['<UNK>']) for token in seq]
    encoded_sequences.append(encoded_seq)

# 打印结果
# encoded_sequences 将会是：
# [[2, 3, 4, 5], [2, 8, 9], [5, 6, 7, 9]]
print(encoded_sequences)
```

请注意`vocabulary.get(token, vocabulary['<UNK>'])`的用法。这是一种常见模式：它会尝试在词汇表中查找分词。如果分词存在，则返回其ID。如果未找到分词（即“词汇表外”或OOV分词），它会默认返回分配给特殊`<UNK>`分词的ID（在此示例中为`0`）。这确保了在测试或部署过程中遇到新词时，您的过程不会崩溃，尽管模型处理许多未知词的能力取决于其训练情况。

### 为什么使用整数编码？

这一步非常重要，因为数学模型不能直接处理字符串。通过将分词 (tokenization)转换为整数，我们创建了可输入到神经网络 (neural network)后续层的数值表示。

需要理解的是，这些整数ID（$2$、$3$、$4$、$5$等）是任意的标识符。模型最初并不知道整数`3`（'quick'）在语义上比`9`（'dog'）更接近`5`（'fox'）。ID本身的数值不带有词语之间的关系含义。这个局限性将在流水线的下一步得到处理：嵌入 (embedding)层。

这些整数序列构成了通常会传递给嵌入层的输入，我们将在接下来介绍嵌入层。嵌入层将在模型训练期间为每个整数ID学习密集向量 (vector) (dense vector)表示，捕获分词之间的语义关系。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson; Pages: Chapter 2: Words and Tokens
  一本全面的教科书，介绍了文本处理的基本概念，包括分词、词汇表创建以及将文本转换为计算模型所需的数值表示。
- [CS224n: Natural Language Processing with Deep Learning, Lecture Notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) — Stanford CS224n Teaching Team (2024)
  Publisher: Stanford University; Pages: Lecture 1 & 2: Introduction, Word Vectors and Word Senses
  提供关于为深度学习模型准备文本数据的教育内容，详细介绍了分词、词汇表构建和整数编码等步骤，作为嵌入层的输入基础。
- [Keras preprocessing layers: Text preprocessing](https://www.tensorflow.org/guide/keras/preprocessing_layers#text_preprocessing) — Francois Chollet, Mark Omernick (2024)
  Publisher: Google
  官方文档，展示了 Keras 中文本预处理步骤的实际实现，包括如何使用 `TextVectorization` 等层将原始文本转换为数值序列作为模型输入。
