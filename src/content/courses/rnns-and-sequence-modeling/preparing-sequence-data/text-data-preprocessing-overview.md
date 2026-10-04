---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "text-data-preprocessing-overview"
sourceId: 2612
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/text-data-preprocessing-overview"
title: "文本数据预处理概述"
description: "概述为自然语言处理任务清洗和准备文本数据的常见步骤。"
order: 1
plots: []
sourceHash: "94bc2a1af97452703b1a13e037974a6946a71ec5c262233f971f57b1d2d22896"
sourceCorrections: []
---

循环神经网络 (neural network) (RNN)（与大多数机器学习 (machine learning)模型一样）需要数值输入。原始文本由字符、单词和标点符号组成，与这些网络所需的结构化张量相去甚远。因此，处理文本的一个重要部分是将其转换为合适的数值格式。此处概述了为LSTM和GRU等序列模型准备文本数据常用的流程。

总目标是将符号序列（单词、字符）转换为数值序列，通常表示为多维数组或张量。可以将其视为将人类语言翻译成机器可读的格式，同时保持原始文本的序列特性。此过程通常包含几个阶段：

"1. **清洗（可选但建议）：** 文本常包含噪声，例如HTML标签、特殊字符或无关标点。初步的清洗步骤可以删除或标准化这些元素，简化后续处理。"
2. **分词 (tokenization)：** 这是将原始文本分解成称为“词元 (token)”的更小单元的过程。这些词元通常是单词，但它们也可以是子词 (subword)或单个字符，具体取决于任务和所需的粒度。例如，句子“RNNs process sequences.”可能会被分词为 `["RNNs", "process", "sequences", "."]`。
3. **构建词表：** 分词后，我们创建一个词表，它是训练数据中所有词元的唯一集合。每个唯一词元都被分配一个特定的整数索引。这会创建一个映射，例如 `{"RNNs": 1, "process": 2, "sequences": 3, ".": 4, ...}`。通常会添加特殊词元，例如用于未知词的 `<UNK>` 或用于填充的 `<PAD>`。
4. **整数编码：** 使用词表映射，每个词元序列都被转换为相应的整数序列。我们的示例句子变为 `[1, 2, 3, 4]`。这是从文本中获得的基本数值表示。
5. **处理变长序列（填充/截断）：** 循环神经网络通常以批次形式训练以提高效率。然而，一个批次中的序列通常需要具有相同的长度。由于实际文本的长度各异，我们应用填充（添加特殊 `<PAD>` 词元，通常用0表示）或截断（删除词元）来标准化批次内所有整数序列的长度。掩码是一种与填充结合使用的技术，用于向模型指示序列的哪些部分是实际数据，哪些只是填充，从而确保填充不会不适当地影响模型的学习过程。

这些步骤的输出通常是整数序列的批次，准备好进入下一阶段。虽然这些整数序列有时可以直接输入到循环神经网络中，但更常见的是，尤其是在自然语言处理（NLP）中，首先将它们通过*嵌入 (embedding)层*。该层通常作为主模型的一部分进行训练，将每个整数索引转换为一个密集的低维向量 (vector)（即一个嵌入）。这些向量能够捕捉词元之间的语义相似性（例如，'king' 和 'queen' 可能具有相似的向量），并提供比原始整数或稀疏独热编码更丰富、更有效的表示。本章后面我们将更详细地讨论嵌入层。

下图展示了这种通用的文本预处理流程：

> 一个将原始文本转换为适合循环神经网络输入的数值张量的典型流程。图中虚线所示的嵌入层步骤，通常作为模型本身的一部分而非独立的预处理阶段。

本章后续部分将详细阐述每个重要的步骤，提供使用常用库实现它们的实践指导和代码示例。掌握这些技术对于将循环神经网络有效地应用于任何基于文本的任务来说是不可或缺的。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本综合性教科书，涵盖自然语言处理的基础概念，包括分词、文本规范化以及多种词表示形式。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的基础教科书，为神经网络、循环神经网络等序列模型以及这些模型所需数值数据表示的普遍必要性提供了背景。
- [Natural Language Processing with Transformers: Building Innovative Applications with NLP](https://www.oreilly.com/library/view/natural-language-processing/9781098103231/) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一本关于使用深度学习进行现代自然语言处理的实用指南，详细介绍了当前的文本预处理技术、分词策略（包括子词）以及为序列模型创建模型输入张量。
