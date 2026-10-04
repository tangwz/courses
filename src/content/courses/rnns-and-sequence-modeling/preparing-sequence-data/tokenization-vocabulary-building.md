---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "tokenization-vocabulary-building"
sourceId: 2613
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/tokenization-vocabulary-building"
title: "分词与词汇表构建"
description: "说明如何将文本分解为标记，并创建从标记到整数的映射。"
order: 2
plots: []
sourceHash: "5f8b92c7ebb29097e3729b356dc928cee580f0b940a7b546f40e51e80d7659ba"
sourceCorrections: []
---

循环模型处理数值数据，而文本以字符串形式出现。为序列模型处理文本的最初且基础的步骤是将其分解为可管理单元，称为标记 (token)（tokens），然后建立一种统一的方式来数值表示这些标记。此过程包含**分词 (tokenization)**和**词汇表 (vocabulary)构建**。

### 什么是分词 (tokenization)？

分词是将一段文本分割成更小单元（即标记 (token)）的过程。可以把它想象成把一个句子切分成单词或字符。分词策略的选择会显著影响模型“看”待文本的方式，并可能影响性能。

常见策略有：

1. **词语分词：** 这可能是最直接的方法。文本被分成词语，通常基于空格和标点符号。例如，句子“RNNs process sequences.”可能会被分词为 `["RNNs", "process", "sequences", "."]`。需要考虑的方面包括如何处理标点符号（保留、丢弃、分割？）、大小写敏感性（全部转换为小写？）以及缩写（将“don't”分割为“do”和“n't”？）。对于许多自然语言处理任务，词语级别分词在序列长度和意义表示之间提供了良好的平衡。
2. **字符分词：** 在这种方式下，文本被分割成单个字符。同样的句子“RNNs process sequences.”会变成 `['R', 'N', 'N', 's', ' ', 'p', 'r', 'o', 'c', 'e', 's', 's', ' ', 's', 'e', 'q', 'u', 'e', 'n', 'c', 'e', 's', '.']`。这会得到一个小得多的词汇表 (vocabulary)（仅包含字母、数字、标点符号、空格），并能自然地处理拼写错误或稀有词。然而，它会导致更长的序列，可能使像简单RNN这样的模型更难捕获长距离依赖。
3. **子词 (subword)分词：** 这是一种越来越受欢迎的混合方法，被许多现代自然语言处理模型（如BERT、GPT）采用。像字节对编码（BPE）或WordPiece这样的算法将词语分解成更小、有意义的子单元。例如，“tokenization”可能变为 `["token", "ization"]`。这使得模型能够通过已知子词组合来处理稀有词，保持词汇表大小易于管理，并避免了与字符标记相比过长的序列。尽管功能强大，但实现这些通常需要专门的库。

对于大多数RNN的基础工作，词语分词是一个常见的起始点。

### 构建词汇表 (vocabulary)

分词 (tokenization)你的文本语料库（你的整个训练数据集）后，下一步是构建一个**词汇表**。词汇表是训练数据中遇到的所有独特标记 (token)的明确集合。然后我们创建一个映射，通常是一个字典或哈希表，为每个独特标记分配一个唯一的整数索引。

为什么要用整数？神经网络 (neural network)执行数学运算，它们需要数值输入。此映射将我们的符号标记转换为网络可以理解的数字。

考虑这个小型数据集：

- 句子 1: “the cat sat”
- 句子 2: “the dog sat”

1. **分词：**

   - 句子 1: `["the", "cat", "sat"]`
   - 句子 2: `["the", "dog", "sat"]`
2. **收集独特标记：** 独特标记是 `{"the", "cat", "sat", "dog"}`。
3. **构建词汇表映射：** 为每个独特标记分配一个整数ID。常见的做法是将索引0保留给填充（稍后讨论），有时将索引1保留给未知词。

   ```
   词汇表:
   {
     "<PAD>": 0,  // 为填充保留
     "<UNK>": 1,  // 为未知词保留
     "the": 2,
     "cat": 3,
     "sat": 4,
     "dog": 5
   }
   ```

我们通常在词汇表中包含特殊标记：

- **`<PAD>` (填充标记)：** 用于使批次中的序列具有相同长度。我们将在“填充序列”部分讨论此内容。
  "\* **`<UNK>` 或 `<OOV>` (未知/词汇表外标记)：** 表示在测试或推断期间遇到但*未*出现在训练数据词汇表中的标记。这很重要，因为数据通常包含模型以前未见过的词语。将它们映射到一个专门的`<UNK>`标记，使模型能够处理它们，尽管没有关于该确切词语的特定学习知识。"
- **`<SOS>` / `<EOS>` (序列开始/结束标记)：** 有时使用，尤其是在序列生成任务中，用于指示序列的开始或结束。

词汇表的大小是一个设计选择。更大的词汇表能捕获更多词语，但会增加模型的嵌入 (embedding)层大小（需要学习更多参数 (parameter)）。通常，非常不常用词语（在大型语料库中只出现一两次）会被映射到`<UNK>`，以保持词汇表大小合理并可能提高泛化能力。

> 一个简化的图示，说明将标记（包括`<PAD>`和`<UNK>`等特殊标记）映射到词汇表中唯一整数ID的过程。

分词和词汇表创建是将原始文本转换为适合下一阶段格式的基础步骤：利用这个词汇表映射将标记序列转换为整数序列，此过程称为整数编码。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson Prentice Hall; Pages: Chapter 2
  一本经典的自然语言处理教材，详细解释了分词策略和词汇表构建等基础概念。
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Rico Sennrich, Barry Haddow, and Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 1715-1725; DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)
  介绍了在神经机器翻译中处理罕见词的字节对编码（BPE），是子词分词领域的开创性工作。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Volume: 1; Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  描述了BERT模型中使用的WordPiece分词策略，展示了子词分词在现代自然语言处理中的实际应用。
- [CS224n: Natural Language Processing with Deep Learning, Course Materials](http://web.stanford.edu/class/cs224n/) — Stanford University (2023)
  来自顶尖大学的课程材料，深入理解如何为深度学习模型准备文本数据，包括分词和词汇表构建。
