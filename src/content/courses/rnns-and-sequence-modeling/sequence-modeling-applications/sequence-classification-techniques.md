---
course: "rnns-and-sequence-modeling"
chapter: "sequence-modeling-applications"
lesson: "sequence-classification-techniques"
sourceId: 2635
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-9-sequence-modeling-applications/sequence-classification-techniques"
title: "序列分类技术"
description: "阐述如何调整RNN输出以适应情感分析或主题分类等分类任务。"
order: 3
plots: []
sourceHash: "f21f9a4a46fa83e0bc453e50edeb69b81f2b967f13af1256b2fc1122e6f4ab3c"
sourceCorrections: []
---

序列分类是循环神经网络 (neural network) (RNN)的一种常见且重要的应用。它的目标是为整个输入序列分配一个单一的类别标签。可以设想这样的任务，例如确定电影评论的情感（积极或消极），识别新闻文章的主题，或者对用户查询背后的意图进行分类。

序列分类需要将整个输入序列的信息汇总成一个单一的决定。与预测序列中的下一个元素或生成完整输出序列的任务不同，RNN、LSTM和GRU非常适合此目的，因为它们的隐藏状态充当了已处理序列的不断变化的概括。

### 使用循环层进行分类

基本思想是使用循环层（SimpleRNN、LSTM或GRU）逐步处理输入序列。当网络处理每个元素时，它会更新其隐藏状态，将当前输入和先前状态的信息纳入其中。当网络处理到序列末尾时，最终的隐藏状态（或者在双向RNN的情况下是多个状态）应该能够理想地捕捉到整个序列内容的有意义表示，这与分类任务相关。

然后，这个最终表示通常被送入一个或多个标准的前馈层（常被称为全连接层或密集层）以执行最终的分类。

### 常见架构模式

有几种主要方式可以利用循环层的输出来进行分类：

1. **使用最终隐藏状态：** 这是最常见的方法。RNN处理序列，并且仅将最后时间步的隐藏状态用作后续分类层的输入。这个最终状态被假定为封装了整个序列的必要信息。框架API通常有一个参数 (parameter)（例如Keras中的`return_sequences=False`），它控制该层是只在最后时间步输出状态，还是输出所有时间步的隐藏状态。对于这种模式，您通常只希望从堆栈中*最终*循环层的最后一步获取输出。

   > 一种常见的架构，其中循环层的最终隐藏状态被传递给全连接层进行分类。
2. **使用池化隐藏状态：** 除了仅仅依赖最终隐藏状态外，您还可以使用*所有*时间步的隐藏状态。`return_sequences=True`参数（或等效参数）将设置在最后一个循环层上。这些状态随后通过池化操作进行聚合，然后传递给分类层。常见的池化策略包括：

   - **最大池化：** 获取隐藏状态中每个特征在时间维度上的最大值。这可以捕捉到序列中任何位置检测到的最重要的特征。
   - **平均池化：** 获取隐藏状态中每个特征在时间维度上的平均值。这提供了序列中所有特征的概括。

   如果分类的重要信息可能出现在序列的任何位置，而不仅仅是末尾，池化有时会有益。然而，使用最终隐藏状态通常更简单且表现良好，特别是对于旨在在长序列中保持相关信息的LSTM和GRU而言。

### 实现注意事项

- **输入准备：** 如第8章所述，您的输入序列（例如文本）需要转换为数值格式（整数编码），可能需要通过嵌入 (embedding)层，并进行填充以确保批次中的所有序列具有相同的长度。通常应使用掩码来通知循环层忽略这些填充的步长。循环层的输入形状通常为`(batch_size, time_steps, feature_dimension)`。
- **输出层：** 最终的全连接层需要根据分类任务的性质使用适当的激活函数 (activation function)：
  - **二元分类：** 使用一个输出单元和`sigmoid`激活函数。相应的损失函数 (loss function)通常是`BinaryCrossentropy`。
  - **多类别分类（单一标签）：** 使用 $N$ 个输出单元（其中 $N$ 为类别数量）和`softmax`激活函数。典型的损失函数是`CategoricalCrossentropy`。
- **返回序列：** 请记住根据您是使用最终隐藏状态（最后一层设为`False`）还是池化（最后一层设为`True`），正确配置循环层的`return_sequences`参数 (parameter)。如果堆叠循环层，中间层必须将`return_sequences`设置为`True`，以便将完整的隐藏状态序列传递给下一层。
- **双向RNN：** 对于许多分类任务，特别是在自然语言处理中，使用双向LSTM或GRU（第7章）可以提高性能。通过在正向和反向两个方向处理序列，模型可以基于任何给定元素两侧的上下文 (context)形成表示。最终的正向隐藏状态和最终的反向隐藏状态通常在传递给分类层之前进行连接（或有时平均/求和）。

### 实际应用示例

- **情感分析：** 给定表示产品评论的词语序列，将其分类为“积极”、“消极”或“中性”。RNN逐词读取评论，最终状态概括了所表达的整体情感。
- **主题分类：** 给定来自文档的词语序列，将其分类为预定义类别之一，如“科技”、“体育”、“金融”等。RNN处理文档文本以捕捉其主要主题。
- **意图识别：** 给定表示用户对语音助手命令的词语序列（例如，“明天天气怎么样？”），分类用户的意图（例如，“获取天气”、“播放音乐”、“设置计时器”）。

序列分类是一种强大的技术，其中RNN处理有序数据和保持状态的能力使其能够有效地概括序列信息以进行分类。通过理解如何构建模型架构，特别是如何运用循环层的输出，您可以为广泛的基于序列的问题构建有效的分类器。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本书是深度学习的综合参考资料，其中有专门的章节解释循环神经网络（第10章）及其应用，这些是序列分类的基础。
- [CS224n: Natural Language Processing with Deep Learning](https://cs224n.stanford.edu/) — Christopher Manning, Abigail See (2023)
  Journal: Online Course Materials; Publisher: Stanford University
  本课程广泛涵盖循环神经网络、LSTM、GRU及其在各种自然语言处理任务中的应用，包括情感分析等序列分类问题，并经常详细说明架构考量。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University (Online Draft)
  第三版草案包含了关于自然语言处理深度学习的章节，涵盖了循环神经网络及其在文本分类任务中的应用，并提供了实践示例。
- [Long Short-Term Memory](https://direct.mit.edu/neco/article-abstract/9/8/1735/6100/Long-Short-Term-Memory) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这是引入长短期记忆 (LSTM) 网络架构的原始论文，该架构是包括序列分类在内的现代循环神经网络应用的基础。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179) — Kyunghyun Cho, Bart van Merriënboer, Çağlar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724-1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  这篇论文引入了门控循环单元 (GRU) 作为 LSTM 的简化版本，因其效率和性能而在包括分类在内的序列建模任务中被广泛使用。
