---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "padding-sequences"
sourceId: 2620
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/padding-sequences"
title: "序列填充"
description: "实现填充以使批次中的所有序列具有相同长度。"
order: 6
plots: []
sourceHash: "318cd5e955cb40f1e93e849f011aa0a847ded29b1849aca190c0267fc62a4fe7"
sourceCorrections: []
---

将你的标记 (token)序列（如单词或字符）转换为整数序列后，你很可能会遇到一个实际问题：序列的长度很少一致。例如，文本数据集中的句子词 (subword)数自然不同，而不同的时间序列段可能覆盖不同的持续时间。

请看这些整数编码的句子：

- 序列 1: `[12, 45, 6, 887, 3]` (长度 5)
- 序列 2: `[12, 101, 500]` (长度 3)
- 序列 3: `[99, 2, 76, 1024, 50, 1]` (长度 6)

循环神经网络 (neural network) (RNN)旨在处理序列数据，但深度学习 (deep learning)框架的底层操作通常依赖于以统一张量表示的批次数据处理。将这些变长序列直接堆叠成单个张量并不简单，因为张量需要一致的维度。尝试将不同长度的列表直接输入到大多数RNN层通常会导致错误。

这就是填充的作用所在。填充是用于在批次内所有序列中强制统一长度的标准方法。它通过向较短的序列添加一个特殊保留值（几乎总是0），直到它们都达到指定长度。

### 填充如何操作

核心思想很简单：用占位值增加较短的序列。通常，值 `0` 被保留用于填充。这很方便，因为词汇映射通常从 `1` 开始索引实际标记 (token)。如果你的词汇编码使用 `0` 作为实际标记，则需要选择不同的填充值或调整词汇索引。

让我们再次看我们的示例序列。如果我们决定将它们全部填充到批次中最长序列的长度（即6），它们将变为：

- 序列 1 (已填充): `[12, 45, 6, 887, 3, 0]`
- 序列 2 (已填充): `[12, 101, 500, 0, 0, 0]`
- 序列 3 (已填充): `[99, 2, 76, 1024, 50, 1]` (无需填充)

现在，这三个序列可以整齐地堆叠成一个形状为 (3, 6) 的张量，其中 3 是批次大小，6 是统一的序列长度（或时间步数）。

### 前填充与后填充

你有两种主要方法来添加填充值：

1. **前填充：** 在序列的*开头*添加填充值。

   - 序列 1 (前填充): `[0, 12, 45, 6, 887, 3]`
   - 序列 2 (前填充): `[0, 0, 0, 12, 101, 500]`
2. **后填充：** 在序列的*末尾*添加填充值（如第一个例子所示）。

   - 序列 1 (后填充): `[12, 45, 6, 887, 3, 0]`
   - 序列 2 (后填充): `[12, 101, 500, 0, 0, 0]`

选择有关系吗？有时有。由于RNN按步处理序列，最终的隐藏状态受序列中后面元素的影响更大。使用后填充，实际内容在填充开始前结束，最终的隐藏状态反映了*原始*序列内容的末尾。使用前填充，RNN在看到实际数据之前可能会处理许多填充步骤。

在实践中，特别是当使用像掩码这样的机制时（我们将在下一节讨论），这种选择通常对许多任务的最终性能影响很小。后填充可能稍微更直观，但前填充是某些库（如Keras的 `pad_sequences` 函数）的默认设置，并且根据具体的架构和任务，偶尔也会有益。项目内的一致性通常比前填充和后填充之间的绝对选择更重要。

### 确定填充长度

填充序列应该多长？有两种常用方法：

1. **填充到批次最大长度：** 计算*当前批次*中的最大序列长度，并将该批次中的所有序列填充到此长度。这有助于节省内存，因为张量维度会适应每个批次。但是，输入张量的形状会在批次之间变化，这可能需要在一些自定义模型实现中进行谨慎处理。
2. **填充到固定最大长度 (`maxlen`)：** 为整个数据集中的所有序列定义一个固定的最大长度 (`maxlen`)。任何长于 `maxlen` 的序列都会被截断（从开头或末尾），任何短于 `maxlen` 的序列都会被填充（前填充或后填充）到 `maxlen`。这保证了所有批次输入张量的形状一致，从而简化了模型构建。缺点是，如果 `maxlen` 远大于典型序列长度，可能会导致大量填充的张量，从而降低效率；或者如果许多序列被截断，则会丢失信息。选择合适的 `maxlen` 通常需要分析数据集中序列长度的分布。

### 框架支持

深度学习 (deep learning)框架提供了方便的函数来处理填充。例如，TensorFlow/Keras 提供了 `tf.keras.preprocessing.sequence.pad_sequences` 工具，它接收一个序列列表（作为Python整数列表），并根据指定的选项（如 `maxlen`、`padding` ('pre' 或 'post') 和 `truncating` ('pre' 或 'post')）执行填充。PyTorch 没有一个完全相同的函数，但可以通过其数据加载和批处理机制中的张量操作或实用程序（如 `torch.nn.utils.rnn` 中的 `pad_sequence`）来实现填充。

### 掩码的必要性

填充解决了创建统一张量的技术问题，但它在数据中引入了人工 `0` 值。我们不希望RNN将这些填充零视为有意义的输入，因为它们不代表原始序列中的实际信息。处理这些零可能会对学习到的表示和最终输出产生负面影响。

因此，填充后，让模型知道输入张量的哪些部分对应实际数据，哪些部分只是填充，这一点很重要。这通过**掩码**实现，我们将在下一节中介绍。掩码有效地告诉后续层（如RNN或注意力层）在计算期间忽略填充的时间步。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  一本关于自然语言处理的教材，涵盖了文本表示和机器学习模型数据准备的基础知识。
- [tf.keras.preprocessing.sequence.pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) — TensorFlow Developers (2024)
  Publisher: Google
  Keras用于将序列填充到统一长度的实用函数的官方文档，详细说明了其参数和用法。
- [torch.nn.utils.rnn.pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  PyTorch的官方文档，该实用函数旨在将可变长度序列列表填充成单个填充批次张量。
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  一门备受推崇的大学深度学习自然语言处理课程材料，通常包含实用数据准备技术。
