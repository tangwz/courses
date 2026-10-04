---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "handling-variable-length-sequences"
sourceId: 2619
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/handling-variable-length-sequences"
title: "处理变长序列"
description: "讨论批处理中处理不同长度序列的必要性。"
order: 5
plots: []
sourceHash: "6de801c16539458e91157e20f28207036d4b51a7f9d164ced58e26b1e6dadf87"
sourceCorrections: []
---

您已成功将原始序列数据（如文本语句或时间序列读数）转换为数字格式，这可能是表示单词索引的整数序列，或随时间变化的浮点测量值。但是，您会很快发现一个共同特点：这些序列的长度很少相同。一个句子可能有10个单词，另一个可能有25个。一个传感器读数可能覆盖50个时间步，而另一个可能覆盖100个。

序列数据通常具有可变的长度，这种可变性给训练深度学习 (deep learning)模型带来了实际困难，尤其是在使用小批量数据以提高效率时。TensorFlow 和 PyTorch 等神经网络 (neural network)框架通常期望批次内的数据被组织成张量。张量是多维数组，需要维度一致。如果您尝试将不同长度的序列直接堆叠到批次中，将会遇到错误，因为时间步维度不会统一。

想象一下，您尝试创建一个矩阵，其中每一行代表一个序列。如果行的列数（时间步）不同，则它不是一个有效的矩形矩阵。

### 统一长度的必要性：填充

处理变长序列的标准方法是**填充（padding）**。填充是指向较短序列添加一个特殊的、预定义的值（“填充值”），直到批次中的所有序列都达到一个共同的固定长度。这个固定长度通常被称为 `maxlen`。

想象一下，这就像给较短的文本行添加空格，使它们都能对齐 (alignment)到相同的右边距。

**选择填充值：**
填充值应与任何真实数据值区分开来。对于整数编码的文本数据，`0` 是一个常见选择，前提是您的词汇表 (vocabulary)映射从 `1` 开始分配索引。如果 `0` 是词汇表中的有效标记 (token)索引，则需要选择一个不同的值或重新索引词汇表。对于已标准化（例如，均值为0，标准差为1）的数值时间序列数据，可能会使用一个远超典型数据范围的值（如-99），尽管掩码（接下来讨论）通常即使在这种情况下也能使使用 `0.0` 变得可行。重要的是模型最终应学会忽略这个填充值。

**确定最大长度（`maxlen`）：**
填充后的序列应该多长？您有几种选择：

1. **最长序列：** 将所有序列填充到整个数据集（或至少当前批次）中最长序列的长度。这确保了不会因截断而丢失信息，但如果存在极端异常值（非常长的序列），这可能会耗费大量计算资源和内存。
2. **固定百分位数：** 根据数据集中序列长度的分布选择一个长度，例如，第95个百分位长度。这意味着长于此长度的序列将被截断（缩短），会丢失一些信息，但它在计算成本和数据保留之间提供了平衡。
3. **预设 `maxlen`：** 根据领域知识或资源限制选择一个合理的固定长度（例如，许多NLP任务的512个标记）。短于此长度的序列将被填充；长于此长度的序列将被截断。

选择通常取决于具体的任务、数据的性质以及可用的计算资源。如果最重要的信息通常出现在序列的开头或结尾，那么截断非常长的序列可能是可以接受的。

### 填充策略：前置填充与后置填充

一旦您确定了 `maxlen` 和填充值，就需要决定*在哪里*添加填充：

- **前置填充：** 在序列的*开头*添加填充值。
- **后置填充：** 在序列的*结尾*添加填充值。

以下是视觉比较：

> 这是一个前置填充（在前面添加零）和后置填充（在后面添加零）的例子，应用于三个整数编码序列，以达到统一长度5。灰色框表示添加的填充值（0）。

前置填充和后置填充的选择有时会影响模型性能，尽管这种影响通常很小，尤其是在正确使用掩码时。

- **后置填充**在 Keras 等库中通常是默认选项。直观来看，对于标准RNN，最终的隐藏状态有时被用作序列的概括。后置填充确保此最终状态对应于原始序列的*最后一个实际元素*，而不是填充值。
- **前置填充**可以考虑在您认为最新信息（在原始序列的末尾）最为重要时使用，因为它确保了RNN最后处理此信息，可能对最终隐藏状态产生更大的影响。

实际操作中，通常最好从默认设置（通常是后置填充）开始，只有当性能不理想或您有充分的理论依据来偏好其中一种策略用于您的特定任务时，才进行试验。

类似地，您需要决定是**截断**过长序列的开头（`truncating='pre'`）还是结尾（`truncating='post'`）。后置截断（从末尾移除元素）是常见的，但如果最近的数据点被认为更重要，则前置截断可能更好。

### 实现说明

大多数深度学习 (deep learning)库都提供了方便的函数来处理填充和截断。例如，在 TensorFlow/Keras 中，`tf.keras.preprocessing.sequence.pad_sequences` 函数处理整个过程：

```python
# 使用 TensorFlow/Keras 的示例
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequences = [
    [12, 5, 23, 8],       # 长度 4
    [7, 101, 15],         # 长度 3
    [3, 9, 42, 11, 2]     # 长度 5
]

# 定义最大长度
maxlen = 5
padding_value = 0

# 后置填充（默认）和后置截断（默认）
padded_sequences = pad_sequences(
    sequences,
    maxlen=maxlen,
    padding='post',      # 'pre' 或 'post'
    truncating='post',   # 'pre' 或 'post'
    value=padding_value
)

# padded_sequences 将看起来像上面“后置填充”的例子
# [[ 12   5  23   8   0]
#  [  7 101  15   0   0]
#  [  3   9  42  11   2]]
```

PyTorch 没有直接内置的单一精确序列处理函数，但可以使用张量操作或 `torchtext` 等库中的工具，或者在自定义 `DataLoader` 的 collate 函数中手动填充序列来达到类似的效果。

填充解决了将变长序列放入固定大小的张量中进行批量处理的结构性问题。然而，它引入了不代表真实数据的人工填充值。将这些填充值直接馈送到RNN中，如果它们是真实输入，可能会对学习产生负面影响。下一步是告知模型忽略这些填充的时间步，这通过\*\*掩码（masking）\*\*来实现。

## 参考资料

- [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  提供了使用 Keras 实现深度学习的实用指导，包括对序列模型数据预处理技术（如填充）的详细说明。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本全面的自然语言处理教材，涵盖文本表示、分词以及为神经网络准备变长文本序列的方法等基本概念。
- [tf.keras.preprocessing.sequence.pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) — TensorFlow Developers (2024)
  Keras `pad_sequences` 工具的官方文档，详细说明了其参数和用于统一序列长度的用法，与章节中演示的一致。
- [Writing Custom DataLoader Classes](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#writing-custom-dataloader-classes) — Sasank Chilamkurthy (2017)
  Publisher: PyTorch Foundation
  PyTorch 官方教程，演示如何创建自定义 `DataLoader` 类和 `collate_fn` 来处理变长数据的批处理，这对于在 PyTorch 中实现填充是必要的。
