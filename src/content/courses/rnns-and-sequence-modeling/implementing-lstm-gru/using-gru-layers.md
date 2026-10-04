---
course: "rnns-and-sequence-modeling"
chapter: "implementing-lstm-gru"
lesson: "using-gru-layers"
sourceId: 2600
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/using-gru-layers"
title: "在深度学习框架中使用GRU层"
description: "展示在常用框架中使用内置GRU层API的方法。"
order: 2
plots: []
sourceHash: "7bbd93c0c56db05485f0ad1cefa0fc9391553ebc088a1b762e6c01f13993acf4"
sourceCorrections: []
---

门控循环单元（GRU）是长短期记忆（LSTM）架构的近亲。与LSTM相比，GRU提供了一种简化的门控机制，通常能以更少的参数 (parameter)获得相似的性能，并且计算可能更快。这里将展示如何使用流行的深度学习 (deep learning)框架，如TensorFlow（使用Keras API）和PyTorch，来实现GRU层。

### TensorFlow/Keras中的GRU层

TensorFlow的高级Keras API提供了一种直接的方法，可以使用`tf.keras.layers.GRU`将GRU层添加到您的模型中。它的使用方式与我们已经接触过的`SimpleRNN`和`LSTM`层非常相似。

要创建一个基本的GRU层，您主要需要指定层中的单元数量。这个数量决定了隐藏状态的维度，并且如果`return_sequences=False`，也决定了输出的维度。

```python
import tensorflow as tf

# 定义一个包含64个单元的GRU层
# 'units' 指定了隐藏状态和输出空间的维度
gru_layer = tf.keras.layers.GRU(units=64)

# 示例输入形状：(batch_size, timesteps, features)
# 例如，一个包含32个序列的批次，每个序列长10步，每步有8个特征
sample_input = tf.random.normal([32, 10, 8])

# 将输入通过层
output = gru_layer(sample_input)

# 默认情况下，GRU只返回最后一个时间步的输出
# 默认输出形状：(batch_size, units)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape (default): {output.shape}")
```

以下是`tf.keras.layers.GRU`最常用的一些参数 (parameter)：

- `units`: (必需) 整数，表示输出空间和隐藏状态的维度。
- `activation`: 候选隐藏状态计算的激活函数 (activation function)。默认为`tanh`，这是GRU的标准选择，允许状态值在-1到1之间。
- `recurrent_activation`: 用于重置门 ($r_t$) 和更新门 ($z_t$) 的激活函数。默认为`sigmoid`。这一点很重要，因为门控需要输出0到1之间的值来有效控制信息流动。
- `return_sequences`: 布尔值。如果为`True`，则层会返回输入序列中每个时间步的隐藏状态输出。输出形状变为`(batch_size, timesteps, units)`。如果为`False`（默认），它只返回最后一个时间步的最终隐藏状态输出，导致输出形状为`(batch_size, units)`。通常，除了堆叠中的最后一个循环层，或者当后续层期望序列时，您会为所有循环层设置`return_sequences=True`。
- `return_state`: 布尔值。如果为`True`，则层除了输出外，还会返回最终的隐藏状态张量。对于GRU层，这是一个表示最后一个时间步隐藏状态的单个张量。输出会是一个列表：`[output, final_state]`。
- `go_backwards`: 布尔值（默认：`False`）。如果为`True`，输入序列将按逆序处理，并且如果`return_sequences=True`，则返回逆序的序列。
- `reset_after`: 布尔值（默认：`True`）。决定GRU的计算变体。`True`对应于在候选隐藏状态的矩阵乘法之后应用重置门的变体。`False`则是在之前应用。默认的`True`在实践中通常效果很好。

让我们观察`return_sequences=True`的效果：

```python
# 创建一个返回完整输出序列的GRU层
gru_layer_seq = tf.keras.layers.GRU(units=64, return_sequences=True)

# 传递相同的示例输入
output_seq = gru_layer_seq(sample_input)

# 输出形状现在包含了时间步维度
# 输出形状：(batch_size, timesteps, units)
print(f"Output shape (return_sequences=True): {output_seq.shape}")
```

如您所见，设置`return_sequences=True`会保留输出中的时间维度，使其适合输入到后续的循环层，或用于需要在每个时间步生成输出的任务。

### PyTorch中的GRU模块

在PyTorch中，`torch.nn.GRU`模块提供了等效的功能。初始化此模块时，您需要指定输入特征的大小和所需的隐藏状态大小。

PyTorch中一个需要特别注意的地方是输入序列的默认预期形状。与Keras不同，PyTorch的循环层，包括`torch.nn.GRU`，默认期望输入张量的形状为`(sequence_length, batch_size, features)`。然而，数据加载和预处理流程通常会生成`(batch_size, sequence_length, features)`格式的数据。为了直接处理这种常见格式，在创建`GRU`实例时，您必须将`batch_first`参数 (parameter)设置为`True`。

```python
import torch
import torch.nn as nn

# 定义网络参数
input_features = 8    # 输入中每个时间步的特征数量
hidden_units = 64     # GRU隐藏状态中的单元数量
batch_size = 32       # 批次中的序列数量
seq_length = 10       # 每个序列的长度

# 定义一个GRU模块
# 确保 batch_first=True 以使用 (batch, seq, feature) 输入格式
gru_module = nn.GRU(input_size=input_features,
                    hidden_size=hidden_units,
                    batch_first=True) # 对常见数据形状很重要

# 示例输入：张量形状 (batch_size, seq_length, input_features)
sample_input_pt = torch.randn(batch_size, seq_length, input_features)

# 将输入通过模块。
# 默认情况下，PyTorch GRU返回两项：
# 1. output_seq: 包含每个时间步的输出隐藏状态的张量。
# 2. final_hidden_state: 包含最后一个时间步的隐藏状态的张量。
output_seq_pt, final_hidden_state_pt = gru_module(sample_input_pt)

# 输出序列形状：(batch_size, seq_length, hidden_size)
print(f"PyTorch Input shape: {sample_input_pt.shape}")
print(f"PyTorch Output sequence shape: {output_seq_pt.shape}")

# 最终隐藏状态形状：(num_layers * num_directions, batch_size, hidden_size)
# 对于单层、单向GRU，num_layers=1, num_directions=1。
print(f"PyTorch Final hidden state shape: {final_hidden_state_pt.shape}")
```

`torch.nn.GRU`的重要参数：

- `input_size`: (必需) 输入 $x_t$ 中预期特征的数量。
- `hidden_size`: (必需) 隐藏状态 $h_t$ 中特征的数量。
- `num_layers`: 整数（默认：1）。堆叠的GRU层数量。这个参数可以方便地处理堆叠。我们将在后面的章节中讨论堆叠。
- `bias`: 布尔值（默认：`True`）。如果为`False`，则层将不使用偏置 (bias)权重 (weight) $b_{ir}, b_{hr}$ 等。
- `batch_first`: 布尔值（默认：`False`）。如果为`True`，则输入和输出张量会以批次维度优先的方式提供：`(batch, seq, feature)`。如果您的数据遵循此常见约定，请将其设置为`True`。
- `dropout`: 浮点数（默认：0）。如果非零，则在除最后一层外的每个GRU层的输出上引入一个Dropout层，并使用指定的dropout概率。用于正则化 (regularization)。
- `bidirectional`: 布尔值（默认：`False`）。如果为`True`，则创建一个双向GRU。我们将在本章后面介绍这种架构。

请注意，PyTorch的`GRU`模块自然会将其输出元组的第一个元素作为完整的输出序列返回（类似于Keras中的`return_sequences=True`）。如果您的任务只需要最终隐藏状态的输出（等同于Keras的默认`return_sequences=False`），您可以轻松地从`output_seq_pt`张量中提取它。一种常见的方法是选择批次中每个序列的最后一个时间步的输出：

```python
# 提取批次中每个序列的最后一个时间步的输出
last_step_output_pt = output_seq_pt[:, -1, :]
print(f"PyTorch Last time step output shape: {last_step_output_pt.shape}")
# 形状：(batch_size, hidden_size)
```

第二个返回项`final_hidden_state_pt`包含最终隐藏状态，它对于初始化后续层或进行序列到序列任务非常有用。它的形状包括层数和方向，这在使用堆叠式或双向GRU时会变得重要。

### 实际选择GRU和LSTM

现在您已经可以实现LSTM和GRU，那么何时选择哪一种呢？

- **复杂度和效率：** 与LSTM（遗忘门、输入门、输出门）相比，GRU的门更少（更新门和重置门）。它们也不维护单独的单元状态。这使得相同数量隐藏单元的参数 (parameter)更少，可能导致训练和推理 (inference)稍快，并占用更少内存。
- **性能：** 没有明确的规则说明哪一种模型始终优于另一种。它们的性能高度依赖于任务和数据。LSTM凭借其专门的单元状态，理论上可能更适合需要记忆非常长时间间隔信息的任务。然而，GRU通常非常有效，更容易训练（有时收敛更快），而且其更简单的结构可能在较小数据集上提供更好的泛化能力，因为参数较少（这是一种隐式正则化 (regularization)形式）。
- **建议：** 鉴于深度学习 (deep learning)的经验性质，在模型开发过程中尝试两种架构通常是一个好策略。您可以从GRU开始，因为它们相对简单，并且迭代时间可能更快。如果性能指标需要提升，或者您有充分理由认为独立的记忆单元管理对您的特定问题有益，那么尝试LSTM是一个值得的步骤。

在学会了如何在TensorFlow/Keras和PyTorch中实例化和使用基本的GRU层之后，您已准备好将它们集成到序列建模流程中。以下部分将在此基础上，展示如何进一步配置这些层，通过堆叠将它们组合成更深的网络结构，并使用双向处理来提升它们获取上下文 (context)的能力。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  介绍了门控循环单元 (GRU) 架构，详细说明了其结构和功能，作为 LSTM 在序列建模任务中计算效率更高的替代方案。
- [tf.keras.layers.GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU) — TensorFlow Developers (2023)
  Publisher: TensorFlow
  TensorFlow Keras GRU 层的官方文档，提供 API 细节、参数和实现示例。
- [torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) — PyTorch Developers (2023)
  Publisher: PyTorch
  PyTorch GRU 模块的官方文档，概述了其参数、输入、输出和在序列建模中的用法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖循环神经网络、LSTM 和 GRU，包括其理论基础和比较。
