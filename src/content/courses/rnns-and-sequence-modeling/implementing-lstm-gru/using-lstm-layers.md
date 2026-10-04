---
course: "rnns-and-sequence-modeling"
chapter: "implementing-lstm-gru"
lesson: "using-lstm-layers"
sourceId: 2598
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/using-lstm-layers"
title: "在深度学习框架中使用LSTM层"
description: "演示在常用框架中使用内置LSTM层API。"
order: 1
plots: []
sourceHash: "8c35358827d0e3f3bc11a942f4f1414156adea6910ebd9cfd2d83acef50a920d"
sourceCorrections: []
---

长短期记忆（LSTM）单元是递归神经网络 (neural network)单元，通过其内部结构有效处理序列数据。该结构包括遗忘门、输入门和输出门（$f_t, i_t, o_t$）以及独立的单元状态（$C_t$），用于随时间管理信息。在深度学习 (deep learning)框架中，应用这种架构得到了TensorFlow（及其Keras API）和PyTorch等现代库的简化。这些库提供了高级抽象，使我们能够将LSTM集成到模型中，而无需从头实现门逻辑。

这些库提供了预构建的LSTM层，封装了我们讨论过的复杂计算。我们的注意力从门方程转移到理解如何在更大的神经网络中正确配置和连接这些层。

### TensorFlow中的LSTM层（Keras API）

TensorFlow通过其用户友好的Keras API，提供了`tf.keras.layers.LSTM`层。实例化此层非常简单。

```python
import tensorflow as tf

# 示例：创建一个LSTM层
lstm_layer = tf.keras.layers.LSTM(units=64)
```

在此示例中，`units=64`指定了输出空间的维度，这与隐藏状态（$h_t$）和单元状态（$C_t$）的大小也一致。我们来看一下`LSTM`层的一些重要参数 (parameter)：

- `units`：（必填）正整数，输出空间（以及隐藏/单元状态）的维度。
- `activation`：用于单元状态更新和输出门的激活函数 (activation function)。默认为'tanh'。选择'tanh'有助于将单元状态值保持在-1到1之间。
- `recurrent_activation`：用于输入门、遗忘门和输出门的激活函数。默认为'sigmoid'。Sigmoid函数在这里很适用，因为门通常输出0到1之间的值，表示比例或概率（例如，遗忘多少）。
- `return_sequences`：布尔值。如果为`True`，则层会返回每个时间步的完整隐藏状态序列（$h_0, h_1, ..., h_T$）。如果为`False`（默认），则仅返回最终隐藏状态（$h_T$）。当堆叠LSTM层或输出需要每个时间步的信息时（例如，序列到序列任务），返回完整序列是必需的。
- `return_state`：布尔值。如果为`True`，则除了输出之外，层还会返回最后的隐藏状态*和*最后的单元状态（$h_T, C_T$）。这对于初始化另一个LSTM层的状态很有用，尤其是在编码器-解码器架构中。默认为`False`。
- `input_shape`：（可选，通常是Sequential模型中第一层所需）一个元组，指定输入形状，不包括批大小。对于序列数据，这通常是`(timesteps, features)`。例如，`input_shape=(10, 32)`表示包含10个时间步的序列，每个时间步有32个特征。

**输入形状：** Keras LSTM层期望以3D张量格式的输入数据：`(batch_size, timesteps, features)`。

- `batch_size`：同时处理的序列数量。
- `timesteps`：每个序列的长度。
- `features`：每个时间步表示输入的特征数量。

**输出形状：**

- 如果`return_sequences=False`（默认）：输出是一个形状为`(batch_size, units)`的2D张量。
- 如果`return_sequences=True`：输出是一个形状为`(batch_size, timesteps, units)`的3D张量。
- 如果`return_state=True`：层返回一个包含`[outputs, final_hidden_state, final_cell_state]`的列表。`outputs`的形状取决于`return_sequences`，而`final_hidden_state`和`final_cell_state`的形状均为`(batch_size, units)`。

以下是在Keras Sequential模型中使用LSTM层的最小示例：

```python
# 定义示例输入形状（例如，32个序列，10个时间步，8个特征）
batch_size = 32
timesteps = 10
features = 8
input_data = tf.random.normal((batch_size, timesteps, features))

# 创建一个包含一个LSTM层的简单模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=(timesteps, features), return_sequences=True),
    # 此处可添加更多层
    tf.keras.layers.Dense(1) # 示例输出层
])

# 获取输出
output = model(input_data)

print("输入形状:", input_data.shape)
# 输出形状取决于最后一层的配置（此处为return_sequences=True）
print("LSTM输出形状:", model.layers[0].output.shape)
print("最终输出形状:", output.shape)
```

### PyTorch中的LSTM层

PyTorch提供了`torch.nn.LSTM`层。它的初始化与Keras略有不同，但目的相同。

```python
import torch
import torch.nn as nn

# 示例：创建一个LSTM层
# input_size = 每个时间步的特征数量
# hidden_size = 隐藏/单元状态中的单元数量
input_size = 8
hidden_size = 64
lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
```

`torch.nn.LSTM`的重要参数 (parameter)：

- `input_size`：（必填）在每个时间步的输入$x$中期望的特征数量。
- `hidden_size`：（必填）隐藏状态$h$（和单元状态$C$）中的特征数量。这对应于Keras中的`units`。
- `num_layers`：循环层的数量。通过此参数进行层堆叠。默认为1。
- `batch_first`：布尔值。如果为`True`（推荐且常见），输入和输出张量以`(batch_size, seq_len, features)`格式提供。如果为`False`（默认），格式为`(seq_len, batch_size, features)`。使用`batch_first=True`通常感觉更直观，并且与数据在管道其他部分的常规处理方式以及Keras的默认设置一致。
- `dropout`：如果非零，则在除最后一层之外的每个LSTM层的输出上引入一个Dropout层，其丢弃概率等于`dropout`。默认为0。
- `bidirectional`：如果为`True`，则变为双向LSTM。默认为`False`。我们稍后将讨论这一点。

**输入形状：**

- 如果`batch_first=True`：输入形状为`(batch_size, seq_len, input_size)`。
- 如果`batch_first=False`：输入形状为`(seq_len, batch_size, input_size)`。

**输出形状：** `nn.LSTM`层返回一个元组：`(output, (h_n, c_n))`。

- `output`：包含LSTM最后一层在每个时间步的输出特征（$h_t$）。
  - 如果`batch_first=True`：形状为`(batch_size, seq_len, num_directions * hidden_size)`。
  - 如果`batch_first=False`：形状为`(seq_len, batch_size, num_directions * hidden_size)`。（如果`bidirectional=True`，`num_directions`为2，否则为1）。
- `h_n`：包含批处理中每个元素的最终隐藏状态。形状为`(num_layers * num_directions, batch_size, hidden_size)`。
- `c_n`：包含批处理中每个元素的最终单元状态。形状为`(num_layers * num_directions, batch_size, hidden_size)`。

请注意，PyTorch中的`output`始终包含所有时间步的隐藏状态（类似于Keras中的`return_sequences=True`）。如果您只需要最终的隐藏状态，通常可以对`output`张量进行索引（例如，如果`batch_first=True`，则为`output[:, -1, :]`）或使用`h_n`。

以下是一个最小的PyTorch示例：

```python
# 定义示例输入形状 (batch_first=True)
batch_size = 32
seq_len = 10
input_size = 8 # 特征
hidden_size = 64

input_data = torch.randn(batch_size, seq_len, input_size)

# 创建一个LSTM层
lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

# 将数据通过层
# 我们可以选择提供初始隐藏/单元状态 (h_0, c_0)
# 如果不提供，它们默认为零。
output, (h_n, c_n) = lstm_layer(input_data)

print("输入形状:", input_data.shape)
print("输出形状（所有时间步）:", output.shape) # (批次, 序列长度, 隐藏大小)
print("最终隐藏状态形状 (h_n):", h_n.shape) # (层数*方向数, 批次, 隐藏大小)
print("最终单元状态形状 (c_n):", c_n.shape) # (层数*方向数, 批次, 隐藏大小)

# 从 'output' 张量中仅获取最后一个时间步的输出：
last_step_output = output[:, -1, :]
print("最后一个时间步输出形状:", last_step_output.shape) # (批次, 隐藏大小)
```

通过使用这些高级`LSTM`层，我们可以轻松地将LSTM的能力整合到我们的序列模型中。这些框架处理复杂的门计算，使我们能够专注于整体模型架构、参数调整（如`units`或`hidden_size`的数量），以及以预期`(batch, timesteps, features)`格式准备数据。接下来的章节将在此基础上介绍GRU层、堆叠循环层和实现双向处理。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）架构的原始论文，为其内部机制奠定了基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教材，其中有一章专门介绍循环神经网络，包括LSTM的详细解释及其理论基础。
- [tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) — TensorFlow Developers (2024)
  TensorFlow中Keras LSTM层的官方文档，涵盖其构造函数参数、输入/输出形状及使用示例。
- [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch LSTM模块的官方文档，提供其参数、状态和数据流的详细说明。
