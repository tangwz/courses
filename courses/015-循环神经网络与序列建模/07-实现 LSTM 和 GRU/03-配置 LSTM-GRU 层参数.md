# 配置 LSTM/GRU 层参数

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/configuring-lstm-gru-parameters)

[返回章节目录](README.md) · [返回课程目录](../README.md)

要根据特定任务调整使用 TensorFlow/Keras 或 PyTorch 等框架 API 构建的模型中的 `LSTM` 或 `GRU` 层，需要配置它们的常用参数 (parameter)。正确设置这些配置对于构建高效的序列模型非常重要。

### 指定单元数量

`LSTM` 和 `GRU` 层最基本的参数 (parameter)通常被称为 `units` (在 Keras/TensorFlow 中) 或 `hidden_size` (在 PyTorch 中)。这个整数值决定了隐藏状态 $h_t$ 的维度，对于 LSTM 而言，也决定了单元状态 $c_t$ 的维度。

```python
# TensorFlow/Keras 示例
lstm_layer = tf.keras.layers.LSTM(units=64)
gru_layer = tf.keras.layers.GRU(units=32)

# PyTorch 示例
# 注意：此处也必须指定 input_size
lstm_layer_pytorch = torch.nn.LSTM(input_size=10, hidden_size=64, batch_first=True)
gru_layer_pytorch = torch.nn.GRU(input_size=10, hidden_size=32, batch_first=True)
```

可将 `units` 视为循环层中记忆单元或神经元的数量。增加单元数量可以使层更有效地捕捉序列数据中复杂的模式和依赖关系，从而提升模型的表示能力。然而，就像在前馈网络中一样，过多的单元可能导致过拟合 (overfitting)，尤其是在数据集较小的情况下，并显著增加计算成本。找到合适的数量通常需要反复试验和验证。

### 选择激活函数 (activation function)

LSTM 和 GRU 等循环层涉及多项内部计算，这些计算很多都使用激活函数。框架通常允许您配置两种主要激活函数：

1. **`activation`**：此函数应用于单元状态更新（在 LSTM 中）和候选隐藏状态的计算。默认且最常见的选择是双曲正切函数 `tanh`。它的输出范围为 $[-1, 1]$，有助于调整网络内部的值。
   
   $$
   tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
   $$
   
2. **`recurrent_activation`**：此函数应用于门（在 LSTM 中是遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$；在 GRU 中是重置门 $r_t$、更新门 $z_t$）。标准选择是 sigmoid 函数 $\sigma(x)$，通常称为 `sigmoid` 或有时称为 `hard_sigmoid`（一种计算成本较低的近似）。sigmoid 函数输出 0 到 1 之间的值，这使其非常适合门控机制。值接近 1 表示“允许信息通过”，而值接近 0 则表示“阻止信息”。
   
   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$
   

尽管您在技术上可以更改这些默认值，但主激活函数使用 `tanh`，循环（门）激活函数使用 `sigmoid`（或 `hard_sigmoid`）是标准做法，并且通常效果良好。

```python
# TensorFlow/Keras 带有自定义激活函数的示例（不常见）
lstm_layer = tf.keras.layers.LSTM(
    units=128,
    activation='relu', # 主激活函数的非标准选择
    recurrent_activation='sigmoid' # 门的标准选择
)

# PyTorch：激活函数通常在 nn.LSTM/nn.GRU 模块中隐式地使用 tanh/sigmoid。
# 自定义它们可能需要手动实现单元逻辑。
```

### 控制输出：`return_sequences`

一个重要参数 (parameter)是 `return_sequences`。这是一个布尔标志，决定了层输出的形状。

- **`return_sequences=False` (默认值)：** 该层只输出*最后*时间步的隐藏状态 $h_t$。如果您的输入形状为 `(batch_size, time_steps, features)`，则输出形状将是 `(batch_size, units)`。当循环层是最终 `Dense` 层之前用于对整个序列进行分类或回归（例如，情感分析）的最后一层时，这种情况很常见。
- **`return_sequences=True`：** 该层输出*每个*时间步的隐藏状态 $h_t$。输出形状将是 `(batch_size, time_steps, units)`。在以下情况中这是必需的：

  - 您正在堆叠多个循环层（下一个循环层需要一个序列作为输入）。
  - 您需要在每个时间步进行预测（例如，序列标注、预测多个未来步骤的预测）。
  - 您正在使用对隐藏状态序列进行操作的注意力机制 (attention mechanism)。

```python
# TensorFlow/Keras 示例
# 层只输出最后一个隐藏状态（形状：batch_size, 64）
lstm_last_state = tf.keras.layers.LSTM(units=64, return_sequences=False)

# 层输出所有时间步的隐藏状态（形状：batch_size, time_steps, 128）
lstm_all_states = tf.keras.layers.LSTM(units=128, return_sequences=True)

# PyTorch 示例
# 默认情况下，nn.LSTM/nn.GRU 返回所有时间步的输出以及最终的隐藏/单元状态。
# 您通常会从它们返回的元组中选择所需的部分。
lstm_pytorch = torch.nn.LSTM(input_size=10, hidden_size=64, batch_first=True)
# output, (hn, cn) = lstm_pytorch(input_tensor)
# 'output' 包含所有隐藏状态 (批量大小, 序列长度, 隐藏维度)
# 'hn' 包含最终隐藏状态 (层数 * 方向数, 批量大小, 隐藏维度)
```

> `return_sequences` 参数对 LSTM 或 GRU 层输出形状的影响。

### 获取最终状态：`return_state`

另一个布尔参数 (parameter) `return_state`，控制层是否在序列输出之外返回最终隐藏状态。

- **`return_state=False` (默认值)：** 该层只返回输出序列（可以是最后一个状态或所有状态，取决于 `return_sequences`）。
- **`return_state=True`：** 该层返回一个列表（在 Keras 中）或元组（在 PyTorch 中），其中包含序列输出*以及*最终状态。
  - 对于 **GRU**，额外返回的项是 `[final_hidden_state]`。
  - 对于 **LSTM**，额外返回的项是 `[final_hidden_state, final_cell_state]`。

这在编码器-解码器架构（后续会介绍）中尤其有用，其中编码器的最终状态用于初始化解码器的状态。它对于分析序列最终学习到的表示也很有帮助。

```python
# TensorFlow/Keras 示例
lstm_layer = tf.keras.layers.LSTM(units=32, return_sequences=True, return_state=True)
# 调用时，输出为：[所有隐藏状态, 最终隐藏状态, 最终单元状态]

gru_layer = tf.keras.layers.GRU(units=64, return_sequences=False, return_state=True)
# 调用时，输出为：[最后一个隐藏状态, 最终隐藏状态]
# 注意：当 return_sequences=False 时，前两个元素是相同的。
```

### 输入形状及其他参数 (parameter)

- **输入形状：** 对于序列模型（如 Keras `Sequential`）中的第一个循环层，您通常需要指定*每个实例*的输入形状，不包括批量维度。这通常通过 `input_shape` 参数完成，该参数需要一个元组，例如 `(time_steps, features)`。对于后续层，框架通常会自动推断输入形状。PyTorch 在层初始化时需要 `input_size`（特征数量）。
- **Dropout：** 为缓解过拟合 (overfitting)，可以使用 dropout。循环层通常具有特定的 dropout 参数，例如 `dropout`（应用于输入/输出单元）和 `recurrent_dropout`（应用于循环连接，即隐藏状态更新）。这些参数在循环连接上跨时间步一致地应用 dropout 掩码，这对于训练的稳定性很重要。
- **初始化：** 与其他神经网络 (neural network)层类似，您通常可以自定义各种权重 (weight)矩阵（`kernel_initializer`、`recurrent_initializer`、`bias_initializer`）的初始化策略。诸如 Glorot (Xavier) 均匀分布或正交等标准初始化器通常是不错的起始选择。

理解这些参数使您能够有效地构建和定制 LSTM 和 GRU 层，以满足您的序列建模需求。请记住，最佳配置通常取决于特定的数据集和任务，需要一定程度的实验。在以下部分中，我们将了解如何将这些层组合成更复杂的架构，如堆叠式和双向 RNN。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）循环神经网络架构，并详细说明了其内部机制，包括门控和细胞状态。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  介绍了门控循环单元（GRU）作为一种简化的循环架构，并描述了其结构和门控机制。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了关于循环神经网络、LSTM和GRU的广泛理论内容，包括它们的架构、训练和参数考虑。
- [tf.keras.layers.LSTM and tf.keras.layers.GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) — TensorFlow Developers (2023)
  描述TensorFlow/Keras中LSTM和GRU层参数及用法的官方文档，直接支持代码示例。
- [torch.nn.LSTM and torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — PyTorch Core Team (2023)
  Publisher: PyTorch Foundation
  PyTorch的LSTM和GRU模块实现的官方文档，解释了它们的构造函数、参数和返回值。

---

[上一节](02-%E5%9C%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E4%B8%AD%E4%BD%BF%E7%94%A8GRU%E5%B1%82.md) · [下一节](04-%E5%A0%86%E5%8F%A0%E5%BE%AA%E7%8E%AF%E5%B1%82.md)
