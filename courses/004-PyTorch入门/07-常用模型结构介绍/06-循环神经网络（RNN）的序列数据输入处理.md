# 循环神经网络（RNN）的序列数据输入处理

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/rnn-sequential-data-input)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network)（RNN）专门用于处理序列数据，无论是句子中的词语、音乐作品中的音符，还是随时间变化的测量值。与独立处理固定大小输入的传统前馈网络不同，RNN在序列的每个步骤都会更新一个内部*隐藏状态*，这使得之前步骤的信息能够影响当前和未来步骤的处理。这种序列处理方式需要特定的输入数据格式。

### 标准RNN输入形状

PyTorch的RNN层（如 `nn.RNN`、`nn.LSTM`、`nn.GRU`）默认期望输入数据为一个三维张量，具有以下维度：

**(序列长度, 批次大小, 输入特征)**

我们来逐一分析每个维度：

1. **序列长度 (`seq_len`)**: 这是序列中的时间步数量。例如，如果您正在处理句子，并且批次中最长的句子有15个词，那么您的`seq_len`通常就是15（较短的句子需要填充，稍后讨论）。
2. **批次大小 (`batch_size`)**: 这表示您同时处理的独立序列的数量。分批训练是提高效率和更好地估计梯度的标准做法。
3. **输入特征 (`input_size` 或 `features`)**: 这是表示*每个*时间步输入特征的维度。如果您处理的是用50维嵌入 (embedding)表示的词语，那么`input_size`将是50。如果您处理的是单变量时间序列（每个时间步只有一个值），那么`input_size`将是1。

**示例：** 假设您想处理一个包含32个句子的批次，其中每个句子表示为20个词的序列，并且每个词都转换为100维向量 (vector)嵌入。输入张量的形状将是 `(20, 32, 100)`。

```python
import torch

# 示例参数
seq_len = 20      # 最长序列长度
batch_size = 32   # 批次中的序列数量
input_features = 100 # 每个词嵌入的维度

# 创建一个虚拟输入张量（例如，用随机数填充）
# 形状: (序列长度, 批次大小, 输入特征)
rnn_input = torch.randn(seq_len, batch_size, input_features)

print(f"标准RNN输入形状: {rnn_input.shape}")
# 输出: 标准RNN输入形状: torch.Size([20, 32, 100])
```

### `batch_first` 选项

尽管 `(序列长度, 批次大小, 输入大小)` 是默认设置，但许多人认为将批次维度放在首位更符合直觉，这也与数据常见组织方式以及其他层类型（如卷积层或线性层）通常处理输入的方式保持一致。PyTorch RNN层为此提供 `batch_first` 参数 (parameter)。

如果您使用 `batch_first=True` 初始化RNN层，它将期望输入张量形状为：

**(批次大小, 序列长度, 输入特征)**

**示例（接续前文）：** 如果使用 `batch_first=True`，相同数据的形状将是 `(32, 20, 100)`。

```python
import torch
import torch.nn as nn

# 示例参数（同前）
seq_len = 20
batch_size = 32
input_features = 100
hidden_size = 50 # RNN的示例隐藏大小

# 创建一个批次维度在前的虚拟输入张量
# 形状: (批次大小, 序列长度, 输入特征)
rnn_input_batch_first = torch.randn(batch_size, seq_len, input_features)

# 使用 batch_first=True 初始化RNN层
rnn_layer = nn.RNN(input_size=input_features, hidden_size=hidden_size, batch_first=True)

# 将输入通过层（输出形状也将是批次维度在前）
output, hidden_state = rnn_layer(rnn_input_batch_first)

print(f"批次维度优先的RNN输入形状: {rnn_input_batch_first.shape}")
print(f"批次维度优先的RNN输出形状: {output.shape}")
# 输出: 批次维度优先的RNN输入形状: torch.Size([32, 20, 100])
# 输出: 批次维度优先的RNN输出形状: torch.Size([32, 20, 50])
```

使用 `batch_first=True` 通常可以简化数据准备流程，因为数据集通常在加载时就是批次维度优先。请记住，如果设置了 `batch_first=True`，输出形状也将采用 `(批次大小, 序列长度, 隐藏大小)` 格式。

> RNN输入数据结构的视觉表示。输入通常是一个三维张量，表示多个序列（批次），每个序列包含多个时间步，并且每个时间步具有多个特征。

### 处理变长序列

"一个常见的问题是，数据集中的序列长度很少完全相同（例如，句子的词数不同）。由于张量要求统一的维度，您需要使批次中的序列长度一致。这通常通过以下方式完成："

1. **填充：** 较短的序列会在末尾填充一个特殊值（通常是零），直到它们达到批次中最长序列的长度（`seq_len`）。
2. **打包（可选但推荐）：** 为了防止RNN处理这些无意义的填充值，PyTorch提供了实用工具（`torch.nn.utils.rnn.pack_padded_sequence` 和 `torch.nn.utils.rnn.pad_packed_sequence`）。您可以在将填充序列送入RNN之前“打包”它们，告诉RNN批次中每个序列的真实长度。然后，RNN只处理实际的数据点。之后您再“填充”输出以获取标准张量。虽然我们在此不详细说明打包，但它是使用变长数据进行高效且准确RNN训练的重要技术。

理解期望的输入形状（`(序列长度, 批次大小, 特征)`，如果 `batch_first=True` 则是 `(批次大小, 序列长度, 特征)`）对于正确准备数据并将其送入PyTorch的循环层必不可少。始终查阅您所用特定层的文档，并确保您的数据预处理流程生成所需形状的张量。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10 Sequence Modeling: Recurrent and Recursive Nets
  为循环神经网络提供了全面的理论基础，包括其架构、隐藏状态和通过时间的反向传播。
- [RNN - PyTorch 2.3 Documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch Foundation (2024)
  PyTorch `nn.RNN`模块的官方文档，详细说明了输入/输出形状、`batch_first`等参数以及其他配置选项。
- [PackedSequence - PyTorch 2.3 Documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) — PyTorch Foundation (2024)
  Publisher: PyTorch Foundation
  `torch.nn.utils.rnn.pack_padded_sequence`和`torch.nn.utils.rnn.pad_packed_sequence`的官方文档，对于处理PyTorch RNN中的可变长度序列至关重要。
- [CS224n: Natural Language Processing with Deep Learning - Lecture Notes](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  斯坦福大学备受推崇的深度学习自然语言处理课程资料，涵盖循环神经网络、长短期记忆网络和序列处理。

---

[上一节](05-%E5%9C%A8PyTorch%E4%B8%AD%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84RNN.md) · [下一节](07-LSTM%20%E5%92%8C%20GRU%20%E7%AE%80%E8%A6%81%E4%BB%8B%E7%BB%8D.md)
