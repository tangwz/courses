# 门控循环单元 (GRU)

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/gated-recurrent-units-gru)

[返回章节目录](README.md) · [返回课程目录](../README.md)

门控循环单元 (GRU) 由 Cho 等人于2014年提出，它为处理序列数据提供了一种有效方法。尽管长短期记忆 (LSTM) 网络也以解决梯度消失问题和捕捉长距离依赖而闻名，但它们因其三个不同的门和独立的细胞状态而导致结构较为繁复。GRU 旨在实现类似的能力，但采用更简洁的架构。

GRU 通过将遗忘门和输入门合并为一个“更新门”，并整合细胞状态和隐藏状态，从而简化了 LSTM 中发现的门控机制。这种单元只有两个门：

1. **重置门 ($r_t$)**：这个门决定了有多少先前的隐藏状态 ($h_{t-1}$) 应该与当前输入 ($x_t$) 结合，以计算一个*候选*隐藏状态。本质上，它决定了多少过去的信息与计算新的状态提议相关。如果重置门接近0，则在计算候选状态时，该单元会有效地“遗忘”先前的隐藏状态。
2. **更新门 ($z_t$)**：这个门控制了有多少来自先前的隐藏状态 ($h_{t-1}$) 的信息应该传递到当前的隐藏状态 ($h_t$)。它还决定了应该整合多少新计算的*候选*隐藏状态。如果更新门接近1，则先前状态大部分被传递；如果接近0，则新的候选状态占主导地位。

### GRU与LSTM的比较

主要区别在于门控结构：

- **LSTM**：有三个门（输入、遗忘、输出），并与隐藏状态 ($h_t$) 一起维护一个独立的细胞状态 ($c_t$)。这使得对信息流和记忆有更精细的控制。
- **GRU**：有两个门（重置、更新），并将细胞状态和隐藏状态合并为一个单一的状态向量 (vector) ($h_t$)。

GRU 中的这种简化导致其参数 (parameter)数量少于具有相同隐藏单元数量的 LSTM。这可以使 GRU 在计算上略微更高效（训练更快，内存占用更少），并且在较小数据集上可能更不容易过拟合 (overfitting)。

然而，LSTM 增加的复杂性可能会使其在某些任务中建模特别复杂的长距离依赖方面更具优势。在实践中，LSTM 和 GRU 之间的性能差异通常取决于具体任务，两者都没有普遍的优越性。通常会尝试两种架构，以查看哪种方案在特定问题上表现更好。

### GRU架构图

以下是单个 GRU 单元内数据流的简化视图：

> GRU 单元的简化表示。它呈现了重置门 ($r_t$) 如何影响候选隐藏状态 ($\tilde{h}_t$) 的计算，以及更新门 ($z_t$) 如何在先前状态 ($h_{t-1}$) 和候选状态之间进行平衡以生成最终隐藏状态 ($h_t$)。Sigmoid ($\sigma$) 和 tanh 激活函数 (activation function)通常分别用于门和候选状态。

### 在 Keras 中使用 GRU

在 Keras 中实现 GRU 层非常直接，并且与使用 `SimpleRNN` 或 `LSTM` 类似。您可以从 `keras.layers` 导入它。

```python
import keras
from keras import layers

# 示例：用于序列处理的堆叠 GRU 层
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=16, input_length=100), # 示例嵌入层
    layers.GRU(units=32, return_sequences=True), # 第一个 GRU 层，返回完整序列
    layers.GRU(units=32), # 第二个 GRU 层，只返回最后一个输出
    layers.Dense(units=1, activation='sigmoid') # 用于二元分类的输出层
])

model.summary()
```

重要的参数 (parameter)，如 `units`（输出/隐藏状态的维度）、`activation`（默认通常为 `tanh`）、`recurrent_activation`（通常用于门控为 `sigmoid`）以及 `return_sequences`，其行为与 `LSTM` 层中的对应项类似。

### 何时选择 GRU？

GRU 是 LSTM 的有力替代方案，尤其是在以下情况：

- 您正在寻求可能更快的训练时间以及更低的计算成本。
- 您的数据集大小有限，减少参数 (parameter)数量可能有助于防止过拟合 (overfitting)。
- 您希望从一个稍微简单的循环架构开始。

正如深度学习 (deep learning)中的许多选择一样，建议进行实证验证。在您的特定序列建模任务（例如，文本分类、时间序列预测）上尝试 LSTM 和 GRU 两种架构，并在验证集上评估它们的性能，以确定最合适的选项。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: EMNLP 2014; Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  介绍门控循环单元 (GRU) 架构的原始学术论文。
- [GRU layer](https://keras.io/api/layers/recurrent_layers/gru/) — Keras Team (2024)
  Keras 官方文档，详细说明了如何在 Keras 框架中使用 GRU 层、其参数和功能。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，对循环神经网络（包括 LSTM 和 GRU）及其基本原理进行了全面的理论解释。

---

[上一节](06-Keras%E4%B8%AD%E7%9A%84LSTM%E5%B1%82.md) · [下一节](08-%E7%94%A8%E4%BA%8ERNN%E7%9A%84%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87.md)
