---
course: "getting-started-with-pytorch"
chapter: "introduction-common-architectures"
lesson: "lstm-gru-mention"
sourceId: 2217
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/lstm-gru-mention"
title: "LSTM 和 GRU 简要介绍"
description: "简要介绍长短期记忆 (LSTM) 和门控循环单元 (GRU)，它们是对简单 RNN 的改进。"
order: 7
plots: []
sourceHash: "aa8c15505b37fae1a8fb25315dedb4c08a499c008a9fc844d8f2eb722bc0df64"
sourceCorrections: []
---

简单的循环神经网络 (neural network) (RNN) 层，例如 PyTorch 中的 `nn.RNN`，通过维护隐藏状态来处理序列。然而，这些网络在学习跨越长时间跨度的模式时经常遇到困难。这种困难主要是由于梯度消失问题，即在通过许多时间步进行反向传播 (backpropagation)时，梯度变得极其小，从而阻碍了模型根据早期输入有效更新权重 (weight)的能力。

为了解决这个局限，开发了更复杂的循环单元。PyTorch 中现成的两种最受欢迎且有效的替代方案是长短期记忆 (LSTM) 和门控循环单元 (GRU)。

### 长短期记忆 (LSTM)

LSTM，由 Hochreiter & Schmidhuber 于 1997 年提出，其设计目的是应对梯度消失问题并更好地捕获长距离依赖关系。LSTM 的核心改进在于其内部结构，它不仅包含像简单 RNN 一样的隐藏状态 ($h_t$)，还包含一个独立的 *单元状态* ($c_t$)。

可以将单元状态视为一条信息高速公路，它允许早期时间步的相关信息相对顺畅地流经网络。信息进出此单元状态的流动由三种称为 *门* 的专门机制调控：

1. **遗忘门：** 决定从前一个单元状态 ($c_{t-1}$) 中丢弃哪些信息。
2. **输入门：** 决定将当前输入 ($x_t$) 和前一个隐藏状态 ($h_{t-1}$) 中的哪些新信息存储到当前单元状态 ($c_t$) 中。
3. **输出门：** 决定单元状态的哪些部分应作为新的隐藏状态 ($h_t$) 输出。

这些门使用 sigmoid 激活函数 (activation function)（输出值在 0 到 1 之间）来控制信息通过的程度。这种门控机制允许 LSTM 有选择地长时间记忆信息并遗忘不相关的细节，使它们在涉及复杂序列模式的任务中非常有效，例如机器翻译、语言建模和语音识别。

在 PyTorch 中，可以通过 `torch.nn.LSTM` 层使用 LSTM。其用法在预期的输入/输出形状和初始化参数 (parameter)（如 `input_size`、`hidden_size`、`num_layers`）方面与 `nn.RNN` 非常相似。

```python
import torch
import torch.nn as nn

# 示例：定义一个 LSTM 层
input_size = 10
hidden_size = 20
num_layers = 2

lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# 示例输入 (batch_size, seq_length, input_size)
batch_size = 5
seq_length = 15
dummy_input = torch.randn(batch_size, seq_length, input_size)

# 前向传播需要初始隐藏状态和单元状态 (h_0, c_0)
# 如果不提供，它们默认为零。
# 形状： (num_layers * num_directions, batch_size, hidden_size)
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

output, (hn, cn) = lstm_layer(dummy_input, (h0, c0))

# output 形状： (batch_size, seq_length, hidden_size)
# hn 形状： (num_layers, batch_size, hidden_size) - 每个层的最终隐藏状态
# cn 形状： (num_layers, batch_size, hidden_size) - 每个层的最终单元状态
print("LSTM Output shape:", output.shape)
print("LSTM Final Hidden State shape:", hn.shape)
print("LSTM Final Cell State shape:", cn.shape)
```

### 门控循环单元 (GRU)

GRU，由 Cho 等人于 2014 年提出，是新一代的门控循环单元，它简化了 LSTM 架构。它们也旨在解决梯度消失问题并捕获长期依赖关系，但通过稍微不同且计算量较小的结构实现这一点。

GRU 将单元状态和隐藏状态合并为一个隐藏状态 ($h_t$)。它们只使用两个门：

1. **重置门：** 决定在提出新的候选隐藏状态时，要遗忘多少前一个隐藏状态 ($h_{t-1}$) 的信息。
2. **更新门：** 决定保留多少前一个隐藏状态 ($h_{t-1}$) 的信息，以及将多少新的候选隐藏状态的信息并入最终隐藏状态 ($h_t$)。

由于门更少且没有独立的单元状态，GRU 在相同隐藏大小时比 LSTM 具有更少的参数 (parameter)。这可以使它们训练更快，并且在较小数据集上可能更不容易过拟合 (overfitting)，同时在许多任务上通常能达到与 LSTM 相当的性能。

PyTorch 提供了 `torch.nn.GRU` 层，其用法与 `nn.RNN` 和 `nn.LSTM` 遵循相同的模式。

```python
# 示例：定义一个 GRU 层
gru_layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

# 前向传播需要初始隐藏状态 (h_0)
# 如果不提供，它默认为零。
# 形状： (num_layers * num_directions, batch_size, hidden_size)
h0_gru = torch.randn(num_layers, batch_size, hidden_size)

output_gru, hn_gru = gru_layer(dummy_input, h0_gru)

# output 形状： (batch_size, seq_length, hidden_size)
# hn 形状： (num_layers, batch_size, hidden_size) - 每个层的最终隐藏状态
print("\nGRU Output shape:", output_gru.shape)
print("GRU Final Hidden State shape:", hn_gru.shape)
```

实际上，在处理需要长期依赖的序列数据时，LSTM 和 GRU 都被广泛用于替代简单 RNN。LSTM 和 GRU 之间的选择通常取决于对特定任务和数据集的经验评估，尽管当计算资源或训练时间受限时，GRU 可能因其更简单的结构而更受青睐。PyTorch 使得对两者进行实验变得简单。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）网络架构的原始研究论文，阐述了其旨在克服RNN中梯度消失问题的设计。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179.pdf) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  本文介绍了门控循环单元（GRU），作为机器翻译中RNN编码器-解码器的一部分，将其作为LSTM的简化替代方案。
- [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — PyTorch Documentation (2024)
  Publisher: PyTorch
  PyTorch中`torch.nn.LSTM`模块的官方文档，详细说明了其API、参数、输入和输出，以供实际应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的教科书，其中第10章提供了循环神经网络（包括LSTM和GRU）及其理论背景的详细解释。
