---
course: "how-to-build-a-large-language-model"
chapter: "revisiting-sequence-processing-architectures"
lesson: "long-short-term-memory-lstm"
sourceId: 5959
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures/long-short-term-memory-lstm"
title: "长短期记忆（LSTM）网络"
description: "详细描述LSTM中的门控机制（输入、遗忘、输出）。"
order: 3
plots: []
sourceHash: "26daade675cd72a3f86e29a1bb25fb8632cf5c6b0f7b29d0924178c8bce33556"
sourceCorrections: []
---

简单循环神经网络 (neural network) (RNN)（RNNs）在学习长序列依赖关系时面临困难。反向传播 (backpropagation)过程中涉及时间上的重复乘法，导致梯度要么趋于零（消失），要么不可控地增长（爆炸）。梯度消失问题尤其棘手，因为它阻碍了网络学习序列中相距较远事件之间的关联。

长短期记忆（LSTM）网络专门用于解决这些问题。它由Hochreiter和Schmidhuber于1997年提出，比简单RNN包含更复杂的内部结构，其特点是具有一个专门的*细胞状态*和*门*，用于调节信息流动。这种架构使LSTM能够长时间保持信息，有效捕获长距离依赖关系。

### 核心思想：细胞状态和门

可以将标准RNN的隐藏状态$h_t$看作一种工作记忆，它在每个时间步都会被覆盖。LSTM引入了一个额外组件，即*细胞状态* $C_t$，它作用类似于传送带或记忆高速公路。信息可以在这条高速公路上相对不变地流动，从而更容易长时间保留上下文 (context)。

LSTM的主要创新在于它们能够使用称为*门*的结构，选择性地从细胞状态中添加或移除信息。门由一个S形（sigmoid）神经网络 (neural network)层和一个逐点乘法操作组成。S形层输出0到1之间的数字，描述每个组件应允许通过多少信息。0表示“不让任何信息通过”，而1表示“让所有信息通过”。

一个LSTM细胞通常有三个主要门：

1. **遗忘门：** 决定从细胞状态中丢弃哪些信息。
2. **输入门：** 决定将哪些新信息存储到细胞状态中。
3. **输出门：** 决定细胞状态的哪一部分作为输出。

我们来考察这些组件在单个时间步$t$如何协同工作，给定输入$x_t$、前一个隐藏状态$h_{t-1}$和前一个细胞状态$C_{t-1}$。

### 遗忘门 ($f_t$)

第一步是决定从前一个细胞状态$C_{t-1}$中丢弃哪些信息。这个决定由遗忘门层做出。它查看$h_{t-1}$和$x_t$，并为细胞状态$C_{t-1}$中的每个数字输出一个0到1之间的数值。


$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$


这里，$\sigma$是S形激活函数 (activation function)，$W_f$表示权重 (weight)，$b_f$表示遗忘门的偏置 (bias)。表示法$[h_{t-1}, x_t]$表明前一个隐藏状态和当前输入是拼接在一起的。

### 输入门 ($i_t$) 和候选细胞状态 ($\tilde{C}_t$)

接下来，我们需要决定将哪些新信息存储到细胞状态中。这包含两部分。首先，一个*输入门层*（另一个S形层）决定我们要更新哪些值。


$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$


其次，一个`tanh`层创建了一个新的候选值向量 (vector)$\tilde{C}_t$，这些值可以被添加到状态中。


$$
\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)
$$


### 更新细胞状态 ($C_t$)

现在，我们将旧的细胞状态$C_{t-1}$更新为新的细胞状态$C_t$。我们将旧状态乘以$f_t$，遗忘我们之前决定遗忘的信息。然后我们加上$i_t * \tilde{C}_t$。这是新的候选信息，根据我们决定更新每个状态值的程度进行缩放。


$$
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$


这里使用加法非常重要。与简单RNN中的重复乘法不同，这种加法作用使得梯度更容易在时间上向后流动，而不会那么快消失。

### 输出门 ($o_t$) 和隐藏状态 ($h_t$)

最后，我们需要决定输出什么。这个输出将基于我们的细胞状态，但会是一个过滤后的版本。首先，我们运行一个S形层，它决定了我们将输出细胞状态的哪些部分。


$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$


然后，我们将细胞状态通过`tanh`函数（将值推到-1到1之间），并将其乘以S形门的输出，这样我们只输出我们决定输出的部分。这个最终输出就是隐藏状态$h_t$。


$$
h_t = o_t * \tanh(C_t)
$$


隐藏状态$h_t$随后与新的细胞状态$C_t$一起传递给下一个时间步。

> LSTM细胞在时间步$t$的信息流动简化视图。细胞状态$C_t$充当传送带，由遗忘门和输入门修改。输出门过滤细胞状态以生成隐藏状态$h_t$。

### 在PyTorch中的实现

PyTorch等深度学习 (deep learning)框架提供了LSTM层的有效实现。使用`torch.nn.LSTM`可以抽象掉门的计算细节。

```python
import torch
import torch.nn as nn

# 示例参数
input_size = 10   # 输入特征 x_t 的维度
hidden_size = 20  # 隐藏状态 h_t 和细胞状态 C_t 的维度
num_layers = 1    # 堆叠LSTM层的数量
batch_size = 5
seq_len = 7       # 输入序列的长度

# 创建一个LSTM层
lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

# 模拟输入数据 (batch_size, seq_len, input_size)
input_seq = torch.randn(batch_size, seq_len, input_size)

# 初始隐藏状态和细胞状态 (num_layers, batch_size, hidden_size)
# 如果未提供，则默认为零。
h0 = torch.randn(num_layers, batch_size, hidden_size)
c0 = torch.randn(num_layers, batch_size, hidden_size)

# 前向传播
# output 包含序列中每个时间步的隐藏状态 h_t
# hn 包含最后一个时间步的最终隐藏状态
# cn 包含最后一个时间步的最终细胞状态
output, (hn, cn) = lstm_layer(input_seq, (h0, c0))

print("输入形状:", input_seq.shape)
print("输出形状（所有隐藏状态）:", output.shape)
print("最终隐藏状态形状 (hn):", hn.shape)
print("最终细胞状态形状 (cn):", cn.shape)

# --- 输出 ---
# Input shape: torch.Size([5, 7, 10])
# Output shape (all hidden states): torch.Size([5, 7, 20])
# Final hidden state shape (hn): torch.Size([1, 5, 20])
# Final cell state shape (cn): torch.Size([1, 5, 20])
```

请注意，隐藏状态和细胞状态以元组`(hn, cn)`的形式返回。这反映了LSTM在整个序列处理过程中维护的两个不同内部状态。`output`张量提供每个时间步的隐藏状态$h_t$，这在序列到序列模型中通常很有用。

通过使用门来控制信息流动以及细胞状态的加法更新机制，LSTM有效缓解了梯度消失问题，与简单RNN相比，能够学习跨越更长时间范围的依赖关系。这种能力使它们在基于注意力模型兴起之前，成为许多自然语言处理任务中的主导架构。虽然我们接下来将研究的门控循环单元 (GRU)（GRUs）提供了一种稍微简单的门控机制，但LSTM引入的核心原则仍然具有影响力。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆 (LSTM) 架构的原始研究论文，详细阐述了其单元状态和门控机制，旨在克服 RNN 中的梯度消失问题。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教材，涵盖了深度学习的理论基础和实践方面，其中包含关于循环神经网络和 LSTM 的专门章节。
- [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — PyTorch Authors (2024)
  PyTorch 官方 LSTM 模块文档，提供了 API 细节、参数以及在深度学习项目中实际实现的用法示例。
