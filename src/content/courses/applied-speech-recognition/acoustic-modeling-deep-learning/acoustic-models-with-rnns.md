---
course: "applied-speech-recognition"
chapter: "acoustic-modeling-deep-learning"
lesson: "acoustic-models-with-rnns"
sourceId: 7124
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-3-acoustic-modeling-deep-learning/acoustic-models-with-rnns"
title: "使用循环神经网络构建声学模型"
description: "了解RNN的序列特性如何使其适合处理语音特征序列。"
order: 2
plots: []
sourceHash: "c645b033b57bc30be5b02aeb5ecd8a78ed8ca1c024f49a5c848c88035c907461"
sourceCorrections: []
---

将原始音频转换为特征向量 (vector)序列后，紧接着的问题是如何对该序列中的时间关系进行建模。标准的*前馈神经网络 (neural network)*独立处理每个输入向量，忽略其在时间上的位置。这对于语音来说是不够的，因为声音的顺序决定了单词和含义。例如，“cat”和“act”中的音素是相同的，但它们的顺序决定了它们的区别。为了解决此问题，我们采用一类专门为序列数据设计的模型：*循环神经网络 (RNN)（RNNs）*。

### 时间序列数据的循环机制

RNN特别适用于时间序列数据，因为它具有一种记忆形式。它一次处理序列中的一个元素，并且在每一步中，其计算都包含来自上一步的信息。这是通过*隐藏状态*实现的，隐藏状态充当到目前为止已处理序列的概括。

简单RNN在每个时间步 $t$ 的核心运算可以通过以下方程描述：


$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$


这里：

- $x_t$ 是当前时间步的输入特征向量 (vector)（例如，一个音频帧的MFCCs集）。
- $h_{t-1}$ 是来自前一时间步的隐藏状态。它承载着过去的信息。对于第一个时间步（$t=1$），$h_0$ 通常被初始化为零向量。
- $W_{xh}$ 和 $W_{hh}$ 是网络在训练期间学习的权重 (weight)矩阵。$W_{xh}$ 用于转换当前输入，$W_{hh}$ 用于转换前一个隐藏状态。
- $b_h$ 是一个偏置 (bias)项。
- $\tanh$ 是双曲正切激活函数 (activation function)，它将输出压缩到[-1, 1]的范围。
- $h_t$ 是新的隐藏状态，它同时捕获了当前输入和来自过去的上下文 (context)。这个隐藏状态随后被传递到下一个时间步。

为了生成预测，每一步的隐藏状态会通过另一个层，通常是带有softmax激活的全连接层，来产生一个输出向量 $o_t$：


$$
o_t = \text{softmax}(W_{ho} h_t + b_o)
$$


这个输出 $o_t$ 表示在那个特定时间步上，我们目标词汇（例如，所有字符'a'-'z'、'0'-'9'和特殊符号）的概率分布。下面的图示了这一过程，展示了网络在时间上的“展开”。

> RNN一步步地处理输入序列（$x_1, x_2, ...$）。在每一步，它会产生一个输出（$o_1, o_2, ...$）并更新其隐藏状态（$h_1, h_2, ...$），隐藏状态将上下文传递到下一步。

### 将RNN应用于声学模型

在我们的ASR管线中，从音频中提取的特征向量 (vector)序列作为RNN的输入序列 $X = (x_1, x_2, ..., x_T)$。

- **输入**：每个 $x_t$ 是一个代表音频短时间片段的向量，例如40个MFCCs或对数梅尔频谱图中的一列。
- **处理**：RNN从 $t=1$ 到 $T$ 遍历这些特征向量。隐藏状态 $h_t$ 学习表示声学特征，有效地概括了到目前为止听到的声音。例如，在处理多个帧之后，隐藏状态可能编码了像/s/这样的摩擦音向元音过渡的存在。
- **输出**：在每个时间步 $t$，模型会产生一个输出向量 $o_t$，其中包含我们词汇表 (vocabulary)中各元素的概率。如果我们的词汇表包含26个字母、一个空格和一个撇号，那么每个时间步的输出向量将有28个值。最高的值表示模型认为在该精确音频帧中最有可能的字符。

这一过程导致输出的概率分布序列与输入特征序列（$T$）长度相同。这提出了一个问题：一个10秒的音频片段可能产生1000个特征向量，从而有1000个输出预测，而对应的文本转录可能只有50个字符长。这种长度上的不匹配是一个基本问题，我们将使用*连接主义时间分类（CTC）损失函数 (loss function)*来解决它，本章稍后将对此进行介绍。

### PyTorch中的实践视角

在现代深度学习 (deep learning)框架中，实现一个基本RNN层非常直接。在PyTorch中，您可以像这样定义一个简单的基于RNN的声学模型：

```python
import torch
import torch.nn as nn

class SimpleASR_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleASR_RNN, self).__init__()
        self.hidden_size = hidden_size
        # RNN层处理输入特征序列
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 一个全连接层，将隐藏状态映射到字符概率
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化第一个时间步的隐藏状态
        # 形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # 将输入序列和初始隐藏状态传递给RNN
        # out: 包含每个时间步的输出隐藏状态
        # hidden: 包含序列的最终隐藏状态
        out, hidden = self.rnn(x, h0)
        
        # 将RNN的输出通过全连接层
        # 以获得每个时间步的预测
        out = self.fc(out)
        return out

# 示例用法:
# num_features = 40 (例如, 40个MFCCs)
# rnn_hidden_size = 256
# num_output_classes = 29 (例如, 26个字母 + 空格 + 撇号 + 空白符号)

# model = SimpleASR_RNN(num_features, rnn_hidden_size, num_output_classes)
```

在这段代码中：

- `input_size` 对应于每个输入向量 (vector) $x_t$ 中的特征数量（例如，40）。
- `hidden_size` 是隐藏状态向量 $h_t$ 的维度。这是一个可以调整的超参数 (parameter) (hyperparameter)。
- `num_classes` 是输出词汇表 (vocabulary)的大小。

### 简单RNN的局限性

虽然RNN是一个不错的起点，但简单的RNN在处理长序列时存在困难。早期时间步输入的影响会逐渐减弱，随着网络对序列的持续处理。这就是所谓的**梯度消失问题**，即用于更新网络权重 (weight)的梯度在长距离上变得极其微小，使得模型难以学习长程依赖。在语音识别中，这意味着模型可能在处理完长句的末尾时，已经忘记了句子的开头。

为了克服这一重要限制，人们开发了更复杂的循环架构。下一节将介绍\*长短期记忆（LSTM）*和*门控循环单元（GRU）\*网络，它们旨在更好地捕获和保持长序列中的上下文 (context)。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10: Sequence Modeling: Recurrent and Recursive Nets
  提供了循环神经网络的基础论述，包括其架构、训练和梯度消失问题。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）网络，这是学习序列数据中长期依赖关系的重要进展，解决了梯度消失问题。
- [RNN - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch Developers (2024)
  Publisher: PyTorch
  PyTorch `torch.nn.RNN` 模块的官方文档，详细说明了其参数和用于构建循环层的用法。
