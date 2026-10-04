---
course: "how-to-build-a-large-language-model"
chapter: "revisiting-sequence-processing-architectures"
lesson: "sequence-to-sequence-models-rnns"
sourceId: 5961
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures/sequence-to-sequence-models-rnns"
title: "基于RNN的序列到序列模型"
description: "回顾使用循环架构的编码器-解码器框架。"
order: 5
plots: []
sourceHash: "4d8b1677e86ba0d1184864f5dc3f108f4bdd9f7eef76e5e3234322fbe111854d"
sourceCorrections: []
---

“循环神经网络 (neural network)（RNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）按元素处理序列，但许多问题需要将一个长度的输入序列映射到可能不同长度的输出序列。例如，机器翻译（将法语句子翻译成英语）或文本摘要（将长文章压缩成几句话）。输入和输出的长度通常没有直接关系。标准RNN架构通常为每个输入产生一个输出，因此不直接适用于这些任务。”

为解决此问题，序列到序列（seq2seq）框架应运而生，主要采用长短期记忆网络（LSTM）或门控循环单元（GRU）等循环架构。其核心思想是使用两个独立的循环神经网络：一个处理输入序列（编码器），另一个生成输出序列（解码器）。

### 编码器-解码器架构

seq2seq模型包含两个主要组成部分：

1. **编码器**：这个循环神经网络 (neural network) (RNN)逐个读取输入序列的令牌（例如，单词或子词 (subword)）。它的目标不是在每一步都生成输出，而是将整个输入序列的信息压缩成一个固定大小的向量 (vector)表示。这个向量常被称为“上下文 (context)向量”或“思维向量”，通常由编码器RNN的最终隐藏状态（以及长短期记忆网络 (LSTM)的细胞状态）表示。
2. **解码器**：这个循环神经网络将编码器生成的上下文向量作为其初始隐藏状态。然后，它逐个令牌地生成输出序列。在每一步$t$，解码器接收上下文向量、它自己的前一个隐藏状态$h_{t-1}$，以及前面生成的输出令牌$y_{t-1}$作为输入，以生成下一个输出令牌$y_t$并将其隐藏状态更新为$h_t$。生成过程通常以一个特殊的序列开始符`<SOS>`令牌开始，并持续到生成序列结束符`<EOS>`令牌或达到最大长度为止。

> 基于RNN的序列到序列模型的高层结构。编码器处理输入以生成上下文向量，该向量用于初始化解码器以生成输出序列。

### 信息流与上下文 (context)向量 (vector)

编码器处理输入序列$X = (x_1, x_2, ..., x_n)$并输出一个上下文向量$c$。这个向量$c$旨在总结整个输入序列。


$$
c = \text{编码器}(x_1, x_2, ..., x_n)
$$


通常，对于长短期记忆网络 (LSTM)， $c$ 将是最终的隐藏状态 $h_n$ 和细胞状态 $C_n$。

解码器用这个上下文进行初始化（例如，$h_0^{\text{dec}} = h_n^{\text{enc}}$，$C_0^{\text{dec}} = C_n^{\text{enc}}$）。然后，它一次生成一个令牌，产生输出序列$Y = (y_1, y_2, ..., y_m)$。下一个令牌$y_t$的概率取决于上下文$c$、前一个令牌$y_{t-1}$以及解码器当前的隐藏状态$h_t^{\text{dec}}$：


$$
P(y_t | y_1, ..., y_{t-1}, c) = \text{解码器}(y_{t-1}, h_{t-1}^{\text{dec}}, c)
$$


解码器的第一个输入通常是一个特殊的`<SOS>`令牌（$y_0 = \text{<SOS>}$）。

### PyTorch实现概述

我们来概述使用PyTorch `nn.LSTM`的简化编码器和解码器模块。

```python
import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 注意：假设 input_size 是嵌入维度
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )

    def forward(self, input_seq):
        # input_seq 形状：(batch_size, seq_length)
        embedded = self.embedding(input_seq)
        # embedded 形状：(batch_size, seq_length, hidden_size)

        # 初始化隐藏状态（为简化起见未显示，
        # 默认为零）
        # hidden = self.init_hidden(batch_size)

        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs 形状：(batch_size, seq_length, hidden_size)
        # hidden 形状：(num_layers, batch_size, hidden_size)
        # cell 形状：(num_layers, batch_size, hidden_size)

        # 我们通常使用最终的隐藏状态和细胞状态作为上下文
        return hidden, cell

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 注意：output_size 是目标语言的词汇表大小
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1) # 经常与NLLLoss一起使用

    def forward(self, input_token, hidden, cell):
        # input_token 形状：(batch_size, 1) -> 单个令牌
        # hidden 形状：(num_layers, batch_size, hidden_size)
        # cell 形状：(num_layers, batch_size, hidden_size)

        embedded = self.embedding(input_token)
        # embedded 形状：(batch_size, 1, hidden_size)

        # 上下文向量（编码器最终的隐藏/细胞状态）
        # 在此处作为初始隐藏/细胞状态传入。
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output 形状：(batch_size, 1, hidden_size)

        # 将输出重塑为 (batch_size, hidden_size) 以用于
        # 线性层
        output = output.squeeze(1)
        output = self.out(output)
        # output 形状：(batch_size, output_size)

        # 可选：应用 softmax 以获得
        # 概率/对数概率
        # output = self.softmax(output)

        return output, hidden, cell

# 示例用法
# input_vocab_size = 10000
# output_vocab_size = 12000
# hidden_dim = 256
# n_layers = 2
# batch_size = 32
# input_length = 50

# encoder = EncoderRNN(input_vocab_size, hidden_dim, n_layers)
# decoder = DecoderRNN(hidden_dim, output_vocab_size, n_layers)

# 示例输入批次（索引）
# input_tensor = torch.randint(
#     0, input_vocab_size, (batch_size, input_length)
# )

# 传入编码器
# encoder_hidden, encoder_cell = encoder(input_tensor)

# 解码器输入以 <SOS> 令牌开始（假设索引为 0）
# decoder_input = torch.full((batch_size, 1), 0, dtype=torch.long)
# decoder_hidden = encoder_hidden # 使用编码器最终的隐藏状态
# decoder_cell = encoder_cell   # 使用编码器最终的细胞状态

# 逐步生成输出序列（简化循环）
# max_target_length = 60
# all_decoder_outputs = []
# for _ in range(max_target_length):
#     decoder_output, decoder_hidden, decoder_cell = decoder(
#         decoder_input, decoder_hidden, decoder_cell
#     )
#     all_decoder_outputs.append(decoder_output)
#
#     # 获取最有可能的下一个令牌（贪婪解码）
#     _, top_idx = decoder_output.topk(1)
#     # 使用预测的令牌作为下一个输入
#     decoder_input = top_idx.detach()
```

### 局限性与后续步骤

使用循环神经网络 (neural network) (RNN)的标准编码器-解码器架构在许多任务中被证明是有效的。然而，它依赖于将*整个*输入序列压缩成一个单一的固定大小的上下文 (context)向量 (vector)。这会产生一个信息瓶颈，尤其对于长输入序列而言问题更突出。当模型生成输出序列的末尾时，它很难记住长输入开头部分的细节。

这一局限性是注意力机制 (attention mechanism)发展的重要推动力。注意力机制让解码器在输出生成过程的每一步，能够选择性地关注输入序列的不同部分，而不是仅仅依赖于单一的上下文向量。这种回顾源输入相关部分的能力大幅提升了机器翻译等任务的性能，并为我们将在下一章中介绍的Transformer架构打下了基础。

## 参考资料

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); DOI: [10.48550/arXiv.1409.3215](https://doi.org/10.48550/arXiv.1409.3215)
  一篇基础性论文，介绍了使用长短期记忆网络（LSTMs）进行机器翻译的序列到序列模型，展示了其将不同长度输入序列映射到输出序列的能力。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Pages: 1724-1734; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  这篇论文同时提出了RNN编码器-解码器框架，展示了其在机器翻译中的应用以及对门控循环单元（GRUs）的使用，这些内容是本节讨论的核心。
- [Long Short-Term Memory](https://www.researchgate.net/publication/13853244_Long_Short-Term_Memory) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇原始论文介绍了长短期记忆（LSTM）网络，其被明确提及为序列到序列框架中使用的核心循环架构。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第10章对循环神经网络、长短期记忆网络（LSTMs）以及编码器-解码器框架提供了理论解释，为所涵盖的模型提供了有益的背景信息。
