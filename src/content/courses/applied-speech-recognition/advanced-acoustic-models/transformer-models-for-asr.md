---
course: "applied-speech-recognition"
chapter: "advanced-acoustic-models"
lesson: "transformer-models-for-asr"
sourceId: 7141
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models/transformer-models-for-asr"
title: "自动语音识别中的 Transformer 模型概述"
description: "调整 Transformer 架构及其自注意力机制，用于语音识别任务。"
order: 4
plots: []
sourceHash: "752aedb08f0c56d0c3cf6e57d78e7c8b1574252c2c72d37de94b846ca59a12a0"
sourceCorrections: []
---

Transformer 架构已显著推动了语音识别系统。循环神经网络 (neural network) (RNN) (RNNs) 按顺序处理数据，一次一个时间步，这会导致计算速度慢，并且在捕获非常长的音频序列中的依赖关系时通常表现不佳。为了克服这些局限性，Transformer 架构提供了一种替代方法。它最初用于机器翻译，完全移除了循环机制，转而仅仅依赖注意力机制 (attention mechanism)。

Transformer 的核心思想是**自注意力 (self-attention)**。与 LAS 模型中的注意力机制不同，LAS 模型衡量编码器状态（音频）相对于当前解码器状态（文本）的重要性，而自注意力机制允许模型衡量*同一序列内*所有其他元素的重要性。对于 ASR 模型，这意味着在处理某个音频帧时，自注意力机制可以查看整个音频片段，以确定哪些其他帧对于构建该特定帧的丰富表示最为相关。这种对所有输入位置对之间关系进行建模的能力，使得模型能够比 RNN 更有效地捕获长距离语境。

### 适用于语音的 Transformer 模型

标准 Transformer 架构由一个编码器和一个解码器组成，两者均由多个相同层构成。对于 ASR，此结构经过调整以处理音频特征作为输入并生成文本作为输出。

> 自动语音识别用 Transformer 模型的高级视图。

各部分在语音处理中的协作方式如下：

1. **输入特征**：模型接收一系列音频特征向量 (vector)，如对数梅尔频谱帧，作为其输入。
2. **位置编码 (positional encoding)**：由于自注意力 (self-attention)机制 (attention mechanism)同时处理所有输入，它没有固有的序列顺序感。为解决此问题，我们向输入特征添加*位置编码*。这些是向量，为模型提供关于每个音频帧位置或时间戳的信息。
3. **编码器**：编码器用于创建输入音频的强大上下文 (context)表示。它由一系列相同层组成，每层包含一个多头自注意力模块和一个简单的前馈神经网络 (neural network)。自注意力模块帮助模型辨别音频中哪些部分对于理解其他部分最为重要。例如，它能学会将某个辅音的声学特征与后续元音跨越多个帧进行关联。
4. **解码器**：解码器是自回归 (autoregressive)的，意味着它一次生成一个标记 (token)（例如，一个字符或一个词片段）的输出文本。它也由一系列层组成。每个解码器层有三个主要组成部分：
   - **掩码自注意力**：此模块使得解码器能够关注输出序列中先前生成的标记。“掩码”防止其在训练时提前查看未来标记，这会是训练中的作弊行为。
   - **编码器-解码器注意力**：这类似于 Seq2Seq 模型中的注意力。它使解码器在预测下一个标记时，能够聚焦于编码器输出（上下文感知的音频表示）中最相关部分。
   - **前馈网络**：与编码器中一样，此网络提供额外的处理能力。

### 优点与挑战

Transformer 的主要优点是其可并行化能力。由于没有循环连接，编码器中所有时间步的计算可以同时进行，这显著加快了在 GPU 和 TPU 等现代硬件上的训练速度。这种并行特性，加上出色的长距离上下文 (context)建模能力，使得在许多 ASR 任务中获得了先进成绩。

然而，Transformer 模型也面临挑战。它们的计算开销大，自注意力 (self-attention)的开销随输入序列长度的平方增长（$O(T^2)$，其中 $T$ 是帧的数量）。由于音频输入可能非常长，这会是一个严重的局限。它们也需要大量数据，通常需要非常大的数据集才能从头开始有效训练。

### 实现方式概览

虽然构建一个完整的 Transformer 超出本节范围，但了解核心组件在 PyTorch 等深度学习 (deep learning)框架中的提供方式很有帮助。您无需从零开始构建自注意力 (self-attention)机制 (attention mechanism)。

```python
import torch
import torch.nn as nn

# 模型参数（示例值）
feature_size = 80  # 用于对数梅尔频谱图
nhead = 8          # 注意力头的数量
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1

# 从 PyTorch 实例化一个标准 Transformer 模型
transformer_model = nn.Transformer(
    d_model=feature_size,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True  # 对于 ASR 数据形状很重要
)

# 示例输入形状
# src = 源音频张量 (批量, 序列长度, 特征)
# tgt = 目标文本张量 (批量, 目标长度, 特征)
src = torch.rand((32, 500, feature_size))  # 32 个音频片段，每个长 500 帧
tgt = torch.rand((32, 50, feature_size))   # 32 个文本，每个长 50 个标记

# 模型返回解码器的输出
output = transformer_model(src, tgt)

print(f"Input audio shape: {src.shape}")
print(f"Input text shape: {tgt.shape}")
print(f"Output shape: {output.shape}")

# 预期输出：
# Input audio shape: torch.Size([32, 500, 80])
# Input text shape: torch.Size([32, 50, 80])
# Output shape: torch.Size([32, 50, 80])
```

> 此代码片段演示了如何在 PyTorch 中创建一个 `nn.Transformer` 模块。`d_model` 参数 (parameter)对应于输入的特征维度，您可以配置层数、注意力头数以及其他超参数 (hyperparameter)。

Transformer 架构为现代 ASR 系统提供了强大支持。下一节中，我们将考察 Conformer 模型，它通过重新引入卷积来增强 Transformer 模型，以更好地捕获局部音频模式。

## 参考资料

- [Deep Transformer Models for End-to-End Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1240.pdf) — Lihong Dong, Shuai Xu, and Bo Ren (2018)
  Journal: Proceedings of Interspeech 2018; Publisher: ISCA (International Speech Communication Association); Pages: 1840-1844; DOI: [10.21437/Interspeech.2018-1240](https://doi.org/10.21437/Interspeech.2018-1240)
  Transformer架构在端到端自动语音识别中的早期应用。
- [Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — PyTorch Core Team (2024)
  `torch.nn.Transformer`模块的官方文档，对实现细节有帮助。
- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本全面的教材，涵盖了自然语言处理和语音领域中的序列到序列模型和Transformer架构。
