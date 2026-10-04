---
course: "how-to-build-a-large-language-model"
chapter: "transformer-architecture"
lesson: "encoder-decoder-stacks"
sourceId: 5966
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-4-transformer-architecture/encoder-decoder-stacks"
title: "编码器与解码器堆叠"
description: "详细说明编码器和解码器模块内部的层（注意力、前馈、归一化）。"
order: 5
plots: []
sourceHash: "93345b0f1de10d455fb21416358965ecf31c643972a87349a1044b0ef2ac5a3a"
sourceCorrections: []
---

编码器和解码器堆叠是Transformer架构的主要组成部分，它们通过组合注意力机制 (attention mechanism)和位置编码 (positional encoding)构建而成。原始的Transformer模型为编码器和解码器都提出了N=6个相同层的堆叠，尽管现代的大型语言模型通常采用更多的层。

## 编码器堆叠

编码器的作用是处理输入序列并生成一系列包含丰富上下文 (context)信息的表示。编码器堆叠中的每个层接收一系列嵌入 (embedding)向量 (vector)（或来自上一层的输出）并对其进行变换。一个单独的编码器层由两个主要子层构成：

1. **多头自注意力 (self-attention):** 该机制允许输入序列中的每个位置关注*上一层输出*序列中的所有位置（包括自身）。它根据序列不同部分之间的关联程度计算加权表示。
2. **逐位置前馈网络 (FFN):** 这是一个简单、全连接的前馈网络，独立应用于每个位置。它通常由两个线性变换和一个非线性激活函数 (activation function)（通常是ReLU或GeLU）组成。其目的是进一步处理注意力机制 (attention mechanism)生成的表示。

重要的是，在两个子层周围都使用了残差连接，随后是层归一化 (normalization)。如果 $x$ 是子层（例如，多头注意力 (multi-head attention)或FFN）的输入，并且 $\text{子层}(x)$ 是子层自身实现的函数，则子层块的输出是 $\text{层归一化}(x + \text{子层}(x))$。这种结构通过促进梯度流动和稳定激活，有助于训练更深的模型。

> 单个Transformer编码器层内部的流程。

一个编码器层的输出作为下一个相同编码器层的输入。这种堆叠方式使得模型能够逐步构建输入序列的更复杂表示，在不同层面捕获依赖关系。

以下是编码器层结构的一个简化PyTorch表示：

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(), # 或 GeLU, SwiGLU 等
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # --- 自注意力块 ---
        # Q, K, V 都来自输入 'src'
        attn_output, _ = self.self_attn(
            src,
            src,
            src,
            key_padding_mask=src_mask,
            need_weights=False
        )
        # 残差连接和层归一化
        src = self.norm1(src + self.dropout(attn_output))

        # --- 前馈块 ---
        ff_output = self.feed_forward(src)
        # 残差连接和层归一化
        src = self.norm2(src + self.dropout(ff_output))

        return src

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # 假设输入嵌入（包括位置编码）
        # 在外部处理

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src # 最终输出表示
```

整个编码器堆叠的最终输出是一系列向量，每个输入令牌对应一个，旨在捕获该令牌在序列中的上下文含义。此输出通常由解码器使用。

## 解码器堆叠

解码器的作用通常是基于编码后的输入序列和迄今已生成的令牌，一次生成一个令牌的输出序列。与编码器类似，解码器也由N个相同的层堆叠构成。然而，每个解码器层有*三个*主要子层：

1. **带掩码的多头自注意力 (self-attention):** 这与编码器的自注意力运作方式相似，但有一个重要区别：它包含掩码机制。该掩码确保在预测位置 $i$ 的输出令牌时，自注意力机制 (attention mechanism)只能关注解码器输入序列中位置小于或等于 $i$ 的部分。它不能“向前看”未来的令牌，从而保留了生成所需的自回归 (autoregressive)属性。
2. **多头编码器-解码器注意力:** 这是解码器与编码器输出交互的地方。查询 ($Q$) 来自前一个解码器子层（即带掩码的自注意力）的输出。键 ($K$) 和值 ($V$) 来自*编码器堆叠*的最终输出。这使得解码器中的每个位置都能关注输入序列中的所有位置，整合来自源端的关联上下文 (context)。
3. **逐位置前馈网络 (FFN):** 结构与编码器层中使用的相同，在编码器-解码器注意力之后独立应用于每个位置。

与编码器类似，在这三个子层之后都应用了残差连接和层归一化 (normalization)： $\text{层归一化}(x + \text{子层}(x))$。

> 单个Transformer解码器层内部的流程，突出了三个子层和输入。

解码器层的一个简化PyTorch结构如下：

```python
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(), # 或 GeLU, SwiGLU 等
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # --- 带掩码的自注意力块 ---
        # Q, K, V 都来自目标输入 'tgt'
        # 'tgt_mask' 防止关注未来的位置
        attn_output, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            need_weights=False
        )
        # 残差连接和层归一化
        tgt = self.norm1(tgt + self.dropout(attn_output))

        # --- 编码器-解码器注意力块 ---
        # Q 来自前一个解码器层 ('tgt')，K & V 来自编码器输出 ('memory')
        # 'memory_mask' (可选) 遮蔽源序列中的填充
        attn_output, _ = self.encoder_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_mask,
            need_weights=False
        )
        # 残差连接和层归一化
        tgt = self.norm2(tgt + self.dropout(attn_output))

        # --- 前馈块 ---
        ff_output = self.feed_forward(tgt)
        # 残差连接和层归一化
        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # 假设目标嵌入（包括位置编码）在外部处理

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt # 在线性层/softmax之前的最终输出表示
```

## 最终输出生成

输入通过整个解码器堆叠后，产生的向量 (vector)序列表示预测的输出令牌。为了将这些向量转换为目标词汇表 (vocabulary)上的概率，通常应用一个最终线性层，然后是softmax函数。线性层将解码器输出向量（维度为 $d_{model}$）投射到词汇表大小 ($V$) 的空间。softmax函数将这些得分（logits）转换为概率，表示词汇表中每个词成为序列中下一个令牌的可能性。


$$
P(y_i | y_{<i}, x) = \text{softmax}(\text{线性}(\text{解码器输出}_i))
$$


编码器堆叠（处理输入）与解码器堆叠（根据输入和之前输出生成输出）的组合构成了完整的Transformer架构，能够处理各种序列到序列的任务。理解这些堆叠如何由注意力层、前馈层以及归一化 (normalization)和残差连接构成，对于构建和修改这些高效模型非常重要。

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  这篇奠基性论文介绍了Transformer架构，详细阐述了编码器和解码器堆栈、多头注意力机制、残差连接和层归一化。
- [Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) — Lewis, Leandro, Thomas (2022)
  Publisher: O'Reilly Media
  一本实践指南，深入涵盖了Transformer架构，包括其组件、训练和针对各种NLP任务的微调。它提供了应用Transformer的现代视角。
- [Stanford CS224N: Natural Language Processing with Deep Learning, Winter 2023](http://web.stanford.edu/class/cs224n/archive/winter2023/) — Christopher Manning, Abigail See, and Kevin Clark (2023)
  Publisher: Stanford University
  这门研究生级别的课程提供了深度学习模型在NLP中应用的详细解释，其中包含专门讲解Transformer架构、注意力机制及其实现的讲座。
