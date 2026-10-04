---
course: "how-to-build-a-large-language-model"
chapter: "transformer-architecture"
lesson: "multi-head-attention-mechanism"
sourceId: 5964
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-4-transformer-architecture/multi-head-attention-mechanism"
title: "多头注意力机制"
description: "描述多头注意力如何处理不同的表示子空间。"
order: 3
plots: []
sourceHash: "f5e5c815623e607d0abf678000d9252fc7adbdc4343cf72914524ea557bebf58"
sourceCorrections: []
---

缩放点积注意力机制 (attention mechanism)允许模型在处理某个特定标记 (token)时衡量不同标记的重要性，但它只使用一组学到的查询($Q$)、键($K$)和值($V$)投影。这可能会限制模型同时识别多种关系或关注输入不同方面的能力。例如，一种注意力模式可能需要用来获取句法依赖关系，而另一种则侧重于长距离的语义相似性。

多头注意力 (multi-head attention)通过并行运行缩放点积注意力机制多次来解决这个问题，每次运行都使用自己学到的线性投影。每个并行运行被称为一个“注意力头”。这使得模型能够同时关注来自不同表示子空间、处于不同位置的信息。

### 机制

不同于执行一个使用维度为 $d_{model}$ 的键、值和查询的单一注意力函数，多头注意力 (multi-head attention)首先使用每个头不同的、学到的线性投影对查询、键和值进行 $h$ 次线性投影。假设输入的查询、键和值是矩阵 $Q$、$K$和$V$（在自注意力 (self-attention)层中，它们通常是同一个张量）。对于每个头 $i \in \{1, ..., h\}$，我们计算：


$$
\text{头}_i = \text{注意力}(QW^Q_i, KW^K_i, VW^V_i)
$$


这里的投影是参数 (parameter)矩阵：
$W^Q_i \in \mathbb{R}^{d_{model} \times d_k}$
$W^K_i \in \mathbb{R}^{d_{model} \times d_k}$
$W^V_i \in \mathbb{R}^{d_{model} \times d_v}$

这里的 $\text{注意力}$ 函数是前一节描述的缩放点积注意力。通常，每个头的维度设置为 $d_k = d_v = d_{model} / h$。这种划分确保了总计算成本与具有完整维度的单头注意力相似。

在并行计算每个头的注意力输出后，它们的输出（每个维度为 $d_v$）被拼接在一起：


$$
\text{拼接}(\text{头}_1, \text{头}_2, ..., \text{头}_h) \in \mathbb{R}^{\text{序列长度} \times (h \cdot d_v)}
$$


由于我们选择 $d_v = d_{model} / h$，拼接后的维度是 $h \cdot d_v = d_{model}$。这个拼接后的输出会经过一个最终的线性投影，其参数为 $W^O \in \mathbb{R}^{h d_v \times d_{model}}$（或 $\mathbb{R}^{d_{model} \times d_{model}}$），以生成多头注意力层的最终输出：


$$
\text{多头}(Q, K, V) = \text{拼接}(\text{头}_1, ..., \text{头}_h) W^O
$$


整个过程可以如下方所示：

> 输入的查询、键和值针对每个注意力头独立地进行线性投影。并行缩放点积注意力机制 (attention mechanism)的输出被拼接起来，然后经过一个最终的线性投影。

### PyTorch 中的实现概述

让我们看一下使用 PyTorch 的一个简化实现概述，以突出主要步骤。我们假设输入的张量 `query`、`key` 和 `value` 的形状为 `(batch_size, seq_len, d_model)`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, (
            "d_model 必须能被 num_heads 整除"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头的键/查询维度
        self.d_k = d_model // num_heads

        # 用于初始投影的线性层
        # (所有头的查询、值)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 拼接后的最终线性层
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(
        self, Q, K, V, mask=None
    ):
        # Q, K, V 形状: (batch_size, num_heads, seq_len, d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )
        # attn_scores 形状: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            # 应用掩码（例如，用于填充或解码器中的未来标记）
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        # attn_probs 形状: (batch_size, num_heads, seq_len, seq_len)

        output = torch.matmul(attn_probs, V)
        # output 形状: (batch_size, num_heads, seq_len, d_k)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1. 执行线性投影
        Q = self.W_q(query) # (batch_size, seq_len, d_model)
        K = self.W_k(key)   # (batch_size, seq_len, d_model)
        V = self.W_v(value) # (batch_size, seq_len, d_model)

        # 2. 为多头注意力重塑形状
        # (batch_size, seq_len, d_model) ->
        # (batch_size, seq_len, num_heads, d_k) ->
        # (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)
        K = K.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)
        V = V.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)

        # 3. 对每个头应用缩放点积注意力
        attn_output = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        # attn_output 形状: (batch_size, num_heads, seq_len, d_k)

        # 4. 拼接注意力头并应用最终线性层
        # 重塑回原形: (batch_size, num_heads, seq_len, d_k) ->
        # (batch_size, seq_len, num_heads, d_k) ->
        # (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attn_output) # (batch_size, seq_len, d_model)

        return output

# 示例用法
# d_model = 512
# num_heads = 8
# multihead_attn = MultiHeadAttention(d_model, num_heads)
# seq_len = 100
# batch_size = 32
# input_tensor = torch.randn(batch_size, seq_len, d_model) # 示例输入
# output = multihead_attn(
#     input_tensor, input_tensor, input_tensor
# ) # 自注意力
# print(output.shape) # 结果应为 torch.Size([32, 100, 512])
```

这个概述演示了输入的 Q、K、V 张量如何被投影并重塑，以实现跨头的并行计算。`transpose` 操作对于将头维度与批次维度分组非常重要，从而在 `scaled_dot_product_attention` 函数中实现高效的批量矩阵乘法。最后，输出被重塑回原形，并经过最终的输出投影 $W^O$。

### 优势

使用多个注意力头具有多项优势：

1. **多种表示：** 它使得模型能够同时处理来自不同表示子空间的信息。每个头可能学会关注不同类型的输入特征或关系（例如，短距离依赖与长距离依赖、句法信息与语义信息）。
2. **提升模型容量：** 将输入投影到多个低维子空间（$h$ 次，维度为 $d_k$）然后将它们组合起来，相比于在完整 $d_{model}$ 维度上操作的单个注意力头，这赋予了注意力层更强的表示能力。
3. **稳定性：** 组合多个头所固有的平均效应可以使学习过程更稳定，并且对任何单个注意力头的输出不那么敏感。

多头注意力 (multi-head attention)不仅是原始 Transformer 中的核心组成部分，也几乎是所有后续大型语言模型中的核心组成部分。它提供了一种强大且计算上可管理的方式来增强基本的注意力机制 (attention mechanism)。这些头、残差连接和归一化 (normalization)层（接下来会提及）之间的配合构成了 Transformer 模块的核心。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出Transformer架构和多头注意力机制的原始论文。
- [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — PyTorch team (2024)
  PyTorch多头注意力机制的官方文档，对理解其参数和实际应用有帮助。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Pearson
  这本权威教材的第10章（神经语言模型和Transformer）详细解释了Transformer，包括多头注意力机制。
- [Stanford CS224N: Natural Language Processing with Deep Learning, Course Notes](https://cs224n.stanford.edu/) — Chris Manning, John Hewitt, and Stanford CS224N Course Staff (2023)
  Publisher: Stanford University
  知名自然语言处理课程的综合讲义，其中第7章专门介绍Transformer架构及其多头注意力机制等组件。
