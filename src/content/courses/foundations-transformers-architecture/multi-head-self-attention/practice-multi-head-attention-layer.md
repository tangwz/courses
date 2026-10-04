---
course: "foundations-transformers-architecture"
chapter: "multi-head-self-attention"
lesson: "practice-multi-head-attention-layer"
sourceId: 1969
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-3-multi-head-self-attention/practice-multi-head-attention-layer"
title: "动手实践：构建多头注意力层"
description: "使用深度学习框架实现多头自注意力层。"
order: 8
plots: []
sourceHash: "ee0d08524077718076021ed4020ca2945ad898f3f27e44be08213b9400367b54"
sourceCorrections: []
---

使用 PyTorch 构建多头自注意力 (self-attention)的实际实现方法。从零开始构建此层有助于巩固对数据流动和张量操作的理解。本实践活动要求对 PyTorch 基础模块和张量操作有一定了解。

我们的目标是创建一个 `MultiHeadAttention` 模块，该模块接受输入序列，并将其投射为多个注意力头的查询（Q）、键（K）和值（V），并行为每个头计算缩放点积注意力，将结果拼接，并进行最终的线性投射。

### 定义模块结构

我们将定义一个继承自 `torch.nn.Module` 的 Python 类。构造函数（`__init__`）将初始化用于初始 Q、K、V 投射以及最终输出投射所需的线性层。我们还需要保存嵌入 (embedding)维度（`d_model`）和注意力头数量（`num_heads`）。一个重要条件是 `d_model` 必须能被 `num_heads` 整除，以便投射维度 ($d_k$, $d_v$) 为整数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """实现了多头注意力机制。"""

    def __init__(self, d_model: int, num_heads: int):
        """
        参数:
            d_model (int): 输入和输出嵌入的维度。
            num_heads (int): 注意力头的数量。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的键/查询维度

        # 用于 Q、K、V 投射的线性层（为提高效率可合并）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 拼接后的最终线性层
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """计算缩放点积注意力。"""
        # 矩阵乘法 QK^T
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 如果提供掩码则应用（用于解码器自注意力）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # 使用一个很大的负值

        # SoftMax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 矩阵乘法 Softmax(QK^T/sqrt(d_k)) * V
        output = torch.matmul(attn_probs, V)
        return output, attn_probs # 返回概率以便后续可视化

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        执行多头注意力的前向计算。

        参数:
            query (torch.Tensor): 查询张量，形状 (batch_size, seq_len_q, d_model)
            key (torch.Tensor): 键张量，形状 (batch_size, seq_len_k, d_model)
            value (torch.Tensor): 值张量，形状 (batch_size, seq_len_v, d_model)
                                   (seq_len_k 和 seq_len_v 必须相同)
            mask (torch.Tensor, optional): 用于阻止对特定位置注意力的掩码张量。
                                            形状取决于应用（例如，填充掩码，先行掩码）。
                                            默认为 None。

        返回:
            torch.Tensor: 输出张量，形状 (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # 1. 线性投射
        # 使用各自的权重矩阵 W_q, W_k, W_v 投射 Q, K, V
        # 输入形状: (batch_size, seq_len, d_model)
        # 输出形状: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 为多头重塑形状
        # 重塑 Q, K, V 以分离各头
        # 原始形状: (batch_size, seq_len, d_model)
        # 目标形状: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 应用缩放点积注意力（每个头）
        # 现在形状为 (batch_size, num_heads, seq_len, d_k)
        # 注意力函数在最后两个维度 (seq_len, d_k) 上操作
        # 如果提供了掩码，它需要能进行适当的广播。
        # 例如，一个填充掩码 (batch_size, 1, 1, seq_len_k) 或
        # 一个先行掩码 (batch_size, 1, seq_len_q, seq_len_k)。
        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output 形状: (batch_size, num_heads, seq_len_q, d_k)
        # attn_probs 形状: (batch_size, num_heads, seq_len_q, seq_len_k)

        # 4. 拼接各头
        # 将注意力输出重塑回以组合各头
        # 转置将 seq_len_q 带回到 d_model 组件之前
        # contiguous() 确保内存布局适合 view()
        # 目标形状: (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. 最终线性投射
        # 应用最终的权重矩阵 W_o
        # 输入形状: (batch_size, seq_len_q, d_model)
        # 输出形状: (batch_size, seq_len_q, d_model)
        output = self.W_o(attn_output)

        return output # 在训练/推断期间通常只需要最终输出张量
```

### 代码解析

1. **初始化（`__init__`）**：我们设置了四个 `nn.Linear` 层。`W_q`、`W_k`、`W_v` 将输入嵌入 (embedding) `d_model` 投射为 Q、K、V 向量 (vector)，其大小也为 `d_model`。此处的实现细节是，我们首先投射到 `d_model`，然后重塑，而不是直接为每个头投射到 `d_k`。两种方法都可行。`W_o` 是最终的输出变换层。我们将 `d_k`（每个头的维度）计算为 `d_model / num_heads`。
2. **缩放点积注意力**：为清晰起见，我们包含了一个单独的 `scaled_dot_product_attention` 方法。这封装了注意力算法：$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$。它计算注意力分数，应用缩放因子 $1/\sqrt{d_k}$，可选地应用掩码（在 softmax 之前将掩蔽位置设置为一个很大的负数），使用 softmax 计算注意力概率，最后计算值的加权和。请注意，PyTorch 提供了一个高度优化的版本（`torch.nn.functional.scaled_dot_product_attention`），在生产代码中应优先使用以获得性能，但此处所示的手动实现有助于理解。
3. **前向计算（`forward`）**：
   - **输入**：`forward` 方法接受 `query`、`key` 和 `value` 张量。对于自注意力 (self-attention)（本章的主题），这三个张量将是相同的（来自*同一*输入序列）。我们将它们分开是为了保持通用性，因为此模块也可用于编码器-解码器交叉注意力（稍后讨论）。也可以提供一个可选的 `mask`。
   - **投射**：输入 `query`、`key`、`value` 张量通过各自的线性层（`W_q`、`W_k`、`W_v`）。
   - **重塑**：这一步很重要。投射的输出，形状 `(batch_size, seq_len, d_model)`，需要重塑为 `(batch_size, num_heads, seq_len, d_k)`。这将每个头的计算独立开来。`.view()` 方法重塑张量，而 `.transpose(1, 2)` 则交换 `num_heads` 和 `seq_len` 维度，为注意力函数中的批次矩阵乘法做准备。
   - **注意力计算**：`scaled_dot_product_attention` 方法使用重塑后的 Q、K、V 张量进行调用。批次矩阵乘法同时处理批次维度和头维度上的计算。
   - **拼接/重塑回原形**：注意力头的输出，形状 `(batch_size, num_heads, seq_len_q, d_k)`，需要组合。我们首先 `.transpose(1, 2)` 回到 `(batch_size, seq_len_q, num_heads, d_k)`。`.contiguous()` 确保张量存储在连续的内存块中，这有时在调用 `.view()` 之前是必需的。最后，`.view(batch_size, -1, self.d_model)` 将其重塑回所需的 `(batch_size, seq_len_q, d_model)` 格式，有效地沿嵌入维度拼接了各头的输出。
   - **最终投射**：这个拼接后的张量通过最终的线性层 `W_o` 产生模块的输出。

### 可视化流程

我们可以可视化多头注意力 (multi-head attention)层的数据流：

> 多头注意力模块内的数据流。B=批次大小，Lq/Lk/Lv=Q/K/V 的序列长度，dm=模型维度，h=注意力头数量，dk=每个头的维度。对于自注意力 (self-attention)，Lq=Lk=Lv。

此实现提供了对多个注意力头如何并行工作的具体认识。每个注意力头可能通过使用不同的投射来关注输入关联的不同方面，它们的综合信息通过最终的线性层进行整合。该层是 Transformer 编码器和解码器堆栈中反复使用的基本组成部分。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍 Transformer 架构和多头注意力机制的奠基性论文。
- [torch.nn Module Documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch 神经网络模块的官方文档，包含 `nn.Module`、`nn.Linear` 和构建自定义层所需的张量操作。
- [torch.nn.functional.scaled_dot_product_attention Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch Contributors (2024)
  PyTorch 优化点积注意力机制的官方文档，适用于生产代码。
- [CS224N: Natural Language Processing with Deep Learning (Winter 2023)](https://cs224n.stanford.edu/w23/) — Christopher Manning, Abigail See, John Hewitt, et al. (Course Staff) (2023)
  Publisher: Stanford University
  知名大学课程，提供深度学习自然语言处理的全面讲座和材料，包括对 Transformer 及其组件的详细解释。
