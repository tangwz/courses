---
course: "how-to-build-a-large-language-model"
chapter: "positional-encoding-variations"
lesson: "transformer-xl-relative-positional-encoding"
sourceId: 6014
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-13-positional-encoding-variations/transformer-xl-relative-positional-encoding"
title: "Transformer-XL 相对位置编码"
description: "解释 Transformer-XL 中使用的相对位置编码方案。"
order: 4
plots: []
sourceHash: "1d8a574b1982ad6a522ac26f51064dc156cb8da15fb1e22052f7794845e6fd74"
sourceCorrections: []
---

绝对位置编码 (positional encoding)，无论是正弦型还是学习型，虽然能为Transformer提供必要的序列顺序信息，但它们存在局限。它们通常预设一个最大序列长度，并且其处理训练数据中未见的更长序列的能力可能受限。此外，它们没有明确表示 token 之间的*相对*距离，而这对注意力机制 (attention mechanism)来说可能是更合适的工作方式。

戴等人 (2019) 提出的 Transformer-XL 架构，提出了一种新颖的相对位置编码方案，旨在专门解决这些问题，尤其是在使用段级别循环机制处理更长序列时（尽管编码方法本身独立来看也很有价值）。Transformer-XL 没有将位置信息加到词嵌入 (embedding)中，而是将相对位置信息直接注入到注意力分数计算中。

### Transformer-XL 相对编码的核心思想

核心思想是修改位置 $i$ 的查询 (query) 与位置 $j$ 的键 (key) 之间注意力分数的计算方式。在标准 Transformer 中，分数取决于查询 $q_i$ 和键 $k_j$，两者可能都包含添加到各自嵌入 (embedding)中的绝对位置信息。

Transformer-XL 重新制定了注意力分数计算，使其明确依赖于相对距离 $(i-j)$。它通过进行两项主要修改来做到这一点：

1. **键的相对位置嵌入：** 它不使用键向量 (vector)的绝对位置嵌入 $p_j$，而是使用表示查询和键位置之间偏移的相对位置嵌入 $R_{i-j}$。这些嵌入 $R_{i-j}$ 通常是固定的正弦编码，与原始 Transformer 类似，但它们编码的是相对距离而非绝对位置。重要的是，当考虑距离 $k$ 个位置的键（即 $j = i-k$）时，所有查询位置 $i$ 都使用相同的相对嵌入 $R_k$。这使得模型能够处理未见的相对距离。
2. **查询交互的分解：** 查询向量 $q_i$ 与内容和位置属性的交互方式不同。标准点积 $q_i^T k_j$ 被分解为多个项，这些项使用专门的可训练参数 (parameter)将内容-内容交互、内容-位置交互和位置-位置交互分开。

### 数学公式

设 $q_i = W_Q x_i$ 是位置 $i$ 处 token $x_i$ 的查询向量 (vector)， $k_j = W_K x_j$ 是位置 $j$ 处 token $x_j$ 的基于内容的键向量。设 $R_{i-j}$ 是相对位置 $i-j$ 的正弦嵌入 (embedding)向量。在标准 Transformer 中，softmax 内部的核心项大约是 $q_i^T (k_j + p_j)$。

Transformer-XL 用更精密的计算来替代，得到注意力分数 $A_{i,j}$：


$$
A_{i,j}^{rel} = \underbrace{q_i^T W_K x_j}_{\text{(a) 基于内容的}} + \underbrace{q_i^T W_R R_{i-j}}_{\text{(b) 内容-位置}} + \underbrace{u^T W_K x_j}_{\text{(c) 全局内容偏置}} + \underbrace{v^T W_R R_{i-j}}_{\text{(d) 全局位置偏置}}
$$


此处：

- $W_Q, W_K$ 是查询和基于内容的键的常用权重 (weight)矩阵。
- $W_R$ 是一个应用于相对位置嵌入 $R_{i-j}$ 的*新*可训练权重矩阵。
- $u$ 和 $v$ 是*新*可训练参数 (parameter)向量。
- $R_{i-j}$ 是一个固定的（通常是正弦型）编码，表示相对距离 $i-j$。它不依赖于特定的内容 $x_i$ 或 $x_j$。

我们来逐项分析：

- **(a) 基于内容的寻址：** 这是衡量查询内容 $q_i$ 与内容 $k_j = W_K x_j$ 之间相似度的标准项。
- **(b) 内容-位置寻址：** 此项描述了查询内容 $q_i$ 如何与键的*相对位置* $i-j$ 相关联。查询与转换后的相对位置嵌入 $W_R R_{i-j}$ 进行交互。
- **(c) 全局内容偏置 (bias)：** 此项仅基于键的内容 $x_j$ 添加偏置。可训练向量 $u$ 学习对某些类型的键内容的全局偏好，与查询 $q_i$ 无关。
- **(d) 全局位置偏置：** 此项仅基于*相对位置* $i-j$ 添加偏置。可训练向量 $v$ 学习对某些相对距离的全局偏好，与查询 $q_i$ 无关。

最终的注意力权重通过对这些分数 $A_{i,j}^{rel}$ 应用 softmax 获得（通常在乘以 $\frac{1}{\sqrt{d_k}}$ 缩放后）。

### 实现考量

生成相对位置编码 (positional encoding) $R_{i-j}$ 通常涉及为一个最大预期相对距离（例如，从 $-L$ 到 $+L$，其中 $L$ 是上下文 (context)长度或段长度）创建标准正弦编码。在计算位置 $i$ 处查询的注意力时，您会查找每个位置 $j=i-k$ 处键的相应编码 $R_{k}$。

引入可训练向量 (vector) $u$ 和 $v$ 以及单独的投影矩阵 $W_R$ 增加了参数 (parameter)，与标准 Transformer 注意力相比，但能够对相对位置重要性进行更精细的模型构建。

计算相对注意力分数的简化 PyTorch 风格纲要可能如下所示（侧重于分数计算，省略多头及其他细节）：

```python
import torch
import torch.nn as nn
import math

class RelativeSinusoidalPositionalEncoding(nn.Module):
    # 为相对位置生成固定的正弦编码
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 我们需要相对位置 [-max_len+1, max_len-1] 的编码
        # 创建双倍长度，稍后进行切片。存储为中心偏移。
        pe_full = torch.cat([
            pe.flip(0)[:-1, :],
            pe
        ], dim=0) # size (2*max_len - 1, d_model)
        self.register_buffer('pe', pe_full)
        self.max_len = max_len

    def forward(self, seq_len_q, seq_len_k):
        # 假设查询长度为 seq_len_q，键长度为 seq_len_k
        # 我们需要从 -(seq_len_k - 1)
        # 到 (seq_len_q - 1) 开始的相对位置
        # 在自注意力中，seq_len_q == seq_len_k == L
        # 相对索引范围从 -(L-1) 到 (L-1)
        # 将这些索引映射到存储缓冲区 [0, 2*max_len - 2]
        # 自注意力示例 (L=seq_len_q=seq_len_k)
        relative_indices = torch.arange(
            seq_len_k - 1, -seq_len_q, -1, dtype=torch.long
        )
        # 偏移索引以使其在 'pe' 缓冲区中查找时为正值
        # 中心位于 max_len - 1
        buffer_indices = relative_indices + self.max_len - 1
        relative_encodings = self.pe[buffer_indices]
        # 形状 (seq_len_q + seq_len_k - 1, d_model)
        # 我们需要一个形状为 (seq_len_q, seq_len_k, d_model) 的矩阵 R
        # 其中 R[i, j, :] = 相对距离 (i - j) 的编码
        # 这需要根据用例（自注意力 vs 编码器-解码器）进行仔细的切片/索引
        # 对于自注意力 (seq_len_q=L, seq_len_k=L)：
        # 我们需要 i 在 [0, L-1], j 在 [0, L-1] 时的编码 R_{i-j}
        # 相对距离范围从 -(L-1) 到 L-1
        # relative_encodings 缓冲区保存了距离 k 的编码
        # 从 L-1 到 -(L-1)
        start_idx = self.max_len - seq_len_k
        end_idx = start_idx + seq_len_q + seq_len_k - 1
        # 选择缓冲区中的相关部分
        rel_enc = self.pe[start_idx:end_idx]
        # 对于自注意力，形状为 (L+L-1, d_model)

        # 有效地创建最终矩阵 R (L, L, d_model)
        # 这可能涉及巧妙的切片或矩阵操作
        # 为简单起见，我们假设有一个函数
        # `get_rel_embeddings(rel_enc, L)`
        # 返回 (L, L, d_model) 矩阵。
        # R = get_rel_embeddings(rel_enc, seq_len_q) # 占位符
        # # 用于复杂索引
        # return R

        # 简化：让我们专注于注意力分数计算
        # 假设 R_ij 矩阵可用
        pass
        # 暂时返回缓冲区，实际使用需要更多的
        # 索引逻辑


class TransformerXLRelativeAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_r = nn.Linear(d_model, d_model)
        # 相对嵌入的投影

        # 可训练参数 u 和 v
        # （为简单起见，最初在所有头之间共享）
        self.u = nn.Parameter(torch.Tensor(self.nhead, self.d_head))
        self.v = nn.Parameter(torch.Tensor(self.nhead, self.d_head))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        self.dropout = nn.Dropout(0.1)
        self.scale = 1.0 / math.sqrt(self.d_head)

        # 假设 RelativeSinusoidalPositionalEncoding 模块提供 R
        # self.relative_pos_encoder = RelativeSinusoidalPositionalEncoding(
        #     d_model, max_len)

    def forward(self, query_embed, key_embed, value_embed, R_ij,
                mask=None):
        # query_embed, key_embed, value_embed：
        #   (batch_size, seq_len, d_model)
        # R_ij: 预计算的相对位置编码投影
        #       W_R * R_{i-j}
        #       高效计算的预期形状：
        #       用于项 (b) 的形状 (batch_size, nhead, seq_len_q, d_head)
        #       以及用于项 (d) 的形状 (batch_size, nhead, seq_len_k, d_head)
        #       乘以 v 之后
        # Mask: (batch_size, seq_len_q, seq_len_k)

        batch_size, seq_len_q, _ = query_embed.size()
        seq_len_k = key_embed.size(1)

        Q = self.W_q(query_embed).view(
            batch_size, seq_len_q, self.nhead, self.d_head
        )
        K = self.W_k(key_embed).view(
            batch_size, seq_len_k, self.nhead, self.d_head
        )
        V = self.W_v(value_embed).view(
            batch_size, seq_len_k, self.nhead, self.d_head
        )
        # R_ij 需要由 W_r 投影并适当地重塑
        # 在此处传入之前
        # Projected R_ij = self.W_r(raw_R_ij).view(....)

        # 转置以进行注意力计算：
        # (batch_size, nhead, seq_len, d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 项 (a): 基于内容的
        AC = torch.matmul(Q + self.u.unsqueeze(0).unsqueeze(2),
                          K.transpose(-2, -1))
        # Q 形状: (batch, nhead, seq_len_q, d_head)
        # K^T 形状: (batch, nhead, d_head, seq_len_k)
        # 结果 AC 形状: (batch, nhead, seq_len_q, seq_len_k)
        # 这里我们将全局内容偏置 'u' 添加到查询侧

        # 项 (b) 和 (d): 基于位置的
        # 这需要仔细处理 R_ij 张量形状
        # 以及相对索引
        # 假设 R_proj = W_r(R) 已预计算并整形为
        # (seq_len_q, seq_len_k, nhead, d_head)
        # R_proj = R_proj.permute(2, 0, 3, 1)
        # -> (nhead, seq_len_q, d_head, seq_len_k)
        # 项 (b): (Q + self.v) * R_proj
        # BD = torch.matmul(Q + self.v.unsqueeze(0).unsqueeze(2), R_proj)
        # 伪代码 - 维度需要注意

        # 简化计算，仅展示原理 -
        # 实际实现很复杂
        # 由于使用矩阵移位实现相对项的高效计算
        # 或偏斜。
        # 有关高效实现细节，请参阅 Dai 等人 (2019) 附录 B。

        # 组合分数计算的占位符：
        # scores = AC + BD # 完整分数计算的占位符

# 让我们为分数使用一个占位符：
        # 假设 AC 代表项 (a) 和 (c) 的和
        # 假设 BD 代表项 (b) 和 (d) 的和
        # 高效计算
        # scores = (AC + BD) * self.scale # 占位符

        # 用于演示结构的虚假分数
        scores = AC * self.scale

        if mask is not None:
             # 确保掩码具有兼容的形状
             # (batch_size, 1, seq_len_q, seq_len_k)
             scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        # (批次大小, 头数, 查询序列长度, 头维度)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )

        return output, attn_weights
```

*注意：* 项 (b) 和 (d) 的实际实现需要仔细的张量操作（通常涉及偏斜矩阵），以便高效地执行相对位置计算，而无需在每一步为每对 $(i,j)$ 明确构建完整的 $R_{i-j}$ 矩阵。上面的代码简化了这一部分。

### Transformer-XL 相对编码的优点

1. **对更长序列的处理能力：** 由于使用相对位置，模型不与训练期间学习到的特定绝对位置绑定。它能够处理比训练中见过的更长的序列，因为相对偏移量仍然有意义。
2. **明确的相对信息：** 它直接编码 token 之间的距离，这比单独的绝对位置能为注意力提供更多信息。
3. **支持段级别循环：** 这项编码方案是 Transformer-XL 主要成就的构成部分：逐段处理序列，同时重用先前段的隐藏状态。相对编码确保了跨段边界的位置一致性。

与 Shaw 等人的方法（在主要查询-键点积*之后*添加相对位置偏置 (bias)）相比，Transformer-XL 通过让查询直接与相对位置嵌入 (embedding) ($q_i^T W_R R_{i-j}$) 交互并整合全局位置偏置 ($v^T W_R R_{i-j}$)，将相对位置信息更紧密地融入到分数计算中。

总而言之，Transformer-XL 相对位置编码 (positional encoding)提供了替代绝对编码的方案。通过侧重于相对距离并分解注意力分数计算，它为序列长度提供了更佳的通用能力，并为旨在处理极长上下文 (context)的架构提供了支持。其实现需要对相对位置嵌入和额外的可训练参数 (parameter)进行仔细处理，但对长序列建模的益处是显著的。

## 参考资料

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov (2019)
  Journal: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL); Pages: 32–42; DOI: [10.48550/arXiv.1901.02860](https://doi.org/10.48550/arXiv.1901.02860)
  介绍 Transformer-XL 架构及其相对位置编码方案的原始论文，包含详细的数学公式和实现策略。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入 Transformer 架构及其原始绝对正弦位置编码的开创性论文，为后续相对编码方法提供了背景。
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — Peter Shaw, Jakob Uszkoreit, Ashish Vaswani (2018)
  Journal: NAACL 2018; Pages: 464–470; DOI: [10.48550/arXiv.1803.02155](https://doi.org/10.48550/arXiv.1803.02155)
  介绍了一种在自注意力中进行相对位置编码的早期方法，使用可学习的相对位置偏差，为与 Transformer-XL 方法的比较提供了参考。
