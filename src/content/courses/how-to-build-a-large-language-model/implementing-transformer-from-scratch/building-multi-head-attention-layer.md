---
course: "how-to-build-a-large-language-model"
chapter: "implementing-transformer-from-scratch"
lesson: "building-multi-head-attention-layer"
sourceId: 5997
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-10-implementing-transformer-from-scratch/building-multi-head-attention-layer"
title: "构建多头注意力层"
description: "结合多个注意力头和线性投影。"
order: 3
plots: []
sourceHash: "1509d0bfc989e10c7948ed9a53173dd1f6d1acb192a6511baf5228e8d0039328"
sourceCorrections: []
---

缩放点积注意力机制 (attention mechanism)允许模型关注序列的不同部分。多头注意力 (multi-head attention)机制，由“Attention Is All You Need”论文提出，进一步提升了此功能。多头注意力不是使用$d_{model}$维度的键、值和查询执行单个注意力函数，而是通过不同的、学习到的线性投影，将查询、键和值分别投影到$d_k$、$d_k$和$d_v$维度$h$次。接着对这些投影后的版本并行执行注意力操作。其输出被拼接起来并再次投影，得到最终结果。

基本思路是，每个“头”可以同时学习关注序列中不同类型的信息或关联。比如，一个头可能关注句法依赖，而另一个追踪指代关系。通过并行运行这些注意力机制并结合它们的输出，模型获得对输入更丰富、多层面的理解。

数学上，多头注意力定义为：


$$
\text{多头}(Q, K, V) = \text{拼接}(\text{头}_1, \dots, \text{头}_h)W^O
$$


其中每个$\text{头}_i$计算为：


$$
\text{头}_i = \text{注意力}(QW_i^Q, KW_i^K, VW_i^V)
$$


此处，$Q, K, V$是输入的查询、键和值矩阵。投影矩阵$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$、$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$和$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$是第$i$个头的参数 (parameter)矩阵，而$W^O \in \mathbb{R}^{hd_v \times d_{model}}$是输出投影矩阵。在原版Transformer论文和许多常见实现中，维度设定为$d_k = d_v = d_{model} / h$。这使得计算成本与使用$d_{model}$维度键和值的单头注意力相似。

我们来在PyTorch中实现它。我们将创建一个`MultiHeadAttention`模块，它接收嵌入 (embedding)维度（`embed_dim`）、头数量（`num_heads`）以及可选的dropout概率作为输入。

```python
import torch
import torch.nn as nn
import math

# 假设scaled_dot_product_attention已在上一节中定义
# def scaled_dot_product_attention(q, k, v, mask=None):
#     d_k = q.size(-1)
#     scores = torch.matmul(q, k.transpose(-2, -1)) / \
#              math.sqrt(d_k)
#     if mask is not None:
#         # 使用一个大的负值
#         scores = scores.masked_fill(mask == 0, -1e9)
#     attn_weights = torch.softmax(scores, dim=-1)
#     output = torch.matmul(attn_weights, v)
#     return output, attn_weights

class MultiHeadAttention(nn.Module):
    """ 实现多头注意力机制。 """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q、K、V投影的线性层。
        # 为了效率，我们使用一个单一的线性层，
        # 投影到embed_dim * 3然后分割结果。
        # 另外，也可以使用单独的层。
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) # Dropout层（可选）

        self._reset_parameters()

    def _reset_parameters(self):
        # 对线性层使用Xavier均匀初始化
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, query, key, value, mask=None):
        """
        多头注意力的前向传播。
        Args:
            query (torch.Tensor): 查询张量，
                                  形状 (batch_size, seq_len_q, embed_dim)
            (torch.Tensor): 张量，
                                形状 (batch_size, seq_len_k, embed_dim)
            value (torch.Tensor): 值张量，
                                  形状 (batch_size, seq_len_v, embed_dim)
                                注意：通常seq_len_k == seq_len_v。
            mask (torch.Tensor, optional): 掩码张量，用于阻止
                                           对某些位置的注意力。
                                           形状 (batch_size, 1, seq_len_q,
                                                  seq_len_k) 或类似
                                           可广播形状。
        Returns:
            torch.Tensor: 输出张量，
                          形状 (batch_size, seq_len_q, embed_dim)
            torch.Tensor: 注意力权重，
                          形状 (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.size()
        # 值序列长度必须匹配
        _, seq_len_k, _ = key.size()
        _, seq_len_v, _ = value.size()
        assert seq_len_k == seq_len_v

        # 1. 使用组合线性层投影Q、K、V
        qkv = self.qkv_proj(query) # 投影查询
        # 我们分别投影键和值，以防
        # 它们在编码器-解码器注意力中具有不同的源长度，
        # 尽管此处我们假设自注意力（查询=键=值）
        # 为了通用性，我们假设键和值可能存在独立的输入。
        # 如果查询、值是相同的张量（自注意力），
        # 这比一次投影并分割效率略低，
        # 但更灵活。
        k_proj = self.qkv_proj(key) # 投影键
        v_proj = self.qkv_proj(value) # 投影值

        # 将组合投影分割成Q、K、V
        # qkv形状: (batch_size, seq_len, embed_dim * 3) ->
        #            3个(batch_size, seq_len, embed_dim)形状的张量
        q, k, v = qkv.chunk(3, dim=-1)

        # 如果使用单独的层或只对查询进行不同投影的替代方案：
        # q = self.q_proj(query)
        # k = self.k_proj(key)
        # v = self.v_proj(value)

        # 2. 重塑Q、K、V以进行多头计算
        # 从(batch_size, seq_len, embed_dim)重塑为
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size,
                   seq_len_q,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size,
                   seq_len_k,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size,
                   seq_len_v,
                   self.num_heads,
                   self.head_dim).transpose(1, 2)

        # 3. 对每个头应用缩放点积注意力
        # 掩码需要能够正确广播。
        # 如果掩码形状是(batch_size, seq_len_q, seq_len_k)，则需要
        # 为头维度进行unsqueezing：
        # (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
             # (batch_size, seq_len_q, seq_len_k)
             if mask.dim() == 3:
                 # 添加头维度: (batch_size, 1, seq_len_q, seq_len_k)
                 mask = mask.unsqueeze(1)
             # (seq_len_q, seq_len_k) - 所有批次使用相同掩码
             elif mask.dim() == 2:
                 # 添加批次和头维度: (1, 1, seq_len_q, seq_len_k)
                 mask = mask.unsqueeze(0).unsqueeze(0)
        # 确保掩码形状兼容：
        # (batch_size, num_heads, seq_len_q, seq_len_k)或可广播

        # attn_output形状: (batch_size, num_heads, seq_len_q, head_dim)
        # attn_weights形状: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask
        )

        # 4. 拼接头并将结果投影回embed_dim
        # 转置并重塑以组合头：
        # (batch_size, seq_len_q, num_heads * head_dim)
        # num_heads * head_dim = embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )

        # 应用最终线性投影
        output = self.out_proj(attn_output)

        # 应用dropout（可选）
        output = self.dropout(output)

        return output, attn_weights
```

在此实现中：

- 我们初始化了`embed_dim`、`num_heads`并计算了`head_dim`。重要的是，`embed_dim`必须能被`num_heads`整除。
- 我们使用一个单一的`nn.Linear`层（`qkv_proj`）来同时投影输入的查询、键和值张量以提高效率。然后我们将结果分割为`q`、`k`和`v`。另一种方案是为`q`、`k`和`v`定义单独的`nn.Linear`层。
- `_reset_parameters`方法处理权重 (weight)初始化，采用Xavier均匀初始化，这是Transformer层中的常见做法。
- 在`forward`方法中：
  - 我们使用`qkv_proj`投影输入。
  - 投影后的张量`q`、`k`、`v`被重塑以分离出各个头。维度变为`(batch_size, num_heads, seq_len, head_dim)`。`transpose(1, 2)`操作重新排列维度，使得头维度位于序列长度维度之前，这通常是注意力实现或优化过的核所期望的。
  - 我们对重塑后的`q`、`k`、`v`和可选的`mask`调用我们的`scaled_dot_product_attention`函数（我们假定此函数已在其他地方定义，例如上一节的代码或某个工具文件中）。掩码处理确保它在各个头之间正确广播。
  - 注意力头（`attn_output`）的输出被拼接回来。我们通过先转置回来（`transpose(1, 2)`），然后使用`contiguous().view()`将`num_heads`和`head_dim`维度合并回原始的`embed_dim`来实现。调用`contiguous()`是必要的，因为`transpose`可能返回一个非连续的张量，而`view`不能直接在其上操作。
  - 最后，拼接后的输出通过最终的线性层（`out_proj`）和一个可选的dropout层。

这个`MultiHeadAttention`模块封装了Transformer论文中描述的核心逻辑。它接收查询、键和值输入（在自注意力 (self-attention)层中它们通常是相同的张量），并生成一个与查询形状相同的输出张量，以及用于后续分析的注意力权重。这个模块将是我们接下来构建更大编码器和解码器层的构成单元。

流程图，显示了多头注意力模块内的步骤，从输入的查询、键、值张量到最终输出和注意力权重。`h`表示头的数量，`bs`代表批次大小。

## 参考资料

- [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  介绍了 Transformer 架构和多头注意力机制，这是现代大型语言模型的基础。
- [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — PyTorch Authors (2024)
  PyTorch 内置多头注意力模块的官方文档，有助于理解标准实现模式和用法。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Pages: Chapter 10: Masked Language Models
  一本标准教科书，在自然语言处理的背景下，全面地学术处理了 Transformer 和注意力机制。
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Alexander Rush, Vincent Nguyen, Guillaume Klein (2018)
  对 Transformer 模型的高质量详细解释和 PyTorch 实现，作为理解原始论文代码的指南。
