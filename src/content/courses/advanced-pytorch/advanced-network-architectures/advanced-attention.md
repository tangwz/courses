---
course: "advanced-pytorch"
chapter: "advanced-network-architectures"
lesson: "advanced-attention"
sourceId: 2688
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/advanced-attention"
title: "高级注意力机制"
description: "研究标准自注意力之外的注意力变体，例如稀疏注意力与 Performer。"
order: 2
plots: ["plots/2688-0.json"]
sourceHash: "a3efa4dfb68cc8786458036618177603e9cddbd142a25f3dafda9082aa73587f"
sourceCorrections: []
---

构成Transformer架构核心的标准自注意力 (self-attention)机制 (attention mechanism)，计算序列中所有 token 之间的成对互动。这导致复杂度与序列长度 $N$ 呈平方关系增长，具体为 $O(N^2 \cdot d)$，其中 $d$ 是模型维度。这种平方级增长对于涉及非常长文档、被视为补丁序列的高分辨率图像或扩展时间序列的应用来说，变得难以承受。为解决这些问题，高级注意力机制旨在应对这些特定局限性，最主要的是与长序列相关的计算和内存需求。

高级注意力机制主要目标是降低这种 $O(N^2)$ 复杂度到更易于管理的状态，通常是线性或接近线性的（$O(N)$ 或 $O(N \log N)$），同时试图保持原始注意力公式的建模能力。

### 稀疏注意力模式

一种方法是使注意力矩阵稀疏。不再是每个 token 都关注其他所有 token，每个 token 只关注一个受限的子集。这种限制通常基于预定义模式。

- **局部注意力：** Token 只关注固定大小的相邻 token 窗口。这能有效捕获局部背景信息，但会遗漏窗口外的长距离依赖关系。滑动窗口注意力是一种常见实现方式。
- **步进或膨胀注意力：** Token 关注固定间隔位置的 token（例如，每隔 $k$ 个 token）。这能捕获序列中远距离部分的信息，但可能会遗漏不在步进范围内的相邻 token 之间的互动。
- **组合模式：** Longformer 或 BigBird 等更复杂的方法结合了局部注意力、膨胀注意力，有时还会加入一些全局 token（如 `[CLS]` token），这些 token 关注所有其他 token，也同时被所有其他 token 关注。这试图兼顾两方面优点：局部细节和稀疏的全局背景信息。

实现这些方法通常涉及在 softmax 操作之前精心遮盖注意力分数矩阵，或者使用专门的索引和收集操作来只计算必要的分数。

### 线性化和高效注意力

另一类方法旨在近似标准注意力机制 (attention mechanism)或重新构建其计算方式，以避免显式计算 $N \times N$ 的注意力矩阵 $QK^T$。这些方法通常目标是达到 $O(N)$ 复杂度。

单个头的标准注意力输出为：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


线性注意力方法研究近似或重写此公式的方法。例如，如果我们能使用核函数 $\phi$ 来表示 softmax 函数（或其近似），使得 $\text{softmax}(x_i^T x_j) \approx \phi(x_i)^T \phi(x_j)$，我们就有可能重写计算方式。

考虑一个不带缩放因子和 softmax 的简化版本：$A = QK^T V$。这可以重新排序为 $A = Q(K^T V)$。$K^T V$ 的计算需要 $O(N d_k d_v)$ 时间，乘以 $Q$ 需要 $O(N d_k d_v)$，这导致总体复杂度相对于序列长度 $N$ 为 $O(N)$（假设 $d_k, d_v$ 是固定的）。

难点在于纳入 softmax 非线性，同时保持线性复杂度。

- **Performer：** 使用基于 Fastfood 算法的随机特征映射来近似 softmax 函数中隐含的高斯核。这实现了注意力机制的线性时间近似。
- **Linformer：** 对键 ($K$) 和值 ($V$) 矩阵应用线性投影，有效地在注意力计算*之前*降低序列长度维度，从而用低秩矩阵近似完整的注意力矩阵。
- **其他基于核的方法：** 研究不同的核函数 $\phi$ 来近似 softmax 操作，从而实现 $Q(K^T V)$ 的重新排列。

这些方法以准确性换取效率。近似方法的选择会影响模型与标准注意力相比捕获复杂依赖关系的能力。

### PyTorch 中的实现考量

尽管你可以从头开始实现稀疏遮盖或核近似，但这可能很复杂，并且需要仔细优化才能达到良好性能。幸运的是，PyTorch 生态系统提供了工具和库：

- **自定义遮盖：** 对于像局部或步进注意力这样的稀疏模式，你通常可以使用 `torch.nn.MultiheadAttention` 或 `torch.nn.functional.scaled_dot_product_attention`（在较新的 PyTorch 版本中可用）中的 `attn_mask` 参数 (parameter)。你需要构造一个布尔遮罩，其中 `True` 表示*不应*被关注的位置。
- **专门库：** 像 Meta AI 的 `xformers` 这样的库提供了各种注意力机制 (attention mechanism)的高度优化实现，包括稀疏和内存高效的变体，通常与 CUDA 内核集成以获得最高速度。对于性能要求高的应用，通常建议使用这些库。

```python
import torch
import torch.nn as nn

# 检查 xformers 是否可用于优化注意力
try:
    from xformers.ops import memory_efficient_attention
# 示例用法（API 细节可能有所不同 - 请查阅 xformers 文档）
    # 假设 q, k, v 形状正确（Batch, Seq, Heads, HeadDim 或类似）
    # output = memory_efficient_attention(q, k, v)
    # print("正在使用 xformers memory_efficient_attention")
    XFORMERS_AVAILABLE = True
except ImportError:
    # print("xformers 不可用。需要标准 PyTorch 注意力或手动实现。")
    XFORMERS_AVAILABLE = False

# 在标准 PyTorch 函数式 API 中使用注意力遮罩的示例
# 假设 embed_dim = 64, num_heads = 8, seq_len = 5, batch_size = 2
embed_dim = 64
num_heads = 8
seq_len = 5
batch_size = 2

# 虚拟输入张量 (Batch, SeqLen, EmbedDim)
query = torch.randn(batch_size, seq_len, embed_dim)
key = torch.randn(batch_size, seq_len, embed_dim)
value = torch.randn(batch_size, seq_len, embed_dim)

# 如果函数需要，为多头注意力重塑形状
# 或在 nn.Module 包装器内处理

# 创建因果遮罩（例如，用于解码器）
# 遮罩需要根据注意力函数设置适当的维度
# 对于 scaled_dot_product_attention，(SeqLen, SeqLen) 遮罩通常是可广播的
causal_mask_bool = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

# 使用 torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
# 注意：此函数在内部处理重塑和缩放
# 它期望布尔遮罩，其中 True 表示“遮盖掉”
try:
    output_sdpa = nn.functional.scaled_dot_product_attention(
        query, value, attn_mask=causal_mask_bool, is_causal=False # 显式遮罩示例
        # 或者使用 is_causal=True 进行自动因果遮罩：
        # output_sdpa = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
    )
    # print("已使用 nn.functional.scaled_dot_product_attention")
except AttributeError:
    # print("scaled_dot_product_attention 不可用（需要 PyTorch 2.0+）。")
    # 回退到 nn.MultiheadAttention 或手动实现
    pass

# 使用 nn.MultiheadAttention 的示例（需要特定格式的遮罩）
# MHA 期望布尔遮罩为 (Batch * NumHeads, TargetSeqLen, SourceSeqLen) 或 (TargetSeqLen, SourceSeqLen)
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
# MHA 遮罩：True 表示该位置*将被阻止*关注。
# 创建一个更简单的遮罩用于说明（适用于所有头/批次）
mha_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# attn_output, attn_weights = multihead_attn(query, key, value, attn_mask=mha_mask)
# print("已使用带遮罩的 nn.MultiheadAttention")
```

上述代码片段说明了你可以在哪里集成来自 `xformers` 等库的优化注意力，或者标准 PyTorch 函数如何接受注意力遮罩。确切的 API 调用和遮罩形状取决于使用的特定 PyTorch 版本和函数。请始终参考官方文档以获取精确用法。

### 权衡

选择注意力机制 (attention mechanism)涉及平衡计算效率、内存使用和模型性能。

- **标准注意力：** 表现力最强，捕获所有成对互动，但计算开销大（$O(N^2)$）。
- **稀疏注意力：** 降低复杂度，适用于局部或预定义全局互动就足够的模式。可能会遗漏不在稀疏模式范围内的重要互动。
- **线性/高效注意力：** 通常能达到 $O(N)$ 复杂度，非常适合长序列。依赖近似方法，这可能会在需要高度精确长距离依赖关系的任务上，与标准注意力相比略微降低性能。

最佳选择很大程度上取决于具体任务、涉及的序列长度以及可用的计算资源。通常需要通过实验来找到最合适的方案。



![计算复杂度与序列长度 (N)](plots/2688-0.json)



> 标准 ($O(N^2)$) 与线性 ($O(N)$) 注意力机制的计算成本随序列长度增加的理论增长曲线。请注意两个坐标轴都采用对数刻度。线性注意力复杂度以任意常数因子为例进行呈现，以便比较。

此图显示了与线性替代方案相比，标准注意力的成本增长有多快，使得后者对于有效处理长序列必不可少。当你构建更复杂的模型时，理解和应用这些高级注意力机制将对管理计算资源和扩展你的架构有重要作用。

## 参考资料

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Iz Beltagy, Matthew E. Peters, Arman Cohan (2020)
  Journal: arXiv preprint arXiv:2004.05150; DOI: [10.48550/arXiv.2004.05150](https://doi.org/10.48550/arXiv.2004.05150)
  介绍了一种通过结合局部和全局注意力模式，使复杂度与序列长度呈线性关系的注意力机制。
- [BigBird: A Self-Attention Mechanism for Long Context Transformers](https://arxiv.org/abs/2007.14062) — Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed (2020)
  Journal: Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2007.14062](https://doi.org/10.48550/arXiv.2007.14062)
  提出了一种稀疏注意力机制，实现了线性复杂度，并能近似完整的注意力。
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) — Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2009.14794](https://doi.org/10.48550/arXiv.2009.14794)
  提出了 Performer 模型，该模型使用正交随机特征来近似 Softmax 注意力，实现了线性复杂度。
- [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768) — Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma (2020)
  Journal: arXiv preprint arXiv:2006.04768; DOI: [10.48550/arXiv.2006.04768](https://doi.org/10.48550/arXiv.2006.04768)
  描述了一种通过将键和值矩阵投影到较低维度来实现线性复杂度的注意力机制。
