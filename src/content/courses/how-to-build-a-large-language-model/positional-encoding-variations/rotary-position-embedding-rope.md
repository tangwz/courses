---
course: "how-to-build-a-large-language-model"
chapter: "positional-encoding-variations"
lesson: "rotary-position-embedding-rope"
sourceId: 6015
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-13-positional-encoding-variations/rotary-position-embedding-rope"
title: "旋转位置编码 (RoPE)"
description: "详细说明旋转位置编码（RoPE）的机制与优势。"
order: 5
plots: []
sourceHash: "058675aaa77d6b6857425b73df234dd21d42198d44f1a09d968917baf87e2137"
sourceCorrections: []
---

旋转位置编码 (positional encoding)（RoPE）提供了一种独特方式，将位置信息融入到 Transformer 架构中。不同于通过添加位置向量 (vector)的绝对位置编码，或通常直接修改注意力分数计算的相对位置编码（如 Shaw 等人的方法或 Transformer-XL），RoPE 在注意力分数计算 *之前*，对查询 ($Q$) 和键 ($K$) 向量应用依赖于位置的旋转。这种方式通过旋转变换巧妙地表示了相对位置信息。

其核心思想源于一个发现：两个分别旋转了 $\alpha$ 和 $\beta$ 角度的向量之间的点积，取决于它们的原始点积以及角度差 ($\alpha - \beta$)。RoPE 运用此特性，设计出依赖于令牌绝对位置的旋转矩阵。

### 数学表达

设位于位置 $m$ 的查询向量 (vector)为 $q_m$，位于位置 $n$ 的键向量为 $k_n$。RoPE 旨在变换这些向量，使得它们的内积 $q'_m \cdot k'_n$ 主要依赖于原始向量 $q_m, k_n$ 和它们的相对位置 $m-n$。

实现此目的的方式是将嵌入 (embedding)维度 $d$ 视为若干维度对，并对每对应用二维旋转。对于向量 $x$ 和位置 $m$，变换 $f(x, m)$ 会施加一个旋转。设查询和键向量的维度为 $d$。我们可以将向量分成 $d/2$ 个大小为 2 的块。对于第 $i$ 个块（对应维度 $2i-1$ 和 $2i$），旋转矩阵 $R_{m, i}$ 定义为：


$$
R_{m, i} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix}
$$


这里，$\theta_i$ 是一个频率项，它依赖于块索引 $i$。一个常见选择是 $\theta_i = \text{base}^{-2i/d}$，其中 $\text{base}$ 是一个较大的数字（例如 10000），确保频率在不同维度上有所变化。这类似于正弦绝对位置编码 (positional encoding)中的频率选择。

RoPE 变换随后按块应用于查询 $q_m$ 和键 $k_n$：


$$
q'_m = f(q_m, m) = R_m q_m \\
k'_n = f(k_n, n) = R_n k_n
$$


其中 $R_m$ 和 $R_n$ 分别表示由 $R_{m,i}$ 和 $R_{n,i}$ 块形成的块对角矩阵。

值得关注的属性是，旋转后的查询向量和键向量之间的内积本身就捕捉了相对位置信息：


$$
(q'_m)^T k'_n = (R_m q_m)^T (R_n k_n) = q_m^T R_m^T R_n k_n
$$


由于 $R_m$ 是一个旋转矩阵，其转置是其逆，$R_m^T = R_m^{-1} = R_{-m}$。因此，$R_m^T R_n = R_{-m} R_n = R_{n-m}$。内积变为：


$$
(q'_m)^T k'_n = q_m^T R_{n-m} k_n
$$


这种最终形式表明，位于位置 $m$ 的查询与位于位置 $n$ 的键之间的作用，明确依赖于它们的相对位置 $n-m$（或等效地，$m-n$，因为 $R_{n-m}$ 包含了此差异）以及原始查询和键向量。

另一种方法是，使用复数提供了一种简洁的视角。将每个二维块 $[x_{2i-1}, x_{2i}]$ 表示为复数 $x_i = x_{2i-1} + j x_{2i}$，则旋转 $m\theta_i$ 等价于乘以 $e^{j m \theta_i}$。旋转后的查询和键分量为 $q'_{m,i} = q_{m,i} e^{j m \theta_i}$ 和 $k'_{n,i} = k_{n,i} e^{j n \theta_i}$。它们对注意力分数的贡献涉及其乘积的实部（考虑到复数点积中有一个被共轭）：


$$
\text{实部}(q'_{m,i} \overline{k'_{n,i}}) = \text{实部}((q_{m,i} e^{j m \theta_i}) (\overline{k_{n,i} e^{j n \theta_i}})) \\
= \text{实部}(q_{m,i} \overline{k_{n,i}} e^{j m \theta_i} e^{-j n \theta_i}) \\
= \text{实部}(q_{m,i} \overline{k_{n,i}} e^{j (m-n) \theta_i})
$$


对所有块 $i$ 求和，再次显示了对相对位置 $m-n$ 的依赖。

### 实现

实际应用中，RoPE 应用于多头注意力 (multi-head attention)机制 (attention mechanism)内的查询和键投影，*在*计算注意力分数之前。这通常涉及预先计算所有所需位置和维度的余弦和正弦值。

让我们看一个 PyTorch 实现代码片段。假设 `q` 和 `k` 是形状为 `(batch_size, seq_len, num_heads, head_dim)` 的张量。我们还需要预先计算好的 `cos_cached` 和 `sin_cached` 张量，通常形状为 `(max_seq_len, head_dim // 2)`。

```python
import torch

def rotate_half(x):
    """将输入张量的隐藏维度旋转一半。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 将后半部分取负，然后连接起来：(-x2, x1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos_cached, sin_cached):
    """
    将旋转位置编码应用于查询和键张量。

    参数：
        q (torch.Tensor): 查询张量，形状 (bs, seq_len, num_heads, head_dim)
        k (torch.Tensor): 键张量，形状 (bs, seq_len, num_heads, head_dim)
        cos_cached (torch.Tensor): 预先计算的余弦值，
            形状 (seq_len, head_dim // 2)
        sin_cached (torch.Tensor): 预先计算的正弦值，
            形状 (seq_len, head_dim // 2)

    返回：
        Tuple[torch.Tensor, torch.Tensor]: 旋转后的查询和键张量。
    """
    # 为 num_heads 添加维度，并在需要时沿批次维度扩展
    # cos_cached 形状: (seq_len, 1, head_dim // 2)
    cos = cos_cached[:q.shape[1], ...].unsqueeze(1)
    # sin_cached 形状: (seq_len, 1, head_dim // 2)
    sin = sin_cached[:q.shape[1], ...].unsqueeze(1)

    # 将 cos 和 sin 重复以适配完整的 head_dim: (seq_len, 1, head_dim)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    # 应用旋转
    # q_rot = (q * cos) + (rotate_half(q) * sin)
    # k_rot = (k * cos) + (rotate_half(k) * sin)

    # 替代计算方法，避免在主计算中显式调用
    # rotate_half 函数
    # 重塑 q 和 k，以分离维度对
    # q 形状: (bs, seq_len, num_heads, head_dim / 2, 2)
    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)

    # 使用复数乘法逻辑应用旋转
    # 将 cos/sin 转换为复数: R = cos + j*sin
    # 将 q/k 块转换为复数: Q = q1 + j*q2
    # 旋转后的 Q' = Q * R = (q1 + j*q2)(cos + j*sin)
    #           = (q1*cos - q2*sin) + j*(q1*sin + q2*cos)
    # q_out1 = q1*cos - q2*sin
    # q_out2 = q2*cos + q1*sin

    # 重塑 cos/sin 以便广播：
    # (1, seq_len, 1, head_dim / 2) -> (1, seq_len, 1, head_dim / 2, 1)
    # 只保留前半部分用于配对
    cos = cos[..., :q.shape[-1] // 2].unsqueeze(-1)
    # 只保留前半部分用于配对
    sin = sin[..., :q.shape[-1] // 2].unsqueeze(-1)

    q_out1 = q_reshaped[..., 0:1] * cos - q_reshaped[..., 1:2] * sin
    q_out2 = q_reshaped[..., 1:2] * cos + q_reshaped[..., 0:1] * sin
    q_rot = torch.cat((q_out1, q_out2), dim=-1).flatten(start_dim=-2)

    k_out1 = k_reshaped[..., 0:1] * cos - k_reshaped[..., 1:2] * sin
    k_out2 = k_reshaped[..., 1:2] * cos + k_reshaped[..., 0:1] * sin
    k_rot = torch.cat((k_out1, k_out2), dim=-1).flatten(start_dim=-2)

    return q_rot.type_as(q), k_rot.type_as(k)

# 示例用法：
# 假设您已经预先计算了 cos_cached, sin_cached
# max_seq_len = 2048
# head_dim = 128
# base = 10000.0
# inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() /
#                            head_dim))
# t = torch.arange(max_seq_len, device=inv_freq.device,
#                  dtype=inv_freq.dtype)
# freqs = torch.einsum("i,j->ij", t, inv_freq)
# emb = torch.cat((freqs, freqs), dim=-1)
# cos_cached = emb.cos()[:, :head_dim // 2]
# sin_cached = emb.sin()[:, :head_dim // 2]

# 在注意力层内部：
# q_rot, k_rot = apply_rotary_pos_emb(q, k, cos_cached, sin_cached)
# 使用 q_rot 和 k_rot 计算注意力分数
```

`apply_rotary_pos_emb` 函数接收查询、键和预先计算的余弦/正弦值（源自位置索引和频率）。它重塑最后一个维度以处理维度对，应用旋转逻辑，并返回修改后的查询和键张量。这些旋转后的张量随后用于标准的缩放点积注意力计算。

### 优点与考量

RoPE 在现代大型语言模型中得到广泛应用，缘于其多项优点：

1. **有效的相对位置编码 (positional encoding)**：如数学推导所示，它直接将相对位置信息编码到查询-键的作用中。
2. **良好的长度外推能力**：旋转的正弦特性使得 RoPE 有助于更好地泛化到训练中未见的序列长度，相比之下，绝对位置编码可能难以处理超出范围的位置。
3. **无需学习参数 (parameter)**：不同于可学习的绝对嵌入 (embedding)或某些相对偏置 (bias)方案，RoPE 本身不引入与位置相关的额外可学习参数，有助于简化训练过程。
4. **实现效率**：尽管需要特定的张量操作，它避免直接修改核心注意力分数计算，能整洁地融入标准的 Transformer 块结构。

与其他方法比较：

- **对比绝对嵌入**：RoPE 避免添加向量 (vector)，而是通过旋转乘法式地修改查询和键。它本身侧重于相对位置。
- **对比相对位置偏置（例如 T5、Shaw 等）**：这些方法通常根据相对距离直接向注意力分数添加偏置。RoPE 修改注意力分数计算的输入（$Q, K$）。效果可能相似，但机制不同。RoPE 由于其连续旋转特性，可能提供更好的外推能力。
- **对比 Transformer-XL**：Transformer-XL 采用了一种更复杂的相对编码方案，直接集成到注意力分数计算中。RoPE 可以被视为一种可能更简单的替代方案，用于实现相对位置感知。

频率计算中 `base` 超参数 (hyperparameter)的选择 ($\theta_i = \text{base}^{-2i/d}$) 可能影响性能和外推能力，需要仔细调整。尽管它在数学上精妙且在 Llama 和 PaLM 等模型中取得了实际成效，但理解它与其他模型组件的关联以及在超长序列上的表现，仍是一个活跃的研究方向。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (2021)
  Journal: arXiv preprint arXiv:2104.09864; DOI: [10.48550/arXiv.2104.09864](https://doi.org/10.48550/arXiv.2104.09864)
  这篇开创性论文介绍了旋转位置嵌入 (RoPE)，详细阐述了其数学基础，并展示了其在Transformer架构中的有效性。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇基础论文介绍了Transformer架构，为理解RoPE等位置编码机制提供了必要背景。
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155) — Peter Shaw, Jakob Uszkoreit, Ashish Vaswani (2018)
  Journal: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers); Pages: 1-11; DOI: [10.48550/arXiv.1803.02155](https://doi.org/10.48550/arXiv.1803.02155)
  本文提出了一种将相对位置表示引入自注意力机制的早期方法，为与RoPE的方法进行比较提供了参考。
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov (2019)
  Journal: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL); Pages: 2978-2988; DOI: [10.48550/arXiv.1901.02860](https://doi.org/10.48550/arXiv.1901.02860)
  这项工作引入了相对位置编码方案和段级循环，为Transformer中处理序列长度和相对位置提供了另一种视角。
