---
course: "how-to-build-a-large-language-model"
chapter: "efficient-inference-strategies"
lesson: "key-value-kv-caching"
sourceId: 6098
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-28-efficient-inference-strategies/key-value-kv-caching"
title: "键值（KV）缓存"
description: "实作KV缓存以避免对先前标记的键和值进行重复计算。"
order: 2
plots: []
sourceHash: "287b64a9178b375567987098bf5064ba76da1c05d07d5e1e4f2fd7cc03f3dedc"
sourceCorrections: []
---

自回归 (autoregressive)生成，即基于先前已生成的标记 (token)逐个生成文本的过程，是大型语言模型生成回复的中心方式。然而，简单的实现会遇到一个很大的性能障碍。Transformer架构中的自注意力 (self-attention)机制 (attention mechanism)是此过程的核心。为了生成下一个标记，例如标记 $t+1$，标准的自注意力计算需要基于*所有*之前的标记 $1...t$ 来计算查询（Q）、键（K）和值（V），然后计算注意力分数。当生成随后的标记 $t+2$ 时，这个完整过程会使用标记 $1...t+1$ 再次进行。请注意这种重复：在生成标记 $t+1$ 期间为标记 $1...t$ 计算的键和值向量 (vector)，与生成标记 $t+2$ 时前 $t$ 个标记所需的向量是相同的。在每一步重复这些计算是计算上的浪费，特别是当序列长度增加时。

键值（KV）缓存是一种基本优化技术，其目的在于消除自回归推理 (inference)中的这种重复。其中心思想简单而高效：存储自注意力层中为所有先前标记计算出的键（K）和值（（V）张量，并在随后的生成步骤中重复使用它们。

### KV缓存的工作方式

在Transformer的自注意力 (self-attention)层中，输入序列 $X$ 被投影到三个矩阵：查询（$Q$）、键（$K$）和值（$V$）。


$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$


$W_Q, W_K, W_V$ 为可学习的权重 (weight)矩阵。注意力输出随后按以下方式计算：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


考虑生成标记 (token) $t+1$。模型将标记嵌入 (embedding)序列 $x_1, ..., x_t$ 作为输入。在每个注意力层内，它计算 $K_1, ..., K_t$ 和 $V_1, ..., V_t$。它还基于*最后*一个标记 $x_t$ 的嵌入（或对应于位置 $t+1$ 的位置嵌入）计算查询向量 (vector) $Q_{t+1}$。然后，注意力计算使用 $Q_{t+1}$ 和完整的键集合 $K = [K_1, ..., K_t]$ 以及值集合 $V = [V_1, ..., V_t]$。

现在，考虑生成标记 $t+2$。输入序列为 $x_1, ..., x_{t+1}$。模型需要计算 $K_1, ..., K_{t+1}$ 和 $V_1, ..., V_{t+1}$。重要的是，对 $K_1, ..., K_t$ 和 $V_1, ..., V_t$ 的计算与上一步完全相同，因为它们仅依赖于输入标记 $x_1, ..., x_t$ 以及固定的权重矩阵 $W_K$ 和 $W_V$。

KV缓存善用这个特性。它不是在每一步重新计算所有的键和值，而是：

1. **初始化（步骤1）：** 基于输入 $x_1$ 计算 $K_1, V_1$。将其存入缓存。
2. **步骤 $t$（生成标记 $t+1$）：**
   - 基于 $x_t$ 计算 $Q_t$。
   - 基于 $x_t$ 计算 $K_t, V_t$。
   - 获取缓存中的 $K_{1..t-1}, V_{1..t-1}$。
   - 拼接：$K_{cache} = [K_{1..t-1}, K_t]$，$V_{cache} = [V_{1..t-1}, V_t]$。
   - 使用 $Q_t$、$K_{cache}$、$V_{cache}$ 计算注意力。
   - 用 $K_{cache}$ 和 $V_{cache}$ 更新缓存。
   - 生成标记 $t+1$。
3. **步骤 $t+1$（生成标记 $t+2$）：**
   - 基于新生成的标记 $x_{t+1}$ 计算 $Q_{t+1}$。
   - 仅基于 $x_{t+1}$ 计算*新的* $K_{t+1}, V_{t+1}$。
   - 从上一步获取缓存中的 $K_{1..t}, V_{1..t}$。
   - 拼接：$K_{cache} = [K_{1..t}, K_{t+1}]$，$V_{cache} = [V_{1..t}, V_{t+1}]$。
   - 使用 $Q_{t+1}$、$K_{cache}$、$V_{cache}$ 计算注意力。
   - 更新缓存。
   - 生成标记 $t+2$。
   - ...依此类推。

> 简化的流程图，展示了在步骤 `t` 计算出的键（K）和值（V）如何被缓存并在步骤 `t+1` 复用，这样只需要为新标记 `x_{t+1}` 进行计算。

这大大减少了每个生成标记的计算成本。注意力计算复杂度不再像每个步骤中与序列长度 $t$ 的平方大致成比例（如果考虑完整矩阵乘法是 $O(t^2)$，或者仅将查询应用于现有键是 $O(t)$），而是与过去标记相关的计算实际变成了常数时间（缓存查找和拼接），主要成本变为计算单个新标记的K和V，并将新查询应用于缓存的键（$QK^T$ 部分为 $O(t)$）。

### 内存使用

尽管KV缓存显著加快了推理 (inference)速度，但它也带来了内存成本。缓存需要为批次中的每个序列，存储所有先前标记 (token)在所有层和所有注意力头中的键和值张量。KV缓存的大小可以估算为：

`缓存大小 ≈ batch_size × num_layers × 2（针对 K 和 V） × num_heads × sequence_length × head_dimension × bytes_per_element`

这种内存占用随 `sequence_length` 线性增长。对于具有多层和多头的模型，以及处理长序列或大批次时，KV缓存会占用大量的GPU内存，有时成为可处理最大序列长度的限制因素。管理这种内存使用是一个重要的考量，它促成了分页注意力或缓存本身的量化 (quantization)等技术，尽管这些内容超出了本基本介绍的范围。

### PyTorch中的实作考量

实作KV缓存通常涉及修改Transformer块的 `forward` 方法（或直接修改注意力模块），使其接受一个可选的 `past_key_values` 参数 (parameter)并返回更新后的 `present_key_values`。

下面是一个（高度简化的）草图，对比了标准注意力计算和使用KV缓存的注意力计算：

```python
import torch
import torch.nn as nn

# 假设 'attention_layer' 是一个预定义的多头注意力模块
# 简化的多头注意力占位符
class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim 必须能被 num_heads 整除"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, past_kv=None):
        batch_size, seq_len, _ = query.size()

        # 投影查询、键、值
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 为多头注意力重塑形状
        q = q.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2) # [B, nh, L_q, hd]
        k = k.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2) # [B, nh, L_k, hd]
        v = v.view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2) # [B, nh, L_v, hd]

        present_kv = None
        if past_kv is not None:
            # past_kv 是一个元组 (past_k, past_v)
            # 每个的形状为 [B, nh, L_past, hd]
            past_k, past_v = past_kv
            # 沿着序列长度维度 (dim=2) 拼接
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
            # 存储更新后的 K, V 以供下一步使用
            present_kv = (k, v) # Shape [B, nh, L_past + L_k, hd]
        else:
             # 首次存储 K, V
             present_kv = (k, v) # Shape [B, nh, L_k, hd]

        # 计算注意力分数
        # q: [B, nh, L_q, hd], k.transpose: [B, nh, hd, L_k]
        # -> scores: [B, nh, L_q, L_k]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # 将注意力权重应用于值
        # attn_weights: [B, nh, L_q, L_k], v: [B, nh, L_v, hd]
        # -> output: [B, nh, L_q, hd]
        # (此处假设 L_v == L_k)
        output = torch.matmul(attn_weights, v)

        # 重塑并投影输出
        output = output.transpose(1, 2).contiguous()
        output = output.view(
            batch_size, -1, self.embed_dim
        ) # [B, L_q, embed_dim]
        output = self.out_proj(output)

        # 返回输出和此层更新后的键值缓存
        return output, present_kv

# --- 生成过程中的用法 ---
# model = 你的Transformer模型(...)
# kv_cache = None # 初始空缓存（每层一个列表或元组）
# input_ids = 初始提示词ID

# for _ in range(最大新标记数):
#     # 准备当前步骤的输入
#     # （通常仅为最后一个生成的标记）
#     current_input_ids = input_ids[:, -1:] # Shape [B, 1]

#     # 带缓存的前向传播
#     # 注意：模型的前向传播需要处理缓存的向下传递
#     # 并收集更新
#     outputs = model(
#         input_ids=current_input_ids,
#         past_key_values=kv_cache,
#         use_cache=True
#     )
#     logits = outputs.logits
#     kv_cache = outputs.past_key_values # 为下一次迭代更新缓存

#     # 获取预测的下一个标记ID（例如，使用argmax或采样）
#     next_token_id = torch.argmax(
#         logits[:, -1:, :], dim=-1
#     ) # Shape [B, 1]

#     # 为下一次迭代的完整输入追加新的标记ID
#     # （尽管只有最后一个用于Q）
#     input_ids = torch.cat([input_ids, next_token_id], dim=-1)

#     # 检查停止条件等
```

在实践中，像Hugging Face Transformers这样的框架对这种缓存机制进行了抽象。当调用 `generate` 方法或使用带有 `use_cache=True` 参数的模型前向传播时，框架会自动处理生成步骤之间KV缓存的创建、传递和更新。然而，理解其基本原理对认识性能提升和内存影响很重要。

KV缓存是高效Transformer推理 (inference)的根本。它直接解决了简单自回归 (autoregressive)解码在序列长度方面的二次复杂度瓶颈，使得在实践中生成更长序列成为可能。尽管它带来了内存开销，但计算上的节省几乎总是使其成为一项不可或缺的优化。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构和自注意力机制的原始论文，这是KV缓存的基础。
- [Accelerate Inference](https://huggingface.co/docs/transformers/main/en/en/perf_infer_speed_up_gpu_cpu) — Hugging Face (2024)
  Publisher: Hugging Face
  官方文档，提供优化Transformer推理的实用指导，包括Hugging Face生态系统中KV缓存的实现和使用方式。
- [vLLM: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: SOSP 2023; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  解决大型语言模型服务中KV缓存的内存管理挑战，并介绍Paged Attention作为一种先进的优化技术。
- [Attention Mechanisms and Transformers](https://d2l.ai/chapter_attention-mechanisms/transformer.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2023)
  Publisher: Cambridge University Press
  一本开源深度学习教材的章节，以教育形式清晰地解释了Transformer架构、自注意力机制和相关概念。
