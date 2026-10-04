---
course: "foundations-transformers-architecture"
chapter: "attention-mechanism-core-concepts"
lesson: "practice-implementing-attention"
sourceId: 1947
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-2-attention-mechanism-core-concepts/practice-implementing-attention"
title: "实践：实现缩放点积注意力"
description: "从头开始实际操作实现缩放点积注意力机制。"
order: 7
plots: ["plots/1947-0.json"]
sourceHash: "b607fb12f66a0ef1005edfd39528bbca01abff46cb76676ceefdb14ab6ca88b6"
sourceCorrections: []
---

提供了一个使用 PyTorch 进行的缩放点积注意力实际操作实现。此实践应用侧重于注意力机制 (attention mechanism)的主要计算。在构建像多头注意力 (multi-head attention)这样更复杂的结构之前，理解此实现是必需的。

回顾一下公式：
$\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

我们将逐步实现它。

### 设置

首先，确保您已安装 PyTorch。我们将需要基本的 `torch` 库和 `math` 模块来进行平方根运算。

```python
import torch
import torch.nn.functional as F
import math
import plotly.graph_objects as go # For visualization
import numpy as np # For visualization data handling
```

### 实现注意力函数

让我们定义一个函数 `scaled_dot_product_attention`，它接受查询 ($Q$)、键 ($K$)、值 ($V$) 以及一个可选的掩码作为输入。

为简化本初始示例，我们假设 $Q$、$K$ 和 $V$ 的形状类似于 `[batch_size, sequence_length, dimension]`。在完整的 Transformer 中，这些张量通常会有一个额外的“头”维度，我们将在之后处理。维度 $d_k$ 对应于键张量的最后一个维度。

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力。

    参数：
        query: 查询张量；形状 (batch_size, seq_len_q, d_k)
        key: 键张量；形状 (batch_size, seq_len_k, d_k)
        value: 值张量；形状 (batch_size, seq_len_k, d_v)
               注意：key 和 value 的 seq_len_k 必须匹配。
               d_v（值维度）可以与 d_k（查询维度）不同。
        mask: 可选掩码张量；形状可广播到 (batch_size, seq_len_q, seq_len_k)。
              值为 True 或 1 的位置表示保留值，False 或 0 表示屏蔽掉。

    返回：
        output: 注意力输出张量；形状 (batch_size, seq_len_q, d_v)
        attention_weights: 注意力权重；形状 (batch_size, seq_len_q, seq_len_k)
    """
    # 键的维度
    d_k = key.size(-1)

    # 1. 计算 QK^T
    # (batch_size, seq_len_q, d_k) @ (batch_size, d_k, seq_len_k) -> (batch_size, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. 按 sqrt(d_k) 缩放
    scores = scores / math.sqrt(d_k)

    # 3. 应用掩码（如果提供）
    if mask is not None:
        # 掩码值通常在需要屏蔽的位置为 False
        # 在 softmax 之前，我们需要将屏蔽位置设置为一个很大的负值（-inf）
        # 为数值稳定性，使用 fill_value=-1e9 或类似的大的负数
        scores = scores.masked_fill(mask == 0, -1e9) # PyTorch 约定：0/False 表示屏蔽

    # 4. 应用 softmax 获取注意力权重
    # Softmax 应用于最后一个维度 (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    # 处理当一行中所有分数都为 -inf 时 softmax 可能产生的 NaN
    # 这可能发生在查询位置被完全屏蔽所有键的情况下。
    # 将 NaN 替换为 0。
    attention_weights = torch.nan_to_num(attention_weights)

    # 5. 将权重乘以 V
    # (batch_size, seq_len_q, seq_len_k) @ (batch_size, seq_len_k, d_v) -> (batch_size, seq_len_q, d_v)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

### 示例用法

让我们创建一些示例张量并测试我们的函数。为了演示，我们将使用较小的批量大小、序列长度和维度。

```python
# 示例参数
batch_size = 1
seq_len_q = 3 # 查询的序列长度
seq_len_k = 4 # 键/值的序列长度
d_k = 8       # 键/查询的维度
d_v = 16      # 值的维度

# 生成随机张量（实际应用中替换为真实嵌入）
query = torch.randn(batch_size, seq_len_q, d_k)
key = torch.randn(batch_size, seq_len_k, d_k)
value = torch.randn(batch_size, seq_len_k, d_v)

# --- 无掩码情况 ---
output, attention_weights = scaled_dot_product_attention(query, key, value)

print("--- 无掩码输出 ---")
print("输出形状:", output.shape) # 预期: [1, 3, 16]
print("注意力权重形状:", attention_weights.shape) # 预期: [1, 3, 4]
# attention_weights 中的每一行总和应为 1
print("注意力权重总和（第一个查询）:", attention_weights[0, 0, :].sum())

# --- 有掩码情况 ---
# 创建一个示例掩码。让我们屏蔽所有查询的最后一个位置。
# 掩码形状: (batch_size, seq_len_q, seq_len_k)
# 这里，更简单地: (batch_size, 1, seq_len_k) 可广播
mask = torch.ones(batch_size, 1, seq_len_k, dtype=torch.bool)
mask[:, :, -1] = 0 # 屏蔽最后一个位置（索引 3）

print("\n掩码形状:", mask.shape)
print("掩码内容:\n", mask)

output_masked, attention_weights_masked = scaled_dot_product_attention(query, key, value, mask=mask)

print("\n--- 有掩码输出 ---")
print("输出形状:", output_masked.shape) # 预期: [1, 3, 16]
print("注意力权重形状:", attention_weights_masked.shape) # 预期: [1, 3, 4]
print("屏蔽后的注意力权重（第一个查询）:\n", attention_weights_masked[0, 0, :])
# 请注意，最后一个位置（索引 3）的权重应为 0 或非常接近 0。
print("注意力权重总和（第一个查询，已屏蔽）:", attention_weights_masked[0, 0, :].sum())
```

您应该观察到输出形状符合我们的预期。当应用掩码时，对应于被屏蔽位置（本例中为最后一个）的注意力权重 (weight)变为零，并且剩余的权重通过 softmax 重新归一化 (normalization)，使其总和为 1。

### 可视化注意力权重 (weight)

可视化注意力权重矩阵 ($\text{softmax}(\frac{QK^T}{\sqrt{d_k}})$) 可以帮助我们理解模型如何关联序列的不同部分。热图常用于此目的。让我们可视化来自我们无掩码示例的权重。

```python
# 使用来自无掩码示例的 attention_weights
weights_np = attention_weights[0].detach().numpy() # 获取第一个批次项的权重

# 创建热图数据
fig_data = go.Heatmap(
    z=weights_np,
    x=[f'位置 {i}' for i in range(seq_len_k)],
    y=[f'查询位置 {i}' for i in range(seq_len_q)],
    colorscale='Blues', # 使用蓝色配色方案
    colorbar=dict(title='注意力权重')
)

# 创建布局
fig_layout = go.Layout(
    title='注意力权重（查询 vs 键）',
    xaxis_title="序列位置",
    yaxis_title="查询序列位置",
    yaxis_autorange='reversed', # 将查询 0 显示在顶部
    width=500, height=400, margin=dict(l=50, r=50, b=100, t=100, pad=4)
)

# 生成图形对象为 JSON，用于网页显示
fig = go.Figure(data=[fig_data], layout=fig_layout)
plotly_json = fig.to_json()
```



![注意力权重（查询 vs 键）](plots/1947-0.json)



> 热图显示了每个查询位置（行）对每个位置（列）的注意力权重。值越高（颜色越深），表示注意力越强。每行总和为 1。

这种可视化显示了对于每个查询位置（行），在计算其输出时，它给予每个位置（列）多少“注意力”或权重。在具有有意义数据的实际应用中，此矩阵中的模式可以显示模型学到的句法或语义关系。例如，一个动词可能会强烈关注其主语和宾语。

这种缩放点积注意力的实现构成了 Transformer 中每个注意力头内的核心计算单元。在下一章中，我们将在此之上构建，以实现多头注意力 (multi-head attention)，从而使模型能够共同关注来自不同表示子空间的信息。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  提出Transformer架构和Scaled Dot-Product Attention机制的开创性论文。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Core Team (2024)
  PyTorch官方文档，提供深度学习模型构建中使用的张量操作和函数详情。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2023)
  Publisher: Cambridge University Press
  一本开源教材，涵盖深度学习理论和实践，包含关于注意力机制和Transformer模型的章节。
