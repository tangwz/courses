---
course: "foundations-transformers-architecture"
chapter: "positional-encoding-embedding-layer"
lesson: "practice-positional-encodings"
sourceId: 2006
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-4-positional-encoding-embedding-layer/practice-positional-encodings"
title: "实践：生成与可视化位置编码"
description: "实现并可视化针对不同序列长度和维度的正弦位置编码。"
order: 8
plots: ["plots/2006-0.json", "plots/2006-1.json"]
sourceHash: "71f2f8a5ac54aec018713debbb6209aeb9f9ee0fc0cb4b3ce1a9c31f40d5c134"
sourceCorrections: []
---

正弦位置编码 (positional encoding)对于向Transformer告知序列顺序很重要。直观了解这些编码的特性可以带来有益的直观认识。在此，将实现位置编码函数并可视化生成的向量 (vector)。

我们将使用Python配合NumPy进行数值计算，并使用Plotly进行交互式可视化，这非常适合基于网络的课程材料。

### 实现位置编码 (positional encoding)函数

首先，让我们将正弦位置编码的数学公式转换为代码。回顾这些公式：


$$
PE_{(位置, 2i)} = \sin(位置 / 10000^{2i/d_{model}})
$$


$$
PE_{(位置, 2i+1)} = \cos(位置 / 10000^{2i/d_{model}})
$$


此处 `pos` 是序列中的位置，$i$ 是嵌入 (embedding)向量 (vector)中维度的索引，$d_{model}$ 是嵌入的维度。

下面是一个使用NumPy生成这些编码的Python函数：

```python
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    """
    生成正弦位置编码。

    Args:
        max_seq_len: 最大序列长度。
        d_model: 模型嵌入的维度。

    Returns:
        一个形状为 (max_seq_len, d_model) 的NumPy数组，包含
        位置编码。
    """
    if d_model % 2 != 0:
        raise ValueError("d_model 必须是偶数才能容纳正弦/余弦对。")

    # 初始化位置编码矩阵
    pos_encoding = np.zeros((max_seq_len, d_model))

    # 创建一个位置列向量 [0, 1, ..., max_seq_len-1]
    position = np.arange(max_seq_len)[:, np.newaxis] # 形状：(max_seq_len, 1)

    # 计算除数项：1 / (10000^(2i / d_model))
    # 对应 i = 0, 1, ..., d_model/2 - 1
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)) # 形状：(d_model/2,)

    # 对偶数索引 (2i) 应用正弦函数
    pos_encoding[:, 0::2] = np.sin(position * div_term)

    # 对奇数索引 (2i + 1) 应用余弦函数
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding

# 示例用法：
max_len = 50  # 最大序列长度
d_model = 128 # 嵌入维度（必须是偶数）

positional_encodings = get_positional_encoding(max_len, d_model)

print(f"生成的位置编码形状：{positional_encodings.shape}")
# 输出：生成的位置编码形状：(50, 128)
```

此函数接收最大序列长度和模型的嵌入维度作为输入。它计算偶数索引的正弦值和奇数索引的余弦值，基于位置和表示频率成分的 `div_term`。结果是一个矩阵，其中每行对应序列中的一个位置，每列对应位置编码向量中的一个维度。

### 可视化位置编码 (positional encoding)

可视化这个矩阵有助于理解这些编码的结构。热力图是查看编码值如何跨位置和维度变化的有效方式。我们将生成序列长度为50、嵌入 (embedding)维度为128的编码。



![正弦位置编码](plots/2006-0.json)



> 热力图可视化了长度为50、嵌入维度为128的序列的正弦位置编码。每行代表一个位置，每列代表一个维度索引。颜色强度指示编码值。

### 分析可视化结果

从热力图中，之前讨论的几个特性变得直观显现：

1. **每个位置的独特编码：** 每行（位置）都有独特的颜色模式，代表其独特的编码向量 (vector)。这种独特性使得模型能够区分不同位置。
2. **变化的频率：** 观察维度轴（X轴）上的波长。
   - 最左侧的列（低维度索引，小 $i$）呈现高频变化（沿位置轴快速的颜色变化）。这些维度编码细粒度的位置信息。
   - 最右侧的列（高维度索引，大 $i$）显示低频变化（缓慢的颜色变化）。这些维度编码较粗略的、长距离的位置信息。
3. **平滑过渡：** 正弦特性确保了相邻位置编码 (positional encoding)之间的平滑过渡。
4. **有界值：** 由于正弦和余弦函数的作用，所有值本身都在 [-1, 1] 的范围内。

让我们通过绘制几个特定位置（例如位置0、10和25）在所有维度上的编码向量来进一步检查这种独特性。



![特定位置的位置编码向量 (正弦组件)](plots/2006-1.json)



> 线图比较了位置0、10和25的128维位置编码向量。每条线独特的形状突出了分配给每个序列位置的独特编码。

这些可视化结果证实了正弦位置编码为每个位置提供了独特的信号，并在不同频率的维度上平滑变化。随后，这个位置信号被添加到输入词元 (token)嵌入 (embedding)中，使后续的自注意力 (self-attention)层能够考虑序列中元素的顺序。

在下一章中，我们将组装这些组件以及多头注意力 (multi-head attention)机制 (attention mechanism)，构成完整的Transformer编码器和解码器堆栈。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构及原始正弦位置编码方法。
- [Natural Language Processing with Transformers](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFdGnppcL9JPwigiRUvpbYdwuVhMR5cNTgUBHuY4QIJfU8CiXsgcUBAheMNRtmq-mi7GASkY0wSzWHA2q3JnqoA-tVwyhgTcjt3yJndCFaEhF_qatCTiaQgBFttwNn4nlXdI-y0mzzWM2SL1yrGqL0rKDAvCfptHe4re_leoqJBhZQhsLMrdMwpKQ==) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一本关于Transformer的综合指南，涵盖其架构、实际应用和各种位置编码方案。
- [torch.nn.Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  提供PyTorch中Transformer模块的详细信息，说明位置编码如何集成到深度学习框架中。
- [Plotly Python Graphing Library Documentation](https://plotly.com/python/) — Plotly Technologies Inc. (2024)
  Plotly Python库的官方文档，对理解和创建所示可视化内容很有价值。
