---
course: "how-to-build-a-large-language-model"
chapter: "transformer-architecture"
lesson: "positional-encoding-techniques"
sourceId: 5965
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-4-transformer-architecture/positional-encoding-techniques"
title: "位置编码方法"
description: "解释用于序列顺序的正弦和学习到的位置编码。"
order: 4
plots: ["plots/5965-0.json"]
sourceHash: "33cf4e468b65b81dd56b603c8c6717febafc88d11f1b5745b980172e8619958e"
sourceCorrections: []
---

自注意力 (self-attention)机制 (attention mechanism)，正如我们所见，同时处理所有输入词元 (token)。尽管它在捕获与距离无关的依赖关系方面表现出色，但这种并行处理也有一个不足之处：标准的自注意力操作是排列不变的。如果你打乱输入词元，注意力输出（在添加位置信息之前）将只是原始输出的一个打乱版本。它本身不具备序列顺序的知识。“猫坐在垫子上”和“垫子坐在猫上”在自注意力层看来是完全一样的。显然，对于语言建模和大多数序列任务，顺序是必不可少的。我们需要一种方法将每个词元的位置信息融入模型中。这通过位置编码 (positional encoding)来实现。

位置编码通过创建一个向量 (vector)来实现，该向量表示词元在序列中的位置。这个向量随后被添加到对应词元的输入嵌入 (embedding)中。这个整合了的嵌入，现在既包含语义信息（来自词元嵌入）又包含位置信息，被输入到Transformer堆栈中。

$\text{最终输入嵌入} = \text{词元嵌入} + \text{位置编码}$

有多种方法可以生成这些位置编码向量。

### 学习到的位置嵌入 (embedding)

或许最直接的方法是像学习词元 (token)嵌入一样学习位置编码 (positional encoding)。我们可以定义一个最大序列长度，比如$L_{max}$，并创建一个大小为$(L_{max}, d_{\text{model}})$的嵌入矩阵，其中$d_{\text{model}}$是模型嵌入的维度。对于位置$pos$（其中$0 \le pos < L_{max}$）处的词元，我们只需在这个嵌入矩阵中查找第$pos$个向量 (vector)并将其添加到词元的嵌入中。

在PyTorch中，这可以使用`nn.Embedding`来实现：

```python
import torch
import torch.nn as nn

# 示例参数
max_seq_len = 512
d_model = 768

# 学习到的位置嵌入层
positional_embedding_table = nn.Embedding(max_seq_len, d_model)

# 示例用法：获取位置 0, 1, 2, ..., seq_len-1 的嵌入
seq_len = 100
positions = torch.arange(0, seq_len,
                         dtype=torch.long).unsqueeze(0) # 形状: (1, seq_len)
learned_pe = positional_embedding_table(positions)
# 形状: (1, seq_len, d_model)

print(f"学习到的位置嵌入的形状: {learned_pe.shape}")
# 输出: 学习到的位置嵌入的形状:
# torch.Size([1, 100, 768])
```

这种方法简单，并让模型能够学习表示特定任务和数据位置的最佳方式。然而，它也有不足之处：

1. **参数 (parameter)数量：** 它会为模型增加$L_{max} \times d_{\text{model}}$个参数。对于非常大的模型和长序列来说，这会是很大的开销。
2. **外推能力：** 它通常无法很好地外推到训练时未见过的、长于$L_{max}$的序列长度。模型尚未学习到超过此限制的位置嵌入。

### 固定正弦位置编码 (positional encoding)

原始的Transformer论文（Vaswani等人，2017）提出了一种固定的、非学习的位置编码方法，该方法使用了不同频率的正弦和余弦函数。这样做的出发点是使用一个确定性函数，它可能使得模型更容易关注相对位置，因为对于任何固定偏移量$k$， $PE_{pos+k}$可以表示为$PE_{pos}$的线性函数。它还避免了学习到的嵌入 (embedding)所带来的额外参数 (parameter)，并且可能对未见过的序列长度有更好的泛化能力。

位置编码$PE$的公式，对于位置$pos$和维度索引$i$处的词元 (token)定义为：

$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$
$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$

这里：

- $pos$是词元在序列中的位置（0, 1, 2, ...）。
- $i$是嵌入向量 (vector)中的维度索引（$0 \le i < d_{\text{model}}/2$）。我们对偶数索引（$2i$）计算正弦，对奇数索引（$2i+1$）计算余弦。
- $d_{\text{model}}$是嵌入的维度。

位置编码的每个维度都对应一个正弦曲线。波长形成一个从$2\pi$到$10000 \cdot 2\pi$的等比数列。这种选择使得模型可能学习关注相对位置，因为相对位置信息编码在相位差中。

我们用PyTorch来实现它：

```python
import torch
import math
import matplotlib.pyplot as plt

def get_sinusoidal_positional_encoding(seq_len, d_model):
    """计算正弦位置编码。"""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(
        0, seq_len, dtype=torch.float
    ).unsqueeze(1)
    # 形状: (seq_len, 1)

    # 用于计算频率的项
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )
    # 形状: (d_model/2)

    # 对偶数索引计算正弦，对奇数索引计算余弦
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 添加批处理维度（可选，通常稍后完成）
    # pe = pe.unsqueeze(0) # 形状: (1, seq_len, d_model)
    return pe

# 示例：生成序列长度100、模型维度128的编码
seq_len = 100
d_model = 128
fixed_pe = get_sinusoidal_positional_encoding(seq_len, d_model)
# 形状: (100, 128)

print(f"固定位置嵌入的形状: {fixed_pe.shape}")
# 输出: 固定位置嵌入的形状:
# torch.Size([100, 128])

# 可视化前几个维度
plt.figure(figsize=(10, 5))
# 绘制维度 0, 2, 4, 6
for i in range(0, 8, 2):
    plt.plot(fixed_pe[:, i].numpy(), label=f'维度 {i} (sin)')
# 绘制维度 1, 3, 5, 7
for i in range(1, 9, 2):
     plt.plot(
         fixed_pe[:, i].numpy(),
         label=f'维度 {i} (cos)',
         linestyle='--'
     )
plt.ylabel("值")
plt.xlabel("位置")
plt.title("正弦位置编码（前8个维度）")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
```



![正弦位置编码（前8个维度）](plots/5965-0.json)



> 前10个位置的正弦位置编码的前8个维度。请注意，较低维度（较小的$i$）比高维度变化更快（频率更高）。

尽管正弦编码在原始Transformer和BERT等模型中有效且被广泛采用，但它们是固定的。它们可能不是所有类型序列模式的最佳表示。

### 选择合适的方法

学习到的位置编码 (positional encoding)和固定的正弦位置编码都是常见的起始点。

- **学习式：** 最初实现起来更简单，如果使用足够的数据和适当的最大长度，可能更具适应性。常用于BERT等模型。
- **固定正弦式：** 没有额外参数 (parameter)，潜在地具有更好的长度泛化能力，确定性。用于原始Transformer以及GPT-2/3等模型。

实际应用中，选择可能取决于具体的应用、模型大小和序列长度需求。此外也值得注意的是，这个领域已经发展出了更高级的方法。绝对位置编码，无论是学习式的还是固定的，主要编码词元 (token)相对于序列起点的绝对位置。然而，对注意力而言，词元间的*相对*位置往往才是最重要的。例如相对位置编码和旋转位置嵌入 (embedding)（RoPE）等方法，直接将相对距离信息融入注意力机制 (attention mechanism)本身。这些更高级的方法在第13章中有介绍。

目前，理解学习到的位置编码和正弦编码，为Transformer如何融入序列顺序信息提供了必要的前提，从而克服了核心自注意力 (self-attention)机制的排列不变性。这种位置数据注入是一个简单但必不可少的要素，使得Transformer在处理序列数据上获得成功。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构和固定正弦位置编码机制。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2018)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Pages: 4171-4186; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  介绍了BERT，一个使用学习位置编码的基于Transformer的语言模型。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  一本提供Transformer架构、注意力机制和位置编码方法解释的教科书。
- [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — PyTorch Authors (2024)
  PyTorch `nn.Embedding`模块的官方文档，用于实现学习位置编码。
