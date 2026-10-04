# 将FFN替换为Transformer中的MoE层

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-5-integrating-moe-into-architectures/replacing-ffn-with-moe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准Transformer架构中的前馈网络（FFN），也称作逐位置前馈网络，是一个简单但在计算上不可忽视的一部分。它通常由两个线性变换构成，中间夹着一个非线性激活函数 (activation function)，并在每个位置独立应用。这个FFN子层集中了模型很大一部分参数 (parameter)和浮点运算（FLOPs），因此成为用专家混合（MoE）层替代的主要对象。

目标是用一个稀疏的MoE层替换密集FFN，从而建立一个参数量大得多，但单次前向传播计算成本相当的模型。这种替换不仅仅是理论探讨；它是一种直接的架构修改，旨在有效提升模型容量。

### 替换的结构解析

Transformer块中的FFN充当一个模块化单元。它接收形状为`(batch_size, sequence_length, d_model)`的输入张量，并生成相同形状的输出张量。MoE层被设计为一种“即插即用”型替换，这意味着它必须符合相同的输入和输出张量维度。

下图阐明了这种架构变化。左侧是一个标准Transformer块，右侧是一个FFN已被MoE层替换的块。

> 密集前馈网络被稀疏专家混合层替换。核心注意力机制 (attention mechanism)和残差连接保持不变。MoE层内部包含一个门控网络和一组独立的专家网络。

尽管外部接口保持不变，但内部计算发生显著变化。不再是每个token都通过FFN中相同的权重 (weight)集，MoE层内的门控网络会将每个token引导至一小部分专家（通常只有一个或两个）。

### 实际操作中的实现

让我们查看一下这种变化在代码中如何体现。一个在PyTorch等框架中实现的典型Transformer块可能如下所示：

```python
# 一个标准Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff) # 密集的FFN
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力子层
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x
```

为了嵌入 (embedding)一个MoE层，我们将`FeedForward`模块替换为我们的`MoELayer`。前向传播中出现一个重要区别：MoE层还必须返回其辅助损失，这是训练时必需的，以确保均衡路由。

```python
# 带MoE层的Transformer块
class MoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts, top_k, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN被MoE层替换
        self.moe_layer = MoELayer(d_model, num_experts, top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 注意力子层（保持不变）
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # MoE子层
        # MoE层返回输出和辅助损失
        moe_output, aux_loss = self.moe_layer(x)
        x = self.norm2(x + self.dropout(moe_output))

        # 辅助损失必须传递到训练循环中
        return x, aux_loss
```

请注意，`forward`方法的返回签名已从`x`变为`x, aux_loss`。这种变化会向上层传播，贯穿模型的 `forward` 主方法。主训练循环必须设计为收集每个MoE块的这些辅助损失，对它们求和，并在反向传播 (backpropagation)之前将它们加到主损失函数 (loss function)（例如交叉熵）中。

### 权衡分析

这种替换根本上改变了模型的资源配置。假设一个标准FFN具有10亿个参数 (parameter)的Transformer。用一个包含8个MoE层替换它，每个专家的大小都与原始FFN相同，这将导致仅在该层就形成一个约80亿参数的模型。

然而，如果门控机制设置为`top_k=1`，则计算FLOPs与原始密集模型几乎保持不变。每个token仅由一个专家处理，因此计算量等同于通过一个单一的FFN，再加上门控网络的少量开销。

| 属性 | 标准FFN | MoE层 (N=8, k=1) |
| --- | --- | --- |
| **参数量** | $P_{FFN}$ | $\approx N \times P_{FFN} = 8 \times P_{FFN}$ |
| **每个Token的FLOPs** | $F_{FFN}$ | $\approx k \times F_{FFN} = 1 \times F_{FFN}$ |

这种解耦是核心优势。我们实现了模型参数量的显著增长，这代表了其知识容量，而推理 (inference)或训练的计算成本没有相应增加。这使得在现有硬件上训练具有万亿参数的模型成为可能，这是使用密集架构无法完成的壮举。主要的新挑战是管理存储所有$N$个专家所需的内存，这将在后续章节讨论的并行策略中得到解决。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (2017)
  Journal: arXiv preprint arXiv:1701.06538; DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  介绍了稀疏门控混合专家（MoE）层、其架构以及用于负载均衡的辅助损失。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.55917/cbdd4778](https://doi.org/10.55917/cbdd4778)
  介绍了Transformer架构的论文，详细说明了其组件，包括MoE层所替代的传统前馈网络（FFN）。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/v23/21-0990.html) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 23; Pages: 1-39; DOI: [10.48550/arXiv.2101.03961](https://doi.org/10.48550/arXiv.2101.03961)
  展示了MoE层在大型Transformer模型中替换FFN的应用，详细说明了效率提升和参数扩展。

---

[上一节](../04-%E9%AB%98%E6%95%88%E7%9A%84MoE%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%BC%98%E5%8C%96%E6%8E%A8%E7%90%86%E7%AE%A1%E7%BA%BF.md) · [下一节](02-MoE%20%E5%B1%82%E7%9A%84%E4%BD%8D%E7%BD%AE%EF%BC%9A%E9%A2%91%E7%8E%87%E4%B8%8E%E9%83%A8%E4%BD%8D.md)
