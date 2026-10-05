# 每个注意力头的Q、K、V线性投影

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-3-multi-head-self-attention/linear-projections-qkv-heads)

[返回章节目录](README.md) · [返回课程目录](../README.md)

已然明确，仅依赖单一自注意力 (self-attention)机制 (attention mechanism)，会限制模型获取序列内多重关系的能力。单次注意力计算可能平均化各类关联信号，或会因此遗失特定有益信息。多头注意力 (multi-head attention)通过并行执行多次注意力计算来应对此况，使得模型能同时考量来自不同视角或“表示子空间”的信息。

多头注意力（Multi-Head Attention）机制通过为每个“头”生成不同的查询（$Q$）、键（$K$）和值（$V$）向量 (vector)来执行并行处理。这意味着我们并非直接将原始输入嵌入 (embedding)送入每个头的注意力计算，而是对每个头单独应用学得的线性变换到输入序列上。

设输入序列的表示为一个矩阵 $X \in \mathbb{R}^{n \times d_{model}}$，其中 $n$ 是序列长度，$d_{model}$ 是输入嵌入的维度（以及模型的隐藏状态大小）。对于一个有 $h$ 个头的多头注意力机制，我们引入 $h$ 组可学习的权重 (weight)矩阵：

- 查询权重矩阵：$W^Q_i \in \mathbb{R}^{d_{model} \times d_k}$，适用于 $i=1, ..., h$
- 键权重矩阵：$W^K_i \in \mathbb{R}^{d_{model} \times d_k}$，适用于 $i=1, ..., h$
- 值权重矩阵：$W^V_i \in \mathbb{R}^{d_{model} \times d_v}$，适用于 $i=1, ..., h$

这里，$d_k$ 表示每个头的键和查询的维度，$d_v$ 表示每个头的值的维度。一种常见且高效的设定，在原版Transformer论文《Attention Is All You Need》中有所采用，是将 $d_k = d_v = d_{model} / h$。这可确保所有头的总计算开销，与作用于完整 $d_{model}$ 维度的单一注意力机制大致相同，同时分散了表示能力。

对于每个头 $i$，其相应的查询、键和值矩阵（$Q_i$, $K_i$, $V_i$）通过将输入 $X$ 投影，并使用这些权重矩阵来计算：


$$
Q_i = X W^Q_i
$$


$$
K_i = X W^K_i
$$


$$
V_i = X W^V_i
$$


所得矩阵 $Q_i, K_i \in \mathbb{R}^{n \times d_k}$ 和 $V_i \in \mathbb{R}^{n \times d_v}$ 现在表示输入序列已变换至与头 $i$ 相关的特定子空间。

> 流程图描绘了输入嵌入 $X$ 如何为每个注意力头 $i$ 独立进行投影，使用不同的权重矩阵（$W^Q_i, W^K_i, W^V_i$）以生成头专属的查询 ($Q_i$)、键 ($K_i$) 和值 ($V_i$) 矩阵。投影通常将维度从 $d_{model}$ 降低到 $d_k$ 或 $d_v$。

### 使用线性投影的缘由

这些独立的线性变换之所以具有重要性，缘由如下：

1. **学习不同子空间：** 每组投影矩阵 $(W^Q_i, W^K_i, W^V_i)$ 实际定义了一个不同的线性子空间。将输入嵌入 (embedding)投影到这些不同子空间中，使得每个注意力头能专注于序列中不同类型或方面的关系。例如，一个头可以学会关注短程句法依赖，而另一个则可能侧重于长程语义关联 (semantic relationship)。
2. **增强表示能力：** 采用 $h$ 组投影矩阵，使得模型拥有更多参数 (parameter)，因而比单一组投影更能学习复杂模式。即使每个头在降维空间 ($d_k, d_v$) 上运作，在汇集所有头的信息后，其综合效果也可能比在完整 $d_{model}$ 维度上运行的单个头更为强大。
3. **促成并行专长化：** 由于每个头 $i$ 接收到其自身的投影版本 $Q_i, K_i, V_i$，因此它在训练过程中可以相对独立于其他头来学习专门的注意力模式。这鼓励了学得表示的多样性。

在实践中，这些线性投影通过标准前馈层（在PyTorch或TensorFlow等框架中常称为线性层或全连接层）实现，不含偏置 (bias)项。权重 (weight)矩阵 $W^Q_i, W^K_i, W^V_i$ 随机初始化，并在训练过程中通过反向传播 (backpropagation)与网络其余参数一同习得。

一旦这些头专属的 $Q_i, K_i, V_i$ 矩阵计算完成，它们将作为并行缩放点积注意力计算的输入，此部分我们接下来将查看。这一投影步骤对于多头注意力 (multi-head attention)机制 (attention mechanism)如何同时从多个视角分析输入序列来说，扮演着基础角色。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer模型和多头注意力机制，详细阐述了Q、K、V向量的线性投影。
- [Transformers and Pretrained Language Models (Lecture Notes CS224N, Winter 2023)](https://web.stanford.edu/class/cs224n/readings/cs224n-2023-notes-L8.pdf) — Abigail See, Chris Manning, and Stanford CS224N Staff (2023)
  Publisher: Stanford University
  作为一门面向NLP的深度学习大学课程的一部分，清晰说明了Transformer组件，包括多头注意力中线性投影的用途。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  提供了一份关于自然语言处理的综合论述，其中包含关于Transformer架构、多头注意力及QKV投影的详细章节。

---

[上一节](03-%E5%BC%95%E5%85%A5%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B.md) · [下一节](05-%E5%B9%B6%E8%A1%8C%E6%B3%A8%E6%84%8F%E5%8A%9B%E8%AE%A1%E7%AE%97.md)
