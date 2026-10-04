---
course: "introduction-to-transformer-models"
chapter: "sequence-modeling-attention-fundamentals"
lesson: "attention-context-vectors"
sourceId: 2966
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-1-sequence-modeling-attention-fundamentals/attention-context-vectors"
title: "来自注意力权重的上下文向量"
description: "描述如何使用注意力权重生成输入序列的加权概括（上下文向量）。"
order: 6
plots: []
sourceHash: "5a6d10392456fdf3c78872ccd068cd9d252799783a157ef163b2748a2686b3db"
sourceCorrections: []
---

注意力机制 (attention mechanism)计算分数，它们根据查询，为每个输入元素分配一个相关性权重 (weight)（$\alpha$）。这些分数指示我们对输入序列的不同部分给予*多少*关注。利用这些分数可以生成一个**上下文 (context)向量 (vector)**。

可以将上下文向量看作输入序列的精炼概括，专门针对当前的关注点或查询进行调整。与传统RNN通常将整个输入压缩成一个单一、固定大小的向量不同，注意力机制使得我们能够生成一个动态概括，突出显示最相关的部分。

### 从分数到概括：加权和

核心思路非常直接：上下文 (context)向量 (vector)是通过输入序列中值向量（$v$）的**加权求和**来计算的。在这个求和中使用的权重 (weight)正是我们之前计算的注意力分数（$\alpha$）。

数学上，如果我们有 $n$ 个输入元素，每个元素都有一个对应的值向量 $v_i$ 和一个注意力分数 $\alpha_i$（相对于当前查询），上下文向量 $C$ 的计算方式如下：


$$
C = \sum_{i=1}^{n} \alpha_i v_i
$$


让我们分解来看：

1. **值向量（$v_i$）：** 请记住，这些向量代表每个输入元素的内容或含义。
2. **注意力分数（$\alpha_i$）：** 这些是在上一步计算的非负权重（通常通过softmax），总和为1（$\sum_{i=1}^{n} \alpha_i = 1$）。每个$\alpha_i$反映了输入元素 $i$ 对当前查询的重要性。
3. **加权和：** 我们将每个值向量 $v_i$ 乘以其对应的注意力分数 $\alpha_i$。这会根据每个值向量的计算相关性来调整其贡献。对应高度相关输入（高 $\alpha_i$）的值向量将对总和贡献更多，而较不相关的输入（低 $\alpha_i$）的影响将被减小。
4. **上下文向量（$C$）：** 生成的向量 $C$ 是这些加权值向量的总和。它的维度与值向量相同，但表示输入的一种混合组合，按相关性加权。

### 过程的可视化

想象你有三个输入词，每个词由一个值向量 (vector)（$v_1, v_2, v_3$）表示，并且你已经根据特定查询计算出它们的注意力权重 (weight)（$\alpha_1, \alpha_2, \alpha_3$）。上下文 (context)向量 $C$ 是通过根据这些权重混合这些向量形成的。

> 该图示说明了每个值向量（$v_i$）如何乘以其对应的注意力权重（$\alpha_i$）。这些加权向量随后被求和以生成最终的上下文向量（$C$）。

### 为什么这有用？

这种机制具有很强的能力，因为它允许模型在输出生成的每一步（或处理的每个元素）动态地生成输入的有关概括。

以机器翻译为例。在翻译句子时，特定输出词的含义通常很大程度上取决于输入句子中的一两个词，再加上更广的语境。传统RNN由于其序列性质和固定大小的隐藏状态，可能难以保留来自距离较远输入词的准确信息。

然而，注意力机制 (attention mechanism)在*每个*输出步骤为*所有*输入词计算权重 (weight)。生成的上下文 (context)向量 (vector)直接包含了来自最相关输入词的信息，无论它们在序列中的位置如何。这克服了固定大小上下文向量的信息瓶颈，并使得模型能够更有效地处理长距离依赖。

这个上下文向量，富含来自输入的聚焦信息，成为下一处理阶段的主要输入，例如生成序列中的下一个词或馈送到网络中的后续层。我们将在接下来的章节中清楚地看到它如何融入更大的Transformer架构。

## 参考资料

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  这篇开创性论文在序列到序列模型中引入了注意力机制，详细说明了上下文向量是如何作为编码器隐藏状态的加权和来创建的，以聚焦于相关的输入部分。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇基础性论文介绍了Transformer架构，该架构完全依赖于注意力机制，并明确定义了通过Value向量的加权和计算上下文向量的方式。
- [CS224N: Natural Language Processing with Deep Learning - Lecture 8: Attention and Transformers](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture08-transformers.pdf) — Christopher Manning, John Hewitt (2023)
  Publisher: Stanford University
  提供了清晰易懂的讲义，解释了注意力、自注意力以及Transformer架构中上下文向量的形成，适合入门读者。
- [Speech and Language Processing (3rd ed. draft) - Chapter 9: Attention and Transformers](https://web.stanford.edu/~jurafsky/slp3/9.pdf) — Daniel Jurafsky and James H. Martin (2024)
  这本教科书的章节详细解释了注意力机制和Transformer模型，包括上下文向量的数学定义和作用。
