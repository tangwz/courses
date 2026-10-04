---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "masked-self-attention"
sourceId: 2014
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/masked-self-attention"
title: "解码器中的掩码自注意力"
description: "掩码如何阻止自回归生成过程中对未来词元的关注。"
order: 4
plots: ["plots/2014-0.json"]
sourceHash: "5c875f0fc7f3aa36292f3d202fb4c0b010acb9446cd82295fb2d629db5074443"
sourceCorrections: []
---

在解码器堆栈中，自注意力 (self-attention)机制 (attention mechanism)的运作方式与编码器不同。回顾一下，编码器同时处理整个输入序列，使每个位置能够关注所有其他位置（包括相对于自身的未来位置）。这种双向上下文 (context)对于理解输入序列结构很有帮助。

解码器的主要作用，尤其是在机器翻译或文本生成等任务中，通常是自回归 (autoregressive)的。这表示它一次生成一个输出序列中的词元 (token)，从左到右。当预测位置 $i$ 的词元时，解码器应仅能访问先前生成的词元（位置 $1$ 到 $i-1$）以及完整的编码输入序列。它绝不能“预先查看”当前正在生成的目标序列中位置 $i, i+1, \dots$ 的词元。允许这种访问会使训练期间的生成任务变得过于简单，因为模型可以简单地复制下一个词元，而不是学习去预测它。

为实现这种单向信息流动，解码器采用**掩码自注意力**。其主要思路是通过遮蔽（设置为负无穷）任何对应于未来位置连接的注意力分数，来修改标准的缩放点积注意力计算。

### 掩码机制

掩码操作发生在softmax函数应用于缩放注意力分数之前。标准的缩放点积注意力分数计算如下：


$$
\text{分数} = \frac{QK^T}{\sqrt{d_k}}
$$


其中 $Q$、$K$ 和 $V$ 是从解码器输入（或前一个解码器层的输出）获得的查询、键和值矩阵，$d_k$ 是键的维度。

创建一个掩码矩阵 $M$。该矩阵的维度通常与注意力分数兼容（序列长度 \u00d7 序列长度）。对于位置 $i$ 关注位置 $j$ 的情况：

- 如果 $j \le i$ （当前或之前的位置），掩码值 $M_{ij}$ 为 $0$。
- 如果 $j > i$ （未来位置），掩码值 $M_{ij}$ 为 $-\infty$ （或在实践中为非常大的负数，如 -1e9）。

然后将此掩码矩阵 $M$ 添加到注意力分数中：


$$
\text{掩码分数} = \text{分数} + M
$$


最后，对这些掩码分数应用 softmax 函数：


$$
\text{注意力权重} = \text{softmax}(\text{掩码分数})
$$


添加 $-\infty$ 的作用是，在 softmax 函数内部进行指数运算后，这些分数变为 $e^{-\infty} = 0$。因此，未来位置的注意力权重 (weight)变为零，从而有效地阻止了来自这些位置的任何信息流动。解码器位置 $i$ 只能关注位置 $1$ 到 $i$。



![掩码注意力模式（示意图）](plots/2014-0.json)



> 序列长度为 5 的注意力掩码的可视化。蓝色单元格（值 1）表示查询（行）可以关注的位置（列）。灰色单元格（值 0）表示被遮蔽的位置（未来词元 (token)）。请注意，每个位置都可以关注自身以及所有先前的位置。

这种因果注意力机制 (attention mechanism)，使Transformer解码器得以高效地学习序列生成任务。它确保每一步的预测仅依赖于之前步骤的已知输出，这与实际推理 (inference)或生成时的条件相符。这与编码器的自注意力 (self-attention)形成鲜明对比，后者可以自由地引入整个输入序列的信息。掩码自注意力（用于处理已生成的序列）和交叉注意力（用于整合来自编码器的信息）的结合，使得解码器能够生成连贯且上下文 (context)相关的输出序列。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构及其组成部分的奠基性论文，详述了解码器中的遮蔽自注意力机制。
- [Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms/transformer.html) — Aston Zhang, Zack C. Lipton, Mu Li, and Alexander J. Smola (2024)
  Publisher: Cambridge University Press
  一本全面的开源教科书，提供了深度学习模型的详细解释和实际实现，其中有一章专门介绍Transformer架构及其遮蔽自注意力机制。
