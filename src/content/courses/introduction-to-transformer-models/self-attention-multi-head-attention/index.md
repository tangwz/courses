---
course: "introduction-to-transformer-models"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention"
sourceId: 663
chapter: "self-attention-multi-head-attention"
title: "自注意力与多头注意力"
order: 2
description: "阐述Transformer模型中使用的缩放点积注意力与多头注意力机制。"
hasQuiz: true
---

基于先前介绍的注意力机制，本章将着重介绍Transformer模型中使用的特定注意力机制。我们将研究自注意力，这是一种技术，允许模型在处理某个特定词语时，衡量*同一*输入序列内不同词语的重要性。

您将学习输入嵌入如何投影到查询($Q$)、键($K$)和值($V$)向量，这些向量构成了计算注意力分数的依据。我们将详细说明缩放点积注意力公式：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

这里 $d_k$ 是键向量的维度。

此外，我们将研究多头注意力。这种方法涉及并行运行缩放点积注意力机制多次，使用不同的、经过学习的$Q$、$K$和$V$线性投影。这使得模型能够共同关注来自不同表示子空间的信息，在不同的位置上。我们将介绍其工作原理以及其有效性背后的原理。最后，本章包含一个实践练习，您将使用深度学习库实现缩放点积注意力机制。
