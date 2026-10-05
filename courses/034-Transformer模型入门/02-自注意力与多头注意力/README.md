# 第 2 章：自注意力与多头注意力

来源：[原章节](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention)

[返回课程目录](../README.md)

基于先前介绍的注意力机制，本章将着重介绍Transformer模型中使用的特定注意力机制。我们将研究自注意力，这是一种技术，允许模型在处理某个特定词语时，衡量*同一*输入序列内不同词语的重要性。

您将学习输入嵌入如何投影到查询($Q$)、键($K$)和值($V$)向量，这些向量构成了计算注意力分数的依据。我们将详细说明缩放点积注意力公式：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

这里 $d_k$ 是键向量的维度。

此外，我们将研究多头注意力。这种方法涉及并行运行缩放点积注意力机制多次，使用不同的、经过学习的$Q$、$K$和$V$线性投影。这使得模型能够共同关注来自不同表示子空间的信息，在不同的位置上。我们将介绍其工作原理以及其有效性背后的原理。最后，本章包含一个实践练习，您将使用深度学习库实现缩放点积注意力机制。

## 小节

- 1. [自注意力的原理](01-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 2. [自注意力机制中的查询、键和值向量](02-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%AD%E7%9A%84%E6%9F%A5%E8%AF%A2%E3%80%81%E9%94%AE%E5%92%8C%E5%80%BC%E5%90%91%E9%87%8F.md)
- 3. [缩放点积注意力机制](03-%E7%BC%A9%E6%94%BE%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 4. [自注意力得分可视化](04-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E5%BE%97%E5%88%86%E5%8F%AF%E8%A7%86%E5%8C%96.md)
- 5. [多头注意力简介](05-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%E7%AE%80%E4%BB%8B.md)
- 6. [多头注意力机制如何运作](06-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E5%A6%82%E4%BD%95%E8%BF%90%E4%BD%9C.md)
- 7. [多头注意力机制的优势](07-%E5%A4%9A%E5%A4%B4%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E7%9A%84%E4%BC%98%E5%8A%BF.md)
- 8. [动手实践：实现缩放点积注意力](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E7%BC%A9%E6%94%BE%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention/quiz)
