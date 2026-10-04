---
course: "introduction-to-transformer-models"
chapter: "self-attention-multi-head-attention"
lesson: "multi-head-attention-introduction"
sourceId: 2980
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention/multi-head-attention-introduction"
title: "多头注意力简介"
description: "阐述并行使用多个注意力头的理由。"
order: 5
plots: []
sourceHash: "eb3c919deb3cb5bed33032b3f1f6601e1bd084866f99fe31edd9f0833d1aaea8"
sourceCorrections: []
---

缩放点积注意力使模型能够计算序列中不同位置间的相关性分数，使用查询 ($Q$)、键 ($K$) 和值 ($V$) 向量 (vector)。这种机制很有效，使得模型在为特定位置生成输出表示时，能够关注输入中的相关部分。

但是，一次注意力计算就足够了吗？当我们处理语言时，通常会同时考虑多个方面。比如，在阅读“The quick brown fox jumps over the lazy dog”时，理解“jumps”这个词可能需要查看主语（“fox”）以确定语法一致性，但也可能将其与宾语（“dog”）或方式（“quick”）联系起来。单组注意力权重 (weight)可能难以有效地捕捉这些不同关系。它可能将不同类型的依赖关系平均化，或主要集中在一个方面，从而可能遗漏其他重要的连接。

这一观察引出了**多头注意力 (multi-head attention)**的想法。其核心思想直观而有效：即不再仅仅基于原始的$Q$、$K$和$V$向量执行一次注意力计算，而是并行进行多次注意力计算。每一次并行计算都被称作一个“注意力头”。

可以把它想象成有多位专家在检查同一序列。每位专家（头）可能专门寻找不同的模式或关联。一个头可能侧重于短距离句法连接，另一个侧重于更长距离的语义相似性，还有另一个侧重于位置关系。

重要的一点是，每个注意力头并非简单地在相同的原始$Q$、$K$和$V$上重新计算注意力。相反，在计算注意力之前，模型会为每个头学习独立的线性投影。这意味着原始的$Q$、$K$和$V$向量会被投影到每个头不同的、低维的子空间中。

假设我们有$h$个注意力头。对于每个头$i$（其中$i$的范围是1到$h$），我们学习投影矩阵$W^Q_i$、$W^K_i$和$W^V_i$。然后，输入$Q$、$K$和$V$按如下方式投影：

$Q_i = Q W^Q_i$
$K_i = K W^K_i$
$V_i = V W^V_i$

每个头$i$接着使用其自身的投影$Q_i$、$K_i$和$V_i$执行缩放点积注意力计算：


$$
\text{头}_i = \text{注意力}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i
$$


这里，$d_k$是*单个头内部*向量的维度（即$K_i$的维度）。

直观来说，每个投影（$W^Q_i, W^K_i, W^V_i$）使得一个头能够学习关注原始嵌入 (embedding)中包含的信息的不同方面或“表示子空间”。通过并行执行这些计算，模型能够同时获取序列中不同类型关系的见解。

这种并行处理使得模型能够基于不同的表示准则，同时关注来自不同位置的信息。结合后的输出，正如我们将在下一节中看到的，与单个注意力机制 (attention mechanism)所能产生的相比，提供了更丰富、更多方面的表示。我们现在将查看这些并行计算是如何被管理和结合，以生成多头注意力层的最终输出的。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇开创性论文介绍了Transformer架构和多头注意力机制，为理解该主题的起源和细节提供了基础性见解。
- [Transformers and Self-Attention (Lecture Notes / Slide Deck)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFHMjwR5oxxNi8PlollkpRBbS_yLihrZMh8leZQpnWqQqyf9K1Aau-rBPI-O4Y6p7PukheMcJ2_rqJdbwowDQhFDEnzXjvZ_vMw550Mk144_Su4EnHzRwQ3gD5p8wXcxOjimkRLrQOltxNvtwlwC58S4GPU4XU4BlXgYNsFXzyQrwI_Si3rHmNgJwxvygAMJXAbKtoWVAqFuZbizcJIb7qW_-0omds=) — Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  来自一所顶尖大学课程的综合讲义，在Transformer的背景下，清晰易懂地解释了多头注意力机制。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Draft; Pages: Chapter 11, Section 11.2
  一本在自然语言处理领域广受推崇的教科书，对多头注意力机制及其在Transformer网络中的作用提供了详细的教学解释。
