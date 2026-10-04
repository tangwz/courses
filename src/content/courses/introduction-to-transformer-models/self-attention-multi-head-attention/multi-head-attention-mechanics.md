---
course: "introduction-to-transformer-models"
chapter: "self-attention-multi-head-attention"
lesson: "multi-head-attention-mechanics"
sourceId: 2983
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-2-self-attention-multi-head-attention/multi-head-attention-mechanics"
title: "多头注意力机制如何运作"
description: "描述了Q、K、V的拆分、注意力应用、结果拼接以及最终线性投影的流程。"
order: 6
plots: []
sourceHash: "377a9e4fb2345978a859f5868128b9e1889b6ce9fa281f1f0d9b93660aa59a3f"
sourceCorrections: []
---

缩放点积注意力使得模型能够区分序列中不同token的优先级。然而，仅进行一次这样的计算可能会迫使注意力机制 (attention mechanism)对不同类型的关联进行平均。设想尝试理解一个句子，例如“这只疲惫的动物没有过马路，因为它太宽了。”当关注“它”这个词时，单一的注意力机制可能难以同时良好地捕捉“疲惫的动物”的关联和“街道宽度”的关联。

多头注意力 (multi-head attention)机制通过多次并行运行缩放点积注意力过程来解决此问题，每次都对原始的查询、键和值使用不同的学习变换。这使得每个“头”能够关注信息中不同的方面或表征子空间。

以下是分步过程：

1. **线性投影：**
   多头注意力机制不使用单一的查询（Q）、键（K）和值（V）矩阵集合，而是首先创建 $h$ 个不同的矩阵集合，其中 $h$ 是注意力头的数量（一个超参数 (parameter) (hyperparameter)）。对于每个头 $i$（从 $1$ 到 $h$），原始输入Q、K和V矩阵（在自注意力 (self-attention)的情况下通常源自相同的输入序列嵌入 (embedding)）使用学习到的权重 (weight)矩阵 $W^Q_i$、$W^K_i$ 和 $W^V_i$ 进行投影。

   - $Q_i = Q W^Q_i$
   - $K_i = K W^K_i$
   - $V_i = V W^V_i$

   通常，这些投影矩阵的维度小于原始嵌入维度（$d_{model}$）。如果输入嵌入维度为 $d_{model}$，每个头通常使用维度 $d_k = d_v = d_{model} / h$ 进行操作。这确保了总计算成本与具有完整维度的单一头注意力相似。这些权重矩阵（$W^Q_i, W^K_i, W^V_i$）对于每个头都是独有的，并在训练过程中学习。
2. **并行注意力计算：**
   然后，这些投影集合（$Q_i, K_i, V_i$）中的每一个都同时送入各自的缩放点积注意力机制。这产生了 $h$ 个独立的输出矩阵，我们称它们为 $\text{头}_i$：

   
   $$
   \text{头}_i = \text{注意力}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i
   $$
   

   每个 $\text{头}_i$ 矩阵根据头 $i$ 学习到的特定投影来捕获注意力信息。因为投影不同（对于每个 $i$， $W^Q_i, W^K_i, W^V_i$ 都不同），每个头可能学习关注输入序列中不同类型的关联或特征。
3. **拼接：**
   所有 $h$ 个注意力头的输出沿着特征维度拼接在一起。如果每个 $\text{头}_i$ 的维度是 $d_v$，拼接后的矩阵维度将是 $h \times d_v$。由于我们通常设置 $d_v = d_{model} / h$，拼接后的矩阵维度变为 $d_{model}$，这与原始输入嵌入维度相匹配。

   
   $$
   \text{拼接}(\text{头}_1, \text{头}_2, ..., \text{头}_h)
   $$
   
4. **最终线性投影：**
   此拼接后的输出随后会经过一个最终的线性投影层，由另一个学习到的权重矩阵 $W^O$ 参数化。这个投影混合了不同头学习到的信息，并生成多头注意力层的最终输出，其维度通常为 $d_{model}$。

   
   $$
   \text{多头}(Q, K, V) = \text{拼接}(\text{头}_1, ..., \text{头}_h)W^O
   $$
   

这个完整的多头注意力模块随后可作为更大的Transformer架构中的一个组成部分使用，替换单一的缩放点积注意力机制。

下图展现了信息流经一个包含 $h$ 个头的多头注意力模块的过程。

> 该图显示了输入Q、K和V矩阵如何首先为每个 $h$ 个注意力头独立投影。然后，缩放点积注意力并行应用于每个投影集。产生的注意力输出被拼接并通过一个最终的线性层，以生成多头注意力输出。

通过允许不同头学习不同的投影矩阵（$W^Q_i, W^K_i, W^V_i, W^O$），多头注意力机制使得模型能够共同关注不同位置上来自不同表征子空间的信息，从而相较于使用单一注意力机制，带来了更丰富、更有效的表征。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构和多头注意力机制的原始论文，详细说明了其机制和优势。
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Alexander Rush, Vincent Nguyen, Guillaume Klein (2018)
  对Transformer模型进行的全面逐行解释和PyTorch实现，包含对多头注意力的清晰阐述。
- [Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms/multi-head-attention.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alexander J. Smola (2024)
  Publisher: Cambridge University Press
  一本开源交互式教科书，为深度学习概念提供了详细的解释和代码示例，其中包含专门的多头注意力章节。
