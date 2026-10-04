---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "encoder-layer-structure"
sourceId: 2010
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/encoder-layer-structure"
title: "编码器层结构"
description: "对单个编码器层的详细剖析：多头自注意力机制和前馈网络。"
order: 2
plots: []
sourceHash: "fadd39d35dacd669f9c47db7cc04ba9e4a4ffda21e0805ce54e5b3aff5ab9558"
sourceCorrections: []
---

编码器堆栈由多个相同层组成，常表示为 $N$ 层（例如，在原始 Transformer 论文中 $N=6$）。每个编码器层的主要作用是将输入嵌入 (embedding)序列转换为语境化表示序列。这些表示纳入了整个输入序列的信息，使模型能够理解每个元素在其语境中的作用。

单个编码器层由两个主要子层构成：

1. 一个多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个简单的、按位置全连接的前馈网络（FFN）。

尤其重要的是，每个子层都包裹在残差连接和层归一化 (normalization)之中。这种“添加与归一化”模式是 Transformer 架构的一个典型特点，对于有效训练深度模型来说非常必要。我们来查看单个编码器层内的结构和数据流。

### 数据流经编码器层

假设编码器层的输入是向量 (vector)序列 $X = (x_1, x_2, ..., x_n)$，且 $n$ 是序列长度，每个 $x_i$ 是一个向量（例如，对于第一层，是词元 (token)嵌入 (embedding)和位置编码 (positional encoding)的和，或前一个编码器层的输出）。

1. **多头自注意力 (self-attention)：** 输入序列 $X$ 首先通过多头自注意力子层。如第三章所述，这种机制使序列中的每个位置 $i$ 能够关注序列 $X$ 中的所有位置（包括自身）。它根据从 $X$ 本身得到的查询、键和值计算注意力分数，产生一个输出序列，其中每个向量是值向量的加权和，反映语境信息。将此子层的输出表示为 $\text{MultiHeadAttention}(X)$。
2. **添加与归一化 (normalization)（第一块）：** 自注意力子层的输出随后通过残差连接（相加）与原始输入 $X$ 结合。这有助于避免深度网络中的梯度消失问题，通过允许梯度直接流经网络。为了正则化 (regularization)，通常在加法步骤*之前*将 Dropout 应用于自注意力子层的输出。加法之后，应用层归一化。层归一化稳定激活并改善训练动态。此块的操作可以表示为：

   
   $$
   \text{子层输出}_1 = \text{层归一化}(X + \text{Dropout}(\text{多头注意力}(X)))
   $$
   

   结果 $\text{子层输出}_1$ 是一个与 $X$ 维度相同的中间表示序列。
3. **按位置前馈网络（FFN）：** 该中间序列 $\text{子层输出}_1$ 随后输入到按位置前馈网络。该网络由两个线性变换构成，其间带有一个激活函数 (activation function)（通常是 ReLU 或 GELU）。重要的是，相同的 FFN（具有相同的权重 (weight)）独立地应用于序列 $\text{子层输出}_1$ 中的每个位置 $i$。它提供非线性变换，进一步处理表示。将输出表示为 $\text{FFN}(\text{子层输出}_1)$。其结构通常是：

   
   $$
   \text{FFN}(z) = \max(0, zW_1 + b_1)W_2 + b_2
   $$
   

   其中 $z$ 是特定位置的输入向量，而 $W_1, b_1, W_2, b_2$ 是两个线性层的可学习参数 (parameter)。内部维度通常大于模型的嵌入维度 $d_{model}$。
4. **添加与归一化（第二块）：** 与第一个子层相似，FFN 的输出通过残差连接与它的输入（$\text{子层输出}_1$）结合，随后是 dropout 和层归一化。

   
   $$
   \text{层输出} = \text{层归一化}(\text{子层输出}_1 + \text{Dropout}(\text{FFN}(\text{子层输出}_1)))
   $$
   

   $\text{层输出}$ 是此编码器层的最终向量输出序列。此输出与输入 $X$ 具有相同的维度，并作为堆栈中下一个相同编码器层的输入。

以下图表说明了一个遵循此描述的单个编码器层结构（通常称为 Post-LN，即归一化发生在加法*之后*）：

> 标准 Transformer 编码器层（Post-LN 变体）的结构。输入流经多头注意力 (multi-head attention)，与残差输入相加，并归一化。这个结果随后流经前馈网络，再次与它的残差输入（第一个归一化的输出）相加，并再次归一化以产生该层的输出。在每次加法之前应用 Dropout。

### Pre-LN 变体

值得注意的是，有一种常见的架构修改称为 Pre-LN Transformer。在此变体中，层归一化 (normalization)步骤在输入进入每个子层（自注意力 (self-attention)和 FFN）*之前*应用，而残差连接将子层的输出直接添加到其输入。

Pre-LN 的流程如下所示：

1. $X_{\text{归一化}} = \text{层归一化}(X)$
2. $Z = \text{多头注意力}(X_{\text{归一化}})$
3. $X' = X + \text{Dropout}(Z)$
4. $X'_{\text{归一化}} = \text{层归一化}(X')$
5. $Y = \text{FFN}(X'_{\text{归一化}})$
6. $\text{层输出} = X' + \text{Dropout}(Y)$

Pre-LN 通常带来更稳定的训练，特别是对于非常深的 Transformer，并且与原始 Post-LN 结构相比，可能需要较少仔细的学习率预热。理解这两种配置很有帮助，因为实现和研究论文可能使用任一变体。

堆栈中最后一个编码器层（第 $N$ 层）的输出作为解码器堆栈中每一层交叉注意力机制 (attention mechanism)的键（$K$）和值（$V$）输入，我们接下来将讨论这个。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇原始论文介绍了Transformer架构，详细阐述了编码器-解码器结构、多头注意力机制、位置感知前馈网络、残差连接和层归一化。
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu (2020)
  Journal: International Conference on Machine Learning (ICML); DOI: [10.48550/arXiv.2002.04745](https://doi.org/10.48550/arXiv.2002.04745)
  本文分析了Transformer架构中两种常见的层归一化变体：Post-LN和Pre-LN，并讨论了它们对训练稳定性和性能的影响。
- [CS224N: Natural Language Processing with Deep Learning - Lecture Notes: Transformers and Large Language Models](https://web.stanford.edu/class/cs224n/readings/CS224N_2023_Transformers_LLMs.pdf) — Stanford University, CS224N Course Staff (2023)
  Publisher: Stanford University
  这份全面的讲义从教学角度解释了Transformer架构，包括其编码器层结构、注意力机制、残差连接和归一化。
