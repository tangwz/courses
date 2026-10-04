---
course: "introduction-to-transformer-models"
chapter: "transformer-encoder-decoder-architecture"
lesson: "position-wise-feed-forward"
sourceId: 3004
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-3-transformer-encoder-decoder-architecture/position-wise-feed-forward"
title: "逐位置前馈网络"
description: "描述在每个位置独立应用的前馈网络的结构和功能。"
order: 7
plots: []
sourceHash: "a0efce641998f28c6330862700b46c66716f438d671da2554f04a6d5073a063b"
sourceCorrections: []
---

逐位置前馈网络（FFN）是每个编码器和解码器层中的重要组成部分。这些网络处理向量 (vector)，为模型增加进一步的处理能力。

尽管名称如此，它实际上是一个非常简单的网络。它由两次线性变换组成，中间夹有一个整流线性单元（ReLU）激活函数 (activation function)。“逐位置”的部分很要紧：这个相同的前馈网络在序列中的每个位置（即每个词元 (token)的表示）上独立且相同地应用。

因此，如果在注意力子层之后您的输入序列表示具有 (序列长度, $d_{model}$) 的维度，FFN 会使用同一组权重 (weight)分别处理每个大小为 $d_{model}$ 的 `序列长度` 向量。在此阶段，它不考虑不同位置之间的关系；这种交互发生在注意力层中。

### FFN 的结构

应用于每个位置 $x$ 的变换可以表示为：


$$
FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$


其中：

- $x$ 是前一层（注意力机制 (attention mechanism)后的残差连接与层归一化 (normalization)层）对于特定位置的输出。
- $W_1$ 和 $b_1$ 是第一次线性变换的权重 (weight)矩阵和偏置 (bias)项。
- $W_2$ 和 $b_2$ 是第二次线性变换的权重矩阵和偏置项。
- $\text{ReLU}$ 是整流线性单元激活函数 (activation function)，定义为 $\text{ReLU}(z) = \max(0, z)$。

FFN 内部的维度通常会发生变化。输入和输出维度是 $d_{model}$（模型的嵌入 (embedding)维度），但内部层维度（通常表示为 $d_{ff}$）通常更大。一种常见配置（如最初的 Transformer 论文中所用）是 $d_{ff} = 4 \times d_{model}$。例如，如果 $d_{model}=512$，则 $d_{ff}=2048$。

- 第一个线性层将 $d_{model}$ 维输入向量 (vector) $x$ 投影到更高维空间 (high-dimensional space) ($d_{ff}$)。
- ReLU 激活引入非线性，使模型能够学习更复杂的函数。如果没有像 ReLU 这样的非线性，堆叠多层并不会比单个线性变换增加太多模型能力。
- 第二个线性层将结果投影回原始 $d_{model}$ 维度，为下一层或模块做好准备。

> 应用于单个位置向量表示的逐位置前馈网络示意图。

### 为什么要使用它？

您可能想知道为什么在注意力机制 (attention mechanism)之后还需要这个组成部分。注意力机制处理序列级别的交互和上下文 (context)聚合，而 FFN 则*在每个位置独立地*提供额外的计算深度和非线性。

1. **增加模型容量：** 它增加了可学习参数 (parameter)，增加了模型表示每个词元 (token)特征内部复杂模式的能力。
2. **非线性：** 如前所述，ReLU 激活对于模型学习非线性关系是不可或缺的。注意力层本身内部通常涉及线性变换，因此 FFN 增加了必要的非线性处理。
3. **特征变换：** 它可以被视为将通过注意力机制学习到的特征变换为更合适的表示，以便用于下一层或模块。扩展到 $d_{ff}$ 并收缩回 $d_{model}$ 允许网络在将特征投影回来之前，可能在更高维空间 (high-dimensional space)中学习更丰富的特征组合。

尽管简单，逐位置前馈网络是每个 Transformer 模块不可或缺的部分，与注意力机制以及残差连接与层归一化 (normalization)层配合工作，以有效处理序列信息。它显著提升了 Transformer 架构的整体性能。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的原始论文，其中包含位置前馈网络的详细设计。
- [Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms/transformer.html#positionwise-feed-forward-networks) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2024)
  Publisher: Cambridge University Press
  一本全面的开源教科书，为深度学习模型提供了阐述和代码示例，其中包含关于Transformer位置前馈网络的专门部分。
- [CS224N: Natural Language Processing with Deep Learning - Lecture 12: Transformers and Pretraining](http://web.stanford.edu/class/cs224n/) — Stanford University (2023)
  来自一所顶尖大学课程的讲义，对Transformer架构及其组件（包括FFN）进行了结构化说明。
