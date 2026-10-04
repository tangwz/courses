---
course: "applied-speech-recognition"
chapter: "advanced-acoustic-models"
lesson: "conformer-architecture"
sourceId: 7143
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models/conformer-architecture"
title: "Conformer：结合卷积神经网络与Transformer"
description: "了解Conformer模型，它结合了卷积和自注意力，以捕捉局部和整体依赖关系。"
order: 5
plots: []
sourceHash: "ac993222173349f48083c3ca8aa20fe1ca95ac50e68b387683a3a8ca8ce3f16e"
sourceCorrections: []
---

尽管Transformer架构在捕捉整个音频序列的关联性方面表现出色，但它并非天生就能高效学习语音中具有意义的细粒度局部模式，例如音素转换或协同发音效果。纯粹的自注意力 (self-attention)模型在计算上将每个时间步视为等距，这可能忽视相邻特征向量 (vector)的特殊性。另一方面，卷积神经网络 (neural network)（CNN）通过在输入上滑动卷积核，非常擅长识别局部模式，但它们难以建模长程依赖。

为了兼得两种方法的优势，谷歌的研究人员提出了Conformer架构。它有效地将Transformer的自注意力机制 (attention mechanism)与CNN的局部模式识别能力整合为一个单一且强大的模块。这种混合设计因其能够同时建模语音发音的局部和整体背景信息，已成为许多先进ASR系统的构建基础。

### Conformer模块

这种架构的核心是Conformer模块，它处理输入特征序列。与标准的Transformer模块不同——标准模块由一个自注意力 (self-attention)层和一个前馈网络组成——Conformer模块在中间插入一个卷积模块，并以一种独特的“马卡龙式”结构排列这些组件。马卡龙有两片相同的饼干，中间夹着馅料；类似地，Conformer模块也包含两个半步长的前馈层，将注意力和卷积模块“夹”在中间。

一个模块包含四个主要子模块，每个都带有残差连接和层归一化 (normalization)：

1. 一个前馈模块。
2. 一个多头自注意力模块。
3. 一个卷积模块。
4. 最后一个前馈模块。

> Conformer模块的数据流。输入首先经过一个前馈网络处理，然后是自注意力模块、卷积模块，以及最后一个前馈网络，每一步都带有残差连接。

让我们审视这种结构中每个组件的作用。

#### 前馈模块

该模块使用两个半步长前馈网络，一个在开始，一个在结束。这些是标准的位置级前馈网络，与Transformer中的类似。分割它们的考量在于有助于训练时的梯度传递。每个前馈模块通常由两个线性层和中间的一个非线性激活函数 (activation function)组成，如Swish或GELU。

#### 多头自注意力模块

这是Transformer架构中标准的多头自注意力机制 (attention mechanism)。其作用是为输入序列中每一对时间步计算注意力分数，使模型在处理特定时间步时，能够衡量音频不同部分的权重 (weight)。这是模型捕捉长程的、整体的关联，如语法和句子语境的地方。对于ASR而言，这个模块通常使用相对位置编码 (positional encoding)，这比原始Transformer中使用的绝对编码更适合语音处理。

#### 卷积模块

这是Conformer的特色。自注意力模块处理完整体背景信息后，卷积模块随后用于明确学习局部模式。它通常由逐点卷积、接着是1D深度可分离卷积，以及另一个逐点卷积组成。

- **逐点卷积：** 将输入投射到更高维空间 (high-dimensional space)。
- **1D深度可分离卷积：** 这是一种效率很高的卷积类型，它对每个输入通道独立地应用一个滤波器。对于ASR而言，这意味着它在时间维度上滑动卷积核，以捕捉局部声学-语音特征，例如共振峰在几毫秒内的变化。
- **批量归一化和激活：** 这些操作在卷积之后进行，以稳定训练并引入非线性。

通过引入这个模块，Conformer直接编码平移不变的局部关联性，意味着它能够识别特定的声学模式，无论其在音频流中的位置如何。

### 这种组合为何有效

Conformer的有效性源于其操作的巧妙排序。自注意力 (self-attention)模块首先识别序列中的整体关联。随后，卷积模块通过关注局部结构来优化这些表示。这使模型能够同时学习受益于整体背景信息和局部声学细节的特征。

例如，要转录单词“cat”，自注意力模块可能会利用更广泛的句子语境来提高该词是名词的可能性。与此同时，卷积模块可以专注于/k/、/æ/和/t/音之间的精确声学转换。最后的前馈网络整合这两种信息来源，然后将结果传递给下一个Conformer模块。

与纯Transformer或基于LSTM的模型相比，这种架构在主要的ASR基准测试中持续展现出优越的性能。它能够有效学习更丰富的特征集合，使其成为构建高准确度声学模型的首选。许多当前可用的、功能最强的预训练 (pre-training)模型（你接下来将学习如何使用它们）都是基于Conformer设计构建的。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  引入 Transformer 模型的开创性论文，该模型是 Conformer 架构中提供全局上下文建模能力的核心组成部分。
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) — Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam (2017)
  Journal: arXiv preprint arXiv:1704.04861; DOI: [10.48550/arXiv.1704.04861](https://doi.org/10.48550/arXiv.1704.04861)
  介绍并详细阐述了深度可分离卷积，这是 Conformer 卷积模块中用于局部模式检测的高效卷积形式。
