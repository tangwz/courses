---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "residual-connections-add"
sourceId: 2020
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/residual-connections-add"
title: "残差连接 (相加)"
description: "残差连接在子层中的作用，对于梯度传输和深度模型训练的帮助。"
order: 7
plots: []
sourceHash: "8bb5eabef1cd0f5308f34e46c1295fc8680b674e137045c7921f81b794ae8cd8"
sourceCorrections: []
---

深度神经网络 (neural network)，包括包含多头注意力 (multi-head attention)机制 (attention mechanism)和前馈网络的 Transformer 编码器和解码器层，通常面临一个与训练效果相关的基本挑战。堆叠多层可以增加模型能力，但也可能使优化变得困难。梯度，即学习过程中使用的信号，在通过多层反向传播 (backpropagation)时可能会减弱或消失，阻碍了前面层的更新。同样地，梯度有时也可能变得过大，导致不稳定。

为了解决这些问题，并使得训练 Transformer 典型的深层架构成为可能，采用了残差连接，也叫跳跃连接。这种技术因计算机视觉中的残差网络 (ResNets) 而广为人知，它为梯度在网络中传输提供了一条直接通路。

### 残差连接的原理

在 Transformer 层中，每个主要子层（多头自注意力 (self-attention)机制 (attention mechanism)和逐位置前馈网络）都包裹在残差连接中。核心思路非常直观：将子层的输入与其输出相加。

令 $x$ 表示子层的输入（例如，来自前一层的输出或初始嵌入 (embedding)）。令 $SubLayer(x)$ 表示子层本身实现的函数（例如，多头注意力 (multi-head attention)或 FFN）。该操作的输出，*在* 归一化 (normalization)之前，计算方法如下：


$$
\text{残差输出} = x + SubLayer(x)
$$


这种相加操作通常紧接着层归一化，构成了 Transformer 图示中常见的“加和归一化”模块。

> 残差连接跳过一个子层，并将原始输入 $x$ 添加到子层的输出 $SubLayer(x)$。

### 残差连接为何有效

1. **改善梯度传输：** 在反向传播 (backpropagation)时，残差块的梯度计算包含一条通过加法操作的直接回传路径。输出对输入 $x$ 的梯度包含来自该直接连接的 $+1$ 项。这确保了即使通过 $SubLayer(x)$ 路径的梯度变得非常小，仍然有梯度信号通过恒等连接 ($x$) 相对不受阻碍地回传。这大大缓解了梯度消失问题，使得信息能够更有效地在多层之间传递。
2. **促进恒等映射学习：** 残差连接使得层更容易学习恒等函数。如果给定层的理想变换仅仅是保持输入不变，网络可以通过将 $SubLayer$ 内的权重 (weight)趋近于零来实现这一点。没有残差连接，使用复杂的非线性变换来学习精确的恒等映射会很困难。这一特性使得可以在不一定损害性能的情况下增加层；如果需要，网络可以有效地“忽略”一个层。
3. **实现更深网络：** 通过缓解梯度问题和简化恒等映射的学习，残差连接在成功训练那些代表强大 Transformer 模型特性的非常深的网络（编码器和解码器堆栈中通常有 6、12、24 或更多层）方面发挥着重要作用。

### 在 Transformer 层中的实现

在标准 Transformer 的编码器和解码器层中，你会发现两个主要子层，每个子层后面都跟着这个“加和归一化 (normalization)”步骤：

1. **多头注意力 (multi-head attention)：** 输入 $x$ 传入多头注意力机制 (attention mechanism)。其输出 $Attention(x)$ 随后与原始输入 $x$ 相加。层归一化应用于这个和。
   $Output_{Attn} = LayerNorm(x + MultiHeadAttention(x))$
2. **前馈网络：** 第一个归一化步骤的输出，我们称之为 $y = Output_{Attn}$，作为逐位置前馈网络的输入。FFN 的输出 $FFN(y)$ 与其输入 $y$ 相加。应用最终的层归一化。
   $Output_{Layer} = LayerNorm(y + FFN(y))$

这些残差连接是一种看似简单却极度有效的技术，它们构成了 Transformer 设计中不可或缺的一部分，并对其在建模复杂序列数据方面的成功起到了很大作用。没有它们，训练现代大型语言模型中常见的深层堆栈会困难得多。

## 参考资料

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 770-778; DOI: [10.48550/arXiv.1512.03385](https://doi.org/10.48550/arXiv.1512.03385)
  本文介绍了残差网络（ResNets）和残差连接的概念，通过解决梯度退化问题，显著提升了训练深度神经网络的能力。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本深度学习的标准教科书，解释了训练深度网络面临的挑战，例如梯度消失/爆炸，以及跳跃连接等架构解决方案。
