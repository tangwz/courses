---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoder-anatomy-encoder-decoder"
lesson: "decoder-hidden-layers-decompression"
sourceId: 6409
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder/decoder-hidden-layers-decompression"
title: "解码器隐藏层与数据解压"
description: "了解解码器中的隐藏层如何逐步扩展数据，使其恢复到原始维度。"
order: 7
plots: []
sourceHash: "f9d107d72bb016862d82e0bdd7f9f44425ae41993d4979796e6b40addd2aa730"
sourceCorrections: []
---

自动编码器中的解码器组件负责从压缩表示中重建输入数据。这个压缩后的摘要，通常表示为 $z$，通常由编码器生成。解码器的主要任务是反转压缩过程，接收 $z$ 并尽可能准确地恢复原始数据。解码器内部的隐藏层是这种“解压”或“上采样”过程的主要发生地。

可以将解码器的隐藏层看作与编码器的隐藏层反向工作。如果编码器在其层中逐渐减少神经元数量，以将信息挤压到瓶颈中，那么解码器的隐藏层将逐渐增加神经元数量。这种架构选择通常是为了与编码器形成某种对称结构，这可能是一种有益的设计模式。

### 逆向扩展：解码器隐藏层的作用

从压缩后的潜在空间 $z$ 返回到原始数据维度的路径，在 $z$ 传递给解码器的第一个隐藏层时便已开始。解码器中每个后续的隐藏层通常比前一层拥有更多的神经元。例如，如果瓶颈 $z$ 有32个单元，解码器中的第一个隐藏层可能有64个单元，下一个128个单元，依此类推，逐渐扩展直到维度接近原始输入。

这些层实际*做*什么？

1. **增加维度**：它们最明显的功能是将低维输入映射到高维输出。这是在此背景下数据解压的核心。
2. **学习转换**：就像任何其他神经网络 (neural network)层一样，这些隐藏层学习一组权重 (weight)和偏置 (bias)。在自编码器的训练过程中，会调整这些参数 (parameter)，使得解码器层执行的转换能够有效地“撤销”编码器执行的压缩。它们学习解释 $z$ 中编码的特征，并将其转换回更宽泛的表示。
3. **引入非线性**：激活函数 (activation function)，例如修正线性单元 (ReLU)，常用于解码器的隐藏层。这些函数使解码器能够学习复杂的非线性映射。没有它们，解码器（和编码器）将仅限于学习线性转换，这对于大多数数据来说不够强大。

设想编码器为一幅详细图像创建一个非常简洁的摘要。解码器的隐藏层随后接收这个摘要，并根据它们在训练中学到的知识，一步步地添加回细节、纹理和结构，以重建出与原始图像高度相似的内容。

下图展示了通过解码器隐藏层的这个扩展过程：

> 该图示出了数据从紧凑潜在空间 ($z$) 流经解码器隐藏层，这些层逐渐增加单元数量（从而增加表示的维度）。这种扩展为最终输出层准备数据，该层旨在匹配原始输入 $X$ 的结构。

此扩展路径中的每个层都学习改进前一层的表示，目标是重建编码器捕获和压缩的特征。解码器中隐藏层的数量以及每个层中神经元的数量，是取决于数据复杂性和自编码器所需容量的设计选择。

最终，解码器最后一个隐藏层的输出会传递给输出层。这个最终层被构造成与原始输入数据的维度匹配，并生成重建数据 $X'$。我们将在后续章节中更详细地介绍输出层，但其成功的基础是由解码器隐藏层执行的有效解压和特征细化奠定的。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 14
  全面阐述了自编码器的结构，涵盖了编码器和解码器隐藏层的功能。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes on Autoencoders, VAEs, GANs](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf) — Fei-Fei Li, Justin Johnson, Serena Yeung (2017)
  Publisher: Stanford University
  提供了自编码器架构的易懂概述，包括解码器在重建和维度提升中的作用，并配有清晰图表，适合学习者。
