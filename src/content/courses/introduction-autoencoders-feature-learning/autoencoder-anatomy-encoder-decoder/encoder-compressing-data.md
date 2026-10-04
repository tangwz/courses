---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoder-anatomy-encoder-decoder"
lesson: "encoder-compressing-data"
sourceId: 6403
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder/encoder-compressing-data"
title: "编码器：数据压缩"
description: "了解自编码器中的编码器部分如何将输入数据压缩成低维表示。"
order: 1
plots: []
sourceHash: "d540842f411d382aa5775fdcd7b61f93369b276c892e6f590679610be09ee37e"
sourceCorrections: []
---

编码器是自编码器的首要主要组成部分。可以将其视为网络中负责信息压缩的部分。它的主要作用是将初始的、通常是高维的输入数据，转换成更紧凑、低维度的表示。这个过程类似于为长篇文档创建一份简洁的摘要；摘要应该包含最重要的信息点，同时比原文短得多。

假设我们的输入数据是 $X$。这个 $X$ 可以是任何形式，比如一张平铺图像（其中每个像素的亮度是一个特征），或者是一组科学实验的测量数据。如果一张图像是 28x28 像素，那么它的平铺表示 $X$ 将有 $28 \times 28 = 784$ 个特征。这是编码器开始处理的信息。

编码器本身通常是一个神经网络 (neural network)，由一个或多个层组成。编码器的第一层接收输入数据 $X$。编码器中的每个后续层通常比前一个层拥有更少的神经元。这种神经元数量从一层到下一层的系统性减少，是编码器逐步将信息压缩到更小空间的方式。

想象一下，水通过一系列漏斗，每个漏斗都比前一个更窄。编码器的层对数据的作用方式类似。

> 一张展示编码器结构的图。输入数据 $X$ 经过隐藏层，这些层逐步降低其维度，最终在瓶颈层形成压缩后的潜在表示 $z$。

这种压缩不仅仅是随意丢弃数据。在训练过程中（我们将在“自编码器如何学习”一节中介绍），编码器学习保留输入数据中最重要和有用的部分。它试图找到使数据能高效表示的潜在模式或结构。因此，虽然维度降低了，但我们希望最有信息量的特征能够被保留下来。

编码器的最后一层产生这种高度压缩的低维表示。这个输出是自编码器结构中一个很重要的组成部分，常被称为**瓶颈层**或**潜在空间表示**。我们将这种压缩形式表示为 $z$。 $z$ 的维度（即瓶颈层中的神经元数量）是一个设计上的选择，它决定了数据被压缩的程度。例如，如果我们的输入 $X$ 有 784 个特征，编码器可能会将其压缩到一个只有 64 个特征的 $z$。

为了进行这些变换并学习复杂的模式，编码器中的层使用**激活函数 (activation function)**。这些是应用于每个神经元输出的数学函数。编码器隐藏层中常用的一种激活函数是修正线性单元，即 $ReLU$。 $ReLU$ 很受欢迎，因为它简单且有助于解决训练深度网络中的一些难题。我们将在本章后面更详细地讨论激活函数。目前，只需了解它们使网络能够学习数据中除简单线性关系之外的更多内容。

因此，数据通过编码器的路径如下：

1. 输入数据 $X$ 被馈送到编码器的第一层。
2. 它通过一个或多个隐藏层，每个层通常比前一个层更小。每个层都变换数据并降低其维度。
3. 编码器的最后一层输出压缩表示 $z$。

这种压缩表示 $z$ 是编码器的最终成果。它包含输入经过学习后形成的紧凑摘要。目标是使 $z$ 成为一种丰富且包含信息量的表示，尽管其大小有所减小，因为解码器（自编码器的另一半）将完全依赖 $z$ 来尝试重建原始输入 $X$。编码器在智能压缩方面做得越好，解码器在重建任务上的表现也就越好。

## 参考资料

- [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  这篇基础性论文介绍了用于降维和表示学习的深度自编码器。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍了深度学习，包括自编码器、神经网络组件和数据压缩原理。
