---
course: "generative-adversarial-networks-gans"
chapter: "gan-foundations-revisited"
lesson: "dcgan-refresher"
sourceId: 2663
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-1-gan-foundations-revisited/dcgan-refresher"
title: "深度卷积GANs (DCGANs) 回顾"
description: "简要回顾DCGANs提出的架构指引。"
order: 5
plots: []
sourceHash: "a2f196c2026785485190173ee8c0255598b3ac6f8b0e6ecd69753724b13307ae"
sourceCorrections: []
---

尽管最初的GAN形式展现了对抗训练的潜力，但早期实现常使用多层感知器（MLP）。这些全连接网络在图像生成任务中难以有效扩展，常产生噪声多或不连贯的结果，并深受明显的训练不稳定问题困扰。

随深度卷积生成对抗网络 (GAN)（DCGANs）的出现，取得了重要进展。顾名思义，DCGANs借鉴了卷积神经网络 (neural network)（CNN）的能力，后者在计算机视觉任务中已表现出显著成效，并将其应用到生成对抗框架中。这不仅仅是交换层类型那么简单；DCGAN论文提出了一套特定的架构指引，它们明显提升了训练稳定性和生成图像的质量。这些指引产生了广泛影响，成为许多后续GAN架构的起点。

让我们回顾DCGANs提出的核心架构原理：

### DCGAN架构指引

1. **用带步长的卷积层替代池化层：** 确定性空间池化层（如最大池化）被替换。在判别器中，使用步长大于1的标准卷积进行下采样。在生成器中，使用分数步长卷积（常称为反卷积或转置卷积）进行上采样。这使得网络能够*学习*自己的空间下采样和上采样方式，使其成为可优化模型的一部分，而非固定操作。
2. **使用批量归一化 (normalization)（BatchNorm）：** 批量归一化应用于生成器和判别器，但生成器的输出层和判别器的输入层除外。批量归一化通过归一化每个单元的输入来稳定学习，有助于解决不佳的初始化问题并改善梯度流动。这在GANs的深度网络中尤其有利，能防止梯度消失或梯度爆炸等问题，并通过阻止生成器将所有样本收缩到一个点上，在一定程度上缓解模式崩溃。
3. **移除全连接隐藏层：** 对于较深的网络结构，移除了CNN顶部的传统全连接层。通常，唯一直接连接的层是：输入噪声向量 (vector) $z$ 投射到生成器的第一个卷积层，以及判别器最后一个卷积层的输出被展平并馈送到单个sigmoid输出单元。这减少了参数 (parameter)数量，并有助于在卷积特征中形成空间层次结构。
4. **在生成器中使用ReLU和Tanh激活函数 (activation function)：** 整流线性单元（ReLU）激活函数（$\max(0, x)$）用于生成器所有层（输出层除外），输出层使用Tanh激活函数（$\tanh(x)$）。Tanh将输出限制在 $[-1, 1]$ 范围内，这通常很方便，因为输入图像像素值通常归一化到此范围。
5. **在判别器中使用LeakyReLU激活函数：** 判别器所有层推荐使用渗漏整流线性单元（LeakyReLU）激活函数（$\max(0.01x, x)$）。与标准ReLU对负输入输出零不同，LeakyReLU在单元不活跃时仍允许一个小的非零梯度。这有助于防止训练期间梯度消失，这对判别器来说可能是一个问题，并确保梯度能更稳定地反向传播 (backpropagation)通过网络。

### DCGAN结构示意

整体流程是：生成器通过一系列分数步长卷积层和批量归一化 (normalization)层对随机噪声向量 (vector) $z$ 进行上采样，最终生成图像。判别器接收图像（真实或生成），并使用带步长卷积和LeakyReLU激活函数 (activation function)对其进行下采样，产生一个概率分数，表示输入图像是否真实。

> DCGAN架构的简化流程。生成器使用分数步长卷积进行上采样，而判别器使用带步长卷积进行下采样。请注意典型的激活函数和批量归一化的使用。

### 影响与传承

DCGAN的出现标志着一个转折点。通过提供一种稳定且相对可扩展的架构，建立在已确立的CNN实践之上，它使GANs在图像生成任务中变得更实用。生成的图像比早期基于MLP的GANs生成的图像连贯得多，能捕获CIFAR-10和LSUN等数据集中的物体结构。

这些架构指引成为标准做法，也是无数后续GAN变体的起点。尽管DCGAN带来了显著提升，但它仍面临挑战，尤其是在稳定生成非常高分辨率图像方面，以及完全克服模式崩溃等问题。理解DCGAN的结构及其贡献，为我们后续章节中将要讲解的更先进架构、稳定化技术和评估指标，提供了必要的背景。这些高级方法建立在DCGAN奠定的基础上，解决其局限性，并进一步拓展生成建模的范围。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教科书，涵盖CNN、GAN、批量归一化和激活函数等基础知识。
