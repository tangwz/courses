---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoder-anatomy-encoder-decoder"
lesson: "decoder-reconstructing-data"
sourceId: 6408
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder/decoder-reconstructing-data"
title: "解码器：数据重建"
description: "了解自动编码器中的解码器如何将原始数据从其压缩形式重建。"
order: 6
plots: []
sourceHash: "4838775d13879c3bf42031841ac2058439c2c36bd792fe7c1b0609d7801b9dc1"
sourceCorrections: []
---

自动编码器包含一个编码器，用于将输入数据 $X$ 压缩成紧凑的低维度潜在表示 $z$，其位于瓶颈层。在数据被压缩之后，自动编码器需要将其转换回有意义的形式。**解码器**是自动编码器的一个主要组成部分，负责进行这种重建。

解码器的主要职责是接收潜在表示 $z$，并尝试重建原始输入数据。它旨在反转编码器执行的压缩过程，将 $z$ 扩展回一种理想情况下与原始数据 $X$ 匹配的格式。这个重建过程的输出通常表示为 $X'$（读作“X撇”）。

想象一下，编码器写了一个长故事的非常简洁的摘要。解码器的工作就是阅读这个简短的摘要（$z$），并尝试重新写出完整的故事（$X'$）。当然，在此重建过程中，某些细节可能会丢失或改变，但一个训练有素的解码器将旨在尽可能忠实地再现原始故事。

### 解码器的架构：表示的扩展

在许多自动编码器设计中，解码器的架构通常是编码器的镜像，但操作方向相反。如果编码器由多个层组成，这些层逐渐减少特征数量（例如，从784个特征的输入减少到128个，然后到64个，最终到32维的潜在向量 (vector) $z$），那么解码器通常会拥有逐渐增加特征数量的层（例如，从32维的 $z$ 增加到64个特征，然后到128个，最终到784个特征以产生重建输出 $X'$）。

> 数据从瓶颈层（$z$）流经解码器层，以产生重建输出（$X'$）。解码器的结构通常是编码器的镜像，但方向相反，用于扩展数据表示。

### 从潜在空间到重建：解码器的过程

解码器内部的过程可以分解为几个步骤：

1. **接收潜在向量 (vector)**：解码器从潜在向量 $z$ 开始其工作。这个向量是编码器和瓶颈层的输出，它封装了从输入数据中学习到的压缩信息。
2. **经过解码器层进行扩展**：潜在向量 $z$ 随后会经过解码器内的一个或多个隐藏层。与编码器隐藏层降低维度不同，解码器的隐藏层旨在增加维度。解码器中每个随后的层通常比前一个层拥有更多神经元，有效地进行“上采样”或扩展表示。例如，如果 $z$ 有32维，第一个解码器隐藏层可能会将其扩展到64维，第二个扩展到128维，以此类推，直到数据的维度接近原始输入。
3. **输出层：生成 $X'$**：解码器中的最后一层是输出层。该层中的神经元数量必须精确匹配原始输入数据 $X$ 的维度数量（或特征）。如果原始输入是一张784像素的图像，解码器的输出层也必须有784个神经元，才能生成相同大小的重建图像 $X'$。

### 解码器中的激活函数 (activation function)

激活函数在解码器中与在编码器中一样重要。它们引入非线性，使解码器能够学习从潜在空间到原始数据空间的复杂映射。

- **隐藏层**：对于解码器中的中间隐藏层（即瓶颈层和输出层之间的层），修正线性单元（ReLU）激活函数是一个常用选择。ReLU 的定义为 $f(x) = \max(0, x)$。它的简洁性以及在防止梯度消失等问题上的有效性，使其在许多深度学习 (deep learning)架构中（包括解码器）受到欢迎。
- **输出层**：解码器输出层的激活函数选择尤为重要，因为它直接影响重建数据 $X'$ 的范围和性质。此选择取决于原始输入数据 $X$ 的特点：

  - **Sigmoid**：如果您的输入数据值被归一化 (normalization)到0到1的范围（例如，这是图像像素强度的常见做法），Sigmoid激活函数通常用于输出层。Sigmoid 函数，
    
    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$
    
    输出0到1之间的值。这确保了重建数据 $X'$ 也遵循这个[0, 1]范围，与输入的尺度匹配。
  - **线性**：如果输入数据可以取任何实数值（可能已标准化为零均值和单位方差），那么线性激活函数（本质上是没有激活，即 $f(x) = x$）可能适用于输出层。
  - **Tanh（双曲正切）**：如果输入数据归一化到-1到1之间，Tanh 函数，
    
    $$
    \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
    $$
    
    输出范围在[-1, 1]之间的值，可以是输出层的合适选择。

正如章节背景中提到的，对于入门示例，特别是那些处理归一化到[0,1]的图像数据（如MNIST手写数字数据集），Sigmoid函数是解码器输出层一个非常普遍且实用的选择。

训练自动编码器的总体目标是调整其内部参数 (parameter)（编码器和解码器中的权重 (weight)和偏置 (bias)），使重建输出 $X'$ 尽可能接近原始输入 $X$。这种学习如何实际发生，通过损失函数 (loss function)和优化算法等机制，将在下一章中详细说明。目前，重要的理解是，解码器是负责将压缩知识（$z$）转换回完整数据表示（$X'$）的组件。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍了自编码器、其架构（编码器和解码器）、潜在表示和各种激活函数。
- [Reducing the Dimensionality of Data with Neural Networks](https://pubmed.ncbi.nlm.nih.gov/16873662/) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  一篇经典论文，展示了深度自编码器在有效降维和数据重建方面的应用。
- [Autoencoders (UFLDL Tutorial)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHw35MPyQFZ7e4TSncStHRLIzxgDuudP0EB9t9qD5zgwPuBLhXe_bf8baCNS-yhRxchtTJ3McmJ3G9meWXcTDwKk104S5YLAVcWm4nUEfbUUBhst2BZg8cwNlBF_uwJhu27JqxjWw3iZq48q-HhuetlkjrqCEKmg0A=) — Andrew Ng, Jiquan Ngiam, Chuan Yu, and Yifan Hu (2011)
  Publisher: Stanford University
  一篇出色的教程，提供了自编码器的清晰易懂的说明，侧重于解码器的重建过程。
