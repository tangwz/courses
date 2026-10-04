---
course: "applied-autoencoders-feature-extraction"
chapter: "convolutional-autoencoders-image-data"
lesson: "transposed-convolutional-layers-decoders"
sourceId: 6475
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-5-convolutional-autoencoders-image-data/transposed-convolutional-layers-decoders"
title: "解码器中的转置卷积层"
description: "在解码器中使用转置卷积层（Conv2DTranspose 或 反卷积）进行特征图上采样。"
order: 4
plots: []
sourceHash: "87eb6da061c0e2bb3a9b69bbaaf9f341bf17c661dabe771be681d9e6102a3b2e"
sourceCorrections: []
---

卷积自编码器 (ConvAE) 的编码器部分成功将输入图像压缩成低维潜在表示后，解码器接手。其主要职责是从这种紧凑表示中重构原始图像。此重构过程包括增加特征图的空间维度，从而有效地反转编码器执行的下采样操作（如池化或步幅卷积）。转置卷积层是实现这种可学习上采样的一种常用且有效的方法。

### 解码器中的上采样挑战

编码器通常在每个连续层中减少特征图的高度和宽度。例如，一个 $256 \times 256$ 的图像在瓶颈层可能会被缩减到 $16 \times 16$ 的特征图。解码器必须智能地将这个 $16 \times 16$ 的表示扩展回 $256 \times 256$，填充细节以生成一个连贯的图像。如果没有学习机制，简单地放大图像通常会导致块状或模糊的输出。需要一种方法来学习如何在上采样过程中生成缺失的细节。

### 转置卷积：可学习的上采样

转置卷积层，有时被称为“反卷积层”，提供了一种机制来执行上采样，使网络能够学习填充更高分辨率细节的最佳方式。“反卷积”这个术语可能会引起误解，因为这些层并不执行真正的数学反卷积（卷积的逆运算）。更准确的名称是“转置卷积”，因为它与标准卷积存在数学关联，特别是关于梯度如何计算或操作如何通过矩阵代数进行表达。

本质上，转置卷积将较小的输入特征图映射到较大的输出特征图。它通过将每个输入像素与输出中的较大区域关联来实现这一点，其中具体的映射由可学习的滤波器权重 (weight)决定。

理解转置卷积操作的一种方法是将其视为应用于其输入扩展版本上的标准卷积。对于给定的输入特征图和步幅 $S$:

1. 输入特征图首先通过在其每个元素之间水平和垂直地插入 $S-1$ 个零进行扩展。此扩展图的边界周围也可能应用额外的填充。
2. 然后，一个标准卷积滤波器（核）以步幅为 1 的方式在此扩展且已填充的输入上进行滑动。

核权重在训练期间学习，使得网络能够确定如何最佳地组合来自低分辨率输入的值，以生成合理的更高分辨率输出。

> 转置卷积层接收低分辨率特征图作为输入，并生成高分辨率特征图作为输出，其变换参数 (parameter)在训练期间学习得到。

### 转置卷积层的参数 (parameter)

与标准卷积层类似，转置卷积层也有几个需要配置的重要参数：

- **滤波器（或通道）：** 这定义了该层将生成的输出通道（特征图）的数量。
- **核大小：** 这指定了卷积滤波器（核）的维度（高和宽）。核在输入上滑动以生成输出。
- **步幅：** 这是上采样的一个重要参数。步幅决定了输入被扩展的程度，或者输出大小相对于输入大小增加的倍数。例如，步幅为 (2,2) 会大致使输入特征图的高度和宽度翻倍。
- **填充：** 填充（'valid' 或 'same'）影响输出维度。
  - 使用 `'valid'` 填充时，输出大小由输入大小、核大小和步幅决定。公式通常是 H\_{out} = (H\_{in} - 1) \times \text{stride} + \text{kernel\_size}。
  - 使用 `'same'` 填充时，像 TensorFlow 这样的框架旨在生成一个输出，其中 $H_{out} = H_{in} \times \text{stride}$。这通常简化了网络设计，因为上采样因子直接与步幅关联。这可能涉及框架计算必要的填充，或者在某些情况下，使用 `output_padding` 参数来解决歧义。

例如，如果一个输入特征图是 $8 \times 8$，并且应用步幅为 (2,2) 且使用 'same' 填充的转置卷积，则输出特征图通常会是 $16 \times 16$（在考虑滤波器数量之前）。

### 构建解码器架构

在设计 ConvAE 的解码器时，一种常见做法是创建大致镜像编码器的架构。如果编码器有一系列 `卷积 -> 池化 -> 卷积 -> 池化` 层，解码器可能有一系列像 `转置卷积 -> 转置卷积 -> ...` 的层，其中每个转置卷积（通常步幅为 2）旨在反转编码器中相应池化或带步幅卷积层的空间下采样。

- **激活函数 (activation function)：** 与编码器中一样，激活函数（例如 ReLU（修正线性单元））通常应用于每个转置卷积层之后（最终输出层可能除外），以引入非线性。
- **最终输出层：** 解码器中最后一个转置卷积层的激活函数取决于自编码器试图重构的输入数据的性质。
  - 如果输入图像像素值被归一化 (normalization)到 0 和 1 之间（例如，对于 MNIST 数字），则 `sigmoid` 激活函数适用于最后一层。
  - 如果像素值被归一化到 -1 和 1 之间，则可能使用 `tanh` 激活函数。
  - 如果像素值在不同的范围或没有强制特定范围，则线性激活（即没有激活函数）可能适用。

### “转置”的含义

“转置”一词源于操作的数学表达。如果一个标准的前向卷积操作可以表示为与一个（通常是稀疏的）矩阵 $C$ 的矩阵乘法，$y = C x$，那么相应的转置卷积执行与该矩阵的转置 $C^T$ 的乘法。这也是为什么这种操作有时被称为“分数步幅卷积”，因为它被视为具有分数步幅的卷积，从而导致输出分辨率的增加。

尽管转置卷积是解码器中进行可学习上采样的一种强大且广泛使用的方法，但它们只是现有几种技术之一。其他方法，例如最近邻或双线性上采样后接标准卷积层，也可以采用，这些将在下一节讨论。然而，转置卷积学习上采样过程本身的能力，使其成为在 ConvAE 中生成精细重构的有力选择。

## 参考资料

- [A guide to convolutional arithmetic for deep learning](https://doi.org/10.48550/arXiv.1603.07285) — Vincent Dumoulin and Francesco Visin (2016)
  Journal: arXiv preprint arXiv:1603.07285; DOI: [10.48550/arXiv.1603.07285](https://doi.org/10.48550/arXiv.1603.07285)
  通过视觉和数学方式解释了卷积和转置卷积操作，详细说明了参数的影响。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于卷积网络、自编码器和深度学习基础知识的教科书。
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) — Augustus Odena, Vincent Dumoulin, and Chris Olah (2016)
  Journal: Distill
  阐明了转置卷积的原理并讨论了常见的伪影，有助于实际应用。
- [tf.keras.layers.Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) — TensorFlow Developers (2023)
  Keras转置卷积层的官方文档，概述了实现所需的参数和用法。
