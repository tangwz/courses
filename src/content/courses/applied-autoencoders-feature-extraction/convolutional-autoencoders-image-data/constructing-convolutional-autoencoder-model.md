---
course: "applied-autoencoders-feature-extraction"
chapter: "convolutional-autoencoders-image-data"
lesson: "constructing-convolutional-autoencoder-model"
sourceId: 6477
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-5-convolutional-autoencoders-image-data/constructing-convolutional-autoencoder-model"
title: "构建卷积自编码器模型"
description: "设计和构建完整卷积自编码器架构的分步指南。"
order: 6
plots: []
sourceHash: "29f832309386ec369801f760840a89dc225cbb064f99ff3c3fe9643723b1e9c1"
sourceCorrections: []
---

构建卷积自编码器（ConvAE）需要仔细设计编码器、瓶颈层和解码器，使用适合图像数据的组件，主要是卷积层及相关层。这种架构使得模型能够学习特征的空间层次结构，与难以处理图像高维度和结构的全连接自编码器有所不同。

## 整体架构

卷积自编码器与其它自编码器一样，由两部分构成：将输入图像映射到低维潜在表示（即瓶颈层）的编码器，以及从这个潜在表示重建图像的解码器。

1. **编码器**：编码器的职责是逐步减小输入图像的空间维度（高度和宽度），同时增加其深度（特征图数量）。这通常通过堆叠`Conv2D`层来实现，这些层后面通常跟着`ReLU`激活函数 (activation function)和`MaxPooling2D`层。
2. **瓶颈层**：这是输入的最压缩表示。它可以是一个平铺向量 (vector)，也可以是一个小而深的特征图。这一层的维度决定了压缩程度。
3. **解码器**：解码器的任务是反转编码器的过程。它接收瓶颈层的潜在表示，并逐步将其上采样到原始图像维度。这通常通过使用`Conv2DTranspose`层（也称为反卷积层）来完成，或者通过上采样层（如`UpSampling2D`）与`Conv2D`层的组合来实现。解码器的最后一层通常具有适合输入图像像素值范围的激活函数（例如，对于在0到1之间归一化 (normalization)的像素使用`Sigmoid`）。

一种常见的设计策略是让解码器与编码器大致对称。

## 编码器设计

编码器负责学习从输入图像中提取显著特征，并将它们编码成紧凑的表示。

### 层与结构

一个典型的编码器块包含：

- 一个`Conv2D`层：此层对输入应用一组可学习的滤波器。重要参数 (parameter)包括：
  - `filters`：输出特征图的数量（输出深度）。在编码器较深层中增加滤波器数量很常见（例如，32 -> 64 -> 128）。
  - `kernel_size`：卷积窗口的尺寸（例如，3x3或5x5）。像3x3这样的小核被广泛使用。
  - `strides`：对于标准卷积通常为(1,1)，意味着滤波器一次移动一个像素。
  - `padding`：通常是`'same'`以确保输出特征图与输入具有相同的空间维度（假设步幅为1），或`'valid'`表示无填充。
- 一个激活函数 (activation function)：`ReLU`（修正线性单元）是隐藏层的常见选择，因为它效率高，并且能够缓解梯度消失问题。
- 一个`MaxPooling2D`层：此层执行空间下采样，减小特征图的高度和宽度（例如，`pool_size`为(2,2)时，它将维度减半）。这有助于实现维度缩减，并使表示对于输入中的小位移更具鲁棒性。

您通常会堆叠多个这样的块。例如，输入图像可能会经过`Conv2D` -> `ReLU` -> `MaxPooling2D`，然后是另一个`Conv2D` -> `ReLU` -> `MaxPooling2D`，依此类推。

### 编码器路径示例

如果输入图像是64x64x3（高 x 宽 x 通道）：

1. `Conv2D` (32个滤波器, 3x3, ReLU, padding='same') -> 输出: 64x64x32
2. `MaxPooling2D` (2x2) -> 输出: 32x32x32
3. `Conv2D` (64个滤波器, 3x3, ReLU, padding='same') -> 输出: 32x32x64
4. `MaxPooling2D` (2x2) -> 输出: 16x16x64 (这可以作为瓶颈层的输入)

## 瓶颈层

瓶颈层，也称为潜在空间或代码，是输入的压缩表示所在的位置。它的设计是自编码器的一个重要方面。

- **形式**：最终编码器层的输出（例如，上面示例中的16x16x64）如果是特征图，可以直接作为瓶颈层。或者，此特征图可以被平铺成一维向量 (vector)。
- **维度**：瓶颈层的大小（如果是平铺的单元数量，或特征图的维度）决定了压缩级别。较小的瓶颈层迫使自编码器学习更显著的特征，但可能会使重建更困难。这是一个通常需要调整的超参数 (parameter) (hyperparameter)。

对于特征提取，此瓶颈层的激活值用作下游任务的学习特征。

## 解码器设计

解码器的作用是接收瓶颈层的压缩表示，并重建原始输入图像。它的架构通常镜像编码器的结构，但顺序相反。

### 层与结构

一个典型的解码器块可能包含：

- 一个上采样操作：这会增加特征图的空间维度。常见选择包括：
  - `Conv2DTranspose`层：此层执行一种类似逆卷积的操作。它可以在一步中学习上采样和卷积。重要参数 (parameter)包括`filters`、`kernel_size`和`strides`（例如，步幅为(2,2)通常会将空间维度加倍）。
  - `UpSampling2D`层：此层执行一个更简单的上采样（例如，最近邻或双线性插值），并且后面通常跟着一个`Conv2D`层来改进特征。
- 一个`Conv2D`层：上采样后，可以使用常规的`Conv2D`层（通常带有`ReLU`激活）来进一步处理特征图并学习重建细节。解码器层中的滤波器数量通常会随着我们接近输出层而减少（例如，128 -> 64 -> 32）。
- 一个激活函数 (activation function)：`ReLU`在解码器的隐藏层中很常见。

### 输出层

解码器的最后一层应该生成一个与原始输入图像具有相同维度（高度、宽度、通道）的图像。

- 它通常是一个`Conv2D`或`Conv2DTranspose`层。
- 滤波器数量必须与输入图像中的通道数量匹配（例如，RGB图像为3，灰度图像为1）。
- 此层的激活函数 (activation function)取决于输入像素值的归一化 (normalization)方式：
  - `Sigmoid`：如果像素值缩放到范围[0, 1]。
  - `Tanh`：如果像素值缩放到范围[-1, 1]。
  - 线性（无激活）：如果像素值没有界限或范围更广。

### 解码器路径示例

从16x16x64的瓶颈层继续，目标是重建一个64x64x3的图像：

1. `Conv2DTranspose` (32个滤波器, 3x3, ReLU, strides=(2,2), padding='same') -> 输出: 32x32x32
   (此层将从16x16上采样到32x32，并将深度从64更改为32)
2. `Conv2DTranspose` (3个滤波器, 3x3, Sigmoid, strides=(2,2), padding='same') -> 输出: 64x64x3
   (此层将从32x32上采样到64x64，并将深度从32更改为3，应用Sigmoid以获得[0,1]输出)

或者，可以使用`UpSampling2D`后跟`Conv2D`：

1. `UpSampling2D` (size=(2,2)) -> 输出: 32x32x64
2. `Conv2D` (32个滤波器, 3x3, ReLU, padding='same') -> 输出: 32x32x32
3. `UpSampling2D` (size=(2,2)) -> 输出: 64x64x32
4. `Conv2D` (3个滤波器, 3x3, Sigmoid, padding='same') -> 输出: 64x64x3

## 典型的卷积自编码器结构

下图显示了一种卷积自编码器的常见架构。

> 典型的卷积自编码器架构。编码器使用卷积层和池化层来缩减维度。解码器使用转置卷积层（或上采样后接卷积）来重建图像。F1、F2、F1'代表滤波器数量，S1是卷积步幅，S\_up是上采样步幅（通常为2），C\_in/C\_out是输入/输出通道。

## 构建要点

- **对称性**：虽然并非严格必要，但将解码器设计成编码器的镜像（在层类型和总体结构方面）是一个常见且通常有效的起点。例如，如果编码器有两个`Conv2D`层后接`MaxPooling2D`层，解码器可能会有两个`Conv2DTranspose`（或`UpSampling2D` + `Conv2D`）块。
- **滤波器数量递进**：在编码器中，滤波器数量通常随着深度增加（例如，32 -> 64 -> 128）。在解码器中，滤波器数量通常随着它接近输出层而减少（例如，128 -> 64 -> 32 -> 输出通道）。
- **核尺寸和填充**：3x3核是`Conv2D`和`Conv2DTranspose`层的标准。在卷积层中使用`'same'`填充（可能除了最后一层）有助于在池化或上采样前保持块内的空间维度，从而简化网络设计。对于`Conv2DTranspose`层，需要与步幅一起仔细选择填充（通常是`'same'`），以达到期望的输出维度。
- **损失函数 (loss function)**：对于图像重建，如果像素值是连续的，均方误差（MSE）是一种常用损失函数。如果像素值归一化 (normalization)到[0,1]并可以被视为概率（例如，对于二值图像或每个像素值都是概率的图像），也可以使用二元交叉熵（BCE）。选择取决于图像数据的性质及其预处理方式。

通过精心堆叠这些层，您就可以创建一个能够学习将图像压缩成密集潜在表示并随后重建它们的网络。在瓶颈层中学到的特征通常对各种下游任务有用，这也是使用自编码器进行特征提取的主要目标。接下来的步骤涉及训练此模型，然后使用其编码器部分提取这些特征。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本全面的教材提供了卷积神经网络和自编码器的基础知识，对于理解卷积自编码器的构建模块和原理至关重要。
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28) — Olaf Ronneberger, Philipp Fischer, Thomas Brox (2015)
  Journal: Medical Image Computing and Computer-Assisted Intervention (MICCAI); Publisher: Springer; Pages: 234-242; DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
  介绍了一种先驱性的编码器-解码器架构，包含卷积层和上采样卷积（转置卷积）层，为图像到图像学习任务（包括自编码器中的重建）提供了广泛影响力的蓝图。
- [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) — Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol (2010)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 11; Pages: 3371-3408; DOI: [10.1162/jmlr.2010.1276](https://doi.org/10.1162/jmlr.2010.1276)
  一篇关于自编码器的基础论文，讨论了学习鲁棒且有用的特征表示，与课程的“特征提取”目标直接相关。
- [Keras documentation: Conv2DTranspose layer](https://keras.io/api/layers/convolution_layers/conv2d_transpose/) — Keras team (2024)
  提供了 Conv2DTranspose 层的详细信息和使用示例，该层是卷积自编码器解码器部分中上采样功能的关键组件。
