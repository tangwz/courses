---
course: "cnns-for-computer-vision"
chapter: "image-segmentation-techniques"
lesson: "encoder-decoder-architectures-unet-segnet"
sourceId: 2574
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-4-image-segmentation-techniques/encoder-decoder-architectures-unet-segnet"
title: "编码器-解码器架构：U-Net 和 SegNet"
description: "了解流行的编码器-解码器架构，例如 U-Net 和 SegNet，它们专为生物医学和通用分割设计。"
order: 3
plots: []
sourceHash: "3e08980e5888206216864523993c810a648fc77a1de69718147960e64c11fb73"
sourceCorrections: []
---

全卷积网络 (FCN) 实现了端到端语义分割，能够将输入图像直接映射到像素级预测。然而，后续的发展改进了恢复特征提取过程中经常丢失的空间分辨率的方法。为应对这一挑战而出现的一种主要且高效的模式是**编码器-解码器架构**。

核心思想很简单：

1. **编码器**路径逐步降低输入图像的空间分辨率，同时增加特征通道的数量。这部分通常类似于标准分类网络（例如 VGG, ResNet），作为特征提取器。它的目的是获取图像中多尺度上的语义和上下文 (context)信息。最大池化等下采样操作在这里很常见。
2. **解码器**路径接收编码器生成的低分辨率、高通道特征图，并逐步将其上采样回原始输入分辨率。在上采样的过程中，它减少特征通道的数量，最终生成一个分割图，其中每个像素对应一个类别预测。

这种结构使网络能够首先理解图像中*有什么*（编码器的上下文聚合），然后精确地描绘*它在哪里*（解码器的定位）。U-Net 和 SegNet 是体现这一原理的两种有影响力的架构。

### U-Net：带跳跃连接的对称架构

U-Net 最初是为生物医学图像分割提出的，这一领域通常训练数据有限，且需要非常精确的定位。它的架构显著对称，可视化时呈“U”形。

> U-Net 架构示意图。箭头表示数据流；蓝色虚线箭头表示将编码器特征连接到解码器的跳跃连接。

**U-Net 的特点：**

1. **对称的编码器-解码器：** 解码器路径在层数和对应阶段的特征图大小上与编码器路径基本对应。
2. **跳跃连接：** 这是一个标志性特点。在解码器中的每个上采样卷积步骤之前，特征图会与编码器路径中对应的特征图（具有相同空间分辨率的特征图）进行拼接。这些连接直接提供了网络早期层的高分辨率特征。这很重要，因为空间信息在更深层中经常被稀释。通过重新引入这些特征，解码器可以生成细节更精细、定位更准确的分割掩码。拼接有效地将高级语义信息（来自解码器深层路径）与低级、精细的空间信息（通过跳跃连接来自编码器）结合起来。
3. **上采样：** U-Net 通常在解码器路径中使用学习到的转置卷积（有时称为上采样卷积或反卷积）进行上采样。每次上采样卷积都会使特征图尺寸加倍，并且通常将特征通道数量减半。
4. **最终层：** 最后的 1x1 卷积将每个像素位置的特征向量 (vector)映射到所需的输出类别数量。

跳跃连接在医学图像等应用中特别有效，这些应用中精确的边界描绘通常必不可少。

### SegNet：利用池化索引实现内存高效的解码

SegNet 与 U-Net 共享编码器-解码器结构，但在解码器路径中引入了不同的上采样机制，这主要是出于计算和内存效率的考虑。与 U-Net 类似，它通常使用预训练 (pre-training)的分类网络（如 VGG-16）作为其编码器。

> SegNet 架构示意图。橙色虚线箭头表示最大池化索引从编码器传输到解码器，以便在上采样时使用。

**SegNet 的特点：**

1. **编码器-解码器结构：** 整体结构与 U-Net 类似，通常使用 VGG-16 编码器。
2. **用于上采样的池化索引：** 这是主要区别。在编码器的最大池化操作中，SegNet 会存储在每个池化窗口中选择的最大值的空间位置（索引）。在解码器中，SegNet 不使用学习到的转置卷积或简单的双线性上采样，而是使用*反池化*操作。此操作将输入特征图中的值放置到由编码器存储的相应池化索引指定的位置。其他位置通常填充零。然后，这个稀疏图与学习到的滤波器进行卷积，生成一个稠密的特征图。
3. **内存效率：** 这种方法的主要好处是内存效率。仅传输池化索引所需的内存显著少于 U-Net 跳跃连接中传输整个高分辨率特征图所需的内存。这在训练非常深的模型或在内存受限下处理高分辨率图像时是有益的。
4. **无学习上采样参数 (parameter)（初始）：** 反池化操作本身没有可学习的参数，尽管解码器中的后续卷积有。
5. **特征信息：** 尽管高效，但仅使用池化索引意味着解码器不像 U-Net 那样直接接收来自编码器相应阶段的丰富特征表示。它主要获得上采样的空间位置引导。在某些情况下，这可能导致边界不如 U-Net 精细，尽管后续卷积会尝试学习重建细节。

### U-Net 与 SegNet 比较

| 特点 | U-Net | SegNet |
| --- | --- | --- |
| **上采样** | 转置卷积 (学习) | 反池化 (使用索引) + 卷积 |
| **信息传输** | 完整特征图拼接 (跳跃) | 最大池化索引 |
| **内存使用** | 较高 (由于拼接特征) | 较低 (仅存储索引) |
| **边界细节** | 可能更高 (可访问更丰富特征) | 可能更低 (需要重建) |
| **参数 (parameter)数量** | 解码器跳跃路径中参数更多 | 直接与上采样相关的参数更少 |

U-Net 和 SegNet 都代表了在设计用于语义分割等密集预测任务网络方面的重要进步。U-Net 的跳跃连接提供了一种有效的机制来整合多尺度信息，带来了出色的性能，特别是在精细细节重要的场合。SegNet 通过使用池化索引进行引导式上采样，提供了一种更注重内存的选择。理解这些模式很重要，因为它们构成了许多后续和更复杂分割架构的起点。在它们之间或受其启发变体中进行选择，通常取决于任务的具体要求、可用数据和计算资源。

## 参考资料

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Olaf Ronneberger, Philipp Fischer, and Thomas Brox (2015)
  Journal: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015; Publisher: Springer; Pages: 234-241; DOI: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28)
  提出U-Net架构的原始论文，强调其对称编码器-解码器结构和跳跃连接，以实现精确的定位，特别是在医学影像领域。
- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561) — Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla (2017)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Volume: 39; Pages: 2481-2495; DOI: [10.1109/TPAMI.2016.2628904](https://doi.org/10.1109/TPAMI.2016.2628904)
  介绍SegNet的基础论文，其通过使用最大池化索引进行上采样的方法，提高了内存效率。
- [Fully Convolutional Networks for Semantic Segmentation](https://doi.org/10.48550/arXiv.1411.4038) — Jonathan Long, Evan Shelhamer, Trevor Darrell (2015)
  Journal: IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 3431-3440; DOI: [10.48550/arXiv.1411.4038](https://doi.org/10.48550/arXiv.1411.4038)
  这项工作确立了全卷积网络作为端到端语义分割的基础，直接影响了U-Net和SegNet等后续编码器-解码器设计。
