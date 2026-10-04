---
course: "applied-autoencoders-feature-extraction"
chapter: "convolutional-autoencoders-image-data"
lesson: "pooling-layers-spatial-downsampling"
sourceId: 6474
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-5-convolutional-autoencoders-image-data/pooling-layers-spatial-downsampling"
title: "使用池化层进行空间降采样"
description: "在编码器中加入池化层（例如，最大池化）以进行特征图降采样。"
order: 3
plots: []
sourceHash: "72355209f27f3d9e8acaaacc74747611034753a647f57059eb00675920aa9853"
sourceCorrections: []
---

应用卷积层后，我们的自编码器编码器会生成一组特征图。这些图保留了空间信息，并显示了输入图像中学习到的各种模式。但是，它们在高度和宽度方面可能仍然很大，尤其是在网络的初始层。如果编码器始终保持这些全尺寸的特征图，我们将很快面临大量的参数 (parameter)和计算密集的网络。池化层在这里发挥作用。

池化层是卷积神经网络 (neural network)（CNN）中的一个基本组成部分，它们在卷积自编码器中也扮演着重要角色。它们的主要功能是**空间降采样**：在目标保留最重要信息的同时，减小特征图的高度和宽度。

### 为什么使用池化？

在编码器架构中使用池化层进行降采样有几个优点：

1. **维度降低**：通过减小特征图的空间维度（高度和宽度），池化层显著减少了后续层的参数 (parameter)数量。这使得网络计算效率更高，并有助于缓解过拟合 (overfitting)。
2. **特征抽象**：池化概括了局部区域内的特征。这种概括有助于网络构建更抽象的表示。例如，如果一个特定特征在小区域内的任何位置被检测到，池化操作就能捕获它的存在。
3. **平移不变性（小程度）**：池化提供了一定程度的局部平移不变性。这意味着如果输入图像中的对象轻微移动，池化后的特征图不太可能发生剧烈变化。网络对特征位置的微小变化更具鲁棒性。

### 常见池化类型

尽管存在多种池化策略，但最常用的是两种：

- **最大池化**：这是最常用的池化操作。对于输入特征图中的每个块（或窗口），最大池化会选择该块中的最大值。其直观解释是，最大值代表了该局部区域中最活跃、也因此最重要的特征。
- **平均池化**：平均池化不是取最大值，而是计算块内所有值的平均值。这提供了该区域特征的更平滑表示。尽管有时使用，但在识别特征最强存在更重要的任务中，最大池化通常表现更好。

### 池化层参数 (parameter)

定义池化层时，通常会指定两个主要参数：

1. **池化大小（或核大小）**：这定义了执行池化操作的窗口尺寸（例如，$2 \times 2$ 像素）。$2 \times 2$ 的池化大小非常常见。
2. **步长**：这个参数决定了池化窗口在特征图上水平和垂直移动的像素数量。
   - 如果步长等于池化大小（例如，池化大小 $2 \times 2$，步长 $2$），池化窗口将不重叠，并且输出特征图的高度和宽度将减半（假设采用“有效”填充，即不添加填充，如果不能完美整除，尺寸可能会略微减小）。
   - 如果步长小于池化大小，窗口将重叠。这在激进的降采样中不太常见。

例如，如果一个输入特征图的尺寸是 $28 \times 28 \times 64$（高 x 宽 x 通道），应用一个步长为 $2$ 的 $2 \times 2$ 最大池化层，将得到一个尺寸为 $14 \times 14 \times 64$ 的输出特征图。注意，通道数（深度）不会因池化操作而改变；池化是独立作用于每个通道的。

> CNN编码器中的一个典型顺序是：输入特征图通过一个卷积层，然后一个池化层减小其空间维度。

### 在编码器中的位置

池化层通常放置在编码器的一个或多个卷积层之后。一个常见模式是 `卷积 -> 激活 -> 池化`。这个序列可以重复多次，随着数据在编码器中流动，逐渐减小空间维度并增加深度（特征图/通道的数量，通常由卷积层控制）。这形成了一个特征的层次结构，从低级细节到更抽象、更高级的表示，同时管理着计算复杂性。

编码器中池化层实现的降采样是重要步骤。它将空间信息压缩成更紧凑的形式，然后才到达瓶颈层。因此，自编码器的解码器部分需要执行反向操作，对特征图进行上采样以重建原始图像尺寸。这种上采样过程将在后续关于转置卷积层和其他上采样技术的章节中介绍。

## 参考资料

- [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems 25 (NIPS 2012); Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105
  一篇基础性论文，展示了深度卷积神经网络的有效性，并突出强调了使用最大池化进行空间下采样。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 9: Convolutional Networks
  这本教科书提供了卷积网络的全面理论理解，包括池化层的原理和类型。
- [Convolutional Neural Networks (CNNs) for Visual Recognition](http://cs231n.github.io/convolutional-networks/) — Stanford University (2023)
  这些被广泛引用的课程资料直观而实用地解释了卷积神经网络和池化的作用。
