---
course: "cnns-for-computer-vision"
chapter: "cnn-foundations-modern-architectures"
lesson: "residual-connections-skip-architectures"
sourceId: 2518
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/residual-connections-skip-architectures"
title: "理解残差连接与跳跃架构"
description: "详细阐释残差学习原理以及深度网络中跳跃连接的运作机制。"
order: 3
plots: []
sourceHash: "cfc652f86a316bdd7de5de948746b6e999eb504fa812bb5b1cb9f77206c00f20"
sourceCorrections: []
---

VGG等卷积神经网络 (neural network) (CNN) 架构采用更多层后，研究人员面临一个重大问题：网络准确率会达到饱和点，然后迅速下降。这不一定是过拟合 (overfitting)引起的，因为训练误差本身随着层数的增加而升高。这种现象被称为*退化问题*，它表明标准深度网络变得越来越难以优化。简单地堆叠层数使得求解器更难找到好的解，部分原因是反向传播 (backpropagation)过程中出现了梯度消失或梯度爆炸等问题。

## 核心思想：学习残差

残差网络（ResNets）的创新之处在于重新构建了问题，而不是寄希望于网络层能够直接学习一个期望的潜在映射，例如$H(x)$。其基本思想是让堆叠的层学习一个相对于层输入$x$的*残差函数* $\mathcal{F}(x)$。原始期望的映射$H(x)$则通过将输入加回去而获得：

$H(x) = \mathcal{F}(x) + x$

这种重新表述是基于一个假设：优化残差映射 $\mathcal{F}(x)$ 比优化原始的、未引用的映射 $H(x)$ 更容易。在极端情况下，如果恒等映射是最优的（$H(x) = x$），那么堆叠的层将更容易学习将 $\mathcal{F}(x)$ 的权重 (weight)推向零，而不是试图通过多个非线性层从头开始学习恒等函数。

## 残差块结构

这种思路通过*残差块*实现。一个典型的残差块包含几个堆叠的层（例如，两到三个卷积层，通常带有批量归一化 (normalization)和ReLU激活），以及一个*捷径*或*跳跃连接*，它绕过这些层并与堆叠层的输出进行逐元素相加。

残差块的输出$y$的等式通常写作：

$y = \mathcal{F}(x, \{W_i\}) + x$

这里：

- $x$是块的输入。
- $\mathcal{F}(x, \{W_i\})$表示块内加权层（例如，Conv-BN-ReLU-Conv-BN）学习到的残差映射，由权重 (weight) $\{W_i\}$ 参数 (parameter)化。
- $+$运算表示逐元素相加。
- $y$是块的输出。

捷径连接通常执行*恒等映射*，这意味着它直接将输入$x$传递给加法运算。这种恒等映射很重要：它不引入额外的参数，也不增加计算复杂性。

> 常见残差块的示意图。输入 $x$ 流经主路径（$\mathcal{F}(x)$ 涉及卷积、批量归一化和激活），同时通过恒等捷径绕过这些层。输出通过逐元素相加合并，随后进行最终激活（如ReLU）。

## 处理维度变化

如果输入 $x$ 的维度与残差函数 $\mathcal{F}(x)$ 的输出维度不匹配，会发生什么？这通常发生在 $\mathcal{F}(x)$ 中的卷积层使用大于1的步幅或改变滤波器数量时。在这种情况下，恒等捷径 $x$ 不能直接相加。

有两种常用策略：

1. **零填充：** 用额外的零填充输入 $x$，以增加其维度，使其与 $\mathcal{F}(x)$ 的输出匹配。
2. **投影捷径：** 在捷径连接中使用投影（通常是1x1卷积）来明确匹配维度（包括空间和深度/通道维度）。等式变为：

   $y = \mathcal{F}(x, \{W_i\}) + W_s x$

   其中 $W_s$ 是用于线性投影的矩阵（通过1x1卷积实现）。虽然这会增加参数 (parameter)和计算量，但这种方法可以提供更强的表示能力。原始ResNet论文考察了这两种方法，发现投影捷径略好，但恒等捷径在计算上更高效，并且通常也足够。

## 残差连接有效的原因

残差连接通过多种方式解决退化问题并有助于训练更深的网络：

1. **改善梯度流动：** 恒等捷径提供了一条直接路径，使得梯度在反向传播 (backpropagation)过程中能够回流到网络中。这有助于缓解梯度消失问题，即梯度在通过多层后变得指数级小，导致早期层训练非常缓慢。有了捷径，梯度可以相对畅通地流动，保证即使是深层网络也能接收到有效的更新信号。
2. **更简单的恒等映射学习：** 如前所述，如果一个最优函数接近恒等映射，残差块可以通过将 $\mathcal{F}(x)$ 的权重 (weight)推向零来轻松学习这一点。这被认为比直接使用一堆非线性层学习恒等映射更容易。
3. **打破对称性并促进特征多样性：** 一些工作指出，跳跃连接有助于打破训练过程中可能出现的对称性，并鼓励层学习更多样化的特征。

## 变体和预激活

原始的ResNet架构在逐元素相加*之后*应用最终激活（ReLU）。随后的工作引入了“预激活”ResNet块等变体（He 等人，2016）。在此变体中，批量归一化 (normalization)和ReLU激活在残差路径 $\mathcal{F}(x)$ 内的卷积层*之前*应用。这种设计可以提供更清晰的网络信息路径，并可能使优化更容易，从而带来更好的正则化 (regularization)和性能。

## 跳跃架构：一个更广的视角

尽管ResNet专门针对 $y = \mathcal{F}(x) + x$ 形式推广了“残差连接”这一术语，但*跳跃连接*的普遍思路——即绕过一个或多个层的连接——也出现在其他成功的架构中。

例如，U-Net（常用于图像分割，将在第四章讨论）采用了长跳跃连接，将收缩路径（编码器）的特征图与扩展路径（解码器）中对应的层连接起来。这些连接有助于解码器恢复在编码器池化操作中丢失的精细空间信息。DenseNets（本章接下来会谈到）则使用了另一种更广泛的跨层特征连接形式。

这些例子表明，在不同深度战略性地连接层是一种强大的架构模式，能够改善信息流动，并使训练用于各种计算机视觉任务的深层、高性能网络成为可能。对残差连接的理解为领会这些更复杂的连接模式提供了依据。

## 参考资料

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 770-778; DOI: [10.48550/arXiv.1512.03385](https://doi.org/10.48550/arXiv.1512.03385)
  介绍残差网络（ResNets）以克服超深度神经网络中退化问题的基础论文，详细阐述了残差块的架构。
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: European Conference on Computer Vision (ECCV); Pages: 630-645; DOI: [10.48550/arXiv.1603.05027](https://doi.org/10.48550/arXiv.1603.05027)
  这篇论文进一步分析了ResNet的机制，并提出了“预激活”残差块设计，改进了信息流和正则化效果。
- [Convolutional Neural Networks for Visual Recognition (CS231n) Lecture Notes](http://cs231n.stanford.edu/) — Fei-Fei Li, Justin Johnson, Serena Yeung (2017)
  Publisher: Stanford University
  斯坦福大学备受推崇的CS231n课程的综合讲义，涵盖了CNNs的基础概念和包括残差网络在内的高级架构。
