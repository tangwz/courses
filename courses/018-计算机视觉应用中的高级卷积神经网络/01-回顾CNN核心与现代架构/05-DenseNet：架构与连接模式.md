# DenseNet：架构与连接模式

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/densenet-architecture-connectivity)

[返回章节目录](README.md) · [返回课程目录](../README.md)

密集连接卷积网络（DenseNets）采用一种非常有效的连接模式。一些神经网络 (neural network)架构通过求和组合特征，而DenseNets则通过拼接方式集成特征。其主要特点是，在同一块内，每个层都接收来自*所有*前面层的特征图作为输入，并且其自身的输出特征图也会作为该块中所有后续层的输入。

### 密集块：主要组成部分

DenseNet的基本单元是密集块。在一个包含 $L$ 个层的密集块中，第 $l$ 个层接收来自所有前面层 $x_0, ..., x_{l-1}$ 的特征图作为输入。其输出 $x_l$ 计算如下：

$x_l = H_l([x_0, x_1, ..., x_{l-1}])$

这里，$[x_0, x_1, ..., x_{l-1}]$ 表示层 $0, ..., l-1$ 中生成的特征图的拼接。函数 $H_l(\cdot)$ 是一个复合函数，表示层内的操作，通常由批归一化 (normalization)（BN）、接着是修正线性单元（ReLU）激活，最后是3x3卷积（Conv）构成。这种BN-ReLU-Conv的顺序在DenseNets中很常见。

> 包含三个层（$L=3$）的密集块内部的连接情况。每个层 $H_l$ 都接收来自所有前面层的拼接特征图 $[x_0, ..., x_{l-1}]$。虚线表示用于拼接的数据流，它不经过层 $H_l$ 的计算。

### 增长率

DenseNets中的一个重要超参数 (parameter) (hyperparameter)是*增长率*，记为 $k$。每个层中的复合函数 $H_l$ 生成 $k$ 个输出特征图。由于每个层都接收来自所有前面层的特征图，因此层 $l$ 的输入将有 $k_0 + (l-1) \times k$ 个通道，$k_0$ 指的是密集块的输入特征图的通道数。

增长率 $k$ 控制着每个层向块内集体知识贡献多少新的“信息”。小的增长率（例如 $k=12$ 或 $k=32$）使得网络在参数使用上非常高效，因为每个层只添加少量特征图。这种设计选择基于以下假设：前面层的特征图可以直接被后续层访问，减少了层重新学习冗余特征的需要。DenseNets可以以比ResNet等架构少得多的参数达到有竞争力的准确率，这主要归因于这种特征复用和小的增长率。

### 过渡层

密集块不能无限期地延续，主要原因是拼接的特征图数量随深度线性增长（在 $L$ 层后有 $k_0 + (L-1) \times k$ 个通道），这会使计算成本很高。此外，还需要池化操作来下采样特征图的空间维度，这在CNN中是一种标准做法。

DenseNets在连续的密集块之间引入*过渡层*来解决这些问题。一个过渡层通常由以下部分组成：

1. 一个批归一化 (normalization)层。
2. 一个1x1卷积层：这作为瓶颈层以减少特征图的数量。如果一个密集块输出 $m$ 个特征图，1x1卷积可能会将其减少到 $\lfloor \theta m \rfloor$ 个特征图，其中 $\theta$ ($0 < \theta \le 1$) 是*压缩因子*。一个常用值是 $\theta = 0.5$。
3. 一个步幅为2的2x2平均池化层：这会下采样特征图的空间分辨率（宽度和高度）。

> 高层结构图，显示过渡层连接两个密集块，执行压缩和下采样。

### 整体架构

一个典型的DenseNet架构首先是一个初始卷积和池化层，接着是多个密集块，中间穿插过渡层。最后，一个全局平均池化层和一个全连接分类器（通常只是一个softmax层）生成最终输出预测。每个密集块中的层数可以根据具体的DenseNet配置（例如DenseNet-121，DenseNet-169）而变化。

### 优点与考量

DenseNets提供多项优点：

- **梯度流良好：** 从所有前面层到后续层的直接连接为梯度创建了更短的路径，减轻了梯度消失问题。
- **特征复用度高：** 拼接特征图明确鼓励网络复用早期学习到的特征，从而生成更紧凑、参数 (parameter)更高效的模型。
- **隐式深度监督：** 由于连接较短，前面层能更直接地从最终损失函数 (loss function)接收监督信号。

然而，存在一个重要的实际考量：

- **内存占用：** 在训练期间，为了拼接而存储密集块内的所有中间特征图会消耗大量GPU内存。尽管DenseNets在参数上高效，但它们可以是内存密集型的。深度学习 (deep learning)框架中存在巧妙的实现策略，通过选择性地重新计算特征图而不是全部存储来减少这种内存占用。

### 与ResNet的比较

尽管ResNet和DenseNet都使用跳跃连接来改善信息和梯度流动，但它们的机制有着根本的不同。ResNet使用逐元素相加（$y = \mathcal{F}(x) + x$）来组合跳跃连接和转换后的特征。这种相加允许恒等映射，以便在需要时轻松跳过层。DenseNet使用按通道拼接（$x_l = H_l([x_0, ..., x_{l-1}])$），强制层汇集所有前面层的特征。这促进了积极的特征复用，但导致在块内深度增加时层变得更宽（就通道而言）。两种方法都被证明在实现非常深的网络训练方面非常有效。

总之，DenseNet引入了一种新颖的连接方式，通过以前馈方式将块内的每个层直接与所有其他层连接起来，从而最大限度地增加层之间的信息流动。这种设计鼓励特征复用，从而形成紧凑的模型，这些模型通常以更少的参数 (parameter)达到当前最佳表现，尽管可能以训练期间更高的内存消耗为代价。

## 参考资料

- [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  提出了残差网络（ResNet），引入了快捷连接以促进深度网络的训练。本文为理解DenseNet的替代方法提供了背景。
- [CS231n: Deep Learning for Computer Vision - Lecture Notes on Architectures](http://cs231n.stanford.edu/) — Stanford University (2024)
  提供了来自领先学术课程的现代CNN架构（包括DenseNet）的教育性概述。

---

[上一节](04-Inception%20%E6%A8%A1%E5%9D%97%E5%92%8C%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E7%BD%91%E7%BB%9C%E6%80%9D%E6%83%B3.md) · [下一节](06-EfficientNet%EF%BC%9A%E6%A8%A1%E5%9E%8B%E5%A4%8D%E5%90%88%E7%BC%A9%E6%94%BE.md)
