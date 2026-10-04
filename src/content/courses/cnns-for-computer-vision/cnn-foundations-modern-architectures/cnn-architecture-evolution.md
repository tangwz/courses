---
course: "cnns-for-computer-vision"
chapter: "cnn-foundations-modern-architectures"
lesson: "cnn-architecture-evolution"
sourceId: 2515
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/cnn-architecture-evolution"
title: "CNN 架构的演变：从 AlexNet 到 ResNet"
description: "回顾卷积神经网络架构的历史发展和创新，直至残差网络 (ResNet)。"
order: 2
plots: []
sourceHash: "32033bc1551a7e1c230de2f786eddaf249d6f581b674bccb847729e5237462eb"
sourceCorrections: []
---

CNN 架构在 2012 年至 2015 年间取得了显著进展。这些发展包含了卷积层、池化和激活函数 (activation function)（如 $ReLU(x) = max(0, x)$）等基本原理。这一时期进展迅速，主要受年度 ImageNet 大规模视觉识别挑战赛 (ILSVRC) 的推动，该挑战赛为图像分类任务提供了严格的基准。这一时期的演变为当今许多先进模型的发展打下了铺垫。

### AlexNet：深度学习 (deep learning)的再兴

尽管 LeNet-5 等早期网络展现了卷积神经网络 (neural network) (CNN)的潜力，但由 Alex Krizhevsky、Ilya Sutskever 和 Geoffrey Hinton 开发的 **AlexNet** 架构在 2012 年标志着一个转折点。它在 ILSVRC 2012 竞赛中的压倒性胜利重新点燃了人们对计算机视觉方面深度学习的广泛兴趣。

促成 AlexNet 成功的因素有以下几点：

1. **增加的深度：** 与早期卷积神经网络相比，AlexNet 深度显著增加，具有 8 个可学习层（5 个卷积层和 3 个全连接层）。
2. **ReLU 激活函数 (activation function)：** 它大量使用了修正线性单元 ($ReLU$) 激活函数。ReLU 有助于缓解梯度消失问题，该问题曾困扰使用 sigmoid 或 tanh 激活函数的早期深度网络，从而实现更快、更有效的训练。
3. **GPU 加速：** 该模型使用 NVIDIA GPU 进行训练，显著加速了计算密集型训练过程。这使得训练更深、更大的网络变得实用。
4. **数据增强和 Dropout：** AlexNet 采用了积极的数据增强技术（如图像平移、水平翻转和图像块提取），并引入 `Dropout` 作为一种正则化 (regularization)方法。Dropout 在训练期间随机将一部分神经元激活值设为零，阻止复杂的协同适应并减少过拟合 (overfitting)。
5. **重叠池化：** 它使用了步长小于池化窗口大小的最大池化层，导致感受野重叠，这被发现可以略微提升性能并提供一定的平移不变性。

AlexNet 的成功不只在于赢得一场比赛；它有力地证明了，在大型数据集上用足够算力 (compute)训练的深度卷积网络，能够在具有挑战性的计算机视觉任务上取得显著性能。

### VGGNets：通过简洁实现深度

继 AlexNet 之后，研究人员研究了网络深度如何影响性能。由牛津大学的 Karen Simonyan 和 Andrew Zisserman 于 2014 年开发的 **VGG 网络**，提供了一个令人信服的答案。VGG 背后的核心思想是架构的简洁性和增加的深度。

VGGNet（如 VGG-16 和 VGG-19，因其权重 (weight)层数量而得名）的特点：

1. **同构架构：** VGG 在整个网络中仅使用小型 $3 \times 3$ 卷积核，并顺序堆叠。这形成了一个统一且易于理解的结构。
2. **小卷积核，更深层：** 使用 $3 \times 3$ 卷积核意义重大。两个 $3 \times 3$ 卷积层（中间带有非线性）的堆叠具有与单个 $5 \times 5$ 层等效的有效感受野，而三个 $3 \times 3$ 层的堆叠则对应于一个 $7 \times 7$ 感受野。然而，堆叠方法使用的参数 (parameter)更少，并引入了更多的非线性激活函数 (activation function)，增强了网络的判别能力。例如，三个 $3 \times 3$ 层需要 $3 \times (3^2 \times C \times C) = 27 C^2$ 个权重（假设输入/输出通道为 C），而一个 $7 \times 7$ 层则需要 $7^2 \times C \times C = 49 C^2$ 个权重。
3. **显著深度：** VGG-16 和 VGG-19 将网络深度显著推向比 AlexNet 更远，证明了显著的深度对图像分类准确性有益。

虽然 VGG 取得了出色的结果，并且由于其简洁的结构，其预训练 (pre-training)权重在迁移学习 (transfer learning)中仍然受欢迎，但它也伴随着缺点：计算成本高昂，并且参数数量非常庞大（主要集中在全连接层），使其内存密集。

### GoogLeNet (Inception v1)：通过宽度提升效率

同样于 2014 年发布（并赢得了当年的 ILSVRC 竞赛）的 **GoogLeNet**（或 Inception v1），由 Google 的 Christian Szegedy 及其同事开发，采取了不同的方法。它不只是增加深度，而是侧重于计算效率，并引入了 `Inception 模块`的思路。

其动机是图像中的特征以不同尺度出现。最佳卷积核大小可能因所检测特征而异。Inception 模块通过并行执行多个不同卷积核大小（$1 \times 1$、$3 \times 3$、$5 \times 5$）的卷积和最大池化，然后将它们的输出拼接起来，解决了这个问题。

GoogLeNet 的重要特点：

1. **Inception 模块：** 这个模块作用类似于一个多层次特征提取器，在同一层内同时捕获不同尺度的模式。
2. **通过 $1 \times 1$ 卷积进行降维：** 为控制计算成本，特别是在 Inception 模块中较大 $3 \times 3$ 和 $5 \times 5$ 卷积之前，使用了 $1 \times 1$ 卷积。这些“瓶颈”层显著减少了输入通道（特征图）的数量，这是一种借鉴自 Network-in-Network 论文的技术。这极大减少了参数 (parameter)和计算量。
3. **更深但更高效：** GoogLeNet 比 VGG 更深（22 层），但参数数量少得多（约 500 万，而 VGG-16 约 1.38 亿）。
4. **辅助分类器：** 为对抗如此深的网络中的梯度消失问题，GoogLeNet 在训练期间包含了连接到中间层的辅助分类器。它们的损失被加到总损失中，为网络早期部分提供额外的梯度信号。这些在推理 (inference)时被移除。

GoogLeNet 证明了，网络性能的提升不仅可以通过纯粹的深度，还可以通过精心设计更宽、计算高效的构建块来实现。

### ResNet：实现真正深度网络

尽管 VGG 和 GoogLeNet 取得了成功，但简单地堆叠更多层最终导致了一个称为**退化**的问题。随着网络变得更深，训练准确率会饱和，然后迅速下降。这并非由过拟合 (overfitting)引起（训练误差本身增加了），而是由于优化非常深的网络存在困难。如果某个块的最优函数是简单的恒等映射，那么堆叠的非线性层似乎更难学习到这种映射。

2015 年，Kaiming He 和微软研究院的合作者提出了 **残差网络 (ResNet)**，根本性地改变了非常深网络的构建方式。其核心创新是 `残差连接` 或 `跳跃连接`。

1. **残差学习：** ResNet 没有期望一系列层学习一个潜在映射 $H(x)$，而是明确地重新定义了问题。它让层去学习一个 *残差* 函数 $\mathcal{F}(x) = H(x) - x$。块的输出变为：
   $y = \mathcal{F}(x) + x$
   这里，$x$ 是块的输入，通过一个“捷径”或“跳跃”连接传递，而 $\mathcal{F}(x)$ 是块内层学习到的映射。相加通常是逐元素进行的。
2. **优化简便性：** 这种表述使得优化更容易。如果恒等映射是最佳的（$H(x) = x$），网络可以通过简单地将 $\mathcal{F}(x)$ 中层的权重 (weight)推向零来实现这一点，这比通过复杂的非线性变换学习恒等函数更容易。
3. **突破深度障碍：** 残差连接使研究人员能够成功训练比以前深得多的网络（例如，50、101、152 层，甚至实验性地超过 1000 层），而不遭受退化问题。更深的 ResNet 在 ImageNet 和其他基准测试上持续取得更好的结果。
4. **瓶颈设计：** 对于更深的网络（ResNet-50+），使用了更高效的“瓶颈”块设计，采用 $1 \times 1$ 卷积在 $3 \times 3$ 卷积之前降低维度，然后是另一个 $1 \times 1$ 卷积来恢复维度，这在理念上与 GoogLeNet 为提高效率所采取的方法相似。

> VGG、Inception 和 ResNet 架构中基本构建块的简化比较。VGG 使用顺序卷积。Inception 使用带有不同卷积核大小和 1x1 瓶颈的并行路径，最后进行拼接。ResNet 引入了跳跃连接，将输入 $x$ 添加到卷积路径 $\mathcal{F}(x)$ 的输出。

ResNet 引入残差学习是一项具有里程碑意义的成就。它提供了一种训练极深网络的方法，克服了之前的局限性，并为网络设计树立了新标准。许多我们稍后将讨论的后续架构，都是在 ResNet 确立的原则之上构建的。

从 AlexNet 的突破到 VGG 的深度、GoogLeNet 的效率以及 ResNet 处理极端深度的能力，这一系列演变凸显了由经验结果和创新的架构思维推动的快速发展。这些核心架构为理解后续章节中更复杂的模型和技术提供了背景。

## 参考资料

- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105
  介绍了AlexNet，这是一个开创性的深度CNN，在ILSVRC 2012中取得了破纪录的成绩，极大地推动了计算机视觉深度学习领域的发展。
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) — Karen Simonyan and Andrew Zisserman (2014)
  Journal: International Conference on Learning Representations (ICLR); Publisher: OpenReview; DOI: [10.48550/arXiv.1409.1556](https://doi.org/10.48550/arXiv.1409.1556)
  介绍了VGG架构，证明使用小型3x3卷积核增加网络深度能显著提升图像分类精度。
- [Going Deeper with Convolutions](https://ieeexplore.ieee.org/document/7298594) — Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich (2015)
  Journal: IEEE Conference Publication; Publisher: IEEE; Pages: 1-8; DOI: [10.1109/CVPR.2015.7298594](https://doi.org/10.1109/CVPR.2015.7298594)
  介绍了GoogLeNet（Inception v1），该架构通过创新的Inception模块和1x1卷积进行降维，侧重于计算效率。
- [Deep Residual Learning for Image Recognition](https://ieeexplore.ieee.org/document/7780459) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: IEEE Conference Publication; Publisher: IEEE; Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  介绍了ResNet，通过引入残差连接，实现了训练极深神经网络的突破，有效解决了网络退化问题。
