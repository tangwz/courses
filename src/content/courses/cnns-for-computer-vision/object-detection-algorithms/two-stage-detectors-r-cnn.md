---
course: "cnns-for-computer-vision"
chapter: "object-detection-algorithms"
lesson: "two-stage-detectors-r-cnn"
sourceId: 2552
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms/two-stage-detectors-r-cnn"
title: "两阶段检测器：R-CNN 系列"
description: "考察两阶段目标检测器的发展及其工作原理，着重介绍R-CNN系列。"
order: 1
plots: []
sourceHash: "fbc810cc1f764105b22a40278d54474e5875313b45c82fe9a96fb1d66dca4b7f"
sourceCorrections: []
---

目标检测不仅需要对图像进行分类，还需要定位并识别图像中可能存在的多个物体。一类主要的方法分两个不同步骤处理此问题：首先，在图像中提出可能包含物体的候选区域；其次，对每个提出的区域内的物体（如果有）进行分类，并调整其边界框。这些被称为**两阶段检测器**。R-CNN系列是此方法最初的、颇具影响力的代表。

### R-CNN：基于CNN特征的区域

最初的R-CNN（基于卷积神经网络 (neural network) (CNN)特征的区域）论文通过成功应用深度CNNs，在目标检测准确性上实现了显著提升，而深度CNNs当时已在图像分类方面表现出色。然而，CNN分类器通常在固定大小的输入上运行，要使其适应检测不同尺寸和位置的任意大小物体，并非易事。R-CNN提出了一种多步骤的流程：

1. **区域提议：** R-CNN没有分析每个可能的窗口，而是首先生成了一组数量可控的候选物体区域。它使用了一种外部算法，例如**Selective Search**，该算法基于颜色、纹理和包含等低层特征，为每张图像生成约2000个与类别无关的区域提议（边界框）。
2. **特征提取：** 每个提出的区域都被形变（非等比例缩放）到预训练 (pre-training)CNN（如AlexNet）所需的固定输入尺寸。形变后的区域随后通过CNN进行处理，以提取固定长度的特征向量 (vector)（例如，来自最终分类器之前的层）。这一步是计算瓶颈。由于提议区域经常大量重叠，相同的图像像素需要通过耗时的CNN多次处理。
3. **分类：** 对于每个类别（加上一个背景类别），使用提取的CNN特征作为输入，训练一个独立的线性支持向量机（SVM）分类器。给定区域提议的特征，SVMs将预测该区域内物体的类别。
4. **边界框回归：** 为了提高定位准确性，R-CNN还训练了针对特定类别的线性回归模型。这些模型学习预测偏移量，以根据该区域的CNN特征调整提议边界框的坐标。

> R-CNN流程概述。请注意其独立阶段以及每个形变区域重复的CNN特征提取。

尽管R-CNN显著提升了当时的先进水平，但其流程存在主要缺点：

- **训练：** 它涉及多个独立的阶段：在形变区域上微调 (fine-tuning)CNN，为每个类别训练一个SVM，以及训练边界框回归器。这既复杂又非最优。
- **速度：** 推理 (inference)速度非常慢（当时GPU上每张图像约40-50秒），主要是因为CNN必须在每张图像的约2000个形变区域提议上独立运行，导致大量的重复计算。
- **存储：** 所有提议的提取特征都需要存储，占用大量磁盘空间。

### Fast R-CNN：共享计算

**Fast R-CNN**的主要动机是解决R-CNN的速度瓶颈。主要思路是，通过共享计算，可以避免对重叠区域进行重复的CNN计算。

Fast R-CNN并非对每个形变区域提议单独运行CNN，其工作方式如下：

1. **全图特征图：** 整个输入图像**一次性**通过主干CNN，以生成卷积特征图。
2. **投射提议：** 区域提议（仍然由外部生成，例如通过Selective Search）被投射到这个共享的卷积特征图上。
3. **RoI池化：** 引入了一个新颖的层，称为**感兴趣区域（RoI）池化**。对于每个投射的区域提议（现在对应于特征图上的一个矩形区域），RoI池化提取一个小的、固定尺寸的特征图（例如7x7）。它通过将特征图上的提议区域划分为固定大小的网格（例如7x7子窗口），并在每个子窗口内进行最大池化来实现此目的。这巧妙地处理了区域提议的可变尺寸，同时生成了适合后续全连接层的固定尺寸输出。
4. **统一头部：** 来自RoI池化的固定尺寸特征图被送入一系列全连接层。最后，它分支为两个并行的输出层：
   - 一个Softmax层，输出类别概率（K个物体类别 + 1个背景类别）。
   - 一个边界框回归层，输出调整后的框坐标（通常每个物体类别有4个值）。

> Fast R-CNN架构。特征提取只进行一次。RoI池化连接了可变尺寸提议与分类/回归头部固定尺寸输入之间的差异。

Fast R-CNN与R-CNN相比具有显著优势：

- **速度：** 在训练和推理 (inference)过程中都显著更快（训练快约9倍，推理快约200倍），因为大部分计算（CNN主干网络）在所有提议中共享。
- **端到端训练（大部分）：** 网络，包括分类和边界框回归层，可以在一个阶段内使用多任务损失（结合分类损失和回归损失）进行联合训练，与R-CNN的多阶段方法相比，简化了训练过程。
- **准确性：** 联合训练通常会带来准确性的提升。

然而，Fast R-CNN仍然依赖外部的、通常较慢的区域提议方法，例如Selective Search，这成为了推理时新的计算瓶颈。

### Faster R-CNN：迈向端到端检测

**Faster R-CNN**通过将区域提议机制**集成到**深度网络本身中，解决了Fast R-CNN的最后一个瓶颈。这是通过引入\*\*区域提议网络（RPN）\*\*实现的。

RPN是一个小型、全卷积网络，它以卷积特征图（由共享主干CNN生成）作为输入，并输出一组矩形物体提议，每个提议都带有一个相关的“物体性”分数（表示包含**任何**物体而非背景的概率）。

Faster R-CNN的工作方式如下：

1. **共享主干网络：** 与Fast R-CNN中一样，输入图像由主干CNN（例如VGG，ResNet）处理，以生成深度卷积特征图。
2. **区域提议网络（RPN）：**
   - 该网络在共享卷积特征图上滑动一个小的 $n \times n$ 空间窗口（例如 $3 \times 3$）。
   - 在每个滑动窗口位置，它同时考虑多个潜在提议。这些提议是根据预定义的**锚框**（或简称为“锚点”）生成的。锚框是以滑动窗口位置为中心的参考框，通常具有多种尺度和宽高比（例如，3种尺度 x 3种宽高比 = 每个位置9个锚框）。
   - 对于每个锚框，RPN通过并行全连接层（实现为$1 \times 1$卷积）输出两种预测：
     - **物体性分数：** 2个分数，表示锚框包含物体或背景的概率。
     - **框调整：** 4个值，表示参数 (parameter)化的坐标调整（中心 $x, y$ 的偏移量以及宽度、高度的对数空间调整），以使锚框更好地适应潜在物体。
   - 在所有位置生成提议后，根据物体性分数应用非极大值抑制（NMS）以减少冗余。
3. **RoI池化与最终头部：** RPN生成的高分区域提议随后被使用，就像Fast R-CNN中的外部提议一样。它们被投射到**相同**的共享特征图上，RoI池化为每个提议提取固定尺寸的特征，这些特征再送入最终的分类和边界框回归层，以预测具体的物体类别并进一步调整框坐标。

> Faster R-CNN架构。RPN使用共享特征图在内部生成提议，使得系统几乎端到端，并消除了Selective Search的瓶颈。

RPN的引入意义重大，因为它：

- **效率：** 它与下游检测网络共享耗时的卷积特征，使得区域提议的生成几乎没有计算成本。
- **端到端可训练性：** 整个系统（主干网络、RPN、检测头部）可以联合训练，尽管原始论文描述了一种四步交替训练方案来管理RPN训练和Fast R-CNN检测器训练之间的依赖关系。现代实现通常采用近似联合训练。
- **提议质量提升：** RPN学习生成专门针对检测网络的提议，这可能带来比Selective Search等固定算法更高质量的提议。

Faster R-CNN成为了许多后续目标检测模型的核心架构。虽然其速度常被单阶段检测器（我们将在后续章节中介绍）超越，但两阶段方法，特别是Faster R-CNN框架，在复杂场景的定位准确性上通常保持优势。了解R-CNN这一发展脉络，对于理解现代目标检测系统中的设计选择和权衡具有重要意义。

## 参考资料

- [Rich feature hierarchies for accurate object detection and semantic segmentation](https://doi.org/10.1109/CVPR.2014.81) — Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik (2014)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition; Publisher: IEEE; Pages: 1770-1777; DOI: [10.1109/CVPR.2014.81](https://doi.org/10.1109/CVPR.2014.81)
  介绍R-CNN架构的开创性工作，通过区域提议和SVM分类器将CNN应用于对象检测。
- [Fast R-CNN](https://doi.org/10.1109/ICCV.2015.169) — Ross Girshick (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 1440-1448; DOI: [10.1109/ICCV.2015.169](https://doi.org/10.1109/ICCV.2015.169)
  通过在不同提议间共享CNN计算并引入RoI池化层以加快处理速度，改进了R-CNN的速度限制。
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://doi.org/10.48550/arXiv.1506.01497) — Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (2015)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 28; DOI: [10.48550/arXiv.1506.01497](https://doi.org/10.48550/arXiv.1506.01497)
  引入区域提议网络（RPN），将提议生成整合到深度学习框架中，实现了几乎端到端的目标检测系统。
