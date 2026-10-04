---
course: "cnns-for-computer-vision"
chapter: "object-detection-algorithms"
lesson: "region-proposal-networks"
sourceId: 2554
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms/region-proposal-networks"
title: "区域候选网络解析"
description: "详细讲解区域候选网络（RPN）在Faster R-CNN中的工作方式。"
order: 2
plots: []
sourceHash: "13446e83bbb1e7e8ab7fcf730595ceee6fa6eac1c45af47c04527e25e9e4dbc3"
sourceCorrections: []
---

两阶段目标检测器，例如R-CNN系列，首先在图像中提出可能包含对象的候选区域，然后对这些区域进行分类。早期方法如R-CNN和Fast R-CNN依赖于外部算法，例如Selective Search，来生成这些区域候选。虽然有效，但这一外部步骤通常计算成本高昂，并成为瓶颈，阻碍了真正的端到端训练和推理 (inference)。

Faster R-CNN引入了一项重要创新来解决这一瓶颈：区域候选网络（RPN）。RPN是一个全卷积网络，旨在直接从主干网络（如VGG或ResNet）生成的卷积特征图中高效预测对象候选。这种集成使得候选生成步骤能够与下游检测网络共享卷积特征，大大提高了速度。

### RPN的工作原理

设想主干网络处理输入图像后生成的深层卷积特征图。此特征图保留了空间信息，尽管分辨率低于原始图像。RPN通过在特征图上滑动一个小型卷积网络（通常使用3x3卷积核）来工作。

在特征图上的每个滑动窗口位置，RPN同时执行两项任务：

1. **预测“对象性”：** 它判断是否存在潜在对象，位于以该位置为中心的预定义边界框内，这些框称为*锚框*。
2. **调整框坐标：** 对于被认为可能包含对象的锚框，它预测对锚框坐标的调整（回归），以更好地匹配潜在对象。

### 锚框：起点

RPN不从零开始预测边界框（这是一个复杂任务），而是使用一组预定义的参考框，称为锚框（或简称锚点）。在RPN滑动窗口操作的每个位置，它会考虑多个锚框，这些锚框通常在尺度（大小）和长宽比（宽度与高度比）上有所不同。例如，一种常见配置是在每个位置使用9个锚框：即3种尺度与3种长宽比（例如1:1、1:2、2:1）的组合。

这些锚框作为潜在对象位置和形状的初始猜测或先验信息。RPN的任务不是凭空创建框，而是将每个预定义的锚框分类为“对象”或“背景”，并微调 (fine-tuning)有希望的锚框的坐标。

> Faster R-CNN内部流程。RPN从主干网络获取特征，使用锚框，并通过分类和回归层生成候选。这些候选与主干特征一同送入最终的检测头部。

### RPN的输出与训练

对于每个滑动窗口位置的$k$个锚框，RPN具有两个并行的输出层：

1. **分类层：** 此层输出$2k$个分数。对于每个锚框，它提供两个分数，分别代表该锚框是背景的概率$P(\text{bg})$和是前景（任何对象）的概率$P(\text{fg})$。这本质上是每个锚框的二分类任务。
2. **回归层：** 此层输出$4k$个值。对于每个锚框，它预测四个参数 (parameter)化的坐标调整值（$t_x, t_y, t_w, t_h$），以调整锚框的中心坐标（$x, y$）、宽度（$w$）和高度（$h$），使其更好地匹配潜在的真实对象框。

训练期间，锚框根据其与真实对象框的交并比（IoU）进行标记 (token)。锚框可能被标记为：

- **正例（对象）：** 如果它与真实框的IoU很高（例如 > 0.7），或者有时，如果它是与给定真实框IoU最高的锚框。
- **负例（背景）：** 如果它与所有真实框的IoU都很低（例如 < 0.3）。
- **忽略：** IoU值居中的锚框通常在训练期间被忽略，以避免模糊性。

RPN随后使用多任务损失函数 (loss function)进行端到端训练，该函数结合了：

- 针对正负锚框的对象性分数的分类损失（如交叉熵）。
- 针对预测坐标调整的回归损失（如Smooth L1损失），仅适用于正锚框。

### 为第二阶段生成候选

RPN处理特征图后，我们获得大量潜在的对象候选，每个都带有对象性分数和调整后的坐标。这些候选中有许多会高度重叠和冗余。为了精简这组候选，通常会执行两个步骤：

1. **按分数筛选：** 对象性分数非常低的候选会被丢弃。
2. **非极大值抑制（NMS）：** NMS根据对象性分数应用，以消除与高分数候选高度重叠的候选。这大大减少了候选的数量。

剩余的候选（通常每张图像几百到几千个，例如Faster R-CNN在NMS后大约300个）随后与原始特征图一同被传递到Faster R-CNN检测器的第二阶段（通常称为RoI头部或Fast R-CNN检测器）。该第二阶段对每个候选对应的特征执行RoIPooling（或RoIAlign），然后使用最终的分类和回归层以分配特定的类别标签（例如“汽车”、“人物”、“狗”）并进一步调整边界框坐标。

### RPN的优势

RPN的核心优势在于其计算效率。通过与最终检测网络共享高成本的卷积特征提取层，生成区域候选的成本变得几乎可以忽略不计，与依赖独立算法（如Selective Search）的旧方法相比。这种集成使Faster R-CNN能够实现显著更高的速度，实现接近实时的目标检测同时保持高准确性，并促进了整个检测流程的真正端到端优化。理解RPN对于理解许多现代高效两阶段目标检测器的架构非常重要。

## 参考资料

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) — Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (2015)
  Journal: Advances in Neural Information Processing Systems (NIPS); DOI: [10.48550/arXiv.1506.01497](https://doi.org/10.48550/arXiv.1506.01497)
  这篇原始论文介绍了区域提议网络（RPN），这是本节的核心内容。它详细阐述了RPN的架构、训练以及如何将其整合到Faster R-CNN框架中。
- [Fast R-CNN](https://arxiv.org/abs/1504.08083) — Ross Girshick (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1440-1448; DOI: [10.48550/arXiv.1504.08083](https://doi.org/10.48550/arXiv.1504.08083)
  这篇论文介绍了Fast R-CNN架构，通过在提议之间共享计算，显著提高了R-CNN的训练和推理速度。RPN后来被开发出来，以取代其外部提议生成步骤。
- [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) — Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik (2014)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 580-587; DOI: [10.48550/arXiv.1311.2524](https://doi.org/10.48550/arXiv.1311.2524)
  这项基础工作确立了使用卷积神经网络和外部区域提议的两阶段目标检测范式。它为RPN的发展提供了必要背景。
