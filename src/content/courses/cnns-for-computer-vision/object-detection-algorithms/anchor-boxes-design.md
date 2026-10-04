---
course: "cnns-for-computer-vision"
chapter: "object-detection-algorithms"
lesson: "anchor-boxes-design"
sourceId: 2561
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-3-object-detection-algorithms/anchor-boxes-design"
title: "锚框：设计与优化"
description: "讨论物体检测中锚框的原理、设计考量及优化技术。"
order: 5
plots: []
sourceHash: "60b1c58b1c6655eb7cb39d417a96852cadcf965da8095bf819864106e1a41563"
sourceCorrections: []
---

检测物体不仅需要对图像区域进行分类，还需要使用边界框对其进行精确的定位。早期方法通常依赖于计算成本高昂的滑动窗口方法。现代检测器，特别是YOLO和SSD等单阶段检测器，以及Faster R-CNN等两阶段检测器的提议阶段，采用了一种更有效的方法：**锚框**（有时也称为默认框或先验框）。这些预定义的框提供了网络学习细化调整的参考模板。

可以把锚框看作是边界框的一组初始预测，它们策略性地放置在图像的不同位置，并具有不同的尺寸和形状。网络不是从零开始预测框的坐标，而是学习预测：

1. 每个锚框的**物体得分**：这个锚框包含物体中心的可能性有多大？
2. **偏移量**：应如何调整（移动、缩放）此锚框，使其紧密贴合实际物体？
3. **类别概率**：如果存在物体，它属于哪个类别？

这些锚框通常是相对于卷积骨干网络输出特征图的单元格来定义的。对于尺寸为 $W \times H$ 且具有 $C$ 个通道的特征图，每个 $W \times H$ 的空间位置（单元格）都与一组 $k$ 个锚框关联。每个锚框都有一个预定义的尺寸（大小）和长宽比（宽度与高度之比）。

> 具有不同尺寸和长宽比的锚框与特征图上的单个空间位置相关联，并投影到输入图像的相应区域。

### 设计锚框集合

基于锚框的检测器性能在很大程度上取决于锚框尺寸和长宽比的恰当选择。这些选择直接影响模型检测不同尺寸和形状物体的能力。常见策略包括：

1. **手动选择**：根据通用知识或初步数据集分析，定义一系列尺寸（例如，2的递增幂次方）和长宽比（例如，1:1、1:2、2:1、1:3、3:1）。这在SSD和Faster R-CNN中很常见。
2. **数据集统计（聚类）**：分析训练数据集中真实边界框的尺寸。使用聚类算法（如k-means）对框的尺寸（通常是归一化 (normalization)的宽度和高度）进行处理，以找到最具代表性的形状和大小。YOLOv2及后续版本推广了这种方法，旨在根据目标数据集定制锚框。聚类距离度量通常基于交并比（IoU），而不是欧几里得距离，以便更好地反映边界框的相似性。

每个位置的锚框数量 ($k$) 也是一个设计选择。使用更多的锚框可以增加覆盖范围和潜在的召回率，特别是对于形状不规则的物体，但同时也会增加计算成本和网络必须进行的预测数量。

### 训练时锚框与真实框的匹配

在训练期间，每个锚框都需要被标记 (token)为包含物体（正样本）或背景（负样本）。这种分配通常通过计算锚框与真实边界框之间的交并比（IoU）来完成。

一种常见的匹配策略包括：

1. 计算每个锚框与每个真实框之间的IoU。
2. 如果锚框与*任何*真实框的IoU超过高阈值（例如，IoU > 0.7），则将其指定为**正样本**。为了确保覆盖每个真实框，每个真实框中具有最高IoU的锚框也可能被指定为正样本，即使其IoU低于高阈值。
3. 如果锚框与*所有*真实框的最大IoU低于低阈值（例如，IoU < 0.3），则将其指定为**负样本**。
4. 介于低阈值和高阈值之间的锚框在训练期间通常会被忽略，不参与分类和回归损失计算，以避免模糊性。

### 通过边界框回归优化锚框

对于与真实框正向匹配的锚框，网络学习预测优化偏移量。网络不直接预测绝对坐标 ($x, y, w, h$)，而是预测相对于锚框属性 ($a_x, a_y, a_w, a_h$) 的四个增量值 ($t_x, t_y, t_w, t_h$)。

一种广泛使用的参数 (parameter)化方法（类似于Faster R-CNN）是：


$$
t_x = (x_{gt} - a_x) / a_w
$$


$$
t_y = (y_{gt} - a_y) / a_h
$$


$$
t_w = \log(w_{gt} / a_w)
$$


$$
t_h = \log(h_{gt} / a_h)
$$


这里，$(x_{gt}, y_{gt}, w_{gt}, h_{gt})$ 是真实框的中心坐标、宽度和高度，而 $(a_x, a_y, a_w, a_h)$ 是锚框的相应属性。网络使用回归损失（如Smooth L1损失）进行训练，以最小化其预测的 $t$ 值与从匹配的真实框计算出的目标 $t$ 值之间的差异。

在推理 (inference)时，网络预测 $(t_x, t_y, t_w, t_h)$。然后，这些预测被用于将初始锚框 $(a_x, a_y, a_w, a_h)$ 转换为最终预测的边界框 $(p_x, p_y, p_w, p_h)$：


$$
p_x = t_x \cdot a_w + a_x
$$


$$
p_y = t_y \cdot a_h + a_y
$$


$$
p_w = \exp(t_w) \cdot a_w
$$


$$
p_h = \exp(t_h) \cdot a_h
$$


这种回归机制使得网络能够精确调整预定义锚框的尺寸和位置，以准确匹配检测到的物体。最终优化后的框集合，连同其置信度和类别分数，通常会使用非极大值抑制（NMS）进行处理，以消除冗余检测，我们将在下一节讨论这一点。

选择合适的锚框配置需要在不同物体尺寸/长宽比的检测性能和计算效率之间进行权衡。尽管锚框一直是许多成功物体检测器的基础，但持续的研究也在关注无锚框方法，这些方法旨在更直接地预测物体位置，从而不再需要预定义的锚框集合及其相关的超参数 (hyperparameter)。然而，理解锚框对于使用许多先进的检测模型仍然很重要。

## 参考资料

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGtBfp-lN7e1UVoYkcr5MysqwfoZ--jzsYpu46TbfYE4Xf4xcVxAe9uphOnj7DOqTwf4-zDGnVaiLbHSMyRZ-UvICDP2v0yq1ej7ZVOTWu9LwjPNZFW8C-Xa4KkbAm4DS85XPwUx2XppItfh0RI04m3lMDLiwj07VWHRDOY3m1tEdFNRqqZZ4VLMKzSgm8U20j467s1LezQqCPlZQ7zhEfPAMXgpFi3UAw=) — Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun (2015)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 28; Publisher: NeurIPS Proceedings; Pages: 91-99; DOI: [10.48550/arXiv.1506.01497](https://doi.org/10.48550/arXiv.1506.01497)
  这篇论文提出了锚框及其边界框回归方法，成为许多后续目标检测架构的标准。
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) — Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg (2016)
  Journal: European Conference on Computer Vision (ECCV); Pages: 21-37; DOI: [10.1007/978-3-319-46448-0_2](https://doi.org/10.1007/978-3-319-46448-0_2)
  这项工作建立了一个流行的一阶段目标检测框架，有效利用了在多个特征图层上预定义的“默认框”（锚框）来检测不同尺度的对象。
