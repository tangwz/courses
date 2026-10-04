---
course: "cnns-for-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-4-image-segmentation-techniques"
sourceId: 612
chapter: "image-segmentation-techniques"
title: "图像分割技术"
order: 4
description: "学习使用CNN的图像分割高阶技术，内容涵盖FCN、U-Net、DeepLab和Mask R-CNN的语义分割与实例分割。"
hasQuiz: false
---

图像分割不再仅仅对整张图像进行分类或为目标绘制边界框，其目标是对场景内容实现像素级的理解。其目的是为图像中的每个像素分配类别标签，提供一个精细的掩模，勾勒出对象的精确形状。

本章中，您将学习使用深度学习进行图像分割的基础原理和方法。我们将首先区分语义分割（按类别标记像素，如“道路”或“人”）和实例分割（标记不同的对象实例，如“人1”和“人2”）。

您将学习专门为此类密集预测任务而开发的几种重要架构和方法，其中包括：

*   全卷积网络（FCNs）作为一种基本方法。
*   编码器-解码器架构，例如U-Net，它在医学图像处理及其他方面应用广泛。
*   使用空洞（膨胀）卷积的技术，例如DeepLab系列中的方法，用于管理感受野。
*   结合检测和分割的方法，用于实现实例级别的结果，特别是Mask R-CNN。

我们还将讨论用于分割任务的常见评估指标，例如交并比（IoU），通常计算为 $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$，并以实践练习作为本章的结束。
