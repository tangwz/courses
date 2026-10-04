---
course: "introduction-to-computer-vision"
chapter: "getting-started-computer-vision"
lesson: "common-tasks-computer-vision"
sourceId: 1227
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-1-getting-started-computer-vision/common-tasks-computer-vision"
title: "计算机视觉的常见任务"
description: "概述图像分类、物体检测和分割等任务（仅限于说明）。"
order: 4
plots: []
sourceHash: "4eefab7d5e7e84f457226241a45361b644e671470a39db4308c44ab8a346ba3c"
sourceCorrections: []
---

既然我们明白了计算机视觉的目标，现在来看看计算机视觉系统旨在应对的一些具体问题或“任务”。可以把这些看作我们向计算机询问图像或视频的常见问题。在本入门课程中，我们仅说明几个基本任务。后面的章节会介绍处理其中一些任务所用的方法。

### 图像分类

假设你有一张图片，希望计算机能识别出图中唯一的、最显著的主体是什么。是猫？是狗？是汽车？还是房子？这个任务叫做**图像分类**。

目标很直接：为整张图片分配一个标签（一个类别名称）。系统会查看整张图片，然后确定它属于哪个预设类别。

比如，给定一张输入图片，一个分类系统可能会输出“猫”的标签。它不会说明猫在图片中的*具体位置*，只是说图片中有一只猫。这通常是人们在计算机视觉中学习的首批任务之一，因为它有助于对机器如何处理视觉资料形成初步认识。

> 图像分类系统接收一张图片，并为整张图片输出一个单一的类别标签。

### 物体检测

如果你不仅想知道图片中*有*一辆车，还想知道它*在*哪里呢？如果图片中有多个你关注的物体怎么办？这时，**物体检测**就派上用场了。

物体检测比分类更进一步。它的目标是识别图片中特定类别的物体实例，并确定它们的位置。通常，位置通过在每个被识别物体周围绘制矩形边界框来表示。

因此，对于一张繁忙街道的图片，物体检测系统可能会识别出多辆汽车、行人和交通灯，并为每个识别到的物体画一个框并进行相应标记 (token)（例如，“汽车”、“行人”、“交通灯”）。这比分类提供了更详细的图片内容信息。

> 物体检测系统在图片中识别出多个物体，并标示它们的位置，通常使用边界框和标签。

### 图像分割

有时，边界框不够精确。如果你需要知道物体的确切形状，细化到单个像素级别怎么办？例如，在医学影像中，你可能需要描绘肿瘤的准确轮廓。这个任务叫做**图像分割**。

图像分割是将图片分成多个部分或区域。它的目标通常是为图片中的*每个像素*分配一个类别标签。属于同一物体类别的像素会获得相同的标签。

设想给一张照片上色：所有属于汽车的像素可能被涂成红色，所有属于道路的像素被涂成蓝色，所有属于树木的像素被涂成绿色。这样就得到了一张图片细节图，其中每个像素的类别都已明确。它比分类或物体检测提供了更细致的图片内容识别。

> 图像分割为图片中的每个像素分配一个类别标签，从而精确描绘出不同物体或区域的形状。

### 其他任务

分类、检测和分割是基本任务，但计算机视觉涉及的远不止这些。其他常见任务有：

- **姿态估计：** 识别特定物体部位（如人体的四肢和关节）的位置和姿态。
- **光学字符识别 (OCR)：** 从图片中识别文字。
- **场景理解：** 对物体间的关联以及场景的背景建立更全面的语义识别。
- **图像生成：** 创建新图片，有时依据文字描述。

这些只是简要的说明，以便你了解计算机视觉的常见目标。随着学习的进行，你将进一步学习这些任务的应对方法和所用技术。目前，主要目的是明白计算机视觉旨在从视觉数据中获取特定类型的信息，而这些任务代表了我们可能寻求的不同细节层次和识别类型。接下来的部分，我们将开始准备所需工具，以便实际操作这些想法。

## 参考资料

- [Computer Vision: Algorithms and Applications](https://doi.org/10.1007/978-3-030-34372-9) — Richard Szeliski (2022)
  Publisher: Springer; DOI: [10.1007/978-3-030-34372-9](https://doi.org/10.1007/978-3-030-34372-9)
  一本全面的计算机视觉教科书，包含了分类、检测和分割等基本算法和应用。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Course Notes](http://cs231n.stanford.edu/) — Fei-Fei Li, Ehsan Adeli, Justin Johnson, Zane Durante (2025)
  Publisher: Stanford University
  详细解释了深度学习概念及其在图像分类、目标检测和分割等计算机视觉任务中的应用。
- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems (NIPS) 25; Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105
  本文介绍了AlexNet，一个突破性的深度卷积神经网络，极大地推动了图像分类领域的先进水平。
