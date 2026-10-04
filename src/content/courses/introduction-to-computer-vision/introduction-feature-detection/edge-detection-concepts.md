---
course: "introduction-to-computer-vision"
chapter: "introduction-feature-detection"
lesson: "edge-detection-concepts"
sourceId: 1269
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-4-introduction-feature-detection/edge-detection-concepts"
title: "边缘检测原理"
description: "介绍寻找边界和剧烈强度变化的方法。"
order: 3
plots: ["plots/1269-0.json"]
sourceHash: "2385d0772f82686a88781430bc1ecc6f939209a31af53f0efac2dbf5e6a3112e"
sourceCorrections: []
---

我们来思考一下，是什么让图像的某些部分显得突出。想象一下，你正在看一张包含一个简单黑方块在白色背景上的图片。你的眼睛会立即注意到边界，即黑白交界的那条线。在计算机视觉中，这些边界被称为**边缘**。

边缘代表图像中像素强度发生剧烈、显著变化的位置。可以将其想象成当你从一个像素移动到其相邻像素时，亮度或颜色值突然跳变或下降。这些变化通常对应着场景中的重要情况：

- 两个不同物体之间的边界。
- 单个物体相对于其背景的轮廓。
- 表面方向或材料属性的变化。
- 场景光照的变化，例如阴影的边缘。

识别这些边缘是理解图像内容的基础一步。为什么呢？因为边缘通常勾勒出物体的形状。如果能可靠地找到边缘，我们就能开始识别物体的位置、它们的基本形状以及它们在空间上的相互关系。边缘检测简化了图像，减少了需要处理的数据量，同时保留了关于物体边界的重要结构信息。

那么，计算机如何*找到*这些强度变化呢？主要思路是寻找像素值快速变化的区域。我们可以使用**图像梯度**来数学化地衡量这种变化。梯度衡量图像中每个点强度变化的*速率*（幅度）和*方向*。

想象一个简单的一维信号，比如沿着一行像素的强度值。边缘在这种信号中会表现为陡峭的上升或下降台阶。



![穿过边缘的简化一维强度曲线](plots/1269-0.json)



> 沿穿过边缘的直线上的像素强度简化视图。在像素位置5附近的急剧跳跃显示出强烈的强度变化，这标志着一个边缘。

梯度在二维图像中具有两个分量：一个衡量水平方向的强度变化（记作 $G_x$），另一个衡量垂直方向的变化（记作 $G_y$）。

- 强的水平梯度 ($G_x$) 表示垂直边缘（强度随从左到右移动而变化）。
- 强的垂直梯度 ($G_y$) 表示水平边缘（强度随从上到下移动而变化）。

梯度的**幅度**告诉我们边缘的*强度*（强度变化的剧烈程度）。幅度越大意味着边缘越明显。梯度的**方向**或朝向告诉我们强度变化最快的方向，这个方向垂直于边缘本身。

因此，边缘检测算法旨在计算图像中的这些梯度，并找出梯度幅度高的像素。不同的算法，如Sobel算子（我们接下来会看到），提供了近似这些梯度的方法，而其他算法，如Canny边缘检测器，则基于此来精炼检测到的边缘，使它们更细、更连贯。理解这种寻找剧烈强度变化的基本思路是这些技术的基础。

## 参考资料

- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer
  这本教材全面介绍了计算机视觉。它包含图像梯度、边缘检测原理和各种边缘检测算法的讨论，是理解基本概念的相关资源。
- [A Computational Approach to Edge Detection](https://ieeexplore.ieee.org/document/4767851) — John Canny (1986)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: PAMI-8; Pages: 679-698; DOI: [10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851)
  这篇开创性论文介绍了Canny边缘检测算法。它概述了有效边缘检测的标准，包括检测、定位和对边缘的单一响应，基于梯度幅度和方向分析。
