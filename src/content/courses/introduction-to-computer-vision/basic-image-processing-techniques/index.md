---
course: "introduction-to-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-3-basic-image-processing-techniques"
sourceId: 435
chapter: "basic-image-processing-techniques"
title: "基本图像处理技术"
order: 3
description: "学习基本图像处理操作：亮度/对比度调整、直方图、几何变换和基本滤波。"
hasQuiz: true
---

在了解了数字图像的数值表示方式之后，我们现在将重点转向修改和优化这些表示。基本图像处理操作是计算机视觉中的重要工具，它们可用于从提高图像质量到为特征检测等更复杂的分析准备图像的各种任务。

在本章中，你将学习几项重要的图像处理技术：

*   **点操作：** 通过修改单个像素值来调整图像的亮度与对比度。
*   **图像直方图：** 理解并使用直方图来可视化像素强度的分布，包括使用直方图均衡化进行自动对比度增强。
*   **几何变换：** 对图像进行缩放（调整大小）、旋转和平移（移动）等操作。
*   **图像滤波：** 介绍使用核（小型矩阵）根据相邻像素修改像素值的思路，重点介绍基本平滑滤波器（如均值滤波和高斯滤波）以减少噪声。

我们将讲解这些技术背后的原理，并演示如何使用常用库实现它们，最后通过动手实践来应用这些变换。
