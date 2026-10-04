---
course: "introduction-to-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-4-introduction-feature-detection"
sourceId: 437
chapter: "introduction-feature-detection"
title: "特征检测入门"
order: 4
description: "了解图像特征是什么，以及边缘（Sobel、Canny）和角点（Harris）的初步检测方法。"
hasQuiz: true
---

在回顾了基础图像表示和通常影响整个图像的处理操作后，我们现在将重点转向识别图像中特定且有意义的结构。这些结构，即特征，代表着独特的点、线或区域，可以作为地标。

本章将呈现特征检测的基本原理。您将学到：

*   图像中特征的构成要素，例如边缘和角点。
*   在各类计算机视觉任务中检测这些特征的理由。
*   边缘检测的初步方法，包括用于梯度计算的Sobel算子和广泛使用的Canny边缘检测器。
*   角点检测的基本思路，侧重介绍Harris检测器的原理。
*   对特征描述符的简要说明，它们以数值方式描述所检测到的特征。

我们将介绍这些方法的理论基础，并以一个使用OpenCV应用边缘检测技术的实践练习作为结尾。理解特征是更细致地分析图像内容的一步，为物体匹配和识别等任务做准备。
