---
course: "introduction-to-computer-vision"
chapter: "introduction-feature-detection"
lesson: "what-are-features-image"
sourceId: 1266
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-4-introduction-feature-detection/what-are-features-image"
title: "图像中的特征是什么？"
description: "理解计算机视觉中特征（边缘、角点、斑点）的原理。"
order: 1
plots: []
sourceHash: "99a088cc17d67670e1c611fbcace5d6027ce7d6b48d81dc1ca249848bd24c9ed"
sourceCorrections: []
---

在之前的章节中，我们主要将图像视为像素值的网格，并应用通常影响图像大部分甚至全部区域的操作。我们学习了如何调整亮度、对比度，甚至重新调整大小或旋转整个场景。现在，我们将重心从全局处理图像转移到识别突出或包含重要信息的特定局部区域。这些区域就是我们在计算机视觉中称为**特征**的部分。

想想你是如何识别物体或场景的。你不一定会以统一的方式单独处理每一个像素。相反，你的视觉系统会被独特部分所吸引：建筑物在天空下的清晰轮廓、书本的角落、一块木头的独特纹理。这些类似于计算机视觉中的特征。

那么，图像的哪一部分究竟算是“特征”呢？

特征本质上是图像中被认为对特定任务有趣或提供信息的模式或独特结构。特征是像素数据中的“地标”。它们通常对应：

- **兴趣点：** 例如角点，即线条相交或急剧改变方向的位置。
- **边缘：** 图像强度突然变化的边界，通常表明物体的轮廓或表面属性的变化。
- **区域：** 具有特定属性的区域，如一块独特纹理或均匀颜色的斑块，常被称为“斑点”。

考虑一张普通白墙的图像。从计算机视觉的角度来看，墙壁的大部分区域看起来非常相似；像素强度变化很小。这种均匀区域通常缺乏独特特征。现在，想象墙上画了一个黑色方块。方块的**边缘**（白色与黑色交界处）和方块的**角点**信息量很大。它们代表图像数据的明显变化，是很好的特征候选。

### 好的特征的属性

要使特征在计算机视觉任务中有用，它们理想情况下应具备某些特质：

1. **可重复性/可检测性：** 好的特征即使图像发生一些变化，也应该能可靠地检测到。如果我们稍微改变光照或从稍微不同的角度观察物体，我们理想地仍然能够找到相同的特征。
2. **独特性/信息量：** 特征应提供关于其位置的独特信息。角点通常比直边上的点更独特，因为强度在多个方向上发生变化。这种独特性有助于在不同图像之间匹配特征或随时间跟踪它们。
3. **局部性：** 特征通常占据图像中相对较小的区域。这使得它们对遮挡（物体部分被遮挡）或图像其他部分的杂乱情况具有抵抗力。
4. **数量：** 理想情况下，我们希望有足够的特征来充分代表我们的任务所需的图像内容，但又不要多到导致处理效率低下。“正确”的数量取决于应用。有纹理的表面可能会产生许多特征，而简单的几何形状可能只有少数角点和边缘。

### 特征示例

让我们可视化一些常见特征：

- **边缘：** 想想你的电脑显示器轮廓、照片中的地平线，或者标志上分隔不同颜色区域的线条。这些都是边缘的例子，其中像素强度在边界处明显地发生变化。
- **角点：** 考虑相框的角、星星的尖角，或者棋盘格方块边缘相交的位置。这些点通常在多个方向上显示强度梯度的急剧变化。

在本章中，我们将主要侧重于检测边缘和角点，因为它们是许多计算机视觉应用中使用的基本特征类型。理解如何识别这些点和线条是使计算机能够以更有意义的方式理解视觉场景的基础步骤，从而从简单的像素操作转向内容理解。以下部分将介绍旨在找到图像中这些信息丰富的部位的特定算法。

## 参考资料

- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf) — Richard Szeliski (2010)
  Publisher: Springer
  一本全面的教科书，提供图像特征的基础知识，包括边缘、角点及其特性，适合计算机视觉入门。
- [A Computational Approach to Edge Detection](https://doi.org/10.1109/TPAMI.1986.4767851) — John Canny (1986)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: PAMI-8; Pages: 679-698; DOI: [10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851)
  一篇开创性论文，介绍了广受认可的Canny边缘检测算法，是理解边缘特征的基础。
- [A Combined Corner and Edge Detector](http://www.bmva.org/bmvc/1988/avc-88-023.pdf) — Christopher G. Harris, M. J. Stephens (1988)
  Journal: Proceedings of the 4th Alvey Vision Conference; Pages: 147-151; DOI: [10.5244/C.2.23](https://doi.org/10.5244/C.2.23)
  本文介绍了有影响力的Harris角点检测器，该算法是识别图像中点特征的基石算法。
