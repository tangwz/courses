---
course: "introduction-to-computer-vision"
chapter: "basic-image-processing-techniques"
lesson: "point-operations-contrast-adjustment"
sourceId: 1251
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-3-basic-image-processing-techniques/point-operations-contrast-adjustment"
title: "点操作：对比度调整"
description: "了解并应用改变数字图像对比度的方法。"
order: 2
plots: ["plots/1251-0.json"]
sourceHash: "a0b9d1ad7591f5e3d407be4fc5c74d407ad932c8e5ea10b216962ae78e9e1512"
sourceCorrections: []
---

亮度调整是改变图像整体明暗程度，而对比度调整则处理像素强度值之间的*差异*。可以将对比度视为图像中较暗和较亮区域之间的分离程度。高对比度图像具有显著差异（深黑色、亮白色），而低对比度图像则显得平淡或模糊，像素强度值集中在一起。

调整对比度是一项重要的技术，用于使图像特征更加分明和美观。提高对比度有助于突出在亮度相似区域中可能模糊的细节。降低对比度有时可用于柔化图像，或为强度变化过大的特定分析做准备。

### 对比度调整的运作方式

调整对比度最常用的方法是线性变换，通常称为对比度拉伸。与亮度调整相似，这是一种*点操作*，意味着每个像素的值都根据其原始值独立修改。

核心思想是缩放像素强度与某个参考水平（通常是中等灰度，如8位图像的128）之间的差异。我们将像素值乘以一个因子，通常用alpha ($\alpha$) 表示，该因子充当*增益*或*斜率*。

一个结合了亮度和对比度调整的通用公式如下：

$p_{out} = \alpha \cdot p_{in} + \beta$

其中：

- $p_{in}$ 是输入像素值。
- $p_{out}$ 是输出像素值。
- $\alpha$ 是控制对比度的增益因子。
- $\beta$ 是控制亮度的偏差因子（如前一节所述）。

如果纯粹调整对比度，我们关注 $\alpha$ 因子：

- **增加对比度 ($\alpha > 1$):** 当 alpha 大于1时，像素强度之间的差异会被放大。暗像素变得更暗，亮像素变得更亮，有效拉伸了强度范围。这使得细节更加突出。
- **降低对比度 ($0 < \alpha < 1$):** 当 alpha 介于0和1之间时，像素强度之间的差异会减小。强度范围被压缩，导致图像更平坦、缺乏活力。
- **无变化 ($\alpha = 1$):** 如果 alpha 为1，对比度保持不变（尽管亮度仍可能通过 $\beta$ 进行调整）。
- **负对比度 ($\alpha < 0$):** 使用负 alpha 会反转图像强度（类似于生成照片底片），这是一种不太常见的对比度调整方法，但有时用于特定效果或分析。

### 效果可视化

我们可以通过绘制输出像素值与输入像素值的关系图来显示对比度因子 $\alpha$ 的效果。公式 $p_{out} = \alpha \cdot p_{in} + \beta$ 描述了一条直线。这条直线的斜率由 $\alpha$ 决定。斜率越陡，表示对比度越高。



![对比度因子 (α) 对像素值的影响](plots/1251-0.json)



> 该图显示了不同对比度因子 ($\alpha$) 如何改变应用于像素值的变换。更陡峭的斜率 ($\alpha > 1$，蓝线) 拉伸强度范围，增加对比度。更平缓的斜率 ($0 < \alpha < 1$，橙线) 压缩范围，降低对比度。虚线灰色线表示没有变化 ($\alpha = 1$)。请注意，确切的输出值还取决于亮度偏移 $\beta$，并且所示结果是在应用裁剪之前的。

### 值裁剪处理

与亮度调整一样，应用对比度变换可能会导致计算出的像素值 ($p_{out}$) 超出标准图像格式的有效范围（通常8位灰度或彩色通道的有效范围为 [0, 255]）。

例如，增加对比度 ($\alpha > 1$) 可能会将原始的暗值推到0以下，或将原始的亮值推到255以上。降低对比度 ($\alpha < 1$) 可能会将整个原始范围映射到 [0, 255] 内的一个更小的子范围。

为了处理这种情况，我们需要对结果进行*裁剪*或*饱和*处理。任何低于0的计算值都设置为0，任何高于255的值都设置为255。

$p_{out, 裁剪后} = \max(0, \min(255, p_{out}))$

这确保了所有最终像素值都保持在可显示的有效范围内。但请注意，裁剪可能导致调整后图像的极暗或极亮区域细节丢失，因为多个不同的输入值可能会映射到相同的输出值（0或255）。

对比度调整，特别是与使用 $p_{out} = \alpha \cdot p_{in} + \beta$ 公式进行的亮度控制结合使用时，提供了一种简单而有效的方法来修改图像的外观，通常作为更复杂的计算机视觉任务前的一个重要预处理步骤。

## 参考资料

- [Digital Image Processing](http://www.ImageProcessingPlace.com) — Rafael C. Gonzalez, Richard E. Woods (2018)
  Publisher: Pearson; Pages: 95-115
  涵盖图像处理基本操作的经典教材，包括对比度与亮度调整等强度变换。
- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer; Pages: 60-65
  详细介绍图像增强点操作的计算机视觉综合教材，包括线性对比度调整。
- [CS231A: Computer Vision: Foundations and Applications](https://cs231a.stanford.edu/) — Silvio Savarese, Jeannette Bohg (2025)
  Publisher: Stanford University
  一流大学的在线课程，涵盖计算机视觉基础概念，包括图像处理和强度变换。
