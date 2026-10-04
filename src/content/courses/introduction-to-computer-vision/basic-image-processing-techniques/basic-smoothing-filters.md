---
course: "introduction-to-computer-vision"
chapter: "basic-image-processing-techniques"
lesson: "basic-smoothing-filters"
sourceId: 1262
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-3-basic-image-processing-techniques/basic-smoothing-filters"
title: "基本平滑滤波器"
description: "了解用于降噪的简单模糊滤波器，如均值滤波器和高斯滤波器。"
order: 8
plots: ["plots/1262-0.json"]
sourceHash: "ba5280f73701fcd146fcfffc071754a350a75ed50e07ef03524c8e089ef9ad0d"
sourceCorrections: []
---

数字图像，特别是那些在弱光或使用廉价传感器拍摄的图像，通常包含不必要的亮度或颜色变化，这通常被称为*噪声*。这种噪声会影响后续的图像分析任务。平滑滤波器旨在通过平均像素与其邻近像素的值来减少这种噪声，从而有效地“平滑”由噪声引起的剧烈强度变化。

许多滤波操作（包括平滑）的原理是*卷积*。想象一个小的矩阵，称为*核*或*滤波器*，在输入图像上逐像素滑动。在每个位置，核覆盖一个特定的像素区域。核中的值被用作权重 (weight)，以计算该区域像素的加权平均值。这个计算结果将成为输出图像中中心像素的新值。

我们来看两种适合初学者的常见平滑滤波器。

### 均值滤波器（盒式滤波器）

最简单的平滑滤波器是均值滤波器，也称为盒式滤波器。它的核由相等的权重 (weight)组成。例如，一个3x3均值滤波器核看起来是这样的：


$$
K = \frac{1}{9} \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
$$


当这个核以输入图像中的某个像素为中心时，它会计算该3x3区域的简单平均值（将所有9个像素值相加并除以9）。这个平均值将替代输出图像中原始中心像素的值。

**工作方式：** 通过平均局部区域的像素值，由噪声引起的突变会减少，因为噪声像素的值会与其邻近像素混合。

**影响：** 这会使图像变得模糊或平滑。核的尺寸越大（例如5x5，7x7），模糊效果越明显。

**不足之处：** 尽管均值滤波器在减少噪声方面很有效，但它平等对待区域中的所有像素。这意味着它也会模糊尖锐的边缘，而这些边缘通常是图像中重要的特征。

### 高斯滤波器

一种更精细的方法是高斯滤波器。与均值滤波器使用相等权重 (weight)不同，高斯滤波器使用的核其权重由高斯函数（钟形曲线）确定。核中心附近的像素比远离中心的像素获得更高的权重。

一个3x3高斯核可能看起来是这样的（实际值取决于标准差参数 (parameter)，$\sigma$）：


$$
K \approx \begin{bmatrix}
\text{低} & \text{中} & \text{低} \\
\text{中} & \text{高} & \text{中} \\
\text{低} & \text{中} & \text{低}
\end{bmatrix}
$$


其中所有权重之和等于1。

**工作方式：** 高斯核将更高的权重分配给中心像素及其紧邻的像素，减少了远离像素的影响。这种加权平均基于高斯分布，由参数$\sigma$（sigma，标准差）影响。更大的$\sigma$会导致更宽的钟形曲线，从而产生更多的模糊。

**影响：** 高斯滤波也能平滑图像并减少噪声。但是，因为它给予中心像素更高的权重，所以它比同等大小的简单均值滤波器更能保持边缘。它产生一种更平滑、看起来更自然的模糊效果。

**为何使用它？** 当保持边缘信息有一定益处时，在高斯滤波器的噪声减少方面，通常比均值滤波器更受欢迎。它能有效减少高频噪声（如高斯噪声）。

以下是效果对比：



![平滑滤波器对比](plots/1262-0.json)



> 显示了一个带噪声的图像块，应用均值滤波器（均匀模糊）后的结果，以及应用高斯滤波器（更平滑的模糊，可能更好地保留图像块以外的边缘）后的结果。

在实际操作中，像OpenCV这样的库提供了函数（例如，用于均值滤波的`cv2.blur()`和用于高斯滤波的`cv2.GaussianBlur()`），它们能高效地处理卷积过程。通常你需要提供输入图像和核尺寸（高斯滤波器还需要sigma参数）。

在这些滤波器之间进行选择取决于噪声类型和期望的结果。对于简单的模糊或中度噪声，均值滤波器可能就足够了。若要获得更平滑且能更好保持边缘的效果，高斯滤波器通常是更好的选择。这些滤波器是进行边缘检测等更复杂分析之前的重要步骤。

## 参考资料

- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer
  计算机视觉领域的权威教材，涵盖了图像滤波和降噪方法，并提供了这些方法在视觉任务中应用的背景。
- [Image Filtering (Lecture 2, CS131: Computer Vision Foundations and Applications)](https://cs131.stanford.edu/lectures/lecture02.pdf) — Juan Carlos Niebles, Jiajun Wu (2022)
  Journal: Stanford University, Computer Science Department; Publisher: Stanford University, Computer Science Department
  大学课程讲义，提供了图像滤波概念的易懂介绍，包括卷积、平均滤波器和高斯滤波器。
