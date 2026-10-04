# 第 4 章：特征检测入门

来源：[原章节](https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-4-introduction-feature-detection)

[返回课程目录](../README.md)

在回顾了基础图像表示和通常影响整个图像的处理操作后，我们现在将重点转向识别图像中特定且有意义的结构。这些结构，即特征，代表着独特的点、线或区域，可以作为地标。

本章将呈现特征检测的基本原理。您将学到：

*   图像中特征的构成要素，例如边缘和角点。
*   在各类计算机视觉任务中检测这些特征的理由。
*   边缘检测的初步方法，包括用于梯度计算的Sobel算子和广泛使用的Canny边缘检测器。
*   角点检测的基本思路，侧重介绍Harris检测器的原理。
*   对特征描述符的简要说明，它们以数值方式描述所检测到的特征。

我们将介绍这些方法的理论基础，并以一个使用OpenCV应用边缘检测技术的实践练习作为结尾。理解特征是更细致地分析图像内容的一步，为物体匹配和识别等任务做准备。

## 小节

- 1. [图像中的特征是什么？](01-%E5%9B%BE%E5%83%8F%E4%B8%AD%E7%9A%84%E7%89%B9%E5%BE%81%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F.md)
- 2. [为什么要识别特征？](02-%E4%B8%BA%E4%BB%80%E4%B9%88%E8%A6%81%E8%AF%86%E5%88%AB%E7%89%B9%E5%BE%81%EF%BC%9F.md)
- 3. [边缘检测原理](03-%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B%E5%8E%9F%E7%90%86.md)
- 4. [边缘检测的Sobel算子](04-%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B%E7%9A%84Sobel%E7%AE%97%E5%AD%90.md)
- 5. [Canny边缘检测器](05-Canny%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B%E5%99%A8.md)
- 6. [角点检测原理](06-%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%8E%9F%E7%90%86.md)
- 7. [Harris角点检测器的思路](07-Harris%E8%A7%92%E7%82%B9%E6%A3%80%E6%B5%8B%E5%99%A8%E7%9A%84%E6%80%9D%E8%B7%AF.md)
- 8. [特征描述符介绍](08-%E7%89%B9%E5%BE%81%E6%8F%8F%E8%BF%B0%E7%AC%A6%E4%BB%8B%E7%BB%8D.md)
- 9. [动手实践：图像边缘检测](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9B%BE%E5%83%8F%E8%BE%B9%E7%BC%98%E6%A3%80%E6%B5%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-4-introduction-feature-detection/quiz)
