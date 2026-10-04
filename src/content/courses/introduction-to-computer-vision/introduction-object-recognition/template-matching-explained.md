---
course: "introduction-to-computer-vision"
chapter: "introduction-object-recognition"
lesson: "template-matching-explained"
sourceId: 1282
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-5-introduction-object-recognition/template-matching-explained"
title: "模板匹配详解"
description: "学习一种在大图中找出模板图像出现位置的简单方法。"
order: 2
plots: []
sourceHash: "ab0f3f34b343c0b3a82b9f27e33949444755e87956d2d75eaecaa1db1ef2812b"
sourceCorrections: []
---

我们已经明确了目标：在图像中找出特定物体。但我们如何实际地告诉计算机去*找到*某个特定内容，比如一个特定标志或游戏角色，在一张大图中呢？一种最直接的方法称为**模板匹配**。

想象你有一张你正在寻找的精确物体的小图——这就是你的**模板**。接着，你有一张大图——**源图像**——你怀疑这个物体可能藏在里面。模板匹配的工作方式很像你在堆满拼图碎片的桌子上寻找一块特定拼图：你拿起你已知的碎片（模板），系统地将其滑过整个桌面（源图像），在每个可能的位置检查它是否吻合。

### 滑动窗口方法

模板匹配的主要思路是**滑动窗口**。具体操作如下：

1. **获取模板：** 你从模板图像开始（例如，一个特定按钮的小图）。
2. **放置到源图上：** 你虚拟地将此模板放置到较大源图像的左上角。
3. **比较：** 算法直接将模板图像与当前其覆盖的源图像区域进行比较。它计算一个**相似度分数**，量化 (quantization)模板中的像素与下方源图像区域中的像素的匹配程度。高分表示匹配良好；低分表示匹配不佳。有不同的数学方法来计算此分数（例如比较像素差异或寻找相关性），但基本思路是衡量模板和当前区域的“相似程度”。
4. **滑动：** 算法随后将模板向右滑动一个像素（或几个像素），并重复比较，为这个新位置计算一个新的相似度分数。
5. **扫描：** 它继续沿当前行滑动模板，一旦到达行尾，它会向下移动一个像素（或几个像素），并开始滑动下一行。这样一直持续，直到模板与源图像中每个可能的区域位置都进行了比较。
6. **找出最佳匹配：** 随着模板的滑动，算法会记录为每个位置计算的所有相似度分数。找到最高相似度分数的位置被认为是最佳匹配——也就是模板模式最有可能出现在源图像中的位置。

> 使用滑动窗口方法的模板匹配流程。

这个过程的结果通常可视化为一个**结果图**，一张图像，其中每个像素的亮度对应于模板在该像素在源图像中位置居中时计算出的相似度分数。结果图中的亮点表示可能的匹配。

模板匹配在需要在一个大图中找到特定模式的精确或非常接近精确的副本时特别有用，假设模式在大小或方向上没有太大变化。它是一种基本方法，有助于说明在图像中定位物体的基本挑战。

## 参考资料

- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book) — Richard Szeliski (2022)
  Publisher: Springer; DOI: [10.1007/978-3-030-80209-9](https://doi.org/10.1007/978-3-030-80209-9)
  一本综合性教材，提供计算机视觉的知识，其中有专门一章介绍目标检测，包括模板匹配。
- [Template Matching](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html) — OpenCV (2023)
  OpenCV库中模板匹配实际应用的官方文档，详细介绍了各种比较方法。
- [Computer Vision: A Modern Approach](https://books.google.com/books?id=QJ8d-Qf41_QC&printsec=frontcover&dq=%22Computer+Vision:+A+Modern+Approach%22+Prentice+Hall+David+A.+Forsyth+and+Jean+Ponce&hl=en&newbks=1&newbks_redir=0&sa=X&ved=2ahUKEwjG3I2G1_SDAxWqIUQIHRXfA9AQ6AF6BAgMEAI) — David A. Forsyth, Jean Ponce (2002)
  Publisher: Prentice Hall
  一本经典教材，介绍了计算机视觉的算法和数学基础，提供了关于基于相关性模式识别的见解。
