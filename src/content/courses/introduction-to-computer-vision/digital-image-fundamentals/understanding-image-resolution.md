---
course: "introduction-to-computer-vision"
chapter: "digital-image-fundamentals"
lesson: "understanding-image-resolution"
sourceId: 1241
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-2-digital-image-fundamentals/understanding-image-resolution"
title: "理解图像分辨率"
description: "了解图像分辨率、尺寸和宽高比。"
order: 5
plots: []
sourceHash: "789d2e6fed724311116804a3ec47f8fabca36291b7f1ed0167a142496a1e622c"
sourceCorrections: []
---

既然我们了解了数字图像本质上是像素的集合，接下来我们来讨论如何衡量图像的尺寸和细节。这使我们开始讨论**图像分辨率**。

### 什么是图像分辨率？

图像分辨率指数字图像中包含的像素总数。它通常表示为图像宽度上的像素数乘以其高度上的像素数。

将图像想象成由微小方块（每个方块就是一个像素）组成的网格或马赛克。分辨率表示这个网格的尺寸。例如，一张分辨率为640x480的图像，其宽度为640像素，高度为480像素。


$$
\text{总像素数} = \text{宽度} \times \text{高度}
$$


因此，对于我们640x480的例子：


$$
\text{总像素数} = 640 \times 480 = 307,200 \text{ 像素}
$$


这意味着图像由超过300,000个独立像素组成，每个都带有其自身的颜色或强度值。

> 低分辨率图像网格（宽4像素，高3像素）的简化示意图。

### 分辨率与图像细节

图像的分辨率直接关系到其能捕捉和显示的细节量。更高分辨率的图像（像素更多，例如1920x1080）比低分辨率的图像（例如320x240）使用更精细的网格。

想象一下，尝试表示一个细节丰富的场景，如人脸或文字，只用少量大块马赛克瓷砖与使用大量小块瓷砖的对比。用更多、更小的瓷砖（更高分辨率）制作的图像可以表示更精细的细节和更平滑的过渡，看起来更清晰锐利。反之，像素较少（分辨率较低）的图像在近距离查看或放大时会显得有方块感或“像素化”，因为每个独立像素覆盖了原始场景中更大的区域。

- **更高分辨率：** 更多像素，可能更多细节，文件尺寸更大。
- **更低分辨率：** 更少像素，更少细节，文件尺寸更小。

需要注意的是，分辨率仅定义了像素的*数量*。实际感知到的质量还取决于以下因素：用于捕获图像的镜头质量、传感器的能力、压缩伪影（将在文件格式部分讨论）以及观看条件。然而，分辨率设定了图像文件所能容纳细节的上限。

### 宽高比

与分辨率紧密相关的是**宽高比**，它描述了图像宽度与高度之间的比例关系。它通常表示为比例，如4:3或16:9。

您可以通过用宽度除以高度并进行约分来计算宽高比。

- 一个 640x480 的图像，其宽高比为 $640 / 480 = 4 / 3$，通常写作 4:3。
- 一个 1920x1080 的图像，其宽高比为 $1920 / 1080 = 16 / 9$，通常写作 16:9（高清视频常用）。
- 一个 1080x1080 的图像，其宽高比为 $1080 / 1080 = 1 / 1$，即 1:1（正方形图像）。

宽高比决定了图像的形状，与其实际像素尺寸无关。

理解分辨率是很重要的，因为它告诉我们正在处理数据的空间尺寸。当我们之后讨论图像处理操作，例如缩放或使用坐标识别像素位置时，图像的分辨率（以像素表示的宽度和高度）将是一个不变的参考点。

## 参考资料

- [Digital Image Processing](https://www.pearson.com/us/higher-education/program/Gonzalez-Digital-Image-Processing-4th-edition/PGM220708.html) — Rafael C. Gonzalez, Richard E. Woods (2018)
  Publisher: Pearson; Pages: Chapter 2: Digital Image Fundamentals
  这本基础教材广泛涵盖数字图像处理的数学和算法基础，包括像素、采样、量化和图像分辨率等概念。
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer; Pages: Chapter 2: Image Formation
  计算机视觉领域的权威教材，介绍图像形成的核心概念，包括数字图像的表示方式，这有助于理解分辨率。
- [Computer Vision: A Modern Approach](https://www.amazon.com/Computer-Vision-Modern-Approach-2nd/dp/013608592X) — David Forsyth and Jean Ponce (2015)
  Publisher: Pearson; Pages: Chapter 2: Image Formation and Representation
  提供计算机视觉的全面介绍，详细阐述图像采集、数字表示的原理以及分辨率在图像质量中的作用。
