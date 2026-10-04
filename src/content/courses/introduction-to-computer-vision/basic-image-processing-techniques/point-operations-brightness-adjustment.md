---
course: "introduction-to-computer-vision"
chapter: "basic-image-processing-techniques"
lesson: "point-operations-brightness-adjustment"
sourceId: 1249
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-3-basic-image-processing-techniques/point-operations-brightness-adjustment"
title: "点操作：亮度调整"
description: "了解如何使用点操作修改图像的整体亮度。"
order: 1
plots: ["plots/1249-0.json"]
sourceHash: "ee55dc3d774fb79f0c862e580c5c62a4ff36c98941c40adbb97bacbc1e1517f7"
sourceCorrections: []
---

让我们从一种最直接的图像处理方法开始：改变其整体亮度。这属于**点操作**的范畴，这意味着我们独立于相邻像素，修改每个像素的值。可以把它想象成调整电视机的亮度旋钮；图像的每个部分都会同时变亮或变暗。

亮度调整涉及给图像中的每个像素加上或减去一个常数值。如果图像由每个坐标 $(x, y)$ 处的像素强度值 $P(x, y)$ 表示，亮度调整操作可以定义为：

$P_{out}(x, y) = P_{in}(x, y) + b$

这里：

- $P_{in}(x, y)$ 是原始图像中位置 $(x, y)$ 处像素的强度值。
- $P_{out}(x, y)$ 是输出（亮度调整后）图像中相同位置像素的强度值。
- $b$ 是我们用来控制亮度的常数值。

如果 $b$ 为正，像素值增加，使图像看起来更亮。如果 $b$ 为负，像素值减小，使图像看起来更暗。如果 $b$ 为零，图像保持不变。

### 彩色图像的应用

对于灰度图像，每个像素只有一个强度值，上述公式可以直接应用。对于彩色图像，例如常见的 RGB（红、绿、蓝）色彩空间中的图像，亮度调整通常独立应用于每个颜色通道：

$R_{out} = R_{in} + b$
$G_{out} = G_{in} + b$
$B_{out} = B_{in} + b$

给所有通道添加相同的值 $b$ 会增加或减少整体亮度，而不会显著改变颜色本身。

### 处理像素值限制：饱和

数字图像通常在固定范围内存储像素值。例如，8位灰度图像存储从 0（黑色）到 255（白色）的值。如果加上 $b$ 后像素值超出此范围会发生什么？

- 如果 $P_{in}(x, y) + b > 255$，结果值无法原样存储。它通常被**裁剪**或**饱和**到最大值 255。
- 如果 $P_{in}(x, y) + b < 0$，该值被裁剪到最小值 0。

这种裁剪可以防止值溢出或引起错误，但这也表示如果调整值 $b$ 过大，原始图像中非常亮或非常暗区域的细节可能会丢失。例如，如果你大幅增加亮度，多个不同的深灰色值在输出中可能会都变成 0（黑色），从而失去它们之间微小的差异。同样，不同的浅灰色值可能会都变成 255（白色）。

因此，考虑饱和运算的有效操作是：

$P_{out}(x, y) = \text{饱和处理}(P_{in}(x, y) + b)$

其中 $\text{饱和处理}$ 确保结果保持在有效 [0, 255] 范围内（或图像类型的适当范围）。

### 效果可视化

想象图像中像素强度的分布，通常使用直方图进行可视化（我们将在下文介绍）。添加一个正值 $b$ 会有效地将整个分布向右移动（朝着更高的强度值）。添加一个负值 $b$ 会将其向左移动。



![亮度调整对强度分布的影响](plots/1249-0.json)



> 亮度增加 ($b > 0$) 或减少 ($b < 0$) 时像素强度分布的示例移动。请注意整个分布是如何移动的。

### 实际实现

大多数图像处理库提供高效的函数来执行亮度调整，自动处理像素遍历和饱和运算。例如，在OpenCV（一个常用的计算机视觉库）中，你可以使用向图像数组的每个元素添加标量值的函数。

让我们考虑一个使用 Python 和 OpenCV 的例子：

```python
import cv2
import numpy as np

# 加载图像（假设已加载到 'image' 变量中）
# image = cv2.imread('你的图片.png') 

# 定义亮度调整值
brightness_value = 50 # 正值表示更亮，负值表示更暗

# 创建一个与图像大小和类型相同的矩阵，并用亮度值填充
# 注意：我们需要处理加法可能导致的数据类型问题
if brightness_value >= 0:
    # 使用 cv2.add 进行高效的饱和运算
    # 创建一个填充亮度值的矩阵
    brightness_matrix = np.ones(image.shape, dtype=image.dtype) * brightness_value
    bright_image = cv2.add(image, brightness_matrix)
else:
    # 使用 cv2.subtract 进行高效的饱和运算
    # 创建一个填充亮度绝对值的矩阵
    brightness_matrix = np.ones(image.shape, dtype=image.dtype) * abs(brightness_value)
    bright_image = cv2.subtract(image, brightness_matrix)

# 'bright_image' 现在包含亮度调整后的图像

# 你通常会显示或保存 'bright_image'
# cv2.imshow('Original', image)
# cv2.imshow('Brightened', bright_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
```

这个简单的操作是更复杂调整的依据。通过理解添加常数值如何影响每个像素以及饱和处理的重要性，你就有了一个用于基本图像增强的基本工具。

## 参考资料

- [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer
  一本全面的计算机视觉书籍，提供了基本图像处理操作及其在视觉系统中的背景知识。
- [cv.add (Adds two arrays or an array and a scalar)](https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga6a7ce3162544ad640a2336098256338e) — OpenCV contributors (2024)
  Publisher: OpenCV
  详细说明cv.add函数的官方文档，该函数执行带饱和的数组加法，与在OpenCV中实现亮度调整相关。
