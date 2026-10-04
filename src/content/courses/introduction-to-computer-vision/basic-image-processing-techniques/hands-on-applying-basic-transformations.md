---
course: "introduction-to-computer-vision"
chapter: "basic-image-processing-techniques"
lesson: "hands-on-applying-basic-transformations"
sourceId: 1264
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-3-basic-image-processing-techniques/hands-on-applying-basic-transformations"
title: "动手实践：应用基本图像变换"
description: "练习对图像进行亮度、对比度、缩放、旋转和滤镜处理。"
order: 9
plots: []
sourceHash: "c68808de92db4ef15918d1694f9203a6d7628a7f305ff0eae60ac274ac95e735"
sourceCorrections: []
---

这里提供了关于实现基础图像处理方法的实用指导，这些方法包括调整亮度与对比度、进行几何变换以及图像平滑处理，将使用 Python 和 OpenCV 库进行。我们假设您已按照第 1 章的说明，搭建好 Python 环境并安装了 OpenCV。

首先，请确保您已导入所需的库，并准备好一张示例图片进行操作。如果您手头没有现成的示例图片，可以很方便地从网上找到（搜索“免版税图片”）或使用标准的测试图片，例如“Lena”或“Peppers”图片，它们通常在计算机视觉的数据集中可以找到，或方便下载。将您选择的图片保存到与 Python 脚本或 Jupyter Notebook 相同的目录下。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 以彩色模式加载图片
# 将 'your_image.jpg' 替换为您的图片文件路径
image_path = 'your_image.jpg'
image = cv2.imread(image_path)

# 检查图片是否成功加载
if image is None:
    print(f"Error: Could not load image from {image_path}")
    # 您可能希望在此处退出或引发错误
else:
    # OpenCV 默认以 BGR 格式加载图片。
    # Matplotlib 需要 RGB 格式。我们为了显示目的进行转换。
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 Matplotlib 显示原始图片
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title('原始图片')
    plt.axis('off') # 隐藏坐标轴刻度
    plt.show()
```

请务必将 `'your_image.jpg'` 替换为您图片文件的实际路径。运行此代码应能显示您的原始图片。如果遇到错误，请仔细检查文件路径，并确保 OpenCV 已正确安装。

### 调整亮度与对比度

请记住，亮度通过给每个像素添加一个值来调整，而对比度通过乘法来调整。我们来试试看。

**亮度调整：** 我们将通过向每个像素添加一个常数值（例如 50）来增加亮度。我们需要注意不要超过最大像素值（255）。OpenCV 使用 `cv2.add` 等函数会自动处理这一点。

```python
# --- 亮度调整 ---
# 创建一个与图片大小相同的矩阵，用亮度值填充
brightness_value = 50
brightness_matrix = np.ones(image.shape, dtype=image.dtype) * brightness_value

# 将亮度矩阵添加到原始图片
# cv2.add 执行饱和运算（将值裁剪到 0 和 255 之间）
brighter_image = cv2.add(image, brightness_matrix)

# 将 BGR 转换为 RGB 以供显示
brighter_image_rgb = cv2.cvtColor(brighter_image, cv2.COLOR_BGR2RGB)

# 显示增亮后的图片
plt.figure(figsize=(6, 6))
plt.imshow(brighter_image_rgb)
plt.title(f'亮度增加 {brightness_value}')
plt.axis('off')
plt.show()
```

**对比度调整：** 我们将通过将像素值乘以一个系数（例如 1.5）来增加对比度。同样，`cv2.multiply` 有助于处理潜在的溢出。

```python
# --- 对比度调整 ---
contrast_factor = 1.5
# 注意：cv2.multiply 在饱和处理方面类似
# 对于对比度，如有必要请确保系数为浮点型
contrast_image = cv2.multiply(image, np.array([contrast_factor]))

# 如果不使用特定的 cv2 函数，可能需要手动裁剪
# 或者确保乘法运算不会导致意外的类型更改
# 一种常用方法是 convertScaleAbs，它进行缩放、计算绝对值，然后转换回 8 位
contrast_image_scaled = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

# 将 BGR 转换为 RGB 以供显示
contrast_image_rgb = cv2.cvtColor(contrast_image_scaled, cv2.COLOR_BGR2RGB)

# 显示对比度调整后的图片
plt.figure(figsize=(6, 6))
plt.imshow(contrast_image_rgb)
plt.title(f'对比度增加 (系数: {contrast_factor})')
plt.axis('off')
plt.show()
```

尝试使用不同的 `brightness_value`（正值增加，负值减少）和 `contrast_factor`（大于 1 增加对比度，小于 1 减小对比度）值，以查看它们的效果。

### 几何变换

现在我们来对图片进行缩放、旋转和平移操作。

**缩放（调整大小）：** 我们可以使用 `cv2.resize` 来使图片变大或变小。

```python
# --- 缩放（调整大小） ---
# 获取原始尺寸
height, width = image.shape[:2]

# 定义新尺寸（例如，原始大小的一半）
new_width = int(width * 0.5)
new_height = int(height * 0.5)
new_dimensions = (new_width, new_height)

# 调整图片大小 - cv2.INTER_LINEAR 是一种常用插值方法
resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

# 将 BGR 转换为 RGB 以供显示
resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# 显示调整大小后的图片
plt.figure(figsize=(4, 4)) # 如有需要，调整图像大小
plt.imshow(resized_image_rgb)
plt.title(f'调整大小为 {new_width}x{new_height}')
plt.axis('off')
plt.show()
```

尝试不同的缩放因子和插值方法，例如 `cv2.INTER_CUBIC`（放大时通常效果更好）或 `cv2.INTER_AREA`（缩小时通常效果更好）。

**旋转：** 要旋转图片，我们首先使用 `cv2.getRotationMatrix2D` 计算旋转矩阵，然后使用 `cv2.warpAffine` 应用它。

```python
# --- 旋转 ---
# 获取图片中心
(cX, cY) = (width // 2, height // 2)

# 旋转角度（单位：度，正值表示逆时针）
angle = 45
# 旋转时的缩放因子（1.0 表示大小不变）
scale = 1.0

# 计算旋转矩阵
M_rotate = cv2.getRotationMatrix2D((cX, cY), angle, scale)

# 执行旋转
rotated_image = cv2.warpAffine(image, M_rotate, (width, height))

# 将 BGR 转换为 RGB 以供显示
rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

# 显示旋转后的图片
plt.figure(figsize=(6, 6))
plt.imshow(rotated_image_rgb)
plt.title(f'旋转 {angle} 度')
plt.axis('off')
plt.show()
```

**平移（偏移）：** 平移也使用 `cv2.warpAffine`，但需要一个不同的变换矩阵。我们定义沿 X 轴和 Y 轴移动的量。

```python
# --- 平移 ---
# 定义偏移量 (tx: 水平方向, ty: 垂直方向)
tx = 50  # 向右平移 50 像素
ty = 25  # 向下平移 25 像素

# 创建平移矩阵 M
# [[1, 0, tx],
#  [0, 1, ty]]
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])

# 应用平移
# (width, height) 定义输出图片大小
translated_image = cv2.warpAffine(image, M_translate, (width, height))

# 将 BGR 转换为 RGB 以供显示
translated_image_rgb = cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB)

# 显示平移后的图片
plt.figure(figsize=(6, 6))
plt.imshow(translated_image_rgb)
plt.title(f'平移 ({tx}, {ty})')
plt.axis('off')
plt.show()
```

请注意，图片的部分可能会移出画面，而新露出的区域会填充黑色。

### 基本图像平滑（滤波）

最后，我们来应用基本的平滑滤镜来减少噪点。

**均值模糊：** 此滤镜将每个像素替换为其在定义好的内核大小内的邻域像素的平均值。

```python
# --- 均值模糊 ---
# 定义内核大小（例如，5x5）
# 内核尺寸越大，模糊效果越明显
kernel_size_avg = (5, 5)
average_blurred = cv2.blur(image, kernel_size_avg)

# 将 BGR 转换为 RGB 以供显示
average_blurred_rgb = cv2.cvtColor(average_blurred, cv2.COLOR_BGR2RGB)

# 显示均值模糊后的图片
plt.figure(figsize=(6, 6))
plt.imshow(average_blurred_rgb)
plt.title(f'均值模糊 ({kernel_size_avg[0]}x{kernel_size_avg[1]} 内核)')
plt.axis('off')
plt.show()
```

**高斯模糊：** 这使用高斯内核，赋予中心像素更大的权重 (weight)，通常会产生更自然的模糊效果。

```python
# --- 高斯模糊 ---
# 定义内核大小（必须为正奇数）
kernel_size_gauss = (5, 5)
# SigmaX（X 方向标准差）。如果为 0，则根据内核大小计算。
sigmaX = 0
gaussian_blurred = cv2.GaussianBlur(image, kernel_size_gauss, sigmaX)

# 将 BGR 转换为 RGB 以供显示
gaussian_blurred_rgb = cv2.cvtColor(gaussian_blurred, cv2.COLOR_BGR2RGB)

# 显示高斯模糊后的图片
plt.figure(figsize=(6, 6))
plt.imshow(gaussian_blurred_rgb)
plt.title(f'高斯模糊 ({kernel_size_gauss[0]}x{kernel_size_gauss[1]} 内核)')
plt.axis('off')
plt.show()
```

比较均值滤镜和高斯滤镜的效果。高斯模糊通常更适合降噪，因为它在相同平滑量下，比简单的均值模糊更能稍微保持边缘。

### 总结

在本实操环节中，您学习了如何使用 OpenCV 实际应用基础图像处理方法：

- 使用简单的像素算术运算（`cv2.add`，`cv2.multiply`）调整了亮度与对比度。
- 进行了几何变换：缩放（`cv2.resize`）、旋转（`cv2.getRotationMatrix2D`，`cv2.warpAffine`）和平移（使用平移矩阵的 `cv2.warpAffine`）。
- 应用了基本的平滑滤镜：均值模糊（`cv2.blur`）和高斯模糊（`cv2.GaussianBlur`）。

这些操作构成了许多计算机视觉应用的基本组成部分。它们常作为预处理步骤，用于增强图片、使其外观标准化，或为特征提取和分析做准备，我们将在下一章开始讨论这些内容。请持续尝试不同的参数 (parameter)和图片，以巩固您的理解。

## 参考资料

- [OpenCV: Learning OpenCV 4](https://docs.opencv.org/4.x/) — OpenCV community (2025)
  OpenCV 库的官方文档，提供了图像处理和计算机视觉中所有功能的详细 API 参考、教程和示例，包括用于亮度、对比度、几何变换和滤波的功能。
- [Digital Image Processing](https://www.pearson.com/us/higher-education/product/Gonzalez-Digital-Image-Processing-4th-Edition/9780133356724.html) — Rafael C. Gonzalez and Richard E. Woods (2017)
  Publisher: Pearson
  一本基础教科书，涵盖了数字图像处理的理论和算法基础，包括对强度变换、几何操作和空间滤波的深入解释。
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer
  一本权威教科书，全面概述了计算机视觉，其中有关于图像形成、滤波和几何变换的专门章节，为图像处理技术提供了更广阔的背景。
