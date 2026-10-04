---
course: "intro-synthetic-data-ml"
chapter: "tools-libraries-overview"
lesson: "libraries-simple-image-manipulation"
sourceId: 5712
sourceUrl: "https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-6-tools-libraries-overview/libraries-simple-image-manipulation"
title: "简单图像处理库 (Pillow, Scikit-image)"
description: "提及Pillow或Scikit-image等库，用于基础图像生成和增强任务。"
order: 4
plots: []
sourceHash: "ac608f50bf3eb9577584c333d5d048bc7a3b18e92b0a8af0c24e8530aba24da1"
sourceCorrections: []
---

虽然生成高度逼真的合成图像通常需要复杂的技法，例如3D渲染或生成对抗网络 (GAN) (GANs)，但许多基础的合成图像任务和增强操作可以使用标准的Python图像处理库完成。这些工具是创建简单数据集或对现有数据进行增强的优秀起点。Pillow和Scikit-image是两个常用的选择。

### Pillow: 基础图像处理的首选

Pillow是原始Python图像库 (PIL) 的一个友好分支，广泛用于打开、处理和保存多种图像文件格式。对于合成数据生成，Pillow特别适用于：

- **创建空白图像：** 您可以轻松生成特定尺寸和背景颜色（例如，白色画布）的图像作为基础。
- **绘制简单图形：** Pillow的`ImageDraw`模块允许您通过编程方式绘制基本的几何图形，如矩形、椭圆、线条和多边形，或在图像上添加文本。这对于为对象检测或分类任务创建简单数据集很有帮助（例如，只包含正方形或圆形的图像）。
- **基本变换：** 应用简单的几何变换，如旋转、调整大小、裁剪或翻转图像。尽管这些技术常用于对真实图像进行数据*增强*，但它们也可以生成您所创建的简单合成图像的不同版本。
- **颜色处理：** 调整亮度、对比度，或在颜色模式之间转换（例如，RGB到灰度）。
- **图像粘贴：** 通过将一张图像粘贴到另一张图像上进行组合，这对于创建合成场景或将合成对象放置到背景中很有用。

可以将Pillow看作是Python的数字画布和画笔工具集。它允许您使用基本组件从头开始构建图像，或通过直接操作修改现有图像。

```python
# 示例：使用Pillow创建简单图像（说明性）
from PIL import Image, ImageDraw

# 创建一个200x100的白色画布
img = Image.new('RGB', (200, 100), color = 'white')
d = ImageDraw.Draw(img)

# 绘制一个蓝色矩形
d.rectangle([(20, 20), (80, 80)], fill='blue', outline='black')

# 绘制红色文本
d.text((100, 50), "简单合成", fill='red')

# img.save('simple_synthetic_image.png') # 可以保存图像
```

### Scikit-image: 借助于NumPy进行图像处理

Scikit-image是科学Python生态系统的一部分（与NumPy、SciPy和Matplotlib并列），并提供了一系列图像处理算法。它的主要优势在于将图像视为NumPy数组，这与其他科学库能够良好配合。对于合成数据，Scikit-image适用于：

- **生成噪声：** 您可以轻松地向图像添加各种类型的噪声（例如，高斯噪声、椒盐噪声、泊松噪声）。添加噪声可以使合成图像看起来稍微更真实，或测试机器学习 (machine learning)模型的鲁棒性。
- **应用滤镜：** 使用滤镜进行模糊处理（例如，高斯模糊）、边缘检测或其他效果。这些可以模拟不同的相机条件或处理伪影。
- **生成图形和纹理：** 虽然Pillow也能绘制图形，但Scikit-image提供了直接将图形甚至程序化纹理作为NumPy数组生成的函数。
- **形态学操作：** 应用膨胀或腐蚀等操作，这可以修改图像中对象的形状。
- **分割和特征提取：** 尽管这些工具常用于分析，但有时可以被重新用于*创建*合成掩码或特定图像特征。

当您需要在像素数据本身上执行更多算法操作时，Scikit-image表现出色，它借助于NumPy在数值计算方面的优势。

```python
# 示例：使用Scikit-image添加噪声（说明性）
# 假设 'image_array' 是一个表示图像的NumPy数组
# （可以通过skimage.io.imread加载或用Pillow/NumPy创建）

# from skimage.util import random_noise
# import numpy as np

# 应用高斯噪声
# noisy_image_array = random_noise(image_array, mode='gaussian', var=0.01)

# 如有需要，转换回标准像素范围
# noisy_image_array = np.clip(noisy_image_array * 255, 0, 255).astype(np.uint8)

# 之后可以使用skimage.io.imsave或Pillow保存这个 noisy_image_array
```

### 工具组合

通常，您可能会将这些库结合使用。例如，您可以使用Pillow创建带有简单图形的基础图像，然后使用Scikit-image添加特定类型的噪声或应用模糊滤镜。

> 一个结合Pillow和Scikit-image进行基础合成图像生成的工作流程。

这些库为编程图像创建和处理提供了易于使用的入口。它们是生成本课程前面讨论过的基础合成图像和增强操作的基础工具，为更高级的生成技术奠定了基础。

## 参考资料

- [Pillow Documentation](https://pillow.readthedocs.io/en/stable/) — Pillow Contributors (2024)
  Pillow库的官方文档，提供Python中基础图像操作的全面指南和API参考。
- [scikit-image Documentation](https://scikit-image.org/docs/stable/) — The scikit-image team (2025)
  scikit-image的官方文档，提供基于NumPy数组的图像处理算法的详细信息。
- [scikit-image: image processing in Python](https://peerj.com/articles/453/) — Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D. Warner, Nicolae Y. Galli, Andrea S. Giugliano, Timothy D. Nathaniel, Olav Alex Eklund, Alexander R. Sofroniew, Maarten Becker, Louis Bonnett, Harald Möller, Eric Ben Shushan, Christian von Faber, Catherine R. Herhold, and Tony Yu (2014)
  Journal: PeerJ; Volume: 2; Pages: e453; DOI: [10.7717/peerj.453](https://doi.org/10.7717/peerj.453)
  介绍scikit-image库的原始学术论文，阐述了其在Python中进行图像处理的架构和功能。
- [Hands-On Image Processing with Python](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGyGDforXrjoIvbM0JFGsE2kfw_nlmrt7tsOuuYWjrYvdhOQDrUe74dq1VU3edR779F6ti2eYQNrR2LaT1IkJyKZlxeTVptkpe9XUTiLgFpA43KZU1U25uKAVdjMgaCx99tkFQ2vc_0QEGJSOkilvO3PK8BBbfJToeK7ZhQHOh4032SsGFzugPoPkVfyfAas_Ec2Mu5xw==) — Sandipan Dey (2018)
  Publisher: Packt Publishing; Pages: 492
  一本指导读者使用Python进行图像处理的实践书籍，包含Pillow和scikit-image等库的使用示例。
