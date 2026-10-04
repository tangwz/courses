---
course: "introduction-to-computer-vision"
chapter: "introduction-object-recognition"
lesson: "overview-machine-learning-approaches"
sourceId: 1298
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-5-introduction-object-recognition/overview-machine-learning-approaches"
title: "机器学习方法概述"
description: "高级说明机器学习常用于更可靠的对象识别（无具体细节）。"
order: 6
plots: []
sourceHash: "676977424c65b9e5287baec10493d7a011bc1833c4add1e58b003c7c6595734a"
sourceCorrections: []
---

如我们所见，模板匹配提供了一种直接的方式，可以在较大图像中找到图案的精确副本。然而，当寻找的对象外观发生变化时，其有效性会迅速降低。尺度（大小）、旋转（方向）、光照条件、视角乃至部分遮挡等变化常使简单的模板匹配失效。对象识别需要更加灵活且能容忍这些变化的方式。

机器学习 (machine learning)（ML）为对象识别提供了处理对象外观变化的有效方法。机器学习方式不需明确编程计算机来寻找固定模式，而是使计算机能够从示例中*学习*对象的外观。

可以这样理解：我们不是给计算机一张特定猫咪的照片（模板），并让它找到完全相同的匹配项，而是向它展示成千上万张不同姿态、不同光照条件下的猫咪图片，并简单地将它们都标记 (token)为“猫”。机器学习算法随后分析这些例子，找出猫咪的共同特征，即其本质的“猫性”，这些特征能够区分猫和狗、汽车或背景画面。

### 从数据中学习

将机器学习 (machine learning)用于对象识别的主要思想包含两个主要阶段：

1. **训练：** 在此阶段，机器学习模型会使用大量标注图像数据集进行“训练”。例如，一个用于识别汽车的数据集可能包含数千张图片，每张图片都清楚地标注了其中汽车的位置和类型。模型会处理这些数据并调整其内部参数 (parameter)，以学习与汽车相关的视觉模式。它会学习哪些特征（边缘、纹理、颜色、形状的组合）在统计学上对识别汽车很重要，即使汽车很小、很大、从侧面看到、部分隐藏，或在明亮阳光下或阴影中。
2. **推理 (inference)（或预测）：** 模型训练完成后，即可用于分析新的、未曾见过的图像。当接收到新图像时，经过训练的模型会应用所学模式来预测其“看到”的对象是否与任何受训类别匹配（例如：“这张图片中有汽车吗？如果有，在哪里？”）。

### 相比简单方法的优势

机器学习 (machine learning)模型，特别是像**深度学习 (deep learning)**（它使用称为神经网络 (neural network)的复杂结构）这样的现代方式，在对象识别方面已证明非常有效。它们相比模板匹配等方式的主要优势包括：

- **对变化的适应性：** 它们通常能识别对象，即使在尺度、旋转、光照、视角以及细微变形或部分遮挡发生显著变化时也是如此。
- **泛化能力：** 一个在大量“汽车”样本上训练过的模型通常能识别出在训练期间从未见过的特定汽车模型，因为它已经掌握了汽车的一般视觉属性。
- **处理复杂性：** 它们可以同时学习区分许多不同的对象类别。

### 这对计算机视觉的意义

尽管模板匹配是目标外观高度一致的特定任务中一个有价值的工具，但机器学习 (machine learning)是当今通用对象识别的主流方式。构建和训练这些模型需要大量数据、强大的计算能力，以及对底层算法更深的理解。

这些机器学习模型如何设计、训练和部署的细节，是更高级学习的主题。然而，理解这个基本思想——计算机可以从数据中学习识别对象，而不是仅仅依赖于僵化、预设的模式——是你在继续学习计算机视觉时的一个重要收获。这些学习型方式支持着你日常生活中可能遇到的许多应用，从照片标签建议到自动驾驶汽车感知系统。

## 参考资料

- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/) — Richard Szeliski (2022)
  Publisher: Springer
  一本涵盖传统计算机视觉技术和现代机器学习物体识别方法的综合性资源。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本阐述深度学习、神经网络及其应用原理的奠基性教科书，这些内容是现代物体识别的核心。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738; DOI: [10.1007/b100721](https://doi.org/10.1007/b100721)
  一本经典教科书，全面介绍机器学习原理，包括从数据中训练和模式识别，这些是物体识别的基础。
