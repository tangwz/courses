---
course: "introduction-to-computer-vision"
chapter: "getting-started-computer-vision"
lesson: "brief-history-overview"
sourceId: 1226
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-1-getting-started-computer-vision/brief-history-overview"
title: "历史简述"
description: "简要介绍计算机视觉发展历程中的重要节点。"
order: 3
plots: []
sourceHash: "cf86283585e4ba0354921cddd770aa363a32b2c33bfc95374304b4ce2f7234b6"
sourceCorrections: []
---

了解计算机视觉的起源，有助于我们认识到它的巨大进步以及当前技术所依赖的根本。尽管机器“看”的想法听起来很未来化，但它的起源可追溯到几十年前。

### 早期萌芽与宏伟目标 (20世纪60年代)

计算机视觉在20世纪60年代正式成型，受到人工智能热潮的推动。1966年的麻省理工学院夏季视觉项目是一个主要时刻。研究人员乐观地期望在一个夏天内构建一个能分析场景并识别其中物体的系统。尽管这个目标被证明过于宏大，但它标志着结构化研究的开端。早期工作通常集中在高度受限的环境中，比如解释堆叠积木的图像（即“积木环境”）。这简化了问题，使得拉里·罗伯茨等先驱能够开发出从2D图像中寻找边缘和理解基本3D形状的初步算法。

### 转向理解过程本身 (20世纪70年代 - 80年代)

早期方法的局限性促使研究人员更深入地思考视觉本身的*过程*。神经科学家兼心理学家大卫·马尔在20世纪70年代末提出了一个有影响力的框架。他提出视觉处理分阶段进行：

1. **原始草图 (Primal Sketch):** 检测图像中的基本元素，如边缘、条形和斑点。
2. **2.5D草图 (2.5D Sketch):** 推断相对于观察者的表面方向和深度。
3. **3D模型 (3D Model):** 构建与视角无关的完整三维物体表示。

马尔的观点强调了理解视觉输入几何和结构的重要性，引导研究采用更具原则性的方法从图像中提取有意义的信息。这个时期也更加注重开发检测图像特征（如边缘和角点）的技术，这些内容我们将在本课程后续部分进行介绍。

> 一个简化的时间线，勾勒出计算机视觉发展中的主要时期和转变。

### 特征检测与统计学习 (20世纪90年代 - 21世纪初)

随着计算能力的提升，出现了更复杂的算法。大量精力投入到开发对视角、光照和尺度变化不那么敏感的特征检测器上。像20世纪90年代后期开发的尺度不变特征变换（SIFT）等算法，使计算机能够在不同条件下找到同一物体或场景不同图像间的对应点。

这一时期也出现了应用于视觉问题的机器学习 (machine learning)技术。研究人员不再仅仅依赖于手工设计的规则，而是开始在大规模数据集上训练系统。一个重要例子是Viola-Jones人脸检测框架（2001年），它使得消费级相机上的人脸实时检测成为可能，并成为首批广泛部署的计算机视觉应用之一。这显示了直接从数据中学习模式的潜力。

### 深度学习 (deep learning)时代 (2010年代 - 至今)

最显著的转变发生在2012年左右，伴随着深度学习，特别是卷积神经网络 (neural network) (CNN)（CNNs）的出现。一个名为AlexNet的系统在ImageNet大规模视觉识别挑战赛（ILSVRC）中取得了突破性表现，这是一项针对图像分类和物体检测的年度比赛。

深度学习如此高效的原因在于它能够直接从原始像素数据中自动学习分层特征。工程师不再需要设计复杂的特征提取器，网络在训练过程中会为任务学习出最佳特征。这使得几乎所有计算机视觉任务都取得了快速进展，从以显著准确度分类图像，到识别和分割复杂场景中的多个物体。如今，深度学习是大多数高性能计算机视觉系统的主流方法。

### 我们身处何处

计算机视觉已从受限环境中的宏大实验，发展成为一项融入现代生活无数方面的技术，包括智能手机相机、医学图像分析、机器人技术和自动导航系统。尽管仍存在挑战，特别是在理解语境、场景推理 (inference)以及确保公平性和鲁棒性方面，但该领域仍在快速发展。

这份简短的历史回顾为我们理解将要介绍的基本原理奠定了根基。您将学到的技术，从基本的图像处理到特征检测，都是贯穿这段历史而形成的重要组成部分，对于理解当今计算机如何处理和解读视觉信息仍然具有重要意义。

## 参考资料

- [Rapid Object Detection using a Boosted Cascade of Simple Features](https://ieeexplore.ieee.org/document/990517) — Paul Viola and Michael Jones (2001)
  Journal: Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2001); Publisher: IEEE; Volume: 1; Pages: I-511 - I-518; DOI: [10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)
  这篇论文介绍了 Viola-Jones 框架，一个实时目标检测系统，特别是面部检测，极大地推动了计算机视觉在消费应用中的普及。
- [Distinctive Image Features from Scale-Invariant Keypoints](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) — David G. Lowe (2004)
  Journal: International Journal of Computer Vision; Publisher: Springer; Volume: 60; Pages: 91-110; DOI: [10.1023/B:VISI.0000029664.99615.94](https://doi.org/10.1023/B%3AVISI.0000029664.99615.94)
  这篇论文介绍了尺度不变特征变换（SIFT）算法，这是一种检测和描述图像局部特征的方法，广泛应用于物体识别和图像匹配。
- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105
  这项工作介绍了 AlexNet，一个深度卷积神经网络，在 ImageNet LSVRC-2012 竞赛中取得了显著突破，标志着计算机视觉深度学习时代的开始。
