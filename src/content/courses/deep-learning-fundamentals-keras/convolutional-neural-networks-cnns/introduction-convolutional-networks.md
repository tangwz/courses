---
course: "deep-learning-fundamentals-keras"
chapter: "convolutional-neural-networks-cnns"
lesson: "introduction-convolutional-networks"
sourceId: 4932
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/introduction-convolutional-networks"
title: "卷积网络简介"
description: "阐述卷积神经网络背后的原因及其对图像等网格状数据的适用性。"
order: 1
plots: []
sourceHash: "f60c6271713d9db9567e052922d323745d1a18f67654a6f5d19e8e1578862b97"
sourceCorrections: []
---

标准神经网络 (neural network)，特别是那些使用`Dense`层构建的网络，处理输入数据时，`Dense`层中的每个神经元都与前一层中的*每个*神经元相连。尽管这种设计对于输入特征相对独立或其顺序没有固有重要性（如表格数据）的许多任务都有效，但当处理具有自身空间排列的数据，尤其是图像时，这种方法存在明显的局限性。

设想将一幅2D图像展平为一个长向量 (vector)，以输入到`Dense`层。这个过程会丢失像素之间重要的空间联系。原本相邻、构成边缘或纹理的像素，现在在输入向量中被视为遥远、不相关的特征。网络失去了某些像素在水平、垂直或对角线上相邻的信息。因此，标准网络难以学习依赖于这些局部结构识别特征。

此外，思考参数 (parameter)的快速增加。一幅相对较小的256x256灰度图像有65,536个像素。如果第一个隐藏层只有512个神经元，输入层与此层之间的连接将需要$65,536 \times 512 \approx 33.5$百万个权重 (weight)，再加上偏置 (bias)！对于更大的图像或彩色图像（具有多个通道），这个数量会大幅增加。训练这样的网络在计算上会很昂贵，并且由于参数数量庞大而容易过拟合 (overfitting)。

另一个困难是平移变异性。如果我们训练一个标准网络来识别图像左上角的物体（例如，一只猫），那么当同样的猫出现在右下角时，它不会自动识别出来。因为展平后的输入向量看起来完全不同，网络需要为新位置再次学习猫的模式。这不高效；我们希望我们的网络能够识别模式，无论它们出现在图像中的哪个位置。

### 卷积神经网络 (neural network) (CNN)的方案：局部模式与参数 (parameter)共享

卷积神经网络（CNN）专门用于处理这些难题，它们结合了两种受生物视觉系统信息处理方式启发的有效方法：

1. **局部连接（局部感受野）：** 卷积神经网络不将每个输入像素连接到第一个隐藏层中的每个神经元，而是使用特殊层（卷积层），其中神经元只对输入图像的一个小的局部区域做出响应。这个区域被称为神经元的*感受野*。这种局部关注使得网络能够直接从附近像素的空间位置学习小的低级模式，例如边缘、角点和纹理。
2. **参数共享：** 一组权重 (weight)，形成一个“滤波器”或“核”，被应用于输入图像中的*所有*位置。这个滤波器在输入上滑动，在每个位置执行相同的操作以检测特定模式（例如，垂直边缘）。如果一个模式在图像的某个部分有用，它在其他地方也可能有用。与全连接层相比，这显著减少了参数的数量（我们只学习滤波器的权重，而不是每个输入-神经元连接的权重），并且内建了*平移不变性*，即无论模式在图像中的哪个位置，都能识别它的能力。

这些核心思想使卷积神经网络能够构建特征的层次结构。早期层识别简单模式（边缘、颜色），随后的层将这些模式组合起来以识别更复杂的模式（纹理、形状），而更深层则将这些模式组合起来以识别物体或物体的一部分。这种分层学习过程与我们理解视觉信息的方式相似，并且非常适用于涉及网格状数据结构的任务。

### 适用于网格状数据

卷积神经网络 (neural network) (CNN)的有效性源于它们能够发挥网格状数据中存在的空间局部性。图像是典型的例子，它们是像素的2D网格。然而，同样的原则也适用于：

- **1D 网格：** 时间序列数据，其中附近的时间步长通常相互影响。文本序列在经过合适的嵌入 (embedding)后也可以被视为1D网格。
- **3D 网格：** 体积数据，例如医学扫描（CT、MRI），或视频数据（其中时间是第三个维度，与空间的高度和宽度一起）。

通过局部连接和参数 (parameter)共享来保留并分析空间关系，卷积神经网络提供了一种比标准全连接网络更高效且有用的架构，用于从这些类型的结构化数据中学习。在接下来的章节中，我们将查看实现这些基本的卷积神经网络思想的特定Keras层，例如`Conv2D`和`MaxPooling2D`。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  一篇开创性的论文，介绍了卷积神经网络（LeNet-5），详细阐述了局部感受野和权重共享等用于模式识别的核心概念。
- [Deep Learning](https://www.deeplearningbook.org/contents/convnets.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 9: Convolutional Networks
  这本权威教材为卷积网络提供了全面的理论基础，解释了其架构和局部连接、参数共享等原则。
- [Convolutional Neural Networks for Visual Recognition (CS231n) Lecture Notes](http://cs231n.stanford.edu/2023/index.html) — Fei-Fei Li, Yunzhu Li, Ruohan Gao (2023)
  Journal: Stanford University CS231n Course; Publisher: Stanford University
  详细的讲义和解释，涵盖了用于图像处理任务的卷积神经网络的基本概念和动机。
- [Keras documentation: \`tf.keras.layers.Conv2D\`](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHyrFRECrAhZ5_sy4efixe0OWN7Jlxk6g7vERVYbz6wX_GBAi0EjwD_R-HCsHlQNh6qIsYo3fDI33Zng78Klnf5Zrsx4afUZrNZNMFGMlxD7rmgfy_aROJ08j3_3snBVqcrLVmtj-Fm0_JYLhSU07JgnckndshwkvxQeR8DDfYybhjEG-J9WED1DflUDjMPKVVEL8MLPyGmWbyJkRdb5SF-7ecAh5Pm1R5zH7ft1-KaQQ==) — Keras team (part of TensorFlow documentation) (2024)
  Publisher: TensorFlow
  Keras `Conv2D` 层的官方文档，提供了其参数和用于构建卷积网络的实践见解。
