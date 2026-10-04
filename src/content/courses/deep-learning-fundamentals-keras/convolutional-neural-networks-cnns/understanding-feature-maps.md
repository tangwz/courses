---
course: "deep-learning-fundamentals-keras"
chapter: "convolutional-neural-networks-cnns"
lesson: "understanding-feature-maps"
sourceId: 4951
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/understanding-feature-maps"
title: "了解特征图"
description: "通过可视化中间特征图来认识CNN学习了什么。"
order: 7
plots: []
sourceHash: "62f978260e89a8a4aa4a9a72298758b948ab55cf9b992f09213ef33f385b6b79"
sourceCorrections: []
---

卷积层在输入体（如图像）上应用滤波器以识别模式。但 `Conv2D` 层的输出究竟是什么？它不是单个处理过的图像。相反，它是一组**特征图**，也称作激活图。每个特征图对应于一个特定滤波器在整个输入上应用后的输出。

回顾 `Conv2D` 层内的滤波器（或核）。每个滤波器基本上都经过训练，以识别输入数据中的特定简单模式。例如，在CNN处理图像的早期层中，一个滤波器可能对垂直边缘敏感，另一个对水平边缘敏感，还有一个对某种颜色梯度敏感，再一个则对简单纹理敏感。

当一个滤波器滑过输入时，它会在其检测到模式存在的区域产生高激活值。对于单个滤波器而言，这些激活值形成的二维数组就是我们所说的**特征图**。它标示出输入中何处发现了特定特征（由该滤波器识别的）。

一个典型的 `Conv2D` 层使用多个滤波器（由Keras中的 `filters` 参数 (parameter)指定）。如果一个 `Conv2D` 层有，比如说，32个滤波器，它将输出32个不同的特征图。每个图都提供了对输入的不同“视角”或解释，侧重于其相应滤波器所学到的特定模式的存在与位置。

> 一个简化视图，显示卷积层中不同滤波器如何处理相同输入以生成不同的特征图，每个特征图都突出显示不同的已学模式。

这些特征图的集合构成了卷积层的输出体。如果输入是一个28x28的灰度图像（形状 `(28, 28, 1)`），并且 `Conv2D` 层使用了32个滤波器，且填充保留了维度，那么输出体的形状将是 `(28, 28, 32)`。第三个维度，32，代表输出的*深度*，对应于特征图的数量（以及滤波器的数量）。

### 分层特征学习

当数据通过CNN中连续的卷积层时，特征图倾向于表示越来越复杂的模式。

1. **早期层（靠近输入端）：** 滤波器通常学习基本特征，如边缘、角点、简单纹理和颜色斑块。由此产生的特征图突显这些基本元素。
2. **中间层：** 滤波器结合早期层的特征以识别更复杂的图案，例如简单形状（圆形、方形）、更精细的纹理，或物体部件（眼睛、车轮）。
3. **更深层（靠近输出端）：** 滤波器通过组合在中间层识别出的模式，学会识别更高级别的、可能是物体特有的特征，甚至整个物体。

这种分层学习过程，即复杂性层层递进，正是CNN能够有效分析图像等复杂输入的原因。

### 池化层的作用

池化层，如 `MaxPooling2D`，对每个特征图独立进行操作。它们的作用通常是降低每个特征图的空间维度（宽度和高度），同时保留最重要的信息（在最大池化的情况下是最大激活值）。这个过程使表示更为紧凑，并引入一定程度的平移不变性，这意味着网络对输入中特征的精确位置变得不那么敏感。虽然池化会减小空间尺寸，但它通常保留深度（特征图的数量）。

### 为什么要了解特征图？

掌握特征图益处良多，原因如下：

- **直觉理解：** 它有助于建立关于CNN如何通过逐步提取和组合模式来分析空间数据的直觉认识。
- **调试：** 如果模型学习效果不佳，可视化中间层的激活值（特征图）有时能提供线索。例如，如果特征图全部空白或看起来像随机噪声，它可能指示初始化、学习率或梯度消失等问题。
- **模型解释：** 可视化训练过的模型的特征图可以提供关于网络在不同阶段学习识别了哪些特定模式的认识，有助于理解其决策过程。（可视化技术通常涉及访问中间层的输出，Keras支持这一点。）

虽然在使用Keras API进行标准模型定义和训练时，通常不会直接与单个特征图交互，但了解它们的作用对于理解CNN如何处理信息以及从网格状数据中学习分层表示非常重要。它们是空间模式被识别、细化并沿网络传递的通道。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  解释了深度学习的理论基础，涵盖了卷积网络、滤波器和特征图，以及它们在跨层特征提取中的作用。
- [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems 25 (NIPS 2012); Publisher: Curran Associates, Inc., Red Hook, NY; Volume: 25; Pages: 1097-1105
  这篇有影响力的论文展示了深度卷积神经网络在图像分类中的应用，证明了通过多层学习到的特征。
- [tf.keras.layers.Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) — Keras team (2024)
  Publisher: TensorFlow team
  `tf.keras.layers.Conv2D`层的官方文档，描述了其参数、输入/输出形状，以及`filters`参数如何决定输出特征图的数量。
