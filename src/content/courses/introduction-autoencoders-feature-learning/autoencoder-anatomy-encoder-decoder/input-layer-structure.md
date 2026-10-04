---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoder-anatomy-encoder-decoder"
lesson: "input-layer-structure"
sourceId: 6404
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder/input-layer-structure"
title: "输入层的结构"
description: "了解自编码器中输入层的作用和结构，它接收初始数据。"
order: 2
plots: []
sourceHash: "ba3ea6dbc5602d23d625eb39c4f3cbb93d367739496934cd7364f0994b8e84af"
sourceCorrections: []
---

输入层是自编码器首次接触它需要学习的数据的地方。可以把它看作是网络的入口。无论你希望自编码器理解、压缩并随后重建什么样的数据，它都必须先经过这一层。

输入层的“结构”简单但十分重要：其设计旨在精确匹配输入数据的格式。如果你的数据包含图像、传感器读数或电子表格中的行，输入层必须为该数据的每个特征设置一个对应的“位置”或神经元。

我们来看几个例子：

- **简单的数值数据：** 设想你有一个数据集，每个样本有三个特征，例如个体的身高、体重和年龄。你的输入层将有三个神经元：一个用于身高，一个用于体重，一个用于年龄。如果单个数据样本表示为向量 (vector) $X = [x_1, x_2, x_3]$，那么 $x_1$（身高）进入第一个输入神经元，$x_2$（体重）进入第二个，而 $x_3$（年龄）进入第三个。
- **图像数据：** 这是自编码器中一种非常常见的数据类型。如果你使用 MNIST 数据集中的灰度图像（一个手写数字的集合），每张图像宽 28 像素，高 28 像素。为了将其输入自编码器，我们通常会“展平”图像。这意味着我们将 2D 像素网格（28x28）转换为一个单独的长向量。因此，$28 \times 28 = 784$ 像素。在这种情况下，你的自编码器输入层将需要 784 个神经元，每个像素值对应一个。如果图像是彩色的（例如，使用红色、绿色和蓝色通道，或 RGB），你通常会有 $width \times height \times 3$ 个输入神经元。

输入层中的神经元数量是一个固定值，其大小直接取决于输入数据的维度，即数据的特征数量。如果输入数据包含 $N$ 个特征，那么输入层将包含 $N$ 个神经元。

值得注意的是，输入层本身不执行任何计算或转换，不像典型的带有学习权重 (weight)和偏差的神经网络 (neural network)层那样。它的主要作用是接收输入样本 $X$ 的各个特征，并将这些值传递给编码器的第一个隐藏层。这是编码过程的起点，自编码器从这里开始学习如何将数据压缩成更紧凑的表示。

这是一个简单图示，说明输入特征如何映射到输入层：

> 输入数据向量 $X$ 中的每个特征 $(x_1, x_2, ..., x_n)$ 被送入输入层中对应的神经元 $(I_1, I_2, ..., I_n)$。这些值随后被传递到编码器的后续层。

理解这种直接映射是理解自编码器如何开始处理信息的第一步。输入层充当网关，确保网络以结构化格式接收数据，为编码器更深层中发生的压缩和特征学习做好准备。从这里开始，数据流入编码器的隐藏层，我们将在接下来进行讨论。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本基础教材提供了关于神经网络的全面理论背景，包括各种架构中输入层的结构和功能。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  一本易于理解的在线书籍，系统地介绍了神经网络概念，解释了输入数据如何表示并馈入网络。
