---
course: "introduction-to-deep-learning"
chapter: "intro-specialized-architectures"
lesson: "feedforward-network-limitations"
sourceId: 5071
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/feedforward-network-limitations"
title: "前馈网络的局限性"
description: "讨论标准前馈网络在图像和序列数据等方面的不足。"
order: 1
plots: []
sourceHash: "0a2392555f5845ce33159666b7d77bfcaa75b6a5c731fce9cda834a9ea545775"
sourceCorrections: []
---

前馈神经网络 (neural network)，也称为多层感知机（MLP），能够从数据中学习复杂的非线性关系。它们的结构，通常包含全连接层，每个神经元都连接到前一层的所有神经元，使得它们在输入是固定大小的特征向量 (vector)时，成为强大的函数逼近器。

然而，正是这种结构引入了局限性，当处理具有固有结构、难以用简单向量表示的特定类型数据时。下面我们来看两个标准前馈网络表现不佳的重要方面：即处理图像等网格结构数据，以及处理文本或时间序列等序列数据。

### 处理图像的空间结构

图像不仅仅是像素的随意集合；它们具有很强的空间结构。相互靠近的像素通常是相关的，共同形成纹理、边缘和物体。当你将图像输入标准 MLP 时，通常的第一步是将图像矩阵（例如，高度 x 宽度 x 通道数）展平为一个单一的长向量 (vector)。

> 2D 图像在输入 MLP 之前通常会被展平为 1D 向量。请注意，原本垂直相邻的像素（例如，1 和 4）在向量中现在是分离的。

这种展平过程丢弃了明确的 2D 空间排列。图像中紧密相邻的像素（例如，一个在另一个正上方）在输入向量中可能会相距很远。MLP 在第一层中对这个向量的每个元素进行一定程度的独立处理，这使得它难以有效地学习基于局部空间模式的特征。

此外，考虑连接方式。在全连接层中，每个神经元都连接到*每个*输入像素。即使对于中等大小的图像（例如，256x256 像素），这也会导致第一隐藏层中参数 (parameter)（权重 (weight)）数量庞大。这种参数数量的急剧增加带来了几个问题：

1. **计算成本**：训练变得非常缓慢且占用大量内存。
2. **过拟合 (overfitting)**：参数过多，模型很容易记忆训练数据，而不是学习可泛化的特征。
3. **忽视局部性**：网络本身不知道相邻像素比相距较远的像素之间更相关。它必须从头开始学习这一点，效率很低。

另一个挑战是**平移不变性**。一个物体（比如一只猫），无论它出现在图像的左上角还是右下角，都应该仍然能被识别为猫。MLP 缺乏内置的平移不变性。因为每个输入像素都由连接它的权重独特处理，网络实质上需要针对猫可能出现的每个位置单独学习检测其特征。这效率极低。

### 处理序列顺序和可变长度

自然语言文本、语音音频或金融时间序列等数据本质上是序列化的。元素的顺序非常重要。例如，“狗咬人”与“人咬狗”的含义截然不同。

标准 MLP 在处理序列时遇到两个主要困难：

1. **固定大小的输入**：MLP 要求输入是预定义、固定大小的向量 (vector)。然而，序列的长度自然是变化的。句子可以有少许单词或许多单词，时间序列可以覆盖不同的持续时间，音频剪辑也各不相同。如何将一个 10 个词的句子和一个 100 个词的句子放入同一个固定大小的输入层？常见的解决方法，例如填充较短的序列或截断较长的序列，可以起作用，但通常会引入噪声或丢弃可能相关的信息。
2. **缺乏记忆/状态**：MLP 在前向传播过程中“一次性”处理其整个输入向量。它没有内在机制来记住序列早期部分的信息，同时处理后期部分。当处理“狗咬人”中的“咬”字时，一个简单的 MLP 本身不会保留“狗”在此之前出现的上下文 (context)。虽然你可以将整个序列（填充/截断后）作为一个大向量输入，但网络难以学习长距离依赖关系。当每个输入元素通过独立权重 (weight)处理，并且没有机制来维持状态或跨时间步共享信息时，长句子中相距较远的词之间的关系很难被捕获。

与图像情况类似，MLP 也缺乏**跨时间步的参数 (parameter)共享**。如果某个特定模式或特征无论其在序列中出现*何处*都很重要（例如，检测特定短语），那么 MLP 需要不同的权重来检测该模式在序列的开头、中间或结尾处。这效率很低，并且需要更多数据才能有效学习。

这些局限性表明需要专门设计的网络架构来处理空间层次（如图像中）和时间依赖（如序列中）。卷积神经网络 (neural network)（CNN）解决了空间数据的挑战，而循环神经网络（RNN）则专为序列数据设计。接下来我们将阐述这些架构的原理。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的教材，全面介绍了前馈网络、其局限性，以及卷积网络和循环网络提供的架构方案。
- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  一篇介绍卷积神经网络（LeNet-5）的开创性论文，详细阐述了其在图像识别中通过利用局部相关性和平移不变性所带来的架构优势，从而克服了全连接层的局限性。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇基础性论文介绍了长短期记忆（LSTM）网络，这是一种能够高效地使循环网络学习和记忆序列数据中长期依赖关系的架构，解决了更简单的前馈或循环模型的主要缺点。
- [Convolutional Neural Networks for Visual Recognition (CS231n)](https://cs231n.github.io/) — Stanford University (2024)
  Publisher: Stanford University
  官方课程讲义，清晰地解释了卷积神经网络，分析了为何前馈网络不足以处理图像数据以及CNN如何通过架构设计应对这些挑战。（2023年春季版）
