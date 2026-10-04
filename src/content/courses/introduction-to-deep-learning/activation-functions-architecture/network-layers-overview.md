---
course: "introduction-to-deep-learning"
chapter: "activation-functions-architecture"
lesson: "network-layers-overview"
sourceId: 5031
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/network-layers-overview"
title: "理解网络层：输入层、隐藏层、输出层"
description: "定义全连接神经网络中输入层、隐藏层和输出层的作用。"
order: 7
plots: []
sourceHash: "344952ff809c278dffb2f8358f81e36b7afe0e8ae414ff2546329d1a797b47ce"
sourceCorrections: []
---

多层感知机（MLP）使用堆叠的层来学习复杂模式。为了理解网络架构，有必要分析这些层各自的作用。在典型的全连接神经网络 (neural network)中，信息沿一个方向流动，依次经过不同类型的层：输入层、一个或多个隐藏层，以及输出层。了解每层的功能对于设计和理解网络行为很重要。

### 输入层：数据入口

输入层是网络的入口。其主要作用是接收原始输入数据，这些数据通常表示为特征向量 (vector)。输入层中的每个神经元（或节点）通常对应数据集中一个单独的特征。

- **结构：** 输入层神经元的数量由输入数据的维度决定。例如，如果你处理的是每样本有10个特征的表格数据，输入层将有10个神经元。如果你处理的是28x28像素的扁平化图像，输入层将有$28 \times 28 = 784$个神经元。
- **作用：** 输入层不执行任何传统意义上的计算（这里不应用加权和或激活函数 (activation function)）。它只是作为通道，将输入样本的特征值传递给第一个隐藏层。可以把它想象成将初始信息分配到网络中。
- **数据表示：** 重要的是，输入到此层的数据是数值化的，并且通常经过预处理（例如，缩放或归一化 (normalization)），因为后续层将对这些值进行数学运算。

### 隐藏层：计算核心

在输入层和输出层之间是隐藏层。它们是神经网络 (neural network)的“主力”，实际的学习和特征提取发生在此处。它们被称为“隐藏层”，因为其输出是中间表示，并不直接对应任务的最终输出。

- **结构：** 一个网络可以有一个或多个隐藏层。拥有多个隐藏层的网络被认为是“深层”的。每个隐藏层中的神经元数量是一个设计选择，一个影响模型学习能力的超参数 (parameter) (hyperparameter)。
- **作用：** 隐藏层中的每个神经元接收来自上一层（或输入层）所有神经元的输入。它计算这些输入的加权和，加上一个偏置 (bias)项，然后应用一个非线性激活函数 (activation function)（如本章前面讨论的Sigmoid、Tanh或ReLU）。这种非线性属性很重要；如果没有它，堆叠多个层将数学上等同于单个层，从而限制了网络对数据中复杂非线性关系进行建模的能力。
  $a^{(l)}_j = f\left( \sum_k w^{(l)}_{jk} a^{(l-1)}_k + b^{(l)}_j \right)$
  这里，$a^{(l)}_j$是第 $l$ 层中第 $j$ 个神经元的激活值，$f$是激活函数，$w^{(l)}_{jk}$是连接第 $l-1$ 层中第 $k$ 个神经元到第 $l$ 层中第 $j$ 个神经元的权重 (weight)，$a^{(l-1)}_k$是前一层（$l-1$）中第 $k$ 个神经元的激活值，$b^{(l)}_j$是第 $l$ 层中第 $j$ 个神经元的偏置。
- **特征层级：** 在深层网络中，较早的隐藏层通常学习简单特征（如图像中的边缘或纹理），而较后面的隐藏层则结合这些简单特征来学习更复杂和抽象的表示（如形状、物体或概念）。

### 输出层：给出结果

网络的最后一层是输出层。其目的是根据隐藏层处理过的信息，产生网络的预测结果。

- **结构：** 输出层神经元的数量直接取决于网络设计的具体任务：
  - **二分类：** 通常使用一个带有Sigmoid激活函数 (activation function)的输出神经元。输出是一个介于0和1之间的概率值，表示属于正类的可能性。
  - **多分类：** 使用 $N$ 个输出神经元，$N$ 是不同类别的数量。Softmax激活函数通常应用于这些神经元。Softmax确保输出是总和为1的概率，表示类别的概率分布。
  - **回归：** 通常有一个带有线性激活函数（或没有激活函数）的输出神经元。这使得网络能够输出一个连续的数值。对于多目标回归，将使用多个带有线性激活函数的输出神经元（每个目标值一个）。
- **作用：** 输出层中的神经元执行与隐藏层神经元相同的计算（加权和加偏置 (bias)，然后是激活函数）。然而，这里激活函数的选择很重要，因为它决定了最终输出格式以符合问题要求（例如，分类的概率，回归的连续值）。

### 结构示意

以下图示为一个简单的全连接神经网络 (neural network)，带有一个输入层、两个隐藏层和一个输出层。信息从左向右流动。

> 全连接神经网络示意图，展示了信息从输入层流出，经过中间隐藏层，进行计算和非线性变换，到产生最终预测的输出层。虚线表示相邻层之间的完全连接（为清晰起见，未单独绘制所有连接）。

总之，输入层接收数据，隐藏层执行非线性变换并学习特征表示，输出层以适合任务的格式产生最终预测。这些层的设计，包括层的数量、每层神经元的数量以及使用的激活函数 (activation function)，构成了网络的架构，我们将在下一节中继续讨论。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本书是深度学习的综合参考资料，其中第6章专门介绍了前馈深度网络及其层结构，包括输入层、隐藏层和输出层以及激活函数。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer
  这本基础教材为神经网络提供了坚实的数学基础，详细介绍了多层感知器、其架构以及不同层的作用。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture 3: Neural Networks and Backpropagation](https://cs231n.github.io/neural-networks-1/) — Andrej Karpathy, Fei-Fei Li, Justin Johnson, Serena Yeung, et al. (2023)
  Journal: Stanford University CS231n Course
  这份在线讲义提供了对神经网络的入门介绍，详细阐述了前馈网络中输入层、隐藏层和输出层的结构与功能。
