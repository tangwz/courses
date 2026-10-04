---
course: "introduction-autoencoders-feature-learning"
chapter: "how-autoencoders-learn"
lesson: "forward-propagation-explained"
sourceId: 6416
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn/forward-propagation-explained"
title: "数据流：前向传播说明"
description: "了解数据在前向传播步骤中如何通过自编码器从输入流向输出。"
order: 4
plots: []
sourceHash: "d1cee77b751401b42a6795ef7346db1a7b5578bacc6e1a3b75b98eb033baeb27"
sourceCorrections: []
---

既然我们已经了解自编码器的主要工作是重建其输入，并且我们使用损失函数 (loss function)来衡量其表现，那么现在我们来看看数据是如何实际通过自编码器移动以产生该重建结果的。这种从输入到输出的单向数据流动被称为**前向传播**。

想象一下你正在发送一条消息 ($X$)，它先通过一系列的翻译器和摘要器处理，然后再通过扩展器和翻译器返回，目的是取回原始消息 ($X'$)。前向传播就是这种完整的发送和接收过程。

以下是数据流动的一个分步说明:

1. **进入输入层**:
   你的数据，无论是图像、一组数字还是传感器读数，首先进入**输入层**。这一层不进行任何计算；它只是将原始数据传递给自编码器的第一部分，即编码器。我们将输入数据称为 $X$。
2. **编码器的运作**:
   数据随后通过**编码器**。编码器通常由一个或多个“隐藏层”构成。每个层都包含处理单元（通常称为神经元）。

   - 当数据进入一个神经元时，它会被**权重 (weight)**相乘。可以把权重看作是控制每条输入信息强度或重要性的旋钮。
   - 然后添加一个**偏置 (bias)**项。偏置就像一个小而恒定的调整，有助于微调 (fine-tuning)神经元的输出。
   - 这些计算（输入加偏置的加权和）的结果随后通过一个**激活函数 (activation function)**。这个函数决定神经元的最终输出是什么，通常将值转换为特定范围或判断神经元是否应该“激活”。
     编码器的作用是压缩输入数据。因此，通常编码器中每个后续层将比前一个层拥有更少的神经元，逐步将信息压缩成更紧凑的形式。
3. **到达瓶颈层（潜在空间）**:
   在通过所有编码器层后，数据到达**瓶颈层**，也称为潜在空间。这一层在自编码器中拥有最少的神经元数量。这一层的输出是输入数据的压缩表示，通常表示为 $z$。这个 $z$ 是一个紧凑的概要，在较低维空间中捕捉了输入最重要的特点。
4. **通过解码器扩展**:
   压缩表示 $z$ 随后被输入到**解码器**中。解码器的结构通常是编码器的镜像。它接收来自瓶颈的紧凑信息，并开始将其扩展回原始数据的形状。

   - 类似于编码器，解码器有带有神经元、权重、偏置和激活函数的隐藏层。
   - 然而，在解码器中，层的大小通常会增加，逐步“解压”压缩后的信息。
5. **产生输出层**:
   最后，数据通过解码器的最后一层，称为**输出层**。这一层的输出是自编码器对原始输入数据的重建。我们可以将这种重建数据称为 $X'$。输出层中的神经元数量必须与原始输入层中的特点数量（例如，图像中的像素，数据集中的列）匹配。

从 $X$ 到 $X'$ 的整个过程，是一个“前向传递”或“前向传播”。自编码器接收输入，将其通过其层和计算网络传递，并产生一个输出。

> 该图显示了数据在前向传播过程中所经过的路径。它以输入 ($X$) 开始，通过编码器在瓶颈处压缩成潜在表示 ($z$)，然后解码器试图从这种表示中重建原始数据 ($X'$)。

通过这种前向传播生成的输出 $X'$，是我们随后与原始输入 $X$ 进行比较的对象。它们之间的差异，正如我们讨论损失函数时所说，告诉我们自编码器在重建任务中的表现如何。这种误差衡量随后被用于下一步——反向传播 (backpropagation)，来调整自编码器的权重和偏置，这将在后续内容中说明。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本基础教科书全面介绍了神经网络架构、前向传播、权重、偏置、激活函数和自编码器。
- [Convolutional Neural Networks for Visual Recognition, Lecture Notes: Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/) — Justin Johnson, Andrej Karpathy, and Fei-Fei Li (2023)
  Journal: Stanford University CS231n Course Materials; Publisher: Stanford University
  对神经网络基础知识（包括前向传播、层计算、权重、偏置和激活函数）进行了清晰的教学解释，对于理解数据流至关重要。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  这是一本易于理解的在线书籍，系统地介绍了神经网络的基本概念，包括前向传播、权重和偏置的机制，并提供了交互式示例。
