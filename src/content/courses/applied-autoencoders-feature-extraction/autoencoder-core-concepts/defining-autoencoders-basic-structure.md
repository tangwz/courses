---
course: "applied-autoencoders-feature-extraction"
chapter: "autoencoder-core-concepts"
lesson: "defining-autoencoders-basic-structure"
sourceId: 6445
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-2-autoencoder-core-concepts/defining-autoencoders-basic-structure"
title: "自编码器的定义：基本构造"
description: "对自编码器、其目的及其基本前馈神经网络构造的介绍。"
order: 1
plots: []
sourceHash: "ddd8463386759d139fd434367deeb6ca28852b88f1d726a75da8854e940e4f6e"
sourceCorrections: []
---

本质上，自编码器是一种用于无监督学习 (supervised learning) (unsupervised learning)的人工神经网络 (neural network)。它的主要目的听起来表面简单：学习如何将其输入复制到其输出。也就是说，如果你输入数据 $x$，它会尝试生成 $\hat{x}$，使 $\hat{x}$ 尽可能接近 $x$。虽然这可能显得微不足道，但自编码器的有效性在于网络架构上的限制，特别是存在一个“瓶颈”，迫使其学习输入的压缩表示。

自编码器的基本构造通常是对称的，包含三个主要部分，我们将在后续章节中更具体地介绍这些部分：

1. **编码器**：网络此部分接收原始高维输入，并将其变成低维内部表示。它学习将输入数据“编码”成压缩形式。
2. **瓶颈层（或编码/潜在空间）**：这是自编码器的中间层，存储着输入的压缩编码表示。此层的维度是一个重要的超参数 (parameter) (hyperparameter)，因为它确定了压缩的程度。我们常用于特征提取的正是这种压缩表示。
3. **解码器**：网络此部分接收瓶颈层的压缩表示，并尝试从中重建原始输入。它学习将压缩形式“解码”回原始数据空间。

这种构造促使自编码器学习数据中重建所必需的最主要特征。设想一下，你尝试将一篇长文档总结成几句重要的话（编码过程），然后请另一个人仅根据你的总结来重构原始文档（解码过程）。要做好这项工作，你的总结必须包含最主要的信息。

下面是说明此基本架构的图表：

> 自编码器通过编码器处理输入数据，将其送至压缩的瓶颈层，然后使用解码器重建数据。目的是使重建数据与原始输入尽可能地接近。

我们简要说明每个部分的作用：

- **输入层**：原始数据（例如，被展平为向量 (vector)的图像，或表格数据集中的一行）被输入到网络的地方。
- **编码器网络**：编码器由一个或多个神经网络层组成，其作用是将输入数据 $x$ 映射到瓶颈层中的隐藏表示 $z$。编码器中的每个后续层通常比前一层具有更少的神经元，从而逐级降低维度。因此，$z = \text{编码器}(x)$。
- **瓶颈层（潜在空间）**：这一层，记作 $z$，表示自编码器学到的关于输入的压缩知识。如果瓶颈维度 $d$ 小于输入维度 $D$，自编码器就会被迫学习一种有效的压缩表示。这些学到的表示通常可用作其他机器学习 (machine learning)任务的特征。
- **解码器网络**：与编码器对称地，解码器接收潜在表示 $z$，并将其映射回输出层中的重建 $\hat{x}$。解码器中的每个后续层通常比前一层具有更多的神经元，将表示扩展回原始输入的维度。因此，$\hat{x} = \text{解码器}(z)$。
- **输出层**：这一层产生重建数据 $\hat{x}$。输出层中的神经元数量必须与输入层中的维度数量相符。此处使用的激活函数 (activation function)取决于输入数据的类型（例如，像素值在0到1之间时使用sigmoid，连续值时使用线性激活）。

整个自编码器是一个前馈神经网络，通过最小化**重建损失函数 (loss function)**进行训练。此损失函数衡量原始输入 $x$ 与重建输出 $\hat{x}$ 之间的差异。例如，如果你的输入是数值向量，均方误差（MSE）是一个常用选项。如果你的输入是二进制（如黑白图像），二进制交叉熵可能更适合。我们将在本章后面部分更具体地讨论损失函数。

这种基本构造的精巧之处在于其能够无须明确标签即可学习有用的表示，使其成为无监督特征学习的一个有益工具。通过训练网络使其通过受限的瓶颈来重现输入，我们促使其找到数据中潜在的模式和结构。这些瓶颈层中包含的学到模式，随后可以被提取并用作各种后续任务的特征，例如分类、聚类或异常检测。这种基本架构是构建更高级自编码器变体（我们也将介绍）的根本。

## 参考资料

- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  这篇开创性论文引入了深度自编码器用于有效的无监督预训练，对神经网络的发展产生了重要影响。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 14: Autoencoders
  这本权威教材全面介绍了自编码器，详细阐述了其结构、训练以及在深度学习背景下的各种形式。
- [CS230: Deep Learning - Lecture 12: Autoencoders and Word Embeddings](http://cs230.stanford.edu/files_winter_2018/lectures/Lecture12_Autoencoders_WordEmbeddings_v5.pdf) — Andrew Ng, Kian Katanforoosh, and the Stanford CS230 Staff (2018)
  斯坦福大学深度学习课程讲义提供了关于自编码器基础知识的清晰简洁的学术概述，包括其基本架构。
