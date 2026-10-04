---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoder-anatomy-encoder-decoder"
lesson: "matching-input-output"
sourceId: 6412
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder/matching-input-output"
title: "输入与输出的匹配"
description: "了解自编码器中输入数据与重建输出之间的关系。"
order: 10
plots: []
sourceHash: "224fbe682476a54c32194762a289c5e790e0e90348a04092f348f59641d178b0"
sourceCorrections: []
---

我们已经了解了编码器如何工作来压缩数据，以及解码器如何试图重建数据。现在，我们来把这两部分联系起来，考察自编码器的初始输入和最终输出之间的关系。这对于弄清自编码器实际学到了什么非常重要。

数据通过标准自编码器的整个过程如下：

1. 原始输入数据，我们称之为 $X$，被送入**编码器**。
2. 编码器将 $X$ 映射到**瓶颈**层中的一个压缩的、低维表示，我们通常用 $z$ 表示它。
3. 这个压缩表示 $z$ 随后被送入**解码器**。
4. 解码器试图从 $z$ 重建原始输入数据，产生输出 $X'$（有时写作 $\hat{X}$）。

自编码器的主要目标是使其重建输出 $X'$ 尽可能与原始输入 $X$ 相似。简单来说，我们希望 $X' \approx X$。

### 为什么是“重建”而不是完美复制？

你可能会疑惑，如果目标是让 $X'$ 像 $X$，为什么不直接做个完美复制呢？答案在于**瓶颈**。因为瓶颈层的维度（或神经元）少于输入（和输出）层，自编码器被迫学习输入数据的压缩摘要。它不能简单地记住所有细节。相反，它必须学会优先处理并捕获数据中最重要的特征，以便能够合理地重建它。

这种压缩和随后的重建过程意味着 $X'$ 很少会与 $X$ 达到绝对完美的像素级（或值对值）匹配，特别是当瓶颈显著更小时。一些信息在压缩过程中不可避免地会丢失。然而，这是有意为之的。通过强制网络使用压缩表示进行工作，我们鼓励它学习有意义的底层模式，而不是仅仅进行简单的复制粘贴操作。

### 确保输入和输出维度对齐 (alignment)

一个重要的架构细节是，**输入层的维度必须与输出层的维度匹配**。如果你的输入数据是28x28像素的图像（展平后为784个值），那么你的输入层将有784个单元。相应地，产生重建图像 $X'$ 的输出层也必须有784个单元。自编码器学习：

1. 将784个输入维度减少到瓶颈层 $z$ 中较小的维度数。
2. 将表示从 $z$ 扩展回输出层 $X'$ 的原始784个维度。

### 输出层：根据输入数据进行调整

解码器的结构，特别是其最终输出层，旨在以与输入相同的格式生成数据。

- **形状**：如前所述，输出层中的神经元数量与输入中的特征（例如，像素、数据点）数量相匹配。
- **激活函数 (activation function)**：输出层激活函数的选择很重要，并且通常取决于输入数据的性质和范围。
  - 如果你的输入数据被归一化 (normalization)到特定范围，例如0到1之间（图像像素强度常见），那么输出层通常使用**Sigmoid激活函数**。Sigmoid函数方便地将其输出值压缩到(0, 1)范围，使网络更容易生成与此尺度匹配的重建结果。
  - 如果你的输入数据可以是任意实数值（正或负）并且没有限制在一个小范围内，那么输出层可能会使用**线性激活函数**（这意味着没有显式激活，或 $f(x)=x$）。
  - 对于二进制数据（0和1），Sigmoid也是输出层的常见选择，表示概率。

### “接近”到何种程度才算足够？相似度测量

自编码器通过尝试最小化原始输入 $X$ 和其重建输出 $X'$ 之间的差异，即**重建误差**来学习。可以把这个误差看作是 $X$ 和 $X'$ 之间“距离”的衡量。误差越小，重建效果越好。

有不同的方法来计算这个误差，它们被称为**损失函数 (loss function)**。例如：

- 如果你的数据是实数（如像素强度或传感器读数），一个常用的损失函数是**均方误差（MSE）**。它计算 $X$ 和 $X'$ 的每个元素之间平方差的平均值。
- 如果你的数据是二进制或表示概率（例如，像素为黑色或白色，缩放到0或1），则通常使用**二元交叉熵（BCE）**。

我们将在下一章详细讨论损失函数和训练过程。目前，需要理解的是，自编码器在训练期间调整其内部参数 (parameter)（权重 (weight)和偏差），以使这个重建误差尽可能小。

下图说明了数据流以及使输出与输入匹配的目标：

> 数据从输入 $X$ 流出，经编码器压缩为 $z$，然后由解码器重建为 $X'$。自编码器的训练旨在最小化 $X$ 和 $X'$ 之间的差异。

如果自编码器能够成功地将 $X'$ 重建得与 $X$ 非常相似，即使数据经过了受限的瓶颈 $z$，这也表明 $z$ 必定捕获了输入数据中最突出、有用和有区分度的特征。这种学习有效表示的能力使得自编码器在特征学习和降维等任务中很有用处，我们将在后续内容中继续考察。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 14: Autoencoders
  这本权威教材全面介绍了自编码器，详细阐述了其架构、瓶颈层的作用、重建目标以及常见的损失函数（如均方误差和交叉熵），是理解自编码器机制的基础。
- [Reducing the Dimensionality of Data with Neural Networks](http://www.ncbi.nlm.nih.gov/sites/entrez?db=pubmed&uid=16873662&cmd=showdetailview&indexed=google) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  这篇开创性论文介绍了深度自编码器在有效降维中的概念，强调学习一个压缩表示（瓶颈），从而可以准确重建输入，直接支持了输入与输出匹配的原则。
- [Lecture 4: Unsupervised Learning (Autoencoders)](http://introtodeeplearning.com/) — Alexander Amini, Ava Soleimany (2021)
  Journal: MIT 6.S191 Introduction to Deep Learning; Publisher: Massachusetts Institute of Technology (MIT)
  这些讲义对自编码器组件、重建过程、输出层激活函数（例如，用于[0,1]数据的Sigmoid）以及常见的损失函数（如均方误差和二元交叉熵）提供了清晰易懂的解释，对于初学者来说非常相关。
