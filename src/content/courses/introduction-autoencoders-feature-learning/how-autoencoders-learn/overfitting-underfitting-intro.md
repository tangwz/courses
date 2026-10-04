---
course: "introduction-autoencoders-feature-learning"
chapter: "how-autoencoders-learn"
lesson: "overfitting-underfitting-intro"
sourceId: 6419
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn/overfitting-underfitting-intro"
title: "过拟合与欠拟合初识"
description: "简要介绍训练自编码器时过拟合与欠拟合的原理。"
order: 7
plots: ["plots/6419-0.json", "plots/6419-1.json", "plots/6419-2.json"]
sourceHash: "f53b5c7ae4a1d11a96fbe7056c8770c68c565227fddc904e67875c17c53eb020"
sourceCorrections: []
---

在训练自编码器时，如前所述，我们的主要目的就是让重建输出尽可能接近原始输入。我们通过损失函数 (loss function)来衡量这一点，并试图将其减小。但就像一个通过背诵特定问答来备考的学生一样，模型有时会对训练数据学习*过度*，或学习不足。这在机器学习 (machine learning)中会引出两种常见情形：过拟合 (overfitting)与欠拟合 (underfitting)。让我们简单看一下这些在自编码器中的含义。

### 了解欠拟合 (underfitting)：学习不足

设想你正在学画一只猫，但你只看了一张非常模糊的图片，或者只花了很少时间研究。之后你画的猫可能就不太像了。这类似于**欠拟合**。

当自编码器过于简单，无法捕获数据中的重要模式时，就会发生欠拟合。它未能学到数据的基础结构，因此不仅在新数据上表现差，在训练过的数据上表现也差。两种情况下的重建误差都会很高。

什么可能导致自编码器欠拟合呢？

- **瓶颈层太小：** 如果瓶颈层限制性过强，试图将数据压缩到极小的空间，可能会丢失太多信息。这就像试图用一个词总结一整本书。
- **模型过于简单：** 编码器或解码器可能没有足够的层或神经元来学习数据的复杂性。
- **训练不足：** 如果你没有对模型进行足够的训练轮次，它就根本没有足够的机会调整权重 (weight)并充分学习。

欠拟合的自编码器不擅长重建输入，而且它在瓶颈层学到的特征可能过于普通或不完整，无法派上用场。



![欠拟合：所有数据上误差均高](plots/6419-0.json)



> 在欠拟合情形下，训练误差和验证误差都保持高位，表明模型没有很好地学习数据。

### 了解过拟合 (overfitting)：学习过多细节

现在，设想相反情况。你正在准备那场考试，但你没有理解主题，而是记住课本例子中的每一个标点符号。你可能在那些使用完全相同例子的测试中取得高分，但如果问题略有不同，你就会很吃力。这就是**过拟合**。

当自编码器对训练数据学习得过于具体，包括该特定数据集中的任何噪声或随机波动时，就会发生过拟合。它变得如此适应训练例子，以至于无法泛化到新的、未见过的数据。因此，你会看到训练数据上的重建误差非常低，但当你用它之前未遇到的数据进行测试时，误差会高得多。

为什么会发生过拟合？

- **模型过于复杂：** 如果自编码器的层数或神经元数量相对于训练数据的数量或复杂性过多，它就有能力记忆而非学习一般模式。
- **瓶颈层限制不足：** 如果瓶颈层没有强制进行足够的压缩，模型可能只是直接通过数据，而未学习到有意义的紧凑表示。
- **训练时间过长：** 给予足够时间，一个复杂的模型会开始记住训练数据中的噪声。

过拟合的自编码器可能会给你训练图像的美丽重建，但它提取的特征可能对那些特定图像过于定制化，对于更广泛的用途不太有用。



![过拟合：训练表现佳，新数据表现差](plots/6419-1.json)



> 过拟合的特点是训练误差减小，但在某个点之后验证误差开始增加。

### 目标：良好拟合

我们的目标是**良好拟合**。这是指自编码器能够从训练数据中很好地学到真实的基础模式，从而泛化到新数据。它不只是记忆训练集，也不会过于简单而无法捕获重要信息。训练误差和未见过数据上的误差（通常称为验证误差或测试误差）都较低并彼此接近。



![良好拟合：均衡学习](plots/6419-2.json)



> 良好拟合的模型表明训练误差和验证误差都收敛到低值。

### 这对自编码器为何重要？

即使自编码器通常以无监督方式训练（只是尝试重建输入），瓶颈层中学到的表示的质量也具有重要作用。

- 如果你的自编码器**欠拟合 (underfitting)**，瓶颈表示将过于粗糙，无法捕获太多关于你数据的有用信息。
- 如果你的自编码器**过拟合 (overfitting)**，瓶颈表示可能对训练数据的噪声和特质过于具体。它不会是一种好的通用特征表示，无法用于其他任务或新数据。

因此，在减小重建误差的同时，我们还需要注意模型是在学习可泛化的特征还是仅仅记忆。

这只是对这些内容的一个快速介绍。有各种方法可以帮助避免过拟合并确保你的模型泛化良好，例如调整模型复杂度、使用正则化 (regularization)方法，或在适当时间停止训练（一种称为早期停止的方法）。我们现在不会详细说明这些，但当你开始使用自编码器及其他机器学习 (machine learning)模型时，了解这些常见问题是有益的。当你准备构建你的第一个自编码器时，记住这些可能的问题将帮助你更好地理解训练过程并评估模型表现。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本基础性教材提供了深度学习的全面理论背景，包含对过拟合、欠拟合以及训练自编码器等神经网络至关重要的泛化概念的详细解释。
- [CS230: Deep Learning - Lecture Notes on Bias/Variance and Regularization](http://cs230.stanford.edu/files_fall_2018/lectures/lecture4_fall2018.pdf) — Andrew Ng (2018)
  Publisher: DeepLearning.AI
  斯坦福大学深度学习课程的讲义，为神经网络中的欠拟合和过拟合提供了学术介绍，以及解决它们的初步策略。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  一本使用 Keras 和 TensorFlow 进行深度学习的实用指南，提供了对过拟合和欠拟合等常见训练问题的实际解释，以及缓解这些问题的实用方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本关于统计学习的经典教材，对模型拟合、偏差-方差权衡和泛化误差进行了严谨处理，这些是过拟合和欠拟合的基础。
