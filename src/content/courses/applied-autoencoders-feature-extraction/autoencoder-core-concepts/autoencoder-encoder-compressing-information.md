---
course: "applied-autoencoders-feature-extraction"
chapter: "autoencoder-core-concepts"
lesson: "autoencoder-encoder-compressing-information"
sourceId: 6446
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-2-autoencoder-core-concepts/autoencoder-encoder-compressing-information"
title: "编码器：信息压缩"
description: "详细说明自编码器的编码器部分，该部分负责将输入数据映射到低维潜在空间。"
order: 2
plots: []
sourceHash: "3ee96b64a96968606c021d0558a3ed74405aaf4bcbd1c905fef074373b9093ae"
sourceCorrections: []
---

编码器是自编码器从数据中学习过程的首要阶段。它的主要职责是接收可能高维且复杂的输入数据，并将其转化为更紧凑、低维度的表示。可以将其视为一个信息精炼器，负责捕捉输入中最显著的特征，同时丢弃噪声或冗余。这种压缩表示通常被称为**潜在空间表示**或**编码**。

通常，编码器是使用一系列神经网络 (neural network)层构建的。对于处理扁平或向量 (vector)数据的标准自编码器（比如我们在第一章中讨论过的），这些层通常是全连接（或密集）层。编码器中这些层的显著特点是它们会逐步降低维度。例如，如果你的输入数据有784个特征，编码器中的第一个隐藏层可能有256个神经元，下一个128个，依此类推，直到编码器的最后一层（瓶颈层）输出所需的低维编码。

> 上图说明了一个典型的编码器结构，其中数据从高维输入流经神经元数量递减的层，最终形成低维的潜在表示。

编码器中的每一层都对其输入进行一次转换。这通常涉及一个线性操作（将输入乘以权重 (weight)矩阵并加上偏置 (bias)向量），接着是一个非线性激活函数 (activation function)。从数学上讲，对于编码器中的单层，给定输入 $x'$（它可以是原始输入 $x$，也可以是前一个编码器层的输出）的输出 $h$ 可以表示为：
$h = \sigma(W x' + b)$
这里，$W$ 是权重矩阵，$b$ 是偏置向量，$\sigma$ 是激活函数。编码器层中常用的激活函数包括：

- **ReLU (修正线性单元):** $f(x) = \max(0, x)$。它计算效率高，并有助于缓解梯度消失问题。
- **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$。它将值压缩到0和1之间，可能有用，但对于非常大或非常小的输入，可能会遇到梯度消失问题。
- **Tanh (双曲正切):** $f(x) = \tanh(x)$。它将值压缩到-1和1之间，通常比Sigmoid更受欢迎，因为其输出是零中心的。

非线性激活函数很重要；没有它们，一堆线性层就只会等同于一个单一的线性变换，从而严重限制了编码器可以学习的函数的复杂性。

编码器的“奥妙”在于它如何学习每层的权重 $W$ 和偏置 $b$。在自编码器的训练过程中，整个网络，包括编码器，都会被优化以最小化重建误差（如在章节介绍中讨论重建损失函数 (loss function)时所述）。这意味着编码器不仅仅是随机压缩数据。相反，它学习执行一种压缩，这种压缩会保留对解码器重建原始输入最为必要的信息。因此，编码器学习了一种有意义的方式，将输入 $x$ 映射到潜在表示 $z$：
$z = \text{编码器}(x)$
这个学得的潜在向量 $z$ 是输入的压缩本质，当我们研究瓶颈层以及如何提取和使用这些特征时，我们将进一步讨论这一点。这种压缩的质量十分重要，因为它直接影响原始数据重建的程度，以及（对于本课程而言更重要的是）这些学得的特征对于其他机器学习 (machine learning)任务的有用程度。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书提供了深度学习基础的广泛内容，包括对自编码器架构、编码器功能、潜在表示和神经网络组件的详细说明。
- [Reducing the Dimensionality of Data with Neural Networks](https://doi.org/10.1126/science.1127647) — Geoffrey E. Hinton, Ruslan R. Salakhutdinov (2006)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 313; Pages: 504-507; DOI: [10.1126/science.1127647](https://doi.org/10.1126/science.1127647)
  这篇论文展示了深度自编码器在非线性降维和学习数据特征方面的有效性，与编码器在数据压缩中的作用直接相关。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  一本易于理解的机器学习指南，其中包含了对自编码器架构和编码器在实际应用中如何运作的清晰解释。第二版。
