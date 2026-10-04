---
course: "introduction-autoencoders-feature-learning"
chapter: "autoencoder-anatomy-encoder-decoder"
lesson: "decoder-activation-functions"
sourceId: 6411
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-2-autoencoder-anatomy-encoder-decoder/decoder-activation-functions"
title: "解码器中常用的激活函数"
description: "了解解码器输出层中常用的激活函数，如 Sigmoid，特别是针对归一化数据。"
order: 9
plots: ["plots/6411-0.json"]
sourceHash: "43e6a217953ecce7f490af8d001f78c60d82ce1826745060a47b12a3a69dc596"
sourceCorrections: []
---

解码器在自编码器中的作用是重建数据。它接收通常由编码器的瓶颈层生成的压缩表示 $z$，并将其转换回类似于原始输入 $X$ 的形式。解码器的最后一层，即输出层，在这一重建过程中扮演着重要角色。此输出层使用的激活函数 (activation function)直接决定了重建值 $X'$ 的性质和范围。因此，其选择受您希望重建数据特征的直接影响。

### 输出层的激活函数 (activation function)

输出层的激活函数需要确保重建数据 $X'$ 与原始输入数据 $X$ 具有相同的格式和范围。如果您的输入数据由介于 0 和 1 之间归一化 (normalization)的像素值组成，则解码器的输出也应落在此范围内。

#### Sigmoid 函数

自动编码器输出层最常用的激活函数之一，特别是在处理归一化到 [0, 1] 范围的输入数据时（例如灰度图像像素强度），是 **Sigmoid** 函数。

Sigmoid 函数定义如下：
$\sigma(x) = \frac{1}{1 + e^{-x}}$
它将任何实值输入 $x$ 压缩到 0 到 1 之间的输出。这种“S”形非常实用，因为它与在此范围内有界的数据自然吻合。例如，如果一个输入像素是 0（黑色）或 1（白色），或介于两者之间，Sigmoid 函数可以确保重建的像素值符合这些边界。

#### Tanh (双曲正切) 函数

输出层的另一个选择是 **双曲正切** 函数，通常缩写为 `tanh`。它与 Sigmoid 类似，但将值压缩到 [-1, 1] 的范围。

`tanh` 函数定义如下：
$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
如果您的输入数据 $X$ 已归一化到 -1 和 1 之间，通常会使用 `tanh`。

下图展示了 Sigmoid 和 `tanh` 函数，显示它们如何将输入值映射到各自的输出范围：



![常用输出层激活函数](plots/6411-0.json)



> Sigmoid 输出介于 0 和 1 之间的值，适用于归一化到此范围的数据。Tanh 输出介于 -1 和 1 之间的值，用于数据按此方式归一化时。

#### 线性函数

如果您的输入数据没有方便地限定在 [0, 1] 或 [-1, 1] 之间怎么办？例如，您可能正在处理可以取任何实数的原始传感器读数。在这种情况下，为输出层使用 **线性** 激活函数（或者等效地，不使用激活函数）是合适的。

线性激活函数很简单：
$f(x) = x$
这意味着神经元的输出只是其输入的加权和，没有任何“压缩”。这使得重建值 $X'$ 可以取任何实数，与原始数据 $X$ 的潜在范围相符。

### 解码器中隐藏层的激活函数 (activation function)

尽管解码器的输出层对输入数据范围有特定要求，解码器中的隐藏层有不同的作用。这些层逐步进行上采样并把压缩表示 $z$ 转换回原始数据的结构。

对于解码器中的这些中间（隐藏）层，通常使用与编码器隐藏层相同的激活函数。**修正线性单元（ReLU）** 是一个很受欢迎的选择。

回忆一下，ReLU 的定义是：
$\text{ReLU}(x) = \max(0, x)$
ReLU 在隐藏层中受到青睐（包括编码器和解码器），因为它有助于更有效地训练更深的网络（通过缓解梯度消失等问题），并且计算效率高。在解码器中，ReLU 使网络能够学习从压缩形式重建数据所需的复杂非线性转换。尽管 Sigmoid 或 `tanh` 也可以用于隐藏层，但 ReLU 通常是一个可靠的默认选择。

总结来说，在设计解码器时：

- **输出层的激活函数** 根据原始数据 $X$ 的范围选择（例如，Sigmoid 用于 [0,1]，`tanh` 用于 [-1,1]，线性函数用于无界数据）。
- **隐藏层的激活函数** 通常使用 ReLU 来帮助学习从瓶颈 $z$ 到 $X'$ 的复杂映射。

理解这些激活函数及其位置是理解自动编码器如何有效地学习重建和表示数据的又一步。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  涵盖深度学习的理论基础，包括激活函数和自编码器架构的详细解释。
- [A Comprehensive Survey on Activation Functions in Deep Learning](https://www.mdpi.com/2076-3417/10/21/7483) — J. D. Ramachandran, K. Chelliah, D. B. Vijendran, P. H. Reddy, B. Bharath (2020)
  Journal: Applied Sciences; Publisher: MDPI; Volume: 10; Pages: 7483; DOI: [10.3390/app10217483](https://doi.org/10.3390/app10217483)
  全面概述了深度学习中使用的各种激活函数，讨论了它们的特点和应用。
- [Extracting and Composing Robust Features with Denoising Autoencoders](https://doi.org/10.1145/1390156.1390294) — Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol (2008)
  Journal: Proceedings of the 25th International Conference on Machine Learning; Publisher: ACM; Pages: 1096-1103; DOI: [10.1145/1390156.1390294](https://doi.org/10.1145/1390156.1390294)
  一篇关于去噪自编码器的开创性论文，展示了解码器在数据重建中的作用，并影响输出层选择。
- [Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/) — Justin Johnson, Andrej Karpathy, Fei-Fei Li, and course staff (2023)
  Publisher: Stanford University
  这份来自顶尖大学课程的资料提供了神经网络架构和常用激活函数（Sigmoid, Tanh, ReLU）特性的易懂概述。
