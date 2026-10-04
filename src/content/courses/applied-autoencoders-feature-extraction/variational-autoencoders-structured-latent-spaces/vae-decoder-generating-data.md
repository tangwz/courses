---
course: "applied-autoencoders-feature-extraction"
chapter: "variational-autoencoders-structured-latent-spaces"
lesson: "vae-decoder-generating-data"
sourceId: 6484
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces/vae-decoder-generating-data"
title: "VAE解码器：从潜在样本生成数据"
description: "描述VAE解码器（也称为生成器）如何从潜在分布中提取的样本重构或生成数据。"
order: 5
plots: []
sourceHash: "da5ea140750933845ef810aad1067aebea4c8a2b99ae7355168778fa69e6d6b0"
sourceCorrections: []
---

一旦VAE的编码器将输入 $x$ 映射到潜在空间中某个概率分布的参数 (parameter)，就会获得潜在样本 $z$。这通常通过重参数化技巧实现，该技巧允许梯度流经采样过程。有了潜在样本 $z$，VAE解码器，通常表示为 $g_\theta(z)$，将这个潜在向量 (vector) $z$ 作为输入。它的主要目标是重构原始数据，或生成与训练数据相似的全新数据样本。

### 作为生成网络的解码器

可将VAE解码器视为一个学习到的生成函数。其主要作用是将连续、有结构的潜在空间中的点映射回高维数据空间。在训练期间，它学习从 $q_\phi(z|x_i)$ 采样得到的潜在向量 (vector) $z_i$ 来重构原始输入 $x_i$。然而，一旦训练完成，其真正的生成能力在于它能够接收任意潜在向量 $z$，尤其是一个从先验分布 $p(z)$（例如，标准高斯分布 $\mathcal{N}(0, I)$）中采样得到的向量，并将其转换为一个合理的数据样本 $\hat{x}$。

解码器的架构通常是编码器的镜像。如果编码器使用一系列层来逐步降低维度，那么解码器将使用一系列层来逐步增加维度。

- 对于**全连接自编码器**，如果编码器包含神经元数量递减的层（例如，784 -> 256 -> 128 -> latent\_dim），那么解码器将包含神经元数量递增的层（例如，latent\_dim -> 128 -> 256 -> 784）。
- 对于**卷积自编码器**（常见于图像数据），如果编码器使用 `Convolutional`（卷积）和 `MaxPooling`（最大池化）层来减小空间维度并增加特征深度，那么解码器将使用 `Transposed Convolutional`（转置卷积，常被称为反卷积）层或 `UpSampling`（上采样）与 `Convolutional`（卷积）层的组合来扩展空间维度并重构图像。

目标是反转编码器执行的压缩操作，将 $z$ 中压缩的信息“解包”为一个完整的数据样本。

### 从潜在样本到生成数据

这个过程始于将潜在向量 (vector) $z$ 输入到解码器网络的第一层。这个向量随后经由后续层进行转换，每一层通常会应用一个仿射变换，紧接着是一个非线性激活函数 (activation function)（例如ReLU、LeakyReLU或tanh）。

解码器的最后一层尤其重要，因为其设计取决于所生成数据的性质：

- **对于像素值归一化 (normalization)到0到1之间的图像数据**（例如MNIST数字），输出层通常对每个像素使用 `sigmoid` 激活函数。这确保了生成的像素值也落在[0, 1]范围内。
- **对于像素值归一化到-1到1之间的数据**，可能会使用 `tanh` 激活函数。
- **对于未限定范围或范围不同的连续数据**，输出层可能适合使用 `linear`（线性）激活函数（即，无激活函数或恒等函数），可能是在将输入数据归一化为零均值和单位方差之后。那么生成的输出将需要进行反归一化。

解码器的输出 $\hat{x} = g_\theta(z)$ 就是生成的样本。在训练期间，这个 $\hat{x}$ 会使用重构损失函数 (loss function)（例如，连续数据的均方误差，二进制数据或[0,1]范围内数据的二元交叉熵）与原始输入 $x$ 进行比较。

下图描述了解码器在从潜在样本生成数据中的作用：

> 潜在向量 $z$ 既可以从近似后验分布 $q_\phi(z|x)$（用于重构）中提取，也可以从先验分布 $p(z)$（用于新数据生成）中提取，然后由解码器网络进行处理。解码器的层对这个向量进行上采样，以生成数据 $\hat{x}$。

### 生成全新样本

一旦VAE得到有效训练，即VAE损失函数 (loss function)中的重构损失和KL散度项都被最小化，解码器 $g_\theta(z)$ 就成为一个强大的生成模型。要生成一个从未见过的新数据样本，你不需要原始输入 $x$。相反，你只需：

1. 从VAE被训练以匹配的先验分布 $p(z)$（通常是 $\mathcal{N}(0, I)$）中抽取一个随机样本 $z_{new}$。
2. 将此 $z_{new}$ 通过已训练的解码器：$\hat{x}_{new} = g_\theta(z_{new})$。

生成的 $\hat{x}_{new}$ 是一个合成数据点。由于VAE促使潜在空间是连续且结构良好的（因为有KL散度正则项），即使是从先验分布中随机采样的样本，在解码后也倾向于产生连贯且有意义的输出，其风格与训练数据相似。这种生成新数据的能力是VAE的一个显著特点，并使它们区别于标准自编码器。这些生成样本的质量和多样性在很大程度上取决于潜在空间在多大程度上捕获了原始数据集中的底层变化。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint arXiv:1312.6114; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇开创性论文介绍了变分自编码器，阐述了解码器在从潜在空间重建数据和生成新样本方面的作用。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  对变分自编码器、其架构、解码器作为生成网络的功能以及转置卷积等不同层类型的应用提供了详尽的背景知识。
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) — Vincent Dumoulin and Francesco Visin (2016)
  Journal: arXiv preprint arXiv:1603.07285; DOI: [10.48550/arXiv.1603.07285](https://doi.org/10.48550/arXiv.1603.07285)
  详细阐述了卷积和转置卷积操作的数学和实践方面，这对于理解卷积变分自编码器解码器中的上采样层至关重要。
- [CS236: Deep Generative Models (Course Website)](http://deepgenerativemodels.github.io/) — Stanford University (2023)
  关于深度生成模型的综合课程，包含变分自编码器、其解码器架构以及从潜在样本生成新数据的详细章节。
