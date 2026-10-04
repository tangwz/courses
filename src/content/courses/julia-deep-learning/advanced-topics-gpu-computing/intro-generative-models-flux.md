---
course: "julia-deep-learning"
chapter: "advanced-topics-gpu-computing"
lesson: "intro-generative-models-flux"
sourceId: 6926
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/intro-generative-models-flux"
title: "使用 Flux 介绍生成模型"
description: "简要介绍如何使用 Flux.jl 构建 GANs 或 VAEs 等生成模型。"
order: 5
plots: []
sourceHash: "e73eb2d864d4f28a843eacbf98312e71c9f3da0b5f04aab76c138162826d36be"
sourceCorrections: []
---

至今，我们一直关注判别模型，这类模型学习将输入映射到输出，例如图像分类或值预测。而生成模型则采用不同的方式。它们不只是为给定输入预测标签，而是致力于理解和学习数据本身的潜在概率分布。这使得它们能够生成与原始数据集相似的新数据样本。想象一个模型，它不仅能识别手写数字，还能画出新的、看起来合理的数字。这就是生成模型的能力范围。

这些模型有多种应用，包括创建逼真图像、合成音频、生成文本、为训练其他模型扩充数据集，甚至通过找出不符合所学分布的数据点来发现异常。

实质上，判别模型可能学习 $P(y|x)$ (给定输入 $x$ 时输出 $y$ 的概率)，而生成模型则通常尝试学习 $P(x)$ (输入 $x$ 的概率) 或有时是 $P(x,y)$ ($x$ 和 $y$ 的联合概率)。Flux.jl 凭借其灵活性和可组合性，为构建这些通常更复杂的架构提供了坚实基础。

接下来我们简要介绍两种主要的生成模型：生成对抗网络 (GAN) (GANs) 和变分自编码器 (VAEs)。

### 生成对抗网络 (GAN) (GANs)

生成对抗网络，简称 GANs，是由 Ian Goodfellow 及其同事提出的一类引人注目的模型。它们基于博弈论方法运行，包含两个神经网络 (neural network)：

1. **生成器 (G)**：此网络接收随机噪声（通常来自简单的分布，如高斯分布）作为输入，并尝试将其转换为看起来像来自真实数据分布的数据。例如，如果对人脸图像进行训练，生成器会学习生成新的人脸图像。
2. **判别器 (D)**：此网络充当评论者。它是一个二元分类器，接收真实数据样本（来自训练集）和伪造数据样本（由生成器生成），并尝试区分它们。它输出一个概率，表示其输入是真实的还是伪造的。

训练过程是对抗性的：

- **生成器**的目标是通过生成越来越逼真的数据来欺骗**判别器**。它的目标是让判别器将其输出分类为真实。
- **判别器**的目标是更好地识别来自生成器的伪造数据。它的目标是正确地将真实样本分类为真实，将伪造样本分类为伪造。

这两个网络同时进行训练。随着生成器表现提升，判别器的任务变得更难，迫使其不断进步。反过来，随着判别器表现提升，它为生成器提供更强的信号，促使其生成更逼真的样本。这种动态持续进行，直到理想状态下，生成器能够生成与真实数据无法区分的样本。

> 一个图表，说明生成对抗网络 (GAN) 的基本架构，显示了生成器和判别器之间的交互过程。

在 Flux.jl 中实现 GANs，需要为生成器和判别器定义两个独立的模型（通常使用 `Chain`）。训练循环比标准的监督学习 (supervised learning)更复杂，因为通常需要交替训练判别器几步，然后训练生成器一步。损失函数 (loss function)根据 GAN 变体（例如，minimax 损失、Wasserstein 损失）进行选择。尽管功能强大，但 GANs 在训练方面有时会比较棘手，通常需要仔细调整超参数 (parameter) (hyperparameter)和架构选择以获得稳定性。

### 变分自编码器 (VAEs)

变分自编码器，简称 VAEs，是生成建模的另一种方式，其根源在于概率图模型和变分推断。与 GANs 的对抗设置不同，VAEs 由两个主要部分组成，它们以更协作的方式一同训练：

1. **编码器（或识别模型）**：此网络接收输入数据点（例如图像），并将其映射的不是潜在空间中的单个点，而是概率分布的参数 (parameter)（通常是具有对角协方差矩阵的多元高斯分布）。也就是说，对于每个输入 $x$，编码器输出一个均值向量 (vector) $\mu$ 和一个对数方差向量 $\log(\sigma^2)$，它们共同定义了潜在空间中的一个分布。
2. **解码器（或生成模型）**：此网络接收从潜在分布中采样得到的点 $z$（由编码器生成的 $\mu$ 和 $\sigma^2$ 定义），并尝试重构原始输入数据点 $x$。

VAE 的训练目标包含两个主要部分：

- **重构损失**：此项衡量解码器从潜在表示中重构原始输入的准确程度。常用选项包括用于实值数据的均方误差 (MSE) 或用于二元数据（如黑白图像）的二元交叉熵。
- **KL 散度正则化 (regularization)项**：此项促使编码器生成的潜在分布接近先验分布，通常是标准正态分布（均值为零，单位方差）。这个正则化项有助于组织潜在空间，使其更连续，并适合生成新样本。这样做是为了，如果潜在空间表现良好，从先验中随机采样一个 $z$ 并通过解码器，应该会生成一个新颖但合理的数据样本。

为了在训练期间以允许反向传播 (backpropagation)的方式采样 $z$，VAEs 使用“重参数化技巧”：不是直接从 $q(z|x) = \mathcal{N}(z; \mu, \sigma^2)$ 中采样，而是采样 $\epsilon \sim \mathcal{N}(0, I)$，然后计算 $z = \mu + \sigma \odot \epsilon$，其中 $\odot$ 是元素级乘法。

> 一个图表，概述了变分自编码器 (VAE) 的结构，显示了编码器、通过重参数化技巧进行的潜在空间采样以及解码器。

在 Flux.jl 中，通常会将编码器和解码器定义为独立的 `Chain`。编码器可能会输出两倍于潜在维度数量的值（用于均值和对数方差）。重参数化技巧直接使用算术运算和随机数生成（例如 `randn!`）来实现。损失函数 (loss function)结合了重构项（例如 `Flux.Losses.mse`）和一个自定义的 KL 散度项。训练包括优化这个组合损失，同时考虑编码器和解码器两部分的参数。

### 使用 Flux.jl 构建生成模型

Flux.jl 的设计使其非常适合生成模型有时非传统的架构和训练过程。

- **灵活性**：您可以使用 `Dense`、`Conv`、`ConvTranspose` 等标准层和各种激活函数 (activation function)，轻松定义生成器、判别器、编码器和解码器的自定义网络结构。
- **自定义训练循环**：正如您可能已经体会到的，训练生成模型通常需要比用于简单监督任务的直接 `Flux.train!` 循环更复杂的方法。您可能需要编写自定义训练循环来管理 GANs 中的交替更新，或者正确计算和组合 VAEs 中的损失组成部分。您对前面章节中梯度、优化器和参数 (parameter)更新的理解将直接应用于此处。
- **自动微分**：Zygote.jl 处理梯度计算，即使对于复杂的损失函数 (loss function)和涉及采样步骤（如 VAEs 中的重参数化技巧）的架构也不例外。

虽然本节只是介绍，但实际实现和训练生成模型需要耐心和反复尝试。与判别模型相比，它们通常对超参数 (hyperparameter)和初始化更敏感。然而，生成新数据的能力为深度学习 (deep learning)带来了许多创意和实际的可能。随着您在深度学习方面的学习，您会发现这些模型能出色地完成多种任务。参考论文和开源实现将为特定的架构和训练技术提供更多指导。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Volume: 27; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇基础论文介绍了生成对抗网络（GANs），详细阐述了其架构、对抗性训练过程和初步应用。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇里程碑式的论文介绍了变分自编码器（VAEs），提出了结合自编码器与变分推断以及重参数化技巧的生成模型框架。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  这本综合性教材详细阐述了深度学习，包含专门章节介绍各种生成模型，如GANs和VAEs，提供了丰富的理论背景和实践见解。
