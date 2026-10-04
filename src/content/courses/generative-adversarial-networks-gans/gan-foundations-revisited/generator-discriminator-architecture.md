---
course: "generative-adversarial-networks-gans"
chapter: "gan-foundations-revisited"
lesson: "generator-discriminator-architecture"
sourceId: 2653
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-1-gan-foundations-revisited/generator-discriminator-architecture"
title: "生成器-判别器架构"
description: "简要回顾生成对抗网络的主要组成部分。"
order: 1
plots: []
sourceHash: "a02a201154b0959b366bef75699cf62fdadcf4a3149744290fbb5951f70ea3d4"
sourceCorrections: []
---

生成对抗网络（GAN）包含两个处于竞争机制中的主要组成部分：生成器和判别器。了解它们各自的职责以及它们之间的互动是掌握GAN如何学习生成合成数据的基本要求。这些组成部分通常以神经网络 (neural network)的形式实现。

### 生成器 (G)

生成器的目标是生成与真实数据无法区分的合成数据。它接收一个随机噪声向量 (vector)（常被称为潜在向量 $z$）作为输入，并将其转换为一个数据样本。这个潜在向量 $z$ 通常从一个简单的先验分布中采样，例如标准高斯分布或均匀分布，表示为 $p_z(z)$。

生成器网络 $G$ 学习从潜在空间到数据空间的映射。它的输出 $G(z)$ 是一个数据样本（例如图像、文本片段、音频波形），旨在模仿真实数据分布 $p_{data}(x)$ 的结构和特征。可以将生成器看作一个有抱负的艺术家或伪造者，试图生产看起来真实的创造物。在早期的GAN以及DCGAN（我们稍后会再次提到）中，生成器常使用转置卷积层（有时称为反卷积）将低维潜在向量上采样为更高维的输出，如图像。

生成样本 $G(z)$ 的质量直接取决于生成器学习真实数据潜在模式和分布的程度。它的成功通过其欺骗判别器的能力来衡量。

### 判别器 (D)

判别器充当生成器的对手。它的职责是检查给定的数据样本 $x$，并确定其来自真实数据集而非由生成器合成的概率。它本质上充当一个二分类器。

判别器网络 $D$ 接收一个数据样本 $x$（这可能是从 $p_{data}(x)$ 中抽取的真实样本，或是由生成器生成的伪造样本 $G(z)$）作为输入。它的输出 $D(x)$ 是一个表示 $x$ 为真实的概率的单个标量值。接近1的值表示判别器认为输入是真实的，而接近0的值表示它认为输入是伪造的。

通常，判别器被实现为标准的前馈或卷积神经网络 (neural network) (CNN)，适用于正在处理的数据类型（例如，图像使用CNN）。它学习识别区分真实数据与生成器当前生成结果的细微特征和统计属性。

### 对抗性互动

生成器和判别器在一同训练的对抗过程中进行，类似于零和博弈。

1. **生成器训练：** 生成器旨在生成判别器错误分类为真实的样本 $G(z)$。也就是说，它试图最大化判别器对其伪造样本的输出，从而有效最大化 $D(G(z))$。当生成器提升时，判别器的任务变得更难。
2. **判别器训练：** 判别器旨在正确分类真实样本（来自 $p_{data}(x)$）和伪造样本（来自 $G(z)$）。它试图对真实 $x$ 最大化 $D(x)$，并对伪造样本 $G(z)$ 最小化 $D(G(z))$。当判别器提升时，它会向生成器提供更清晰的信号（梯度），指导其如何生成更真实的输出。

这种迭代训练过程，其中每个组件根据对方的表现更新其参数 (parameter)，理想情况下会达到平衡。此时，生成器生成的样本与真实数据在统计上无法区分，而判别器基本上是在猜测 ($D(x) \approx 0.5$) 样本是真实还是伪造。

下图展示了GAN架构内的基本信息流。

> 标准GAN内的信息流。随机噪声 $z$ 输入到生成器 $G$ 以生成伪造数据 $G(z)$。判别器 $D$ 同时接收真实数据 $x$ 和伪造数据 $G(z)$，并输出一个概率，表示其输入是否为真实。

这种基本架构及其竞争性网络，是所有GAN变体模型赖以建立的结构。尽管设想上简单，但有效训练这些网络会带来不稳定性、模式坍塌等挑战，我们将在接下来讨论并在本课程中通过更高级的技术来应对。在继续学习之前，牢固掌握这种生成器-判别器动态是必要条件。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇开创性论文介绍了生成对抗网络的概念，确立了生成器和判别器之间的对抗训练框架。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) — Alec Radford, Luke Metz, and Soumith Chintala (2016)
  Journal: International Conference on Learning Representations (ICLR 2016); DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  这篇论文为使用卷积层稳定训练GAN提供了关键的架构指导，显著提升了图像生成能力。
- [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160) — Ian Goodfellow (2016)
  Journal: arXiv preprint arXiv:1701.00160; DOI: [10.48550/arXiv.1701.00160](https://doi.org/10.48550/arXiv.1701.00160)
  GAN发明者提供的一份易懂而全面的教程，解释了核心架构、训练动态和基本概念。
