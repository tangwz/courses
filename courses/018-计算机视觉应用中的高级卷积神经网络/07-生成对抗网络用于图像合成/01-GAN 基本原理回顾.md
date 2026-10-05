# GAN 基本原理回顾

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/gan-fundamentals-revisited)

[返回章节目录](README.md) · [返回课程目录](../README.md)

生成对抗网络 (GAN)（GANs）由 Ian Goodfellow 及其同事于2014年提出，代表了一类强大的生成模型。GANs 不显式地对数据概率分布建模，而是通过一个对抗过程，学习从该分布中生成样本。此过程包含两个相互竞争的神经网络 (neural network)：生成器和判别器。生成对抗网络的主要部分及其协作方式在此进行讲解。

### 生成器 (G)

生成器网络，表示为 $G$，充当创造性引擎。其主要作用是合成与真实数据分布相似的数据样本。通常，$G$ 以从简单先验分布（如高斯分布或均匀分布）中取样的随机噪声向量 (vector) $z$（即 $z \sim p_z(z)$）作为输入。然后，它通过一系列变换处理此噪声，这些变换常使用深度卷积层（特别是用于图像生成的转置卷积或“反卷积”）来实现，以生成候选数据样本 $G(z)$。$G$ 的目标是学习从潜在空间（$z$ 所在的空间）到数据空间的映射，使生成的样本 $G(z)$ 变得与真实数据样本 $x$ 无法区分。

### 判别器 (D)

判别器网络 $D$ 充当评论家或评判者。其作用是评估给定数据样本的真实性。$D$ 本质上是一个二元分类器，通常作为标准的前馈或卷积神经网络 (neural network) (CNN)实现。它将数据样本（来自训练数据集的真实样本 $x$ 或生成器生成的伪样本 $G(z)$）作为输入，并输出一个标量概率 $D(x)$，表示输入样本是真实的（非生成）可能性。接近1的值表明判别器认为样本是真实的，而接近0的值则说明它认为样本是伪造的。

### 对抗训练过程

训练过程组织了一场 $G$ 和 $D$ 之间的竞争性活动。此活动可按以下方式理解：

1. **判别器训练：** $D$ 接受训练以提高其区分真实数据与生成数据的能力。它接收一个批次的数据，包含来自数据集的真实样本 $x$（$x \sim p_{data}(x)$）和当前生成器生成的伪样本 $G(z)$（使用随机噪声 $z \sim p_z(z)$）。$D$ 的权重 (weight)通过梯度上升进行更新，以最大化其分类准确度，有效地使 $D(x)$ 接近1，使 $D(G(z))$ 接近0。在此阶段，生成器的权重保持不变。
2. **生成器训练：** $G$ 接受训练以提高其欺骗判别器的能力。它生成一批伪样本 $G(z)$。这些样本随后经由判别器处理。生成器的权重根据判别器的输出 $D(G(z))$ 通过梯度下降 (gradient descent)进行更新，目标是使判别器将这些伪样本分类为真实样本（使 $D(G(z))$ 接近1）。在此阶段，判别器的权重保持不变。

这两个步骤迭代交替进行。随着时间的推移，$G$ 在生成逼真样本方面表现更好，使 $D$ 的任务变得更难。同时，$D$ 在识别伪造样本方面表现更好，促使 $G$ 生成更具说服力的输出。这种动态竞争最终会使得 $G$ 生成的样本在统计上与真实数据无法区分，而 $D$ 则被迫随机猜测（$D(x) \approx 0.5$）。

> 流程图描绘了 GAN 框架中生成器、判别器、随机噪声、真实数据和生成数据之间的关系。

### 最小最大目标函数

对抗活动通过最小最大目标函数 $V(D, G)$ 正式描述：


$$
\min_G \max_D V(D, G) = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$


我们来细分一下：

- $E_{x \sim p_{data}(x)}[\log D(x)]$: 此项表示判别器对从真实数据分布 $p_{data}(x)$ 中取样的真实数据样本 $x$ 的预期输出。判别器 $D$ 目标是最大化此项，即它希望 $D(x)$ 接近1（正确识别真实样本）。
- $E_{z \sim p_z(z)}[\log(1 - D(G(z)))]$: 此项表示对从噪声 $z$ 生成的伪数据样本 $G(z)$ 的预期输出。判别器 $D$ 也目标是最大化目标函数的此部分，这对应于最小化 $D(G(z))$，使其接近0（正确识别伪样本）。
- $\max_D$: 判别器试图通过更擅长区分真实与伪造样本，来最大化整个表达式 $V(D, G)$。
- $\min_G$: 生成器无法直接影响第一项（$\log D(x)$），因此它试图通过影响第二项来最小化整体目标。最小化 $E_{z \sim p_z(z)}[\log(1 - D(G(z)))]$ 意味着 $G$ 旨在使 $D(G(z))$ 接近1，有效地欺骗判别器，使其认为生成的样本是真实的。

在实践中，当 $D$ 很强并以高置信度拒绝 $G$ 的样本时（$D(G(z))$ 接近0），训练 $G$ 来最小化 $\log(1 - D(G(z)))$ 在训练早期可能导致梯度消失。一个常见的替代方法是修改生成器的目标，转而最大化 $\log D(G(z))$。此替代目标在早期提供更强的梯度，但在对抗活动中保持相同的固定点。

此基本框架构成了我们将要了解的各种 GAN 架构和应用，包括本章后面讨论的特定模型。理解此主要对抗动态对于诊断训练问题和评估更高级 GAN 变体的改进之处非常重要。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27 (NIPS 2014); DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍了GAN框架，详细阐述了生成器和判别器之间的对抗过程，以及最小最大目标函数。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) — Alec Radford, Luke Metz, Soumith Chintala (2015)
  Journal: International Conference on Learning Representations (ICLR 2016); DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  提出了使用卷积网络稳定训练GAN的架构指南，特别适用于图像生成任务。
- [A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications](https://doi.org/10.1007/s10462-020-09861-2) — Jianmin Gui, Zesong Sun, Yujie Wen, Yangtao Tao, Daomin Wang, and Ruochuan Xie (2020)
  Journal: Artificial Intelligence Review; Publisher: Springer; Volume: 54; Pages: 215–251; DOI: [10.1007/s10462-020-09861-2](https://doi.org/10.1007/s10462-020-09861-2)
  全面概述了GAN的架构、训练策略和理论方面，为GAN的发展提供了更广阔的背景。
- [Deep Learning (Chapter 20: Generative Adversarial Networks)](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  该章节提供了GAN的基础性解释，由原始创建者之一撰写，涵盖了核心原理和数学公式。

---

[上一节](../06-%E9%AB%98%E7%BA%A7%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E4%B8%8E%E5%9F%9F%E9%80%82%E5%BA%94/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%9C%A8%E7%89%B9%E5%AE%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B.md) · [下一节](02-%E8%AE%AD%E7%BB%83%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%E7%9A%84%E6%8C%91%E6%88%98.md)
