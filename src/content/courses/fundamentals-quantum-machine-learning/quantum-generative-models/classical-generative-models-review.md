---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-generative-models"
lesson: "classical-generative-models-review"
sourceId: 1071
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/classical-generative-models-review"
title: "经典生成模型回顾"
description: "简要回顾生成对抗网络 (GANs) 和变分自编码器 (VAEs) 等经典生成模型。"
order: 1
plots: []
sourceHash: "4de377a2b89ea00cb78ce53e1ae5a8929ab974898f31363435b1706554458761"
sourceCorrections: []
---

在我们研究生成模型的量子方法之前，让我们简要回顾两个有影响力的经典框架：生成对抗网络 (GAN) (GANs) 和变分自编码器 (VAEs)。了解它们的核心思想为理解其量子对应物提供了重要背景。两种情况的目标都是学习一个模型分布 $p_{\text{模型}}(x)$，使其近似于未知的真实数据分布 $p_{\text{数据}}(x)$，从而生成与原始数据相似的新样本。

## 生成对抗网络 (GAN) (GANs)

GANs 由 Ian Goodfellow 及其同事于 2014 年提出，采用一种巧妙的对抗训练过程，涉及两个神经网络 (neural network)：

1. **生成器 (G)：** 该网络接收一个随机噪声向量 (vector) $z$ 作为输入，通常从高斯或均匀分布这样的简单分布中取样 ($z \sim p_z(z)$)。其目标是将这些噪声转化为看起来像来自真实数据分布 $p_{\text{数据}}(x)$ 的数据样本 $G(z)$。
2. **判别器 (D)：** 该网络充当分类器。它接收一个样本 $x$（可以是训练数据中的真实样本，也可以是生成器生成的伪造样本），并输出一个概率 $D(x)$，表示 $x$ 是真实样本而非伪造样本的可能性。

训练过程被建模为一个最小-最大博弈。判别器经过训练以最大化其区分真实样本和 $G$ 生成的伪造样本的准确性。同时，生成器经过训练以最小化判别器正确识别其输出为伪造的概率。本质上，$G$ 试图欺骗 $D$。

该博弈的目标函数通常表示为：


$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{数据}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$


这里，$\mathbb{E}$ 表示期望。判别器 $D$ 希望最大化此值（使真实样本 $x$ 的 $D(x)$ 接近 1，使伪造样本的 $D(G(z))$ 接近 0）。生成器 $G$ 希望最小化此值（通过使 $D(G(z))$ 接近 1，从而有效地欺骗判别器）。

> 生成对抗网络 (GAN) 的基本架构，展示了生成器和判别器之间的交互。

训练 GANs 具有挑战性，因为可能出现不稳定性，例如模式崩溃（即生成器只产生有限种类的样本）或梯度消失。然而，当成功时，GANs 以生成高保真样本而闻名，尤其是在图像生成等方面。

## 变分自编码器 (VAEs)

VAEs 由 Diederik Kingma 和 Max Welling 于 2013 年提出，从概率角度处理生成模型问题，结合了变分推断和自编码器的思想。一个 VAE 包含两个主要部分，通常以神经网络 (neural network)的形式实现：

1. **编码器 (识别模型)：** 该网络，表示为 $q_\phi(z|x)$，接收输入数据点 $x$ 并输出参数 (parameter)（通常是均值 $\mu$ 和方差 $\sigma^2$），用于低维潜在空间 $Z$ 中的概率分布。此分布表示真实后验分布 $p(z|x)$ 的近似。通常，选择 $q_\phi(z|x)$ 为高斯分布：$\mathcal{N}(z | \mu_\phi(x), \sigma^2_\phi(x)I)$。
2. **解码器 (生成模型)：** 该网络，表示为 $p_\theta(x|z)$，接收从潜在空间中采样的点 $z$（在训练期间从近似后验 $q_\phi(z|x)$ 中或在生成期间从先验分布 $p(z)$ 中采样，通常为 $\mathcal{N}(0, I)$），并重构数据点 $x$。它定义了以潜在变量为条件的数据空间上的分布。

训练 VAE 的目标是最大化数据的对数似然，$\log p(x)$。由于这通常难以处理，VAEs 转而最大化一个下界，称为证据下界 (ELBO)：


$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


ELBO 包含两个项：

- **重构损失：** $\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]$ 促使解码器从其潜在表示 $z$ 精确地重构输入 $x$。对于连续数据，这通常对应于最小化均方误差；对于二元数据，二元交叉熵是常见的做法。
- **KL 散度正则项：** $D_{KL}(q_\phi(z|x) || p(z))$ 衡量编码器学习到的近似后验分布与潜在空间上的预定义先验分布 $p(z)$ 之间（例如，标准高斯分布）的 Kullback Leibler 散度。此项作为正则器，迫使编码分布接近先验，这有助于通过直接从 $p(z)$ 中提取 $z$ 来生成新样本。

使用了一种名为“重参数化技巧”的特定技术，以允许梯度通过编码器内的采样过程反向传播 (backpropagation)，从而通过标准反向传播进行训练。

> 变分自编码器 (VAE) 的基本架构，展示了编码、潜在采样和解码阶段。

与 GANs 相比，VAEs 通常提供更稳定的训练，并提供了一个明确的概率模型，具有定义良好的似然目标（ELBO）。然而，VAEs 生成的样本有时被认为不如最先进 GANs 生成的样本清晰或真实，可能由于所做的简化假设（例如，高斯后验）以及重构项的影响。

这些经典模型确立了生成模型中的基本观念：隐式（GANs）或通过潜在变量显式（VAEs）表示分布，定义目标函数，并处理训练动态。当我们转向量 (vector)子领域时，我们将看到量子电路如何作为类似框架中的组成部分，这可能提供捕捉复杂数据分布的新方法。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27; Publisher: Curran Associates, Inc.; Pages: 2672-2680; DOI: [10.55982/tjc.2014.288](https://doi.org/10.55982/tjc.2014.288)
  介绍生成对抗网络及其架构和最小最大训练框架的原始论文。
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf) — Diederik P. Kingma and Max Welling (2013)
  Journal: arXiv preprint arXiv:1312.6114; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器、概率框架和证据下界目标的原始论文。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，为深度学习提供理论和实践基础，其中包含关于GAN和VAE等生成模型的专门章节。
