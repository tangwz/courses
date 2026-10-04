---
course: "synthetic-data-gans-diffusion"
chapter: "generative-modeling-foundations-revisited"
lesson: "gan-fundamentals"
sourceId: 5637
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-1-generative-modeling-foundations-revisited/gan-fundamentals"
title: "GAN 核心"
description: "重温GAN目标函数、最小-最大博弈及训练动态。"
order: 4
plots: []
sourceHash: "9ef9a0bf2709827a54c1feb8fdc7e0e68f41fc39e46d87042cfe9ae6d0429353"
sourceCorrections: []
---

生成对抗网络 (GAN) (GANs) 由 Ian Goodfellow 及其同事于 2014 年提出，是一种用于构建生成模型的强力架构。GANs 采用一个对抗过程，涉及两个神经网络 (neural network)：生成器 ($G$) 和判别器 ($D$)。这些网络的职责和相互作用将被细致分析。

### 生成器网络 (G)

生成器的任务是合成看起来与真实数据无法区分的数据。它接收一个随机噪声向量 (vector) $z$ 作为输入，通常从一个简单的先验分布 $p_z(z)$ 中采样，例如多元高斯分布或均匀分布。这个噪声向量 $z$ 存在于一个低维隐空间中。生成器作为一个映射函数，将这个隐向量转换为数据空间（例如图像）中的一个高维数据样本 $G(z)$。

$G: \mathcal{Z} \rightarrow \mathcal{X}$

这里，$\mathcal{Z}$ 是隐空间，$\mathcal{X}$ 是数据空间。$G$ 的目标是学习 $\mathcal{X}$ 上的一个分布 $p_g$，使其与真实数据分布 $p_{data}$ 相符。在架构上，对于图像生成等任务，生成器通常使用转置卷积（有时称为反卷积）等层，将低维输入噪声上采样为全尺寸图像。其在训练期间的目标仅是生成判别器分类为真实数据的输出 $G(z)$。

### 判别器网络 (D)

判别器充当二元分类器。它的输入是一个数据样本 $x$（可以来自 $p_{data}$ 的真实样本，也可以是来自 $p_g$ 的生成样本 $G(z)$），其输出 $D(x)$ 是一个单一标量，表示 $x$ 来自真实数据分布 $p_{data}$ 的概率。

$D: \mathcal{X} \rightarrow [0, 1]$

理想情况下，对于真实样本 $D(x)$ 应接近 1，对于生成样本则应接近 0。对于图像数据，判别器通常以标准卷积神经网络 (neural network) (CNN) 的形式实现，输出一个概率。其在训练期间的目标是正确辨别真实样本和生成样本。

### 对抗性最小-最大博弈

训练过程使 $G$ 和 $D$ 在一个零和博弈中相互对抗。这种相互作用由前面介绍的价值函数 $V(D, G)$ 描述：

$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$

我们来分解一下：

1. **最大化 $D$**：判别器 $D$ 希望最大化 $V(D, G)$。它通过以下方式实现：

   - 为真实样本 $x \sim p_{data}(x)$ 分配高概率（接近 1），从而最大化第一项 $\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)]$。
   - 为生成样本 $G(z)$ 分配低概率（接近 0），这会最大化第二项 $\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$，因为 $1 - D(G(z))$ 将接近 1。
     本质上，$D$ 通过最大化标准二元交叉熵损失，准确地辨别真实样本与虚假样本。
2. **最小化 $G$**：生成器 $G$ 希望通过让 $D$ 在生成样本上的表现不佳来*最小化* $V(D, G)$。由于 $G$ 只影响第二项，它试图使 $D(G(z))$ 尽可能接近 1（从而欺骗判别器）。这会使 $\log(1 - D(G(z)))$ 最小化，当 $D(G(z)) \to 1$ 时，它趋向于 $-\infty$。

这种最小-最大公式确立了一个平衡点。从理论上讲，如果 $G$ 和 $D$ 都具有足够的能力，并且训练过程收敛到最佳状态，生成器的分布 $p_g$ 将与真实数据分布 $p_{data}$ 完全吻合。此时，判别器无法比随机猜测更有效地辨别真实样本与生成样本，这意味着对于所有 $x$，都有 $D(x) = 0.5$。价值函数 $V(D, G)$ 收敛到 $-\log 4$。

> 生成对抗网络 (GAN)的基本架构，展示了随机噪声和真实数据通过生成器和判别器的路径。

### 训练动态与实现细节

在实践中，使用标准梯度下降 (gradient descent)同时训练 $G$ 和 $D$ 会不稳定。相反，训练会在更新 $D$ 和 $G$ 之间交替进行：

1. **更新判别器**：采样一个噪声向量 (vector)的小批量 $\{z^{(1)}, ..., z^{(m)}\}$ 和一个真实数据样本的小批量 $\{x^{(1)}, ..., x^{(m)}\}$。通过沿 $V(D, G)$ 的随机梯度上升来更新 $D$ 的参数 (parameter)：
   $\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m [\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))]$
   此步骤可能会重复 $k$ 次迭代，以确保 $D$ 保持高效。
2. **更新生成器**：采样一个噪声向量的小批量 $\{z^{(1)}, ..., z^{(m)}\}$。通过沿 $V(D, G)$ 的随机梯度下降来更新 $G$ 的参数，特别是针对第二项：
   $\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log(1 - D(G(z^{(i)})))$

生成器的损失函数 (loss function) $\log(1 - D(G(z)))$ 带来一个重要的实际问题。当判别器在训练早期变得非常高效时，它会正确地将 $D(G(z)) \approx 0$ 分配给生成样本。在这个区域，$\log(1 - x)$ 关于 $x$ 的梯度非常小（饱和），为生成器提供了微弱的训练信号。

为应对这种饱和现象，一个常见更改是将生成器的目标从最小化 $\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$ 变为最大化 $\mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]$。这通常被称为“非饱和”启发式目标。虽然它并非精确地表示原始的最小-最大博弈，但其目的相同（使 $D(G(z))$ 接近 1），且在训练早期当 $D(G(z))$ 很小时，它提供了更强的梯度。在实践中，这意味着通过梯度上升来更新 $G$：
$\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log D(G(z^{(i)}))$

GAN 的基本框架设计精巧，但要实现稳定的训练和高质量的成果，需要仔细考量架构选择、损失函数变体和优化策略。诸如训练不稳定（振荡或发散）和模式崩溃（生成器只产生有限种类的样本）等困难很常见。这些挑战促使我们将在后续章节中介绍更先进的技术和架构。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS 2014); DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍生成对抗网络框架的原始科学文章。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) — Alec Radford, Luke Metz, Soumith Chintala (2015)
  Journal: International Conference on Learning Representations (ICLR 2016); DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  描述使用深度卷积网络（包括转置卷积）实现稳定GAN训练的架构准则。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 679-708 (Chapter 20)
  这本综合教材的第20章详细解释了GANs。
