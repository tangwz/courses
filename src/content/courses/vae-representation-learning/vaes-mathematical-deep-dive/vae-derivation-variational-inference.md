---
course: "vae-representation-learning"
chapter: "vaes-mathematical-deep-dive"
lesson: "vae-derivation-variational-inference"
sourceId: 6284
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/vae-derivation-variational-inference"
title: "VAE推导：变分推断"
description: "逐步推导变分自编码器，运用变分推断的原理。"
order: 1
plots: []
sourceHash: "0f702be3378bba649adb123d9075aa56e2f9946709ea009509db0e4330c987f6"
sourceCorrections: []
---

生成模型的主要目标是学习观测数据 $x$ 的潜在概率分布 $p(x)$。变分自编码器通过引入隐变量 $z$ 来达成此目的，这些隐变量是未观测变量，它们表示数据的内在结构或属性。我们假定数据 $x$ 是通过生成过程 $p_	heta(x|z)$ 由这些隐变量 $z$ 生成的，其中 $z$ 本身是从先验分布 $p(z)$ 中采样的。观测值 $x$ 的边际似然由以下公式给出：


$$
p_\theta(x) = \int p_\theta(x|z) p(z) dz
$$


我们目标是学习此生成模型（通常是一个神经网络 (neural network)，即“解码器”）的参数 (parameter) $\theta$。直接最大化 $\log p_\theta(x)$ 很困难，因为这个积分通常难以计算，尤其当 $p_\theta(x|z)$ 复杂（如深度神经网络）且 $z$ 维度很高时。

此外，为了执行将数据编码为隐表示等任务，我们经常需要访问后验分布 $p_\theta(z|x) = \frac{p_\theta(x|z)p(z)}{p_\theta(x)}$。这也难以计算，因为其分母 $p_\theta(x)$ 难以计算。

这就是变分推断的作用所在。我们不计算真实后验 $p_\theta(z|x)$，而是引入一个更简单、可计算的分布 $q_\phi(z|x)$ 对其进行近似。这个分布 $q_\phi(z|x)$ 通常被称为变分后验或推断网络（在VAE中，即“编码器”），它通常由 $\phi$ 参数化（例如，另一个神经网络的参数）。我们的目标是使 $q_\phi(z|x)$ 尽可能接近真实后验 $p_\theta(z|x)$。

我们从数据点 $x$ 的对数似然 $\log p_\theta(x)$ 开始。我们可以使用我们的近似后验 $q_\phi(z|x)$ 重写它：


$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x)]
$$


之所以成立是因为 $\log p_\theta(x)$ 不依赖于 $z$，并且 $\int q_\phi(z|x) dz = 1$。现在，我们使用条件概率的定义 $p_\theta(x) = p_\theta(x,z) / p_\theta(z|x)$：


$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{p_\theta(z|x)}\right]
$$


接下来，我们在对数内部乘以和除以 $q_\phi(z|x)$，这是一个常用技巧：


$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \left(\frac{p_\theta(x,z)}{q_\phi(z|x)} \frac{q_\phi(z|x)}{p_\theta(z|x)}\right)\right]
$$


利用性质 $\log(ab) = \log a + \log b$：


$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right] + \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]
$$


我们审视这两个项。第二项是 $q_\phi(z|x)$ 和真实后验 $p_\theta(z|x)$ 之间的Kullback-Leibler (KL) 散度：


$$
\mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right] = D_{KL}(q_\phi(z|x) || p_\theta(z|x))
$$


KL散度总是非负的 ($D_{KL} \ge 0$)，当且仅当 $q_\phi(z|x) = p_\theta(z|x)$ 时为零。

第一项定义为证据下界 (ELBO)，记作 $\mathcal{L}(\theta, \phi; x)$：


$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right]
$$


于是，我们得到了基本恒等式：


$$
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{KL}(q_\phi(z|x) || p_\theta(z|x))
$$

> 边际对数似然分解为ELBO以及近似后验与真实后验之间的KL散度。最大化ELBO是我们的可计算目标。

由于 $D_{KL}(q_\phi(z|x) || p_\theta(z|x)) \ge 0$，因此有：


$$
\log p_\theta(x) \ge \mathcal{L}(\theta, \phi; x)
$$


这就是为什么 $\mathcal{L}(\theta, \phi; x)$ 被称为“证据下界”：它提供了数据对数似然（即“证据”）的下界。通过同时最大化ELBO，并优化生成模型参数 $\theta$ 和变分参数 $\phi$，我们实际上是在：

1. 提高 $\log p_\theta(x)$ 的下界，使其更高。
2. 最小化KL散度 $D_{KL}(q_\phi(z|x) || p_\theta(z|x))$，使我们的近似 $q_\phi(z|x)$ 更接近真实后验 $p_\theta(z|x)$。如果 $q_\phi(z|x)$ 变为一个完全近似，KL散度将变为零，ELBO将等于真实的对数似然。

ELBO可以重写成一种对VAE而言通常更直观的形式。从 $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x,z) - \log q_\phi(z|x)]$ 开始并使用 $p_\theta(x,z) = p_\theta(x|z)p(z)$：


$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)p(z) - \log q_\phi(z|x)]
$$


$$
= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)]
$$


$$
= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x) - \log p(z)]
$$


第二项再次是一个KL散度：$D_{KL}(q_\phi(z|x) || p(z))$。这给我们提供了VAE中最常见的ELBO形式：


$$
\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重构项}} - \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\text{正则化项}}
$$


我们来分析这两个组成部分：

- **重构项：** $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$。此项衡量了解码器 $p_\theta(x|z)$ 能多好地重构输入数据 $x$，给定从编码器近似 $q_\phi(z|x)$ 中采样的隐变量样本 $z$。最大化此项鼓励 VAE 学习有意义的隐表示 $z$，这些表示保留足够信息以重构 $x$。例如，如果 $p_\theta(x|z)$ 是一个高斯分布，此项将变为平方误差损失。如果 $p_\theta(x|z)$ 是一个伯努利分布（用于二值数据），此项将变为二元交叉熵损失。
- **正则化 (regularization)项：** $D_{KL}(q_\phi(z|x) || p(z))$。此项充当正则化器。它衡量了近似后验分布 $q_\phi(z|x)$（由编码器对给定 $x$ 输出的）与隐变量的先验分布 $p(z)$ 之间的散度。先验 $p(z)$ 通常选择为简单的分布，例如标准多元高斯分布 $\mathcal{N}(0, I)$。此KL项鼓励编码器将隐表示 $z$ 分布得与先验类似。这种正则化对于确保隐空间结构良好且可用于生成新数据非常重要。没有它，编码器可能会学习生成 $q_\phi(z|x)$ 分布，这些分布对不同的 $x$ 彼此相距很远，从而导致非平滑或“有空隙的”隐空间。

因此，VAE 由以下部分组成：

1. 一个**编码器网络** $q_\phi(z|x)$，它接收数据 $x$ 并输出 $z$ 分布的参数（例如，如果 $q_\phi(z|x)$ 是高斯分布，则为均值和方差）。其参数为 $\phi$。
2. 一个**解码器网络** $p_\theta(x|z)$，它接收隐变量样本 $z$ 并输出 $x$ 分布的参数。其参数为 $\theta$。
3. 一个选定的隐变量**先验** $p(z)$，通常是 $\mathcal{N}(0,I)$。

训练过程包括同时优化ELBO $\mathcal{L}(\theta, \phi; x)$，并优化 $\theta$ 和 $\phi$，使用诸如随机梯度上升的技术。这个推导为VAE目标函数提供了理论依据，它平衡了数据重构与隐空间正则化。这种平衡使得VAE能够学习丰富、有结构的表示并生成新的数据样本。变分自编码器和变分推断中的“变分”一词指的是变分法中的方法，其中我们优化泛函（函数的函数），在此例中，是在由 $\phi$ 参数化的一系列分布中找到最佳的 $q_\phi(z|x)$。

在后续章节中，我们将探究这个目标的每个组成部分，优化它的实际操作（如重参数化技巧），以及这种公式表达的含义。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）框架及其目标函数，包括证据下界（ELBO）的推导和重参数化技巧。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 10; DOI: [10.1007/bpc100507](https://doi.org/10.1007/bpc100507)
  对变分推断进行了全面的数学阐述，包括ELBO的推导及其在概率模型中的应用。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  在深度生成模型的背景下介绍了变分自编码器的理论，解释了ELBO及其组成部分。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  对变分推断方法进行了广泛回顾，涵盖了其理论基础和广泛应用。
