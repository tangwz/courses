# 推导证据下界（ELBO）

来源：[原文](https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-4-variational-autoencoders-generative/evidence-lower-bound-elbo)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在前几节中，我们已经确定变分自编码器（VAE）从概率角度处理生成模型。VAE编码器（由$\phi$参数 (parameter)化）不是将输入$x$映射到潜在空间中的单个点$z$，而是输出潜在空间上概率分布$q_\phi(z|x)$的参数。解码器（由$\theta$参数化）随后定义了一个似然$p_\theta(x|z)$，代表从给定潜在变量$z$生成$x$的概率。

我们在训练生成模型时的最终目的是使模型观测到实际数据$x$的概率（或似然）最大化。这个边际似然通过对所有可能潜在变量$z$进行积分得到：


$$
p_\theta(x) = \int p_\theta(x|z) p(z) dz
$$


此处，$p(z)$是为潜在变量选择的先验分布，通常是标准多元高斯分布（$N(0, I)$）。遗憾的是，对于像用作解码器的神经网络 (neural network)这样的复杂模型，计算此积分通常是难以处理的，因为它需要对$z$的所有可能值进行积分，而$z$可以是高维的。

### 介绍变分方法

由于我们无法直接优化$p_\theta(x)$，VAE采用一种源自变分推断的方法。其主要思想是引入一个近似分布$q_\phi(z|x)$（我们的编码器），旨在估计真实但难以处理的后验分布$p_\theta(z|x) = p_\theta(x|z)p(z) / p_\theta(x)$。我们希望使$q_\phi(z|x)$尽可能地接近$p_\theta(z|x)$。

衡量两个分布之间“接近程度”的常用方法是 Kullback-Leibler (KL) 散度。让我们考查近似后验$q_\phi(z|x)$与真实后验$p_\theta(z|x)$之间的KL散度：


$$
KL(q_\phi(z|x) || p_\theta(z|x)) = \int q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(z|x)} dz
$$


这表示在使用$q_\phi$近似$p_\theta$时丢失的信息。我们希望最小化此KL散度。让我们使用后验$p_\theta(z|x)$的定义来展开此定义：


$$
KL(q_\phi(z|x) || p_\theta(z|x)) = E_{z \sim q_\phi(z|x)} [\log q_\phi(z|x) - \log p_\theta(z|x)]
$$


根据贝叶斯定理，$p_\theta(z|x) = \frac{p_\theta(x|z) p(z)}{p_\theta(x)}$，我们可以代入$\log p_\theta(z|x) = \log p_\theta(x|z) + \log p(z) - \log p_\theta(x)$：


$$
KL(q_\phi(z|x) || p_\theta(z|x)) = E_{z \sim q_\phi(z|x)} [\log q_\phi(z|x) - (\log p_\theta(x|z) + \log p(z) - \log p_\theta(x))]
$$


重新排列期望内部的项：


$$
KL(q_\phi(z|x) || p_\theta(z|x)) = E_{z \sim q_\phi(z|x)} [\log q_\phi(z|x) - \log p(z) - \log p_\theta(x|z) + \log p_\theta(x)]
$$


由于$\log p_\theta(x)$不依赖于$z$，期望$E_{z \sim q_\phi(z|x)}[\log p_\theta(x)]$即为$\log p_\theta(x)$。我们可以分离这些项：


$$
KL(q_\phi(z|x) || p_\theta(z|x)) = E_{z \sim q_\phi(z|x)} [\log q_\phi(z|x) - \log p(z)] - E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] + \log p_\theta(x)
$$


第一项是近似后验$q_\phi(z|x)$与先验$p(z)$之间的KL散度：$KL(q_\phi(z|x) || p(z))$。因此，我们得到：


$$
KL(q_\phi(z|x) || p_\theta(z|x)) = KL(q_\phi(z|x) || p(z)) - E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] + \log p_\theta(x)
$$


### 证据下界 (ELBO)

现在，让我们重新排列此方程，以分离出难以处理的对数似然$\log p_\theta(x)$：


$$
\log p_\theta(x) = E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - KL(q_\phi(z|x) || p(z)) + KL(q_\phi(z|x) || p_\theta(z|x))
$$


这个方程很重要。它将数据的对数似然分解为三项。请注意，KL散度$KL(q_\phi(z|x) || p_\theta(z|x))$始终是非负的（$KL \ge 0$）。因此，如果我们舍弃这一项，剩余的表达式就构成了对数似然的一个下界。这个下界被称为**证据下界（ELBO）**，通常表示为$\mathcal{L}(\theta, \phi; x)$：


$$
\mathcal{L}(\theta, \phi; x) = E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - KL(q_\phi(z|x) || p(z))
$$


由于$KL(q_\phi(z|x) || p_\theta(z|x)) \ge 0$，我们有：


$$
\log p_\theta(x) \ge \mathcal{L}(\theta, \phi; x)
$$


这种关系是VAE训练的核心。我们不直接最大化难以处理的$\log p_\theta(x)$，而是最大化其可处理的下界，即ELBO $\mathcal{L}(\theta, \phi; x)$。最大化ELBO同时达到两个目的：

1. 它将下界$\mathcal{L}$推向实际的对数似然$\log p_\theta(x)$。
2. 由于$\log p_\theta(x) - \mathcal{L}(\theta, \phi; x) = KL(q_\phi(z|x) || p_\theta(z|x))$，最大化ELBO等同于最小化近似后验$q_\phi(z|x)$与真实后验$p_\theta(z|x)$之间的KL散度。这意味着随着ELBO的增加，我们的编码器$q_\phi(z|x)$对真实后验的近似效果会更好。

> 总对数似然、ELBO和KL散度间隙之间的关系。最大化ELBO会将其向上推向总对数似然，有效地使KL间隙最小化。

### ELBO的组成部分

我们来分析构成ELBO目标函数的两项：

1. **重构项：** $E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)]$
   该项衡量的是原始输入数据$x$由解码器$p_\theta(x|z)$生成的预期对数似然，其中潜在编码$z$是从编码器$q_\phi(z|x)$提出的分布中采样的。最大化该项促使解码器学习从其潜在表示中准确地重构输入。实际上，对于图像等输入，这通常转化为最小化重构损失，例如均方误差（MSE）或二元交叉熵（BCE），具体取决于数据分布假设。
2. **正则化 (regularization)项（KL散度）：** $- KL(q_\phi(z|x) || p(z))$
   该项衡量的是编码器对给定输入$x$产生的近似后验分布$q_\phi(z|x)$与先验分布$p(z)$（例如，标准高斯分布）之间的KL散度。最大化该项（或最小化正的KL散度）促使编码器产生接近先验$p(z)$的分布$q_\phi(z|x)$。这充当正则化器，强制潜在空间采用先验的结构（例如，平滑、居中的高斯云）。这种正则化使VAE能够通过直接从先验$p(z)$中抽取$z$并将其输入解码器来生成新样本。

### 优化目标

VAE通过最大化ELBO来训练，同时优化编码器参数 (parameter)$\phi$和解码器参数$\theta$。这个目标在两者之间取得了平衡：重构项推动实现准确的数据表示，而KL散度项则强制潜在空间具有适用于生成的规整结构。

在实践中，我们通常最小化负ELBO：


$$
\text{损失}_{VAE} = - \mathcal{L}(\theta, \phi; x) = - E_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] + KL(q_\phi(z|x) || p(z))
$$


通过梯度下降 (gradient descent)（由接下来将讨论的重参数化技巧实现）最小化此损失函数 (loss function)，训练VAE以找到能产生良好重构以及与先验$p(z)$对齐 (alignment)的结构良好潜在空间的参数$\phi$和$\theta$。理解ELBO及其组成部分对于理解VAE如何学习生成数据具有重要作用。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  引入变分自编码器并详细阐述证据下界（ELBO）推导和重参数化技巧的奠基性论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  第20章提供了关于变分自编码器的全面且易懂的解释，包括ELBO的清晰推导和阐释。
- [Probabilistic Machine Learning: Advanced Topics](http://probml.github.io/book2) — Kevin P. Murphy (2023)
  Publisher: MIT Press; Pages: Chapter 23
  第23章深入探讨了变分自编码器，包括ELBO的严谨推导及其与更广泛的变分推断原理的联系。

---

[上一节](04-%E9%87%8D%E5%8F%82%E6%95%B0%E5%8C%96%E6%8A%80%E5%B7%A7%E7%9A%84%E8%A7%A3%E9%87%8A.md) · [下一节](06-KL%E6%95%A3%E5%BA%A6%E9%A1%B9%E5%88%86%E6%9E%90.md)
