---
course: "generative-adversarial-networks-gans"
chapter: "gan-training-stabilization"
lesson: "wasserstein-distance-gan"
sourceId: 2690
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/wasserstein-distance-gan"
title: "其他散度：Wasserstein 距离"
description: "介绍 Wasserstein 距离，作为 GAN 损失函数中 JS 散度的一个更可靠的选项。"
order: 3
plots: []
sourceHash: "eae1ba759f784414a93e692fd1f7508dc0110e74710b660453c7043902d754f6"
sourceCorrections: []
---

普通 GAN 的训练常遇到不稳定问题。生成器可能只产生狭窄范围的输出（模式崩溃），或者学习过程可能因梯度消失或梯度爆炸而完全偏离。这些问题的一个主要原因在于所用目标函数的特性：Jensen-Shannon（JS）散度。

尽管 JS 散度是衡量概率分布差异的有效方法，但其属性使其不适用于 GAN 的对抗训练动态，尤其是在图像等高维空间 (high-dimensional space)中。

### Jensen-Shannon 散度的问题

原始 GAN 论文指出，最小化 GAN 目标函数等同于最小化真实数据分布 $P_r$ 和生成数据分布 $P_g$ 之间的 JS 散度。但是，请思考当 $P_r$ 和 $P_g$ 的支撑集没有显著重叠，或者它们位于更高维空间 (high-dimensional space)中的低维流形上时（图像数据的一个常见情况）会发生什么。

在这种情况下，通常可以找到一个完美区分真实样本和生成样本的判别器。当判别器达到接近完美的准确率时：

1. $P_r$ 和 $P_g$ 之间的 JS 散度有效饱和在其最大值 (log 2) 上。
2. 更重要的是，损失函数 (loss function)关于生成器参数 (parameter)的梯度趋近于零。

考虑一下：如果判别器总能区分真实和虚假样本，它就几乎无法向生成器提供关于 *如何* 改进的有用信息。生成器基本上遇到了瓶颈，没有收到有意义的信号来引导其更新。这种梯度消失问题是训练失败和模式崩溃的一个主要原因，因为生成器停止了有效学习。

我们需要一种衡量 $P_r$ 和 $P_g$ 之间距离的方法，即使当分布差异很大或没有完美重叠时，也能提供更平滑、信息量更大的梯度。

### 介绍 Wasserstein 距离 (Earth Mover's Distance)

Wasserstein-1 距离登场，它常被称为 Earth Mover's Distance (EMD)。设想 $P_r$ 和 $P_g$ 是两堆不同的土（概率质量）。EMD 表示将一堆土转换为另一堆所需的最小“成本”。成本定义为移动的土量乘以移动的距离。

> Wasserstein 距离计算将质量从生成分布 $P_g$ 传输到匹配真实分布 $P_r$ 所需的最小成本。

重要的是，即使两堆土（分布）相距遥远（不相交的支撑集），移动这些土的成本也能提供一个有意义的衡量它们差异程度的指标。与 JS 散度不同，Wasserstein 距离不会突然饱和，通常提供一个更平滑的度量。

### Kantorovich-Rubinstein 对偶性

直接使用 EMD 的原始定义进行计算，需要找到一个最优的“传输计划”（即从 $P_g$ 中的每个点向 $P_r$ 中的每个点移动多少土），这在大多数有意义的情况下都是计算上无法处理的。幸运的是，Kantorovich-Rubinstein 对偶性为 Wasserstein-1 距离提供了一种替代表达：


$$
W_1(P_r, P_g) = \sup_{\|f\|_L \le 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{\tilde{x} \sim P_g}[f(\tilde{x})] \right)
$$


我们来细分一下：

- $W_1(P_r, P_g)$ 是真实分布 $P_r$ 和生成分布 $P_g$ 之间的 Wasserstein-1 距离。
- $\sup$ 表示上确界，即最小上界（类似于最大值）。
- $\|f\|_L \le 1$ 意味着上确界是针对所有 **1-Lipschitz 连续函数** $f$ 取的。如果对于其定义域中的所有 $x_1, x_2$ 都有 $|f(x_1) - f(x_2)| \le K \|x_1 - x_2\|$，则函数 $f$ 是 K-Lipschitz 的。本质上，1-Lipschitz 函数的变化率被限制在1以内。这个约束很重要。
- $\mathbb{E}_{x \sim P_r}[f(x)]$ 是当 $x$ 从真实数据分布 $P_r$ 中采样时 $f(x)$ 的期望值。
- $\mathbb{E}_{\tilde{x} \sim P_g}[f(\tilde{x})]$ 是当 $\tilde{x}$ 从生成器的分布 $P_g$ 中采样时 $f(\tilde{x})$ 的期望值（其中 $\tilde{x} = g(z)$，且 $z$ 从某个先验噪声分布 $p(z)$ 中采样）。

这种对偶形式表明，Wasserstein 距离是评估一个 1-Lipschitz 函数 $f$ 在两个分布样本上时所能达到的最大期望差值。

### Wasserstein GAN (WGAN)

WGAN 论文 (Arjovsky et al., 2017) 的见解是利用这种对偶形式来定义一个新的 GAN 目标函数。核心思想如下：

1. **用评判器取代判别器：** 不再使用判别器输出样本为真实的概率（通常通过 Sigmoid 函数得到 0 到 1 之间的值），我们使用一个网络，称为**评判器**（我们将其表示为具有参数 (parameter) $w$ 的 $f_w$），它输出一个无界的实数。
2. **训练评判器以近似上确界：** 评判器 $f_w$ 被训练成使项 $(\mathbb{E}_{x \sim P_r}[f_w(x)] - \mathbb{E}_{\tilde{x} \sim P_g}[f_w(\tilde{x})])$ 尽可能大。通过这样做，在正确条件下（具体来说，如果 $f_w$ 保持在 1-Lipschitz 函数空间内），评判器的目标值就会近似于 Wasserstein 距离 $W_1(P_r, P_g)$。
3. **训练生成器以最小化评判器的输出：** 生成器 $g_\theta$（带参数 $\theta$）被训练生成样本 $\tilde{x} = g_\theta(z)$，使得评判器的输出 $f_w(\tilde{x})$ 更大（更接近真实样本的输出），从而最小化评判器试图最大化的差异。这有效减少了估计的 Wasserstein 距离。

目标函数如下：

- **评判器损失：** 最大化 $L_{\text{Critic}} = \mathbb{E}_{x \sim P_r}[f_w(x)] - \mathbb{E}_{z \sim p(z)}[f_w(g_\theta(z))]$。在实践中，神经网络 (neural network)通过梯度下降 (gradient descent)（最小化）进行训练，所以我们通常最小化 $-L_{\text{Critic}}$。
- **生成器损失：** 最小化 $L_{\text{Generator}} = - \mathbb{E}_{z \sim p(z)}[f_w(g_\theta(z))]$。生成器试图通过生成高分（如同真实样本）的样本来欺骗评判器。

### 这种方法为何有助于稳定性

通过评判器使用 Wasserstein 距离具有以下几个优点：

1. 有意义的梯度：即使当 $P_r$ 和 $P_g$ 具有不相交的支撑集时，Wasserstein 距离也能提供非零且有用的梯度。这大幅缓解了困扰原始 GAN 的梯度消失问题。
2. 与样本质量的相关性增强：经验表明，WGAN 损失（估计的 Wasserstein 距离）在训练过程中通常与生成样本的感知质量有更好的相关性。较低的损失通常意味着更好看的图像，这在原始 GAN 损失中并非总是如此。
3. 模式崩溃风险降低：理论论证和经验结果表明，WGAN 更不容易出现模式崩溃，这很可能是因为更稳定的梯度使得生成器能够更有效地考察数据分布。

### Lipschitz 约束：一个实际的障碍

WGAN 背后的理论很大程度上依赖于评判器 $f_w$ 是（或近似于）一个 1-Lipschitz 函数。如果评判器的权重 (weight)变得过大，它可能会违反这个约束，使得 Wasserstein 距离的近似无效，并可能导致训练再次变得不稳定。

因此，WGAN 实现的一个重要部分是对评判器网络施加这个 Lipschitz 约束。原始 WGAN 论文提出了一种简单但有时存在问题的方法，称为权重裁剪。后续研究引入了更完善且通常更受欢迎的技术，例如梯度惩罚（WGAN-GP）和谱范数归一化 (normalization)。我们将在后续章节中考察这些施加 Lipschitz 约束的方法。

## 参考资料

- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Pages: 5767-5776; DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  提出了梯度惩罚作为在WGAN中执行Lipschitz约束的更有效和稳定的替代方法，取代了权重裁剪。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了深度学习的全面介绍，包括GAN的基础解释、训练问题和不同的散度度量。
