---
course: "generative-adversarial-networks-gans"
chapter: "gan-foundations-revisited"
lesson: "minimax-objective-function"
sourceId: 2656
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-1-gan-foundations-revisited/minimax-objective-function"
title: "最小最大目标函数"
description: "理解原始GAN损失函数及其理论阐释。"
order: 2
plots: []
sourceHash: "91e477b625f8cb59d14f8091c4b4e293b58c1f0a9dcd03a0b441c13ebbf9ce10"
sourceCorrections: []
---

最小最大目标函数是驱动生成器（$G$）和判别器（$D$）对抗训练的机制。此函数以数学形式阐明了这两个网络相互竞争的目标。

### 价值函数 $V(D, G)$

GAN框架的核心通过一个价值函数 $V(D, G)$ 来体现，此函数表示一个双人最小最大博弈中的收益。目标是生成器 $G$ 试图最小化这个值，而判别器 $D$ 同时试图最大化它。Goodfellow 等人（2014）的原始GAN论文提出了以下目标：


$$
\min_G \max_D V(D, G) = E_{x \sim p_{data}(x)}[\log D(x)] + E_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]
$$


我们来拆解此表达式：

1. **$x \sim p_{data}(x)$**: 这表示 $x$ 是从真实数据分布 $p_{data}$ 中取出的样本。这些是我们希望生成器最终能够模仿的真实实例（例如，真实的人脸图像）。
2. **$z \sim p_{z}(z)$**: 这表示 $z$ 是从先验噪声分布中取出的样本，通常是像高斯分布或均匀分布这样的简单分布。这个向量 (vector) $z$ 作为生成器的输入种子。
3. **$G(z)$**: 这是生成器网络在接收噪声向量 $z$ 时的输出。它表示一个生成的，或“伪造的”，样本（例如，合成的人脸图像）。这些生成样本的分布表示为 $p_g$。
4. **$D(x)$**: 这是判别器网络在接收输入 $x$ 时的输出。它表示 $x$ 是来自真实数据 $p_{data}$ 的真实样本而非来自 $p_g$ 的伪造样本的概率。理想情况下，$D(x)$ 对于真实样本应接近1，对于伪造样本应接近0。
5. **$E[\cdot]$**: 这表示期望值，意即从指定分布中提取的所有可能样本的平均值。

### 理解各项

价值函数 $V(D, G)$ 由两个主要项组成：

- **$E_{x \sim p_{data}(x)}[\log D(x)]$**: 此项衡量判别器正确分类真实样本的能力。$D$ 希望通过对真实数据（$x$）输出接近1的值来最大化此项，使 $\log D(x)$ 接近 $\log(1) = 0$。
- **$E_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]$**: 此项衡量判别器正确识别伪造样本的能力。对于伪造样本 $G(z)$，$D$ 希望输出接近0的值。这使得 $1 - D(G(z))$ 接近1，且 $\log(1 - D(G(z)))$ 接近 $\log(1) = 0$。因此，最大化此项也对应着 $D$ 正确识别伪造样本。

### 最小最大博弈

目标函数展现了GAN训练的对抗特性：

1. **最大化 $D$ ($\max_D V(D, G)$)**: 对于一个*固定*的生成器 $G$，判别器 $D$ 被训练来最大化 $V(D, G)$。这涉及调整 $D$ 的参数 (parameter)，使其更善于区分真实样本（$D(x) \to 1$）和伪造样本（$D(G(z)) \to 0$）。这通常通过对 $D$ 的参数进行 $V(D, G)$ 的随机梯度上升来完成。
2. **最小化 $G$ ($\min_G \max_D V(D, G)$)**: 同时（或在实践中常交替进行），对于一个*固定*的判别器 $D$，生成器 $G$ 被训练来*最小化* $V(D, G)$。请注意 $G$ 只影响第二项，$E_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]$。生成器试图生成判别器判断为真实的样本 $G(z)$（$D(G(z)) \to 1$）。如果 $D(G(z))$ 接近1，那么 $\log(1 - D(G(z)))$ 就会变成一个很大的负数（接近 $-\infty$），从而从 $G$ 的角度最小化目标。这通过对 $G$ 的参数进行 $V(D, G)$ 的随机梯度下降 (gradient descent)来实现。

> GAN中最小最大博弈的流程。生成器试图从噪声（$p_z$）中生成逼真的数据（$p_g$）以迷惑判别器。判别器试图最大化其区分真实数据（$p_{data}$）和生成数据（$p_g$）的能力。目标函数驱动两个网络的参数更新。

### 理论平衡与散度的关联

从理论上看，当生成器分布与真实数据分布完全匹配时，此最小最大博弈达到全局最优，即 $p_g = p_{data}$。此时，最优判别器 $D^*$ 区分真实样本和伪造样本的能力不会超过随机猜测，意即对于所有 $x$，都有 $D^*(x) = 1/2$。价值函数此时评估为：


$$
V(D^*, G) = E_{x \sim p_{data}}[\log(1/2)] + E_{z \sim p_{z}}[\log(1 - 1/2)] = \log(1/2) + \log(1/2) = -2\log 2
$$


可以证明，对于一个固定的 $G$，最优判别器是 $D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$。将此代回目标函数，会显示出与概率分布之间差异度量的一种关联。具体来说，当 $D$ 处于最优状态时，目标函数与真实数据分布 $p_{data}$ 和生成数据分布 $p_g$ 之间的\*\*Jensen-Shannon散度（JSD）\*\*相关联：


$$
\max_D V(D, G) = V(D^*_G, G) = 2 \cdot JSD(p_{data} || p_g) - 2 \log 2
$$


JSD是衡量两个概率分布之间相似性的对称度量，且仅当 $P = Q$ 时 $JSD(P || Q) = 0$。因此，相对于 $G$ 最小化目标（在相对于 $D$ 最大化之后）等同于最小化生成器分布与真实数据分布之间的Jensen-Shannon散度。这为GAN按照此目标进行训练应能使生成器生成逼真样本提供了理论依据。

尽管优雅，但这种原始形式在训练期间面临实际挑战，尤其是在训练初期 $p_{data}$ 和 $p_g$ 分布重叠很少时。这可能导致生成器出现梯度消失，从而阻碍学习。我们将在第3章考察这些不稳定性和为缓解它们而发展出的技术。然而，在转向那些高级方法之前，理解这个根本目标函数非常重要。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125994/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media, Inc.; Pages: 525-555 (Chapter 17, 4th Edition)
  第17章“生成对抗网络”清晰地解释了最小最大目标函数、其实际实现以及GAN训练中遇到的常见挑战，适合概念和实践理解。
