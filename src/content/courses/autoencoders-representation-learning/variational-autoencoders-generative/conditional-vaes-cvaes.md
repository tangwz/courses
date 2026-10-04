---
course: "autoencoders-representation-learning"
chapter: "variational-autoencoders-generative"
lesson: "conditional-vaes-cvaes"
sourceId: 2864
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-4-variational-autoencoders-generative/conditional-vaes-cvaes"
title: "条件变分自编码器 (CVAE)"
description: "拓展VAE，使其能根据特定属性或标签生成数据。"
order: 8
plots: []
sourceHash: "a38668392da264252d073a5965fbbcb56abab533202aa42f4f091045bf3cfb5e"
sourceCorrections: []
---

标准变分自编码器提供了一个有效的框架，用于学习潜在表征并通过从先验分布 $p(z)$ 中采样，然后将样本通过解码器 $p_\theta(x|z)$ 生成新的数据点。然而，这种生成过程通常是无条件的。我们采样一个 $z$ 得到一个 $x$，但我们无法对生成的 $x$ *是何种类型* 进行细致的控制。想象一下在手写数字图片（0-9）上训练一个VAE。虽然它可能生成看起来逼真的数字，但我们无法直接要求它只生成，例如，数字‘7’。

条件变分自编码器 (CVAE) 拓展了VAE框架，通过在建模过程中引入条件信息（通常表示为 $y$）来解决这一局限。这个变量 $y$ 可以代表标签、属性或任何与数据 $x$ 相关的辅助信息。通过对编码器和解码器都施加 $y$ 条件，CVAE使我们能够控制生成过程。

### 条件化生成过程

CVAE背后的主要思想是使推断（编码）和生成（解码）过程都依赖于条件变量 $y$。

1. **条件编码器**：编码器的作用从近似后验分布 $p(z|x)$ 转变为近似条件后验分布 $p(z|x, y)$。其输出分布表示为 $q_\phi(z|x, y)$。这意味着潜在表征 $z$ 现在捕获了 $x$ 中针对给定条件 $y$ 的特有变化。
2. **条件解码器**：类似地，解码器学习生成数据 $x$ 不仅仅依赖于潜在变量 $z$，还基于条件 $y$。其分布表示为 $p_\theta(x|z, y)$。

### CVAE 目标函数

CVAE的目标函数以与标准VAE类似的方式得出，但纳入了条件 $y$。我们的目标是最大化条件对数似然 $\log p_\theta(x|y)$。相应的证据下界 (ELBO) 变为：


$$
\mathcal{L}_{\text{CVAE}}(x, y; \theta, \phi) = \mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - D_{KL}(q_\phi(z|x, y) || p(z|y))
$$


我们来拆解这个目标：

- **条件重建损失**：第一项 $\mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)]$ 衡量了在给定从条件编码器分布中采样的潜在变量 $z$ *以及* 条件 $y$ 的情况下，解码器重建原始输入 $x$ 的效果如何。与标准VAE一样，这通常使用均方误差（针对实数值数据）或二元交叉熵（针对二元数据）来实现，计算对象是原始 $x$ 和从 $z$ 和 $y$ 生成的重建 $\hat{x}$。
- **条件KL散度**：第二项 $D_{KL}(q_\phi(z|x, y) || p(z|y))$ 作为正则化 (regularization)项。它促使条件编码器产生的分布 $q_\phi(z|x, y)$ 接近条件先验分布 $p(z|y)$。

一个常见简化是假设潜在变量的先验分布与条件 $y$ 独立，即 $p(z|y) = p(z)$。在许多实际应用中，$p(z)$ 常被选为标准多元高斯分布 $\mathcal{N}(0, I)$。在此假设下，KL散度项变为 $D_{KL}(q_\phi(z|x, y) || p(z))$。ELBO简化为：


$$
\mathcal{L}_{\text{CVAE}}(x, y; \theta, \phi) = \mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - D_{KL}(q_\phi(z|x, y) || p(z))
$$


最大化此ELBO会训练编码器和解码器网络（$\phi$ 和 $\theta$），使其能够准确重建输入，同时确保条件潜在空间结构与简单的先验 $p(z)$ 对齐 (alignment)。

### 架构实现

将条件 $y$ 整合到编码器和解码器的神经网络 (neural network)中通常是直接的。如果 $y$ 是类别型（例如数字标签），它通常被转换为one-hot向量 (vector)或嵌入 (embedding)向量。然后，这个 $y$ 的向量表示会与各自网络的其他输入进行拼接：

- **编码器输入**：输入 $x$ 和条件向量 $y$ 被组合（例如，拼接）并输入到编码器网络。
- **解码器输入**：潜在变量样本 $z$ 和条件向量 $y$ 被组合（例如，拼接）并输入到解码器网络。

以下图表展示了CVAE中的数据流：

> 条件变分自编码器中的数据流。条件 `y` 作为输入提供给编码器和解码器网络，实现了受控的生成和表征学习。

### 生成条件样本

CVAE训练完成后，生成对应于特定条件 $y$ 的样本 $x$ 是直接的：

1. 选择所需条件 $y$。将其转换为训练期间使用的适当向量 (vector)格式。
2. 从先验分布 $p(z)$ 中采样一个潜在向量 $z$（例如，$\mathcal{N}(0, I)$）。
3. 组合 $z$ 和 $y$，并将它们输入到训练好的解码器网络 $p_\theta(x|z, y)$。
4. 解码器的输出是基于 $y$ 条件生成的样本 $\hat{x}$。

例如，使用在MNIST上训练的CVAE，你可以通过向解码器提供数字‘3’的one-hot向量作为 $y$ 以及一个随机样本 $z$ 来生成数字‘3’的图像。通过在保持 $y$ 不变的同时改变 $z$，你可以生成数字‘3’的不同风格变化。

### 应用场景

CVAE在多个方面为受控生成提供了可能性：

- **图像合成**：生成具有特定属性的图像（例如，戴眼镜的人脸、特定服装风格、特定物体类别）。
- **文本生成**：生成具有受控情感、主题或风格的文本序列。
- **音乐生成**：创作符合特定流派或情绪的音乐作品。
- **数据增强**：为数据集中特定代表性不足的类别生成合成数据。

通过允许外部信息引导生成过程，CVAE对于需要目标输出合成的任务来说，相较于标准VAE提供了显著提升。它们是朝着构建更可控和多功能生成模型迈出的重要一步。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  提出了变分自编码器模型，CVAE是其扩展。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本介绍变分自编码器及相关生成模型的综合性教材。
- [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1606.05908) — Carl Doersch (2016)
  Journal: arXiv preprint arXiv:1606.05908; DOI: [10.48550/arXiv.1606.05908](https://doi.org/10.48550/arXiv.1606.05908)
  一篇关于变分自编码器及其扩展（包括CVAE）的清晰教程。
