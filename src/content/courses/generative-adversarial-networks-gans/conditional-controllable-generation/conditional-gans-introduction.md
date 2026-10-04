---
course: "generative-adversarial-networks-gans"
chapter: "conditional-controllable-generation"
lesson: "conditional-gans-introduction"
sourceId: 2709
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-4-conditional-controllable-generation/conditional-gans-introduction"
title: "条件式GAN（cGAN）介绍"
description: "根据特定属性或标签生成数据。"
order: 1
plots: []
sourceHash: "1f21ba196ea8754730dab4b633353d10a1fd83bc316322be2e16ab397b6182f6"
sourceCorrections: []
---

标准生成对抗网络 (GAN)学习将随机噪声向量 (vector) $z$ 映射到与目标分布 $p_{data}(x)$ 中的数据相似的样本 $x$。尽管能力强，这个过程对生成的样本的*种类*几乎没有明确的控制。如果在手写数字图像（例如MNIST）上训练一个GAN，它会生成逼真的数字，但无法简单地要求它生成一个“7”。生成的数字是采样到的 $z$ 恰好映射到的任何数字。

条件式GAN（cGAN）通过加入辅助信息来处理此限制，其扩展了基本GAN架构，这些信息通常称为条件或标签，用 $y$ 表示。条件 $y$ 为生成过程提供情境或指引，使我们能控制输出。cGAN不再学习整体数据分布 $p(x)$，而是学习条件分布 $p(x|y)$。

## 主要理念：对生成进行条件设置

cGAN的基本改变是，生成器和判别器都将条件信息 $y$ 作为额外输入接收。

- **生成器：** 生成器的任务现在是产生一个逼真的样本 $x_{fake}$，并且这个样本*符合*给定条件 $y$。它的输入不再仅仅是噪声向量 (vector) $z$，而是配对 $(z, y)$。我们将生成器函数表示为 $G(z, y)$。
- **判别器：** 判别器现在必须区分真实数据样本 $x_{real}$ 和生成样本 $x_{fake}$，但它必须*在条件 $y$ 的情境下*进行区分。它需要判断输入对 $(x, y)$ 是否是*有效*的配对，即 $x$ 逼真并与标签 $y$ 对应。真实的配对 $(x_{real}, y)$ 来自真实数据分布，而虚假的配对 $(G(z, y), y)$ 来自生成器。我们将判别器函数表示为 $D(x, y)$。

这种条件设置机制实现了定向生成。如果 $y$ 代表MNIST中的一个数字类别，向 $G(z, y)$ 提供 $y=7$ 会指示生成器专门合成一个“7”的图像。

> 标准GAN与条件式GAN（cGAN）中的数据流。cGAN将条件信息 $y$ 引入生成器和判别器。

## 修改目标函数

引入条件 $y$ 需要修改标准的GAN极小极大目标函数。该目标现在反映了生成和判别的条件特性。cGAN的价值函数 $V(D, G)$ 变为：


$$
\min_G \max_D V(D, G) = \mathbb{E}_{(x, y) \sim p_{data}(x, y)}[\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)}[\log(1 - D(G(z, y), y))]
$$


让我们分解说明：

1. $\mathbb{E}_{(x, y) \sim p_{data}(x, y)}[\log D(x, y)]$: 判别器 $D$ 旨在使这一项最大化。它试图为与正确对应条件 $y$ 配对的真实数据样本 $x$ 分配高概率（接近1）。期望值是针对从联合数据分布 $p_{data}(x, y)$ 中提取的真实数据样本 $x$ 及其相关条件 $y$ 计算的。
2. $\mathbb{E}_{z \sim p_z(z), y \sim p_y(y)}[\log(1 - D(G(z, y), y))]$: 生成器 $G$ 旨在使这一项最小化（而 $D$ 则通过使 $D(G(z,y), y)$ 变小来最大化它）。 $G$ 接收随机噪声 $z$ 和条件 $y$（从其分布 $p_y(y)$ 中采样，通常来自训练数据标签）以生成一个虚假样本 $G(z, y)$。判别器 $D$ 随后评估这个生成样本*及其预设条件* $y$。 $D$ 试图为这些虚假配对分配低概率（接近0），而 $G$ 则试图欺骗 $D$ 分配高概率。

本质上，极小极大博弈依然存在，但现在它是在条件分布上进行的。判别器学习对 $p(x|y)$ 进行建模，区分真实条件样本和虚假样本。生成器学习生成与*给定*条件 $y$ 的真实样本无法区分的样本。

## 条件信息如何被引入？

将 $y$ 输入到 $G$ 和 $D$ 的具体机制取决于 $y$ 的性质和网络架构。常见的方法，我们将在“cGAN的架构”一节中进一步阐述，包括：

- **拼接：** 如果 $y$ 是一个简单向量 (vector)（例如，独热编码的类别标签或嵌入 (embedding)），它可以直接与生成器的噪声向量 $z$ 拼接，或者与判别器的输入 $x$（或中间特征图）拼接。
- **嵌入层：** 类别标签 $y$ 通常会通过嵌入层来获得密集向量 (dense vector)表示，然后再与 $z$ 或 $x$ 结合。
- **调制技术：** 更高级的架构可能会使用 $y$ 来调制网络中的权重 (weight)或激活（例如，条件批归一化 (normalization)）。

## 为什么要使用cGAN？

条件式GAN为控制生成过程提供了一种有效方式。你可以明确要求特定类型的输出，而不是从学习到的分布中随机采样。这在多种应用中非常有益：

- **类别条件图像合成：** 生成属于特定类别的图像（例如，生成一张“暹罗猫”的图片，而不仅仅是任何猫）。
- **文本到图像合成：** 根据文本描述生成图像（例如，StackGAN，本章后续会介绍）。这里，$y$ 是输入文本的向量 (vector)表示。
- **图像到图像转换：** 根据目标域标签或示例将图像从一个域转换到另一个域（例如，将草图转换为照片）。
- **属性操作：** 修改图像的特定属性（例如，在人脸图像上添加眼镜）。

通过对生成进行条件设置，cGAN从简单的模仿转向可控和定向的合成，这为生成建模带来了更广阔的实际应用范围。接下来的部分将详细介绍具体的架构和相关技术，例如InfoGAN，它旨在自动学习有意义的条件因子。

## 参考资料

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Mehdi Mirza and Simon Osindero (2014)
  Journal: arXiv preprint arXiv:1411.1784; DOI: [10.48550/arXiv.1411.1784](https://doi.org/10.48550/arXiv.1411.1784)
  这篇原始论文介绍了条件生成对抗网络（cGAN）框架。它详细阐述了将条件信息整合到GAN中的架构修改和目标函数。
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; DOI: [10.1109/CVPR.2017.74](https://doi.org/10.1109/CVPR.2017.74)
  这篇有影响力的论文展示了cGAN在各种图像到图像转换任务中的多功能性，例如将草图转换为照片或日景转换为夜景。
