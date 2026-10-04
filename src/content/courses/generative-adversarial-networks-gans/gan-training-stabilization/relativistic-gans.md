---
course: "generative-adversarial-networks-gans"
chapter: "gan-training-stabilization"
lesson: "relativistic-gans"
sourceId: 2703
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/relativistic-gans"
title: "相对论生成对抗网络"
description: "了解考量真实与虚假数据之间关系的相对损失函数。"
order: 8
plots: []
sourceHash: "3b1adda7da64cf4e7412715c81920531c9eb0a5bfb32f8e9f05c2e15a275d499"
sourceCorrections: []
---

使GAN训练更稳定可以通过修改损失函数 (loss function)的基本距离度量（例如在WGAN中）或采用正则化 (regularization)技术（例如谱归一化 (normalization)）来实现。相对论生成对抗网络 (GAN)提出了一种不同的方法来提升稳定性，它从根本上改变了判别器需要预测的内容。

在标准GAN的表达中，判别器$D$试图估计给定输入$x$为真实的绝对概率。其输出$D(x)$通常被解读为$P(x \text{ 为真实})$。生成器$G$则被训练来生成样本$G(z)$，使其最大化这个概率$D(G(z))$。

相对论GANs认为，判别器预测给定真实数据比随机采样的虚假数据更具真实性的*相对*概率，可能更有效且更稳定。判别器的任务不再是输出一个绝对分数，而是变为比较性的。

设$C(x)$表示判别器在最终激活函数 (activation function)（例如sigmoid）*之前*的输出。在标准GAN (SGAN) 中，判别器损失包含形如$\log(\sigma(C(x_{real})))$和$\log(1 - \sigma(C(x_{fake})))$的项，其中$\sigma$是sigmoid函数。

### 相对平均生成对抗网络 (GAN) (RaGAN)

相对平均生成对抗网络 (RaGAN) 是一个特别有效的变体。RaGAN并非比较单个真实样本与单个虚假样本，而是将一个样本（真实或虚假）与其对立分布中样本的*平均*评估进行比较。

其核心思想在RaSGAN（相对平均标准GAN）的损失函数 (loss function)中得到了形式化。判别器$D$被训练来最大化：


$$
L_D^{RaSGAN} = - E_{x_{real} \sim P_{real}}[\log(\sigma_{real})] - E_{x_{fake} \sim P_{fake}}[\log(1 - \sigma_{fake})]
$$


其中:

- $\sigma_{real} = \sigma(C(x_{real}) - E_{x_{fake} \sim P_{fake}}[C(x_{fake})])$
- $\sigma_{fake} = \sigma(C(x_{fake}) - E_{x_{real} \sim P_{real}}[C(x_{real})])$

这里，$E_{x_{fake} \sim P_{fake}}[C(x_{fake})]$是批次中虚假样本的判别器平均输出，而$E_{x_{real} \sim P_{real}}[C(x_{real})]$是批次中真实样本的判别器平均输出。判别器正在学习使$C(x_{real})$大于平均$C(x_{fake})$，并使$C(x_{fake})$小于平均$C(x_{real})$。

生成器$G$被训练来最小化其*相反*的目标：


$$
L_G^{RaSGAN} = - E_{x_{fake} \sim P_{fake}}[\log(\sigma_{fake})] - E_{x_{real} \sim P_{real}}[\log(1 - \sigma_{real})]
$$


请留意这种对称性。生成器既能从增加其生成样本相对于平均真实样本的感知真实性（$\sigma_{fake}$）中获益，也能从降低真实样本相对于平均虚假样本的感知真实性（$1 - \sigma_{real}$）中获益。这种结构为生成器提供了基于真实和虚假样本的梯度，这有助于更稳定的学习。

### 相对论生成对抗网络 (GAN)的优点

1. **更高的稳定性：** 通过将真实和虚假批次之间的比较直接纳入损失函数 (loss function)，RaGAN与标准GAN目标相比，通常能带来更稳定的训练动态，减少如模式崩溃等问题。
2. **更快的收敛：** 实验结果表明，RaGAN比标准GANs甚至有时比WGAN-GP能更快收敛。
3. **更高的样本质量：** 这种相对比较能更有效地引导生成器，可能带来更高视觉逼真度的生成样本。

### 实现方面的考量

实现RaGAN涉及修改损失计算：

1. 计算真实批次的判别器输出$C(x_{real})$和虚假批次的判别器输出$C(x_{fake})$。
2. 计算这些输出在其各自批次中的平均值：$\text{真实平均值} = \text{平均}(C(x_{real}))$ 和 $\text{虚假平均值} = \text{平均}(C(x_{fake}))$。
3. 计算相对论差异：$C(x_{real}) - \text{虚假平均值}$ 和 $C(x_{fake}) - \text{真实平均值}$。
4. 应用sigmoid函数，并根据上述RaSGAN公式，为判别器和生成器的更新计算二元交叉熵损失。

相对论生成对抗网络 (GAN)，特别是RaGAN，为GAN训练提供了一种不同的方法。通过将判别器的任务从绝对真实性评估转向相对真实性评估，它们提供了一种实用方式来达到更稳定有效的训练，为构建高级生成模型增添了又一项有价值的技术。虽然WGAN-GP或谱归一化 (normalization)等技术通过距离度量或正则化 (regularization)来处理稳定性问题，但RaGAN改变了判别器与生成器对抗游戏的基本目标本身。

## 参考资料

- [The relativistic discriminator: a key element to make GANs work](https://arxiv.org/abs/1807.00734) — Alexia Jolicoeur-Martineau (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1807.00734](https://doi.org/10.48550/arXiv.1807.00734)
  这篇是介绍相对论GANs（RaGAN/RaSGAN）概念的开创性研究论文，详细阐述了其理论基础，并展示了在GAN稳定性与样本质量方面的实证效益。
