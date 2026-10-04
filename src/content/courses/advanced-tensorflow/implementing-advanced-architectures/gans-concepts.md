---
course: "advanced-tensorflow"
chapter: "implementing-advanced-architectures"
lesson: "gans-concepts"
sourceId: 2947
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-7-implementing-advanced-architectures/gans-concepts"
title: "生成对抗网络 (GANs) 原理"
description: "理解 GAN 的理论基础，包括生成器、判别器和损失函数。"
order: 3
plots: []
sourceHash: "7f77871438126cf1f743e8c6dedd6140ee6efb10e4080050aa4dce0e8dd2b24b"
sourceCorrections: []
---

生成对抗网络 (GAN) (GANs) 是生成式建模领域的一项重要进展，使机器能够生成逼真的数据，如图像、文本或音乐。由 Ian Goodfellow 及其同事于 2014 年提出，GANs 采用一种独特的对抗训练过程，涉及两个相互竞争的神经网络 (neural network)。理解其核心思想对于实现这些强大模型非常重要。

### 对抗框架

核心是，GAN 基于两个独立网络之间的零和博弈进行运作：

1. **生成器 (G):** 这个网络试图生成合成数据，以模拟目标数据分布。它接收一个随机噪声向量 (vector)作为输入，该向量通常从高斯分布或均匀分布之类的简单分布中抽取（常称作*潜在空间*，用 $z$ 表示），并将其转换为一个与真实数据相似的数据样本（例如图像）。生成器的目标是变得足够擅长生成能够欺骗第二个网络的假样本。可以将其比作一个艺术品伪造者，试图制作出逼真的赝品。
2. **判别器 (D):** 这个网络充当分类器。它接收来自训练数据集的真实数据样本，或者由生成器生成的假样本。它的目标是准确判断输入样本是真实的（来自真实数据分布）还是合成的（由 G 生成）。继续这个比喻，判别器就像一个艺术评论家或侦探，试图区分真品和赝品。

> GAN 架构的基本图示，显示了生成器从潜在向量创建数据以及判别器对真实数据和生成数据进行分类之间的互动。

### 训练过程：最小最大博弈

训练 GAN 涉及一个迭代交替过程，其中生成器和判别器以对抗方式进行训练：

1. **训练判别器：** 在固定步数（通常只有一步）内，判别器被训练以提高其分类准确性。它会获得一批包含真实样本（来自训练集）和当前生成器生成的假样本的数据。判别器的权重 (weight)通过反向传播 (backpropagation)进行更新，依据是它能否正确地将真实样本标记 (token)为真实，将假样本标记为假。它的目标是最大化其分类表现。
2. **训练生成器：** 接下来，生成器被训练。在此阶段，判别器的权重保持不变。生成器使用新的随机潜在向量 (vector)生成一批假样本。这些假样本被输入到（冻结的）判别器中。生成器的权重随后根据其生成样本欺骗判别器的程度进行更新（即判别器对假样本的预测值与“真实”有多接近）。生成器的目标是*最小化*判别器检测其假样本的能力，从而有效地最大化其生成样本被判别器分类为真实的概率。

这种来回训练持续进行，理想情况下会达到一种平衡，即生成器生成的样本与真实数据无法区分，判别器被迫随机猜测（对真实/伪造输出 0.5 的概率）。

### 损失函数 (loss function)

对抗训练过程通过数学形式化，使用价值函数 $V(D, G)$ 表示一个最小最大博弈：


$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{数据}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]
$$


我们来分析一下：

- $p_{\text{数据}}(x)$ 是真实数据的分布。
- $p_{z}(z)$ 是输入噪声（潜在变量）的分布。
- $G(z)$ 是给定噪声 $z$ 时生成器的输出。
- $D(x)$ 是判别器判断输入 $x$ 为真实的（非伪造）的输出概率。
- $\mathbb{E}$ 表示期望值（样本上的平均）。

判别器 $D$ 希望*最大化*这个价值函数。它旨在使真实样本 $x$ 的 $D(x)$ 接近 1（最大化 $\log D(x)$），并使假样本 $G(z)$ 的 $D(G(z))$ 接近 0（最大化 $\log(1 - D(G(z)))$）。

生成器 $G$ 希望*最小化*这个价值函数。由于 $G$ 只影响第二项，它试图使 $D(G(z))$ 接近 1（欺骗判别器），这会最小化 $\log(1 - D(G(z)))$。

**实际考量：非饱和损失**

在实践中，当判别器容易拒绝糟糕的初始伪造样本时（即 $D(G(z))$ 接近 0），最小化 $\log(1 - D(G(z)))$ 可能导致生成器在训练早期出现梯度消失。$\log(1 - x)$ 的梯度在 $x=0$ 附近是平坦的。

一个常见修改是，将生成器的目标改为最大化 $\log D(G(z))$：


$$
\max_{G} \mathbb{E}_{z \sim p_{z}(z)}[\log D(G(z))]
$$


这个“非饱和”目标在训练早期提供更强的梯度，同时仍鼓励生成器生成判别器会分类为真实的样本。这是最常采用的目标。

### 常见挑战

虽然功能强大，但 GAN 的训练以具有挑战性和不稳定著称。一些常见问题包括：

- **模式崩溃：** 生成器学习只生成几种类型的输出，甚至单一高度逼真的输出，而不是捕捉训练数据分布的全部多样性。它会找到一些能很好地欺骗判别器的“模式”，并停止进一步生成其他模式。
- **训练不稳定：** 生成器和判别器之间精妙的平衡难以维持。训练可能发散，或者损失值可能震荡而无法有效收敛。通常需要仔细调整超参数 (parameter) (hyperparameter)、网络架构和优化算法。
- **评估困难：** 定量评估生成样本的质量和多样性仍然是一个未解决的研究问题。Inception Score (IS) 和 Fréchet Inception Distance (FID) 等指标常用于图像，但它们有局限性。

尽管存在这些挑战，GANs 在生成高保真图像和其他数据类型方面取得了卓越的成就。了解其核心对抗机制、训练动态和潜在困难，为在 TensorFlow 中实现和实践这些高级模型提供了前提，我们将在接下来进行介绍。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: arXiv preprint arXiv:1406.2661; Publisher: arXiv; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇原始论文介绍了生成对抗网络，概述了其架构、对抗性训练和最小最大目标函数。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; Pages: 2816-2826; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  这项研究引入了Fréchet Inception Distance (FID)，这是一种广泛使用的指标，用于客观评估GANs生成图像的质量和多样性。
- [A Review of Generative Adversarial Networks: Algorithms, Theory, and Applications](https://www.sciencedirect.com/science/article/pii/S095219762100523X) — Yongxin Liu, Qingfeng Lan, Xiaoyong Li, Yipeng Wang, Lingfang Wu, Zhihong Li, Guandong Xu (2021)
  Journal: Engineering Applications of Artificial Intelligence; Publisher: Elsevier; Volume: 105; Pages: 104381; DOI: [10.1016/j.engappai.2021.104381](https://doi.org/10.1016/j.engappai.2021.104381)
  这篇综述回顾了各种GAN算法、其理论方面，并讨论了该领域常见的训练困难和进展。
