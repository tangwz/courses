# 原版GAN的局限性

来源：[原文](https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-1-gan-foundations-revisited/limitations-vanilla-gans)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然最初的生成对抗网络 (GAN)理念为生成模型引入了开创性的方法，但直接基于极大极小目标函数的“原版”实现很快展现出一些明显的实际局限。这些难题推动了许多研究，从而产生了我们将在本课程中介绍的高级技术。了解这些不足对于理解更复杂架构和训练策略的背后原因非常重要。

### 训练过程的脆弱性

GAN的基本思想是生成器（$G$）和判别器（$D$）之间的对抗性博弈。其原始目标函数源自真实数据分布（$p_{data}$）与生成数据分布（$p_g$）之间的詹森-香农（JS）散度，理论上设计优美，但在实际应用中迅速暴露出多种明显的局限性。这些挑战推动了后续许多先进技术的研究和发展。认识这些不足有助于理解更复杂架构和训练策略背后的原因。


$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$


然而，在实践中训练这种极大极小博弈出了名的困难。

1. **梯度消失：** 在训练初期，如果判别器过快地变得过于擅长，它可以轻松区分真实样本和虚假样本。对于生成样本，其输出 $D(G(\mathbf{z}))$ 将接近 0。生成器的梯度主要来自 $\log(1 - D(G(\mathbf{z})))$ 项。当 $D(G(\mathbf{z}))$ 接近 0 时，$\log(1-x)$ 函数相对平坦，导致传递回生成器的梯度非常小。生成器学习速度极慢，甚至根本不学。相反，如果生成器变得过于优秀，判别器就会陷入困境，可能导致反方向的不稳定。
2. **不收敛：** 同时优化过程无法保证像典型最小化问题那样收敛。找到这个博弈的纳什均衡很有挑战。训练通常涉及振荡，两者都没有明确“胜出”，生成的样本质量会剧烈波动或无法持续提升。实现稳定训练所需的微妙平衡，需要细致的超参数 (parameter) (hyperparameter)调整，对于原版GAN来说，这往往更像是一门艺术，而不是一门科学。

### 模式崩溃：一个持续的问题

也许最常被提及的局限性是**模式崩溃**。当生成器找到少数几种特定输出（模式）特别擅长欺骗当前判别器时，就会发生这种情况。生成器不是学习表示训练数据分布的全部多样性，而是只生成这些有限的变体。

想象一下，在手写数字（0-9）数据集上训练GAN。模式崩溃可能表现为生成器只生成可信的“1”和“7”的图像，完全忽略了其他数字。虽然这些生成的样本单独看起来可能很真实，但生成器未能捕捉到底层数据分布。原版GAN的目标函数没有内在地惩罚这种多样性的缺乏，尤其是在判别器可以轻易被这少数几种模式欺骗的情况下。

### 衡量进展的困难

原版GAN一个重要的实际问题是缺乏可靠的指标来跟踪训练进展。生成器和判别器的损失，源自极大极小目标，在训练期间经常波动，并且与生成样本的感知质量或多样性不一致地相关。

- 生成器损失下降不总是意味着更好的图像；它可能只意味着判别器当前表现不佳。
- 看起来稳定的损失不能保证收敛到一个好的解决方案；模型可能陷入一种无益的平衡。

这使得仅凭损失值难以判断何时停止训练，或者不同超参数 (parameter) (hyperparameter)或架构改进的效果如何。目视检查成为主要工具，但这是主观且耗时的。这一局限突显了对更多评估指标的需求，例如Inception Score (IS)和Fréchet Inception Distance (FID)，我们将在第5章讨论。

### 生成控制有限

标准GAN框架从随机噪声向量 (vector)$\mathbf{z}$生成样本。虽然不同的输入向量$\mathbf{z}$会产生不同的输出，但没有直接的方法来控制*哪些*具体特征或类型的输出被生成。从潜在空间到数据空间的映射是复杂且常纠缠不清的，这意味着改变$\mathbf{z}$中的一个维度可能影响输出图像中多个不相关的特征。这种缺乏直接控制的情况推动了条件GAN (cGANs) 和学习解耦表示的方法的开发，这些将在第4章中介绍。

这些局限性——训练不稳定、模式崩溃、进展指标不佳以及缺乏控制——表明，尽管GAN的核心思想强大，但初始形式需要显著的改进。后续章节将探讨为解决这些问题而开发出的解决方案，从而产生了当今更稳定、更有能力、更可控的GAN。

## 参考资料

- [Generative Adversarial Networks](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: MIT Press; Volume: 27; Pages: 2672-2680
  提出了生成对抗网络 (GAN) 框架及其原始目标函数。
- [Wasserstein GAN](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) — Martin Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Pages: 214-223; DOI: [10.5555/3305381.3305483](https://doi.org/10.5555/3305381.3305483)
  提出了 Wasserstein GAN (WGAN)，旨在解决 GAN 训练不稳定和模式崩溃问题。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFrUIkHXLJn_MIfR0QPgSy9JjgrGOshiUO1AGYO1h64ZCnuCI25wcVrnMsEQ5CELldcjS4NTz0ud3PrrY6pO9GLIEG0368CKhGfnv599VGjDdg3ru4IwzII6Eo=) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; Pages: 6626-6637
  提出了 Fréchet Inception Distance (FID) 用于评估 GAN 性能和训练进展。
- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — Mehdi Mirza, Simon Osindero (2014)
  Journal: arXiv preprint arXiv:1411.1784; DOI: [10.48550/arXiv.1411.1784](https://doi.org/10.48550/arXiv.1411.1784)
  提出了条件生成对抗网络 (cGAN)，用于生成具有特定属性或条件的样本。

---

[上一节](03-%E5%B8%B8%E8%A7%81%E7%9A%84%E8%AE%AD%E7%BB%83%E4%B8%8D%E7%A8%B3%E5%AE%9A%E7%8E%B0%E8%B1%A1.md) · [下一节](05-%E6%B7%B1%E5%BA%A6%E5%8D%B7%E7%A7%AFGANs%20%28DCGANs%29%20%E5%9B%9E%E9%A1%BE.md)
