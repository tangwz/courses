---
course: "generative-adversarial-networks-gans"
chapter: "gan-training-stabilization"
lesson: "challenge-non-convergence"
sourceId: 2685
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/challenge-non-convergence"
title: "不收敛的难题"
description: "了解为何常见GAN训练会不稳定且无法收敛。"
order: 1
plots: ["plots/2685-0.json"]
sourceHash: "8e021bf1856019c2c690de7cfc534f5f625defc5d3f53fe228cca7cfb707beb0"
sourceCorrections: []
---

理想情况下，训练一个生成对抗网络 (GAN)涉及在生成器（$G$）和判别器（$D$）之间的极小极大博弈中找到一个稳定的纳什均衡点。在这个均衡点，生成器将学会真实数据分布，判别器将无法分辨真实样本和生成样本（对所有输入输出0.5）。然而，在实践中达成这个理论上的理想情况是出了名的困难。常见的GAN训练过程常无法平稳收敛，表现出多种影响性能的不稳定性。

这种缺乏可靠收敛的情况源于这两个竞争性神经网络 (neural network)优化动态的几个相互关联的因素。

### 极小极大优化难题

训练GAN需要同时优化两个网络的参数 (parameter)，每个网络都有自己的目标函数，且此函数依赖于另一个网络的参数。这种设置与常见监督学习 (supervised learning)有根本区别，在监督学习中，我们最小化单个损失函数 (loss function)，仅针对一组参数。

原始GAN表述的目标函数是：


$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$


在深度神经网络 (neural network)高维、非凸的参数空间中找到此目标函数的鞍点 $(\theta_G^*, \theta_D^*)$ 具有挑战性。常见的梯度下降 (gradient descent)方法是为最小化设计的，而非用于在零和博弈中找寻鞍点。交替梯度更新（更新$D$ K步，然后更新$G$一步）是一种常见启发式方法，但在这种非凸环境下，它缺乏强收敛保证。

### 梯度问题：消失与振荡

一个重要的实际问题源于用于更新生成器的梯度。考虑生成器的损失，即 $\mathcal{L}_G = \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$。如果判别器$D$在区分真实样本和虚假样本方面变得非常有效（对虚假样本$G(\mathbf{z})$输出接近0的值），项 $\log(1 - D(G(\mathbf{z})))$ 会饱和。其相对于判别器输出的梯度会变得非常小。

当$D(G(\mathbf{z}))$接近0时，梯度 $\nabla_{\theta_G} \mathcal{L}_G$ 依赖于 $\nabla_{D(G(\mathbf{z}))} \log(1 - D(G(\mathbf{z})))$，而后者趋近于零。这种现象在GAN中被称为**梯度消失问题**，意味着生成器获得的信息很少，无法改进其样本，即使这些样本很容易被判别器识别为假。训练停滞，因为生成器没有有效学习。

另一方面，优化过程也可能导致**振荡行为**。生成器的更新可能会使生成分布发生偏移，从而显著增加判别器的损失。随后的判别器更新可能会过度纠正，积极地改变决策边界。这使得生成器在下一次更新时，被推向数据空间的不同部分。参数 (parameter)可能不会趋向均衡点，而是振荡，损失值也可能大幅波动，而样本质量不一定有改善。



![GAN损失振荡示意图](plots/2685-0.json)



> 图表显示生成器和判别器损失在训练期间如何振荡而非稳定收敛，表明存在不稳定性。

### 詹森-香农散度的难题

原始GAN损失函数 (loss function)，当判别器达到最优时，实际是将真实数据分布$P_{data}$与生成数据分布$P_g$之间的詹森-香农散度（JSD）降至最低。尽管JSD是衡量分布相似性的有效指标，但在GAN训练中它有一个明显的缺点。

如果两个分布$P_{data}$和$P_g$的重叠可忽略不计，或它们位于不相交的流形上（在训练早期，当生成器产生不真实样本时，这种情况很可能发生），它们之间的JSD会变成一个常数值（$\log 2$）。常数的梯度为零。这种理论观点解释了梯度消失问题：当分布差异过大时，JSD无法提供有用的梯度信号来引导生成器如何使$P_g$更接近$P_{data}$。

### 损失值与样本质量

另一个复杂之处在于，训练过程中生成器和判别器的损失值通常不能很好地表示生成样本的实际质量和多样性。你可能会观察到损失值下降，而生成器却坍缩，只产生少数几种类型的输出（模式坍塌），或者相反，即使样本质量似乎在提高，也会看到损失值波动。这种缺乏相关性使得仅仅依靠损失曲线来监测训练进程或决定何时停止训练变得困难。

这些困难，即鞍点优化的难度、梯度消失或不稳定、JSD对低重叠分布的局限性以及损失值的不可靠性，促使人们开发在后续章节中讨论的稳定化技术。弄清这些根本原因，是运用如Wasserstein损失、梯度惩罚和谱归一化 (normalization)等方法，构建更稳定、更有效GAN的第一步。

## 参考资料

- [Generative Adversarial Networks](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEliMAMDuC2eFlrFGRYZrDlpTBLJWpjP0iWPif7db2SAlgzAw0n7dmAn28vFr8D8uoEk5zAEUtasDWfDPfullh8yH91p7BYo1Af6bm7YsFRXzwoNqH_tcNBtWSYtRTT5ZyWVZugDrxS9DMVw32n-ZnzZZmP633M) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 27; Pages: 2672-2680
  介绍GAN的奠基性论文，定义了最小最大目标，并建立了与詹森-香农散度的联系，这些都是本节讨论的非收敛性的根本原因。
- [Wasserstein GAN](http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf) — Martin Arjovsky, Soumith Chintala, and Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Volume: 70; Pages: 214-223
  引入了 Wasserstein 距离作为 JSD 的替代方案，直接解决了当分布重叠可忽略时出现的梯度消失问题，这是本节详述的一个主要挑战。
- [Improved Techniques for Training GANs](https://doi.org/10.48550/arXiv.1606.03498) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems; Volume: 29; Pages: 3430-3438; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  本文讨论了 GAN 中常见的训练不稳定性，并提出了几种稳定训练的实用启发式方法，间接强调了非收敛和模式崩溃的困难。
- [On the Convergence of Adversarial Training](http://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf) — Thomas Mescheder, Andreas Geiger, and Sebastian Nowozin (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 80; Pages: 3489-3498; DOI: [10.5598/v80/mescheder18a](https://doi.org/10.5598/v80/mescheder18a)
  提供了 GAN 训练动态的理论分析，阐明了寻找鞍点的困难以及交替梯度更新时出现的振荡行为。
