---
course: "intro-diffusion-models"
chapter: "generative-modeling-fundamentals"
lesson: "core-idea-noise-denoise"
sourceId: 5365
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-1-generative-modeling-fundamentals/core-idea-noise-denoise"
title: "核心思路：噪声与去噪"
description: "介绍扩散模型的主要理念：迭代地添加噪声并学习逆转该过程。"
order: 3
plots: []
sourceHash: "3228f17e5a09458edecb53d7863002fe282d6c144cd86ec0981c6d92d435b36f"
sourceCorrections: []
---

扩散模型在生成式 AI (Generative AI) 中提供了一种创建新数据的特定方法。其核心策略出乎意料地直观，包含一对过程：一个系统地破坏数据结构，另一个则学习撤销这种破坏。

设想你有一张清晰的高分辨率图像。这是你的起点，我们称之为 $x_0$。第一个过程，称为**前向过程**或**扩散过程**，在大量离散时间步 $t=1, 2, ..., T$ 中逐步向这张图像添加少量噪声（通常是高斯噪声）。在每个步 $t$，我们只添加足够少量的噪声，使得变化是轻微的。如果你观察这个过程展开，你会看到图像缓慢失去其特征和结构，变得越来越嘈杂。经过许多步（其中 $T$ 可能是数百或数千），得到的图像 $x_T$ 与原始的 $x_0$ 毫无相似之处。它有效地变成纯粹的、无结构的噪声，类似于从标准高斯分布中采样。这个前向过程是固定的；它不涉及任何学习。它只是一个预定义的将数据降级为噪声的机制。

真正的效果出现在第二个过程，即**逆向过程**或**去噪过程**。这里，目标是学习如何逆转加噪程序。我们从纯噪声样本 $x_T$ 开始（重要的是，我们可以轻松地从已知分布如高斯分布中采样）。模型随后尝试执行前向过程的逆操作：从 $x_T$ 开始，它迭代地预测一个噪声稍少的版本 $x_{T-1}$，然后用它来预测 $x_{T-2}$，依此类推，一直回到 $x_0$。如果模型能够成功学习这种逐步去噪程序，它就能从随机噪声生成一个看起来逼真的数据样本。

这个逆向过程是学习发生的地方。训练一个神经网络 (neural network)来预测在前向过程的每个步 $t$ 添加的噪声，给定嘈杂数据 $x_t$。更确切地说，网络通常将嘈杂数据 $x_t$ 和当前时间步 $t$ 作为输入，并输出从 $x_{t-1}$ 得到 $x_t$ 时添加的噪声成分的估计。通过减去这个预测的噪声（或用它来估计前一个状态的均值），模型可以近似从 $x_t$ 回到 $x_{t-1}$ 的转换。重复这个程序 $T$ 次，从随机噪声 $x_T$ 开始，就能生成一个新的数据样本 $x_0$。

下图展示了这种两部分结构：

> 此图显示固定的前向过程通过逐步添加噪声将数据 $x_0$ 转换为噪声 $x_T$。学习到的逆向过程从噪声 $x_T$ 开始，并在每一步使用神经网络预测并去除噪声，最终生成样本 $x_0$。

这种噪声与去噪方法与 VAEs 大相径庭，后者使用编码器-解码器结构将数据映射到潜在空间和从潜在空间映射回来；也与 GANs 不同，后者依赖于生成器和判别器相互竞争。扩散模型直接学习逆转数据破坏过程，这通常会带来稳定的训练和高质量的样本生成，解决了早期方法的一些局限。

前向过程是数学上定义明确且易于处理的。主要难题，也是神经网络发挥作用的地方，是学习逆向去噪步骤。在接下来的章节中，我们将细致地考察前向和逆向过程的精确数学表述，讨论常用的神经网络架构（如 U-Net），理解从概率框架推导出的训练目标，最后，我们将看到如何实现采样过程来生成新数据。

## 参考资料

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf) — Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Stefano Ermon (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 37; Pages: 2252-2261; DOI: [10.55989/ah-3](https://doi.org/10.55989/ah-3)
  这篇是首次提出用于生成任务的扩散概率模型概念的论文，为后续进展奠定了理论基础。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS); Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  本文改进并推广了去噪扩散概率模型 (DDPM) 框架，提供了更简单的训练目标，并在图像生成方面展示了优异效果，使扩散模型得到广泛应用。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  这项研究将去噪扩散概率模型与基于分数的生成模型统一在一个连续时间随机微分方程框架下，显著推动了这些模型的理论理解和实际应用。
