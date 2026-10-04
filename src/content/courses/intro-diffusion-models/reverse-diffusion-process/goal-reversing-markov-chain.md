---
course: "intro-diffusion-models"
chapter: "reverse-diffusion-process"
lesson: "goal-reversing-markov-chain"
sourceId: 5388
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-3-reverse-diffusion-process/goal-reversing-markov-chain"
title: "目标：逆转马尔可夫链"
description: "定义逆向过程的目标：估计$p(x_{t-1}|x_t)$。"
order: 1
plots: []
sourceHash: "ea1c31c4649d0ab6addbc1f5066b945fc2fee4be8c6ddeb74983e22047e461e7"
sourceCorrections: []
---

前向扩散过程是一种通过在T个时间步中迭代添加高斯噪声来系统地破坏数据$x_0$的方法。


$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_T
$$


在此过程结束时，$x_T$基本上与纯高斯噪声无法区分。我们现在的目标是生成模型：我们想生成新的数据样本，使其看起来像是来自原始数据分布$q(x_0)$。为了实现此目的，我们需要找出如何逆转加噪过程。

设想从一个简单分布（如标准高斯分布$\mathcal{N}(0, I)$）中抽样得到的$x_T$开始。如果我们可以某种方式逆转前向过程的每一步，从$t=T$向后移动到$t=1$，我们可能将这个初始噪声样本$x_T$转换成一个真实的数据样本$x_0$：


$$
x_T \rightarrow x_{T-1} \rightarrow x_{T-2} \rightarrow \dots \rightarrow x_0
$$


这种逆转定义了扩散模型的*生成*路径。

> 图示了前向（加噪）和逆向（生成）过程，它们是向相反方向移动的马尔可夫链。

前向过程由转移概率$q(x_t | x_{t-1})$定义，它指定了如何通过添加受控量的噪声从$x_{t-1}$得到$x_t$。逆向过程的核心目标是学习相反的转移：概率分布$p(x_{t-1} | x_t)$。这个分布告诉我们，给定时间步$t$的一个带噪声样本$x_t$，在前一个时间步$t-1$上，可能的“更少噪声”样本$x_{t-1}$的分布是怎样的。

如果我们能成功建模所有相关时间步$t$（从$T$到$1$）的逆向转移概率$p(x_{t-1} | x_t)$，我们就可以实现生成过程：

1. 从纯噪声开始：抽样$x_T \sim \mathcal{N}(0, I)$。
2. 迭代去噪：对于$t = T, T-1, \dots, 1$：
   - 抽样$x_{t-1} \sim p(x_{t-1} | x_t)$。
3. 最终样本$x_0$就是生成结果。

因此，构建扩散模型的主要难题在于有效地估计或参数 (parameter)化这些逆向条件概率$p(x_{t-1} | x_t)$。前向过程$q(x_t | x_{t-1})$设计得在数学上很方便（即添加高斯噪声）。正如我们将在接下来的章节中看到，真实的逆向转移$q(x_{t-1} | x_t, x_0)$（注意对$x_0$的条件依赖）是已知的，但计算所需的$q(x_{t-1} | x_t)$需要知道完整的数据分布，而这正是我们试图学习的目标。这种难处理性促使我们使用强大的函数逼近器，特别是神经网络 (neural network)，来学习一个逼近真实逆向转移的模型$p_\theta(x_{t-1} | x_t)$。

我们的目标已明确：学习一个模型，它能够根据当前状态$x_t$预测前一个状态$x_{t-1}$，使我们能够沿着链从噪声向数据逆向推演。接下来的章节将详细说明我们如何构建和训练一个神经网络来执行此任务。

## 参考资料

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli (2015)
  Journal: arXiv preprint arXiv:1503.03585; DOI: [10.48550/arXiv.1503.03585](https://doi.org/10.48550/arXiv.1503.03585)
  这篇基础论文介绍了扩散概率模型，定义了用于通过反转固定的加噪过程来生成数据的正向和反向马尔可夫链。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇论文通过简化目标函数并展示高质量图像生成，显著推动了扩散模型的发展。它详细介绍了学习反向条件概率的训练过程。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  这项工作在随机微分方程的单一框架下统一了基于分数的生成模型和扩散概率模型，为反向生成过程提供了全面的理解。
