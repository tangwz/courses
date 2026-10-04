---
course: "intro-diffusion-models"
chapter: "forward-diffusion-process"
lesson: "defining-markov-chain"
sourceId: 5370
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-2-forward-diffusion-process/defining-markov-chain"
title: "定义马尔可夫链"
description: "将逐步加噪过程建立为马尔可夫链。"
order: 1
plots: []
sourceHash: "725ef0fe668046956b6c60e2e30d1c7e642bbadbdfced0bdf4d45ae6fea0598d"
sourceCorrections: []
---

前向扩散过程将初始数据点（例如清晰图像 $x_0$）在一系列离散时间步 $t=1, 2, ..., T$ 中逐渐损坏成噪声。这种有序的损坏并非随意；它遵循一种特定的概率模型，称为**马尔可夫链**。

马尔可夫链是一个随机变量序列，其中转移到下一状态的概率仅取决于当前状态，而不取决于之前的事件序列。在我们这里，这些“状态”是数据逐渐增噪的版本：$x_0, x_1, x_2, ..., x_T$。其决定性特征是，要得到 $x_t$，我们只需要知道 $x_{t-1}$ 以及在步骤 $t$ 添加噪声的规则。我们不需要直接知道 $x_{t-2}$ 或任何更早的状态。

我们可以这样表示这个序列：


$$
x_0 \rightarrow x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_{t-1} \rightarrow x_t \rightarrow \dots \rightarrow x_T
$$


这里：

- $x_0$ 是从真实数据分布 $q(x_0)$ 中取出的原始数据样本。
- $x_t$ 是经过 $t$ 步加噪后，在时间步 $t$ 的数据样本。
- $x_T$ 是经过 $T$ 步后的最终状态，理想情况下它类似于纯高斯噪声。

从一个状态 $x_{t-1}$ 到下一个状态 $x_t$ 的转移由条件概率分布定义，记作 $q(x_t | x_{t-1})$。这个分布规定了在给定 $x_{t-1}$ 的情况下如何采样 $x_t$。因为这个过程只取决于前一个状态，它满足马尔可夫性质：


$$
q(x_t | x_{t-1}, x_{t-2}, \dots, x_0) = q(x_t | x_{t-1})
$$


这个性质极大简化了前向过程的分析和实现。我们不需要追踪整个加噪历史来确定下一状态；只需要紧邻的前一个状态。

从视觉上看，我们可以描绘出这种依赖链：

> 前向扩散过程被建模为马尔可夫链。每个箭头表示一个转移 $q(x_t | x_{t-1})$，其中噪声的添加仅基于前一个状态。

在后续部分，我们将详细阐述转移 $q(x_t | x_{t-1})$ 的具体数学形式，这涉及在每一步添加经过仔细调整的高斯噪声，并由预定方案控制。理解这种马尔可夫结构是理解扩散模型如何运作，以及最终它们如何学习逆转此过程以进行生成的第一步。

## 参考资料

- [Markov Chains](https://doi.org/10.1017/CBO9780511810633) — James R. Norris (1997)
  Publisher: Cambridge University Press; DOI: [10.1017/CBO9780511810633](https://doi.org/10.1017/CBO9780511810633)
  本书对马尔可夫链的理论和应用进行了基础性介绍，是其数学性质和定义的重要参考资料。
- [Denoising Diffusion Probabilistic Models](https://neurips.cc/virtual/2020/public/paper_by_id.html?id=4c5bcfec8584af0d967f1ab10179a613) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  本文介绍了去噪扩散概率模型，明确将前向扩散过程定义为一个逐步添加高斯噪声的固定马尔可夫链。
