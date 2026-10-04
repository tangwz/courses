---
course: "intro-diffusion-models"
chapter: "sampling-generation-process"
lesson: "understanding-sampling-variance"
sourceId: 5423
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process/understanding-sampling-variance"
title: "理解采样方差"
description: "随机性在采样过程中的作用。"
order: 3
plots: []
sourceHash: "948d9df9ba109d486441087aebedfa87dc6355a646c5c985c6a172a5b508ec92"
sourceCorrections: []
---

使用逆向扩散过程生成数据的大致过程是，从噪声 $x_T$ 开始，并迭代应用学习到的去噪步骤以得到 $x_0$。标准去噪扩散概率模型 (DDPM) 采样过程的一个主要特点是其固有的随机性。即使从完全相同的初始噪声张量 $x_T$ 开始两个生成过程，也可能得到两个不同的最终样本 $x_0$。这种现象源于采样过程本身的特性。

回顾第三章，逆向过程旨在近似真实后验分布 $p(x_{t-1}|x_t)$。在 DDPM 中，我们在每个时间步 $t$ 将这个逆向转移参数 (parameter)化为高斯分布：

$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$

此处，$\mu_\theta(x_t, t)$ 表示给定当前状态 $x_t$ 和时间步 $t$ 时，前一个状态 $x_{t-1}$ 的分布均值。我们训练的神经网络 (neural network) $\epsilon_\theta(x_t, t)$ 用于计算此均值。项 $\sigma_t^2 \mathbf{I}$ 表示方差，其中 $\sigma_t^2$ 通常是从前向过程噪声调度（通常与 $\beta_t$ 或 $\tilde{\beta}_t$ 相关）得出的超参数 (hyperparameter)。

要点在于，从 $x_t$ 获取 $x_{t-1}$ 涉及从此高斯分布中**采样**，而不仅仅是计算均值 $\mu_\theta(x_t, t)$。采样操作如下所示：

$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$

其中 $z$ 是从标准高斯分布中采样的随机向量 (vector)，$z \sim \mathcal{N}(0, \mathbf{I})$。

在 *每个* 时间步 $t$（从 $T$ 到 1）添加缩放的随机噪声 $\sigma_t z$ 是 DDPM 采样过程中变异性的来源。每个步骤都引入少量随机性。

### 逐步随机性的累积效应

可以想象从纯噪声 $x_T$ 到最终样本 $x_0$ 的路径是 $T$ 个步骤的序列。在每个步骤中，模型预测方向（即均值 $\mu_\theta$），但由于添加了噪声 $\sigma_t z$，模型会朝着该方向稍微随机地迈出一步。

> 此图显示了，从相同的状态 $x_t$ 开始，在逆向步骤中采样不同的噪声向量 (vector) $z_t$ 和 $z'_t$ 会导致两种潜在生成路径得到略微不同的状态 $x_{t-1}$。经过多个步骤，这些微小的差异会累积，从而产生不同的最终样本 $x_0$。

由于这些小的随机扰动在整个 $T$ 个步骤序列中累积，最终输出 $x_0$ 反映了这些随机选择的总和。这解释了为什么即使使用相同的超参数 (parameter) (hyperparameter)和训练好的模型，多次运行 DDPM 采样过程也会产生多样化的生成样本。这种固有的随机性通常是受欢迎的，因为它允许单个训练好的模型生成多种多样的输出。

### 方差调度 ($\sigma_t^2$) 的作用

逆向步骤中方差 $\sigma_t^2$ 的大小会影响每个步骤中注入的随机性量。在原始 DDPM 论文中，$\sigma_t^2$ 是根据前向过程中使用的噪声调度 $\beta_t$ 设置的（具体来说，$\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ 或简而言之 $\sigma_t^2 = \beta_t$）。虽然这是与前向过程在理论上相关联的标准选择，但也可以使用不同的值。较大的 $\sigma_t^2$ 值通常会导致生成样本的多样性更高，但如果设置过高，可能会略微降低生成质量或与学习到的数据分布的保真度。较小的值则会减少随机性。

### 与更快采样方法的关联

理解这种方差来源也很重要，当我们考虑去噪扩散隐式模型（DDIM）等更快的采样方法时，我们将在接下来讨论这些方法。DDIM 重新诠释了生成过程，并引入了一个参数 (parameter)（通常表示为 $\eta$）来控制随机性的大小。当 $\eta=0$ 时，给定初始噪声 $x_T$，DDIM 采样变为确定性的。这意味着对于固定的 $x_T$，$\eta=0$ 的 DDIM 将始终产生 *相同* 的 $x_0$。这与 DDPM 形成鲜明对比，DDPM 由于在每个步骤中都添加了 $\sigma_t z$ 项而具有固有的随机性。随机性（多样性）与确定性（可重现性，可能更快的采样）之间的这种权衡是 DDPM 和 DDIM 采样策略之间的一个主要区别。

## 参考资料

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, and Stefano Ermon (2020)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  这项工作提出了去噪扩散隐式模型（DDIM），它概括了扩散模型的采样过程，并引入了一个确定性变体，从而能够控制采样的随机性。
