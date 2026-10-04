---
course: "intro-diffusion-models"
chapter: "forward-diffusion-process"
lesson: "gaussian-noise-schedule"
sourceId: 5372
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-2-forward-diffusion-process/gaussian-noise-schedule"
title: "高斯噪声调度"
description: "理解如何使用预定义的方差调度逐步添加噪声。"
order: 2
plots: ["plots/5372-0.json"]
sourceHash: "912576ca6963c18050ba440c365f74c8b02438f344447ba853ab89045242f98c"
sourceCorrections: []
---

前向扩散过程是一个马尔可夫链，它通过 $T$ 个离散时间步逐步将噪声注入到数据 $x_0$ 中。控制每一步添加的噪声*量*是这个过程的一个重要方面。我们并非随意添加，而是遵循一个精心制定的计划，这被称为**噪声调度**。

噪声调度是一系列方差值，表示为 $\beta_1, \beta_2, ..., \beta_T$。每个 $\beta_t$ 决定了从状态 $x_{t-1}$ 过渡到 $x_t$ 时添加的高斯噪声的方差。这些值是超参数 (parameter) (hyperparameter)，意味着它们是在模型训练*之前*选定的。它们不会在训练过程中被学习。

可以将噪声调度理解为设定了每一步噪声化过程的“强度”。$\beta_t$ 的值通常选择满足以下条件：

1. 它们很小，以确保连续步骤 $x_{t-1}$ 和 $x_t$ 之间的变化受到限制。
2. 它们随时间步逐步增加，因此与早期步骤相比，后期步骤会添加更多的噪声。
3. $T$ 个步骤的累积效果足以将原始数据分布转换为非常接近标准高斯分布（各向同性噪声）的形式。

### 定义调度值

原始的去噪扩散概率模型（DDPM）论文提出了一种**线性调度**，其中 $\beta_t$ 在 $T$ 个步骤中（通常 $T=1000$）从一个较小的起始值 $\beta_{start}$（例如 $10^{-4}$）线性增加到一个较大的结束值 $\beta_{end}$（例如 $0.02$）。

线性调度的公式是：


$$
\beta_t = \beta_{start} + (t-1) \frac{\beta_{end} - \beta_{start}}{T-1} \quad \text{对于 } t = 1, ..., T
$$


尽管这种方法简单有效，但也发展出了其他调度方案。一个常用的替代方案是**余弦调度**，它在“改进的去噪扩散概率模型”论文中被提出。这种调度在过程的开始和结束阶段变化较慢，可能带来更好的表现，并避免信号在早期阶段过快地被破坏。

余弦调度使用相关量 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ 定义。它根据余弦函数设定 $\bar{\alpha}_t$，然后推导出相应的 $\beta_t$：


$$
\bar{\alpha}_t = \frac{f(t)}{f(0)} \quad \text{其中} \quad f(t) = \cos\left( \left( \frac{t}{T} + s \right) / (1+s) \cdot \frac{\pi}{2} \right)^2
$$


这里，$s$ 是一个小的偏移量（例如 $0.008$），用以防止 $\beta_t$ 在 $t=0$ 附近过小。一旦为所有 $t$ 计算出 $\bar{\alpha}_t$，就可以得到单独的 $\beta_t$ 值：


$$
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
$$


调度方案的选择影响了原始数据 $x_0$ 的信息被遮蔽的速度。如果调度方案在早期阶段过于激进地添加噪声，可能会使逆向过程更难学习。反之，如果总体上添加的噪声过少，则可能无法在步骤 $T$ 时充分将数据转换为简单的先验分布。

让我们来看看这两种常用调度方案在 $T=1000$ 个步骤下如何比较，其中线性调度采用 $\beta_{start}=0.0001$ 和 $\beta_{end}=0.02$，余弦调度采用 $s=0.008$，以近似匹配在 $T$ 时的噪声水平。



![噪声调度比较 (Beta_t)](plots/5372-0.json)



> 方差 ($\beta_t$) 在 $T=1000$ 个时间步中，通过线性调度和余弦调度在每个时间步添加。余弦调度在初期添加噪声的速度比线性调度慢，但在后期会加速。

### 在马尔可夫步骤中的作用

回顾上一节，前向过程步骤由条件概率 $q(x_t | x_{t-1})$ 定义。这个转换定义为添加具有特定均值和方差的高斯噪声：


$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
$$


在这里，噪声调度值 $\beta_t$ 直接设定了在时间步 $t$ 添加的高斯噪声的方差。均值通过 $\sqrt{1 - \beta_t}$ 进行缩放，以确保数据的总体方差不会急剧增加。由于 $\beta_t$ 很小，$\sqrt{1 - \beta_t}$ 略小于 1，逐步减小了前一状态 $x_{t-1}$ 的作用。

总而言之，高斯噪声调度 $\{\beta_t\}_{t=1}^T$ 是一系列预定义的超参数 (parameter) (hyperparameter)，控制着前向扩散过程每一步添加噪声的大小。它的设计（例如线性、余弦）和值的范围是重要的选择，会影响从数据到噪声的路径，并影响后续逆向过程的学习。我们稍后将看到这些 $\beta_t$ 值及其相关推导量如何在训练目标和采样过程中出现。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型（DDPM）框架和线性噪声调度的基础性论文。
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Tero Karras, Miika Aittala, Timo Aila, Samuli Laine (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2206.00364](https://doi.org/10.48550/arXiv.2206.00364)
  分析了扩散模型的各种设计选择，包括噪声调度参数化如何影响训练、采样效率和质量。
