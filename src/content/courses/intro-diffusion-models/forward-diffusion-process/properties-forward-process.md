---
course: "intro-diffusion-models"
chapter: "forward-diffusion-process"
lesson: "properties-forward-process"
sourceId: 5380
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-2-forward-diffusion-process/properties-forward-process"
title: "前向过程的性质"
description: "分析前向扩散过程的特征和结果。"
order: 5
plots: ["plots/5380-0.json"]
sourceHash: "24f967861abc40cd8033e1f9a5bda4de7fdd17001e7eaac131021812b22dca8b"
sourceCorrections: []
---

既然我们已经在数学上定义了前向扩散过程，让我们考察它的一些重要特征。理解这些性质对于把握扩散模型的工作方式及其设计缘由非常重要。

### 过程的高斯属性

回想一下前几节内容，前向过程的每一步都会添加高斯噪声。具体来说，其转移分布定义为：

正向扩散过程具有主要特性。理解这些特性对于掌握扩散模型如何运作以及其设计选择的原因至关主要。

这里 $\beta_t$ 是时间步 $t$ 处的方差调度，$I$ 是单位矩阵。由于我们从数据 $x_0$ 开始并重复添加高斯噪声，任何噪声样本 $x_t$ 在起始点 $x_0$ 条件下的边缘分布也是高斯分布。如前所述，此分布具有一个简洁的封闭形式：


$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$


这里，$\alpha_t = 1 - \beta_t$，且 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。这一性质非常实用，因为它表示我们可以在给定 $x_0$ 的情况下，直接为任意时间步 $t$ 对 $x_t$ 进行采样，而无需遍历所有中间步骤 $x_1, x_2, ..., x_{t-1}$。这极大地提升了后续训练过程的速度。

### 收敛到标准高斯分布

前向过程的主要目标是将复杂的数据分布 $q(x_0)$ 逐步转变为一个简单、已知且通常为各向同性高斯分布 $\mathcal{N}(0, I)$。我们定义的过程能做到这一点吗？

让我们再次审视分布 $q(x_t | x_0)$：


$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
$$


噪声调度 $\{\beta_t\}_{t=1}^T$ 通常被设计为 $\beta_t$ 值较小但为正。这确保了 $\alpha_t = 1 - \beta_t$ 略小于 1。因此，累积乘积 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ 是一个从 $\bar{\alpha}_0 = 1$（按约定）开始并随 $t$ 增加而稳定减小的值。

对于足够大的步骤数 $T$（例如 $T=1000$ 或更多）以及合适的调度 $\{\beta_t\}$，$\bar{\alpha}_T$ 的值会变得非常接近零。

- 随着 $\bar{\alpha}_T \rightarrow 0$，均值项 $\sqrt{\bar{\alpha}_T} x_0$ 趋近于 $0$。
- 随着 $\bar{\alpha}_T \rightarrow 0$，方差项 $(1 - \bar{\alpha}_T) I$ 趋近于 $(1 - 0) I = I$。

因此，对于大的 $T$，分布 $q(x_T | x_0)$ 变为：


$$
q(x_T | x_0) \approx \mathcal{N}(x_T; 0, I)
$$


这表示在 $T$ 步之后，得到的样本 $x_T$ 基本上是纯粹的高斯噪声，并且关于原始数据点 $x_0$ 的几乎所有信息都已被消除。前向过程成功地将任何输入数据点转换为来自标准高斯分布的样本，无论起始点 $x_0$ 是什么。



![累积 Alpha 的减小（信息衰减）](plots/5380-0.json)



> 随着时间步 $t$ 的增加，$\bar{\alpha}_t$ 的值从 1 减小到 0，这表示原始数据 $x_0$ 的影响逐渐减弱，而噪声的影响逐渐增强。显示的值是针对 $T=1000$ 步的典型调度进行的说明。

### 易处理性与固定性

前向过程的一个显著性质是其易处理性。如前所述，我们可以使用封闭形式表达式直接计算分布 $q(x_t | x_0)$。这使得我们能够在给定 $x_0$ 的情况下高效地为任意 $t$ 采样 $x_t$，这对于训练学习逆向过程的神经网络 (neural network)非常重要。我们在训练期间无需模拟逐步加噪的过程。

此外，整个前向过程是固定的。它不包含任何可学习的参数 (parameter)。噪声调度 $\{\beta_t\}_{t=1}^T$ 是预先选定的（例如，线性调度、余弦调度），并在整个训练和推理 (inference)过程中保持不变。所有学习都发生在逆向过程中，该过程必须学会如何撤销这种固定的加噪操作。

### 马尔可夫性质

按照定义，前向过程是一个马尔可夫链。这表示状态 $x_t$ 的分布仅取决于紧邻的前一个状态 $x_{t-1}$，而不取决于任何更早的状态 $x_0, ..., x_{t-2}$。


$$
q(x_t | x_{t-1}, x_{t-2}, ..., x_0) = q(x_t | x_{t-1})
$$


虽然我们得出了一个用于 $q(x_t | x_0)$ 的实用表达式，但其底层的逐步过程保留了马尔可夫性质。这种结构简化了数学分析，并在逆向过程中得到了反映（尽管是近似的）。

总而言之，前向过程是一个固定且易于处理的机制，它逐步且可控地将数据转换为符合高斯统计的噪声。它的终点被设计为一个简单、已知的分布 ($\mathcal{N}(0, I)$)，其中间步骤 $q(x_t | x_0)$ 易于计算。这些性质构成了可学习的逆向（去噪）过程的依据。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; Pages: 6840-6851; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍去噪扩散概率模型的原始论文，详细阐述了前向加噪过程、其高斯特性以及封闭式边际分布的推导。
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Yang Song, Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems; Volume: 32; DOI: [10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600)
  介绍了基于分数的生成建模，这是一个通过逐步添加噪声并学习逆转该过程的相关框架。为生成扩散过程中高斯噪声的作用提供了背景。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021)
  Journal: International Conference on Learning Representations; DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  在随机微分方程的框架下统一了去噪扩散概率模型和基于分数的生成模型，提供了对前向过程的普遍理解。
