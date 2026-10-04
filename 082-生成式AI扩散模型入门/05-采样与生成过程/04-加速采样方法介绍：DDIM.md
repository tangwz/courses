# 加速采样方法介绍：DDIM

来源：[原文](https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process/intro-faster-sampling-ddim)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管DDPM采样过程通过细致地逐步反转扩散过程，能够可靠地生成高质量样本，但通常需要大量步骤（通常$T=1000$或更多）。每一步都涉及较大的U-Net模型的正向传播，这使得生成过程计算成本高且缓慢。如果您需要生成大量样本，或在交互式应用中使用扩散模型，这种延迟会成为一个主要瓶颈。

这促使了对更快采样方法的需求。最有影响力且被广泛采用的方法之一是去噪扩散隐式模型（DDIM），由Song、Meng和Ermon在2020年提出。

DDIM提供了一种更灵活的方式来反转扩散过程。回顾一下，DDPM采样定义了一个特定的马尔可夫过程：生成$x_{t-1}$严格只依赖于前一个状态$x_t$。DDIM提出了一种不同的、非马尔可夫生成过程，该过程仍然使用为DDPM训练的*完全相同的神经网络 (neural network)*。它的主要观点是，DDPM的训练目标并未严格强制执行DDPM采样所用的特定马尔可夫链；它主要训练网络来预测噪声$\epsilon_\theta(x_t, t)$。

DDIM采用这种方式，即设计一种采样过程，该过程可以进行更大的“跳跃”回到原始数据$x_0$。DDIM不需要计算所有$T$个中间步骤$x_{T-1}, x_{T-2}, \dots, x_1$，而是允许使用更少的一部分步骤进行采样，例如$S < T$。例如，您可能只使用50或100步而不是1000步，大幅加速了生成过程。

DDIM的一个显著特性是，它在特定参数 (parameter)（通常记为$\eta$，eta）设为0时能够生成确定性输出。给定相同的初始噪声$x_T$和相同的时间步序列，当$\eta=0$时，DDIM总是会生成完全相同的最终样本$x_0$。这与DDPM不同，DDPM在每一步都会添加随机噪声（由方差$\sigma_t^2$控制），这使其输出本质上是随机的。当$\eta > 0$时，DDIM会重新引入随机性，其中$\eta=1$通常会恢复与DDPM非常相似的行为。这种对确定性的控制对于需要可复现结果的应用或在潜在空间中进行插值很有用。

本质上，DDIM提供了一类泛化的采样过程族，DDPM是其一个特定情况。它通过修改用于从$x_t$估计$x_{t-1}$的更新规则来实现了更快的采样，从而允许更大的、可能是确定性的步骤。其权衡是，尽管DDIM快得多，但使用极少步骤的样本质量有时可能略低于完整运行的DDPM，尽管DDIM通常在显著减少的步骤数（例如50-200）下也能生成出色的结果。

下一节将详细阐述DDIM采样的具体数学公式和算法，并强调它与您之前看到的DDPM更新规则有何不同。

## 参考资料

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, and Stefano Ermon (2020)
  Journal: International Conference on Learning Representations (ICLR 2021); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  介绍了去噪扩散隐式模型（DDIM），提出了一种非马尔可夫生成过程，通过允许更大的步长和提供确定性生成来显著加速采样。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型（DDPM）的基础论文，DDIM在此基础上构建，详细阐述了前向扩散和逆向生成过程。
- [Lecture 12: Denoising Diffusion Models](https://ermon.ai/papers/diffusion_notes.pdf) — Stefano Ermon (2023)
  Publisher: Stanford University
  提供了来自顶尖大学深度生成模型课程的详细讲义，解释了DDIM背后的数学公式和直觉。

---

[上一节](03-%E7%90%86%E8%A7%A3%E9%87%87%E6%A0%B7%E6%96%B9%E5%B7%AE.md) · [下一节](05-DDIM%E9%87%87%E6%A0%B7%E7%AE%97%E6%B3%95.md)
