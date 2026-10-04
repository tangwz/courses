---
course: "synthetic-data-gans-diffusion"
chapter: "diffusion-models-theory-implementation"
lesson: "ddim-variance-schedules"
sourceId: 5699
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation/ddim-variance-schedules"
title: "改进技术：DDIM和方差调度"
description: "实现去噪扩散隐式模型（DDIM）以加快采样。"
order: 4
plots: ["plots/5699-0.json"]
sourceHash: "d8627227c680ff6a0da541fb2536aa55aef5a43ce9557fa2ebac81fd1104baea"
sourceCorrections: []
---

尽管去噪扩散概率模型（DDPM）实现了出色的生成质量，但其采样速度慢是一个主要缺点。生成单个样本通常需要数百甚至数千个连续去噪步骤，这与训练中使用的噪声级别数 $T$ 对应。旨在加快采样和改进生成过程的技术得到探讨，重点介绍去噪扩散隐式模型（DDIM）和方差调度的影响。

### 去噪扩散隐式模型（DDIM）实现更快的采样

DDIM 对DDPM的生成（逆向）过程进行了修改，能够显著加快采样，通常可以将所需步骤数减少10到100倍，而无需重新训练模型。其独特之处在于构建了一种*非马尔可夫*逆向过程，该过程仍然使用利用DDPM目标进行训练的相同噪声预测网络 $\boldsymbol{\epsilon}_\theta$。

回顾标准DDPM逆向步骤：


$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \tilde{\beta}_t \mathbf{I})
$$


其中 $\boldsymbol{\mu}_\theta$ 取决于 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$，方差 $\tilde{\beta}_t$ 则根据噪声调度 $\beta_t$ 固定。这个过程是马尔可夫的，意味着 $\mathbf{x}_{t-1}$ 只依赖于 $\mathbf{x}_t$。

DDIM 引入了一类更普遍的非马尔可夫扩散过程。其主要思路是，首先从当前噪声状态 $\mathbf{x}_t$ 预测最终的干净数据点 $\mathbf{x}_0$，然后利用这个预测来引导向 $\mathbf{x}_{t-1}$ 的步骤。预测的 $\mathbf{x}_0$ 是通过重新排列正向过程方程 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 获得的：


$$
\mathbf{x}_0^{\text{pred}}(t) = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t))
$$


这个预测的 $\mathbf{x}_0$ 表示模型对原始数据的最佳估计，考虑到噪声输入 $\mathbf{x}_t$ 和当前时间步 $t$。

DDIM逆向步骤随后利用这个预测的 $\mathbf{x}_0$ 来采样 $\mathbf{x}_{t-1}$：


$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0^{\text{pred}}(t) + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}}_{\text{指向 } \mathbf{x}_t \text{ 的方向}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \underbrace{\sigma_t \boldsymbol{\epsilon}'}_{\text{随机噪声}}
$$


这里，$\boldsymbol{\epsilon}' \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是新的随机噪声，$\sigma_t$ 控制此逆向步骤的随机性。参数 (parameter) $\sigma_t$ 使用超参数 (hyperparameter) $\eta \ge 0$ 定义：


$$
\sigma_t(\eta) = \eta \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}
$$


其作用在于 $\eta$ 的角色：

- **随机情况 ($\eta = 1$):** 当 $\eta = 1$ 时，$\sigma_t^2$ 的值等于 DDPM 方差 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$。在这种情况下，DDIM 采样过程恢复了原始的 DDPM 马尔可夫过程。
- **确定性情况 ($\eta = 0$):** 当 $\eta = 0$ 时，我们有 $\sigma_t = 0$。随机噪声项消失，并且给定 $\mathbf{x}_t$ 时，更新变为完全确定性：
  
  $$
  \mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0^{\text{pred}}(t) + \sqrt{1 - \bar{\alpha}_{t-1}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
  $$
  
  这使得生成过程变为*隐式*，因为 $\mathbf{x}_{t-1}$ 是直接计算得出的，而非从分布中采样。这种确定性使得在采样过程中时间步可以有更大的跳跃。不必使用全部 $T$ 个步骤（例如 $T=1000$），我们可以使用一个时间步子序列 $\tau_1, \tau_2, ..., \tau_S$，且 $S \ll T$（例如 $S=50$ 或 $S=100$）。更新规则按顺序应用于 $t = \tau_S, \tau_{S-1}, ..., \tau_1$。这个确定性变体通常与扩散模型的概率流常微分方程（ODE）公式相关联。

> DDPM 逆向步骤与确定性 DDIM 逆向步骤 ($\eta=0$) 的比较。DDIM 使用对干净数据 $\mathbf{x}_0$ 的中间预测来确定 $\mathbf{x}_{t-1}$。

使用 $\eta=0$（确定性 DDIM）通常能以少得多的步骤产生高质量样本。$\eta$ 值在0到1之间，可在确定性生成和随机性生成之间进行插值，可能增加多样性，但会牺牲部分一致性。DDIM 的一个主要优点是它使用为 DDPM 训练的*完全相同*的网络 $\boldsymbol{\epsilon}_\theta$。只有采样过程发生改变，因此可以轻松地与现有模型一起部署，以实现更快的生成。

### 方差调度

噪声调度的选择（由 $t=1, ..., T$ 的 $\beta_t$ 定义）是影响模型性能的另一个重要方面。此调度决定了在正向过程中添加噪声的速度，控制着每个步骤 $t$ 的信噪比。常见的调度包括：

1. **线性调度：** $\beta_t$ 从一个较小的值 $\beta_1$（例如 $10^{-4}$）线性增加到一个较大的值 $\beta_T$（例如 $0.02$）。这在最初的 DDPM 论文中被使用。
2. **余弦调度：** 提出用于改进训练稳定性和样本质量。累积噪声水平 $\bar{\alpha}_t$ 遵循余弦形状，防止信号在正向过程早期衰减过快。具体来说：
   
   $$
   \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad \text{此处} \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
   $$
   
   这里，$s$ 是一个小的偏移量（例如 $0.008$），以防止 $\beta_t$ 在 $t=0$ 附近过小。然后 $\beta_t$ 被推导为 $\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$。



![噪声调度比较 (信号率)](plots/5699-0.json)



> $\bar{\alpha}_t$ 的平方根（表示信号率）随时间减小。$\beta_t$ 的线性调度导致 $\sqrt{\bar{\alpha}_t}$ 大致呈线性下降，而余弦调度则能维持更高的信号率更长时间，然后才更快地衰减。

除了固定调度外，一些研究还关注了**学习逆向过程的方差** $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$。原始 DDPM 将此方差固定为 $\tilde{\beta}_t \mathbf{I}$ 或 $\beta_t \mathbf{I}$。然而，模型 $\boldsymbol{\epsilon}_\theta$ 可以被修改，使其也能预测一个参数 (parameter) $v$，该参数在最优逆向方差的这些下限和上限之间进行插值。虽然学习方差可以改进对数似然分数，但它通常不会大为改善感知质量（由 FID 等指标衡量），并增加了复杂性。固定的小方差方法（通常由 $\tilde{\beta}_t$ 近似）在实践中通常表现良好。DDIM 框架通过控制 $\eta$ 的随机性来避免显式方差学习，提供了一种灵活的方式来隐式管理逆向过程方差。

"总而言之，DDIM 提供了一种有效的方法来加快扩散模型的采样，它通过定义确定性或近似确定性的逆向路径，利用相同的已训练噪声预测网络。方差调度（$\beta_t$）的选择仍然是影响模型性能的重要设计决策，余弦调度通常比线性调度更受青睐。这些方法共同使得扩散模型在需要高效生成的应用中更具实用性。"

## 参考资料

- [Denoising Diffusion Implicit Models](https://doi.org/10.48550/arXiv.2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2020)
  Journal: ICLR 2021; DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  引入了DDIM，这是一种非马尔可夫生成过程，可实现显著更快且确定性的扩散模型采样，无需重新训练。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍了去噪扩散概率模型（DDPM）的基础论文，定义了其前向和反向过程。
- [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a.html) — Alexander Quinn Nichol, Prafulla Dhariwal (2021)
  Journal: Proceedings of the 38th International Conference on Machine Learning; Publisher: PMLR; Volume: 139; Pages: 8162-8171; DOI: [10.1109/ICCV48922.2021.01050](https://doi.org/10.1109/ICCV48922.2021.01050)
  提出了余弦方差调度及其他DDPM改进，提高了样本质量和训练稳定性。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  提出了基于分数的生成模型和扩散模型的统一框架，强调了确定性采样（如η=0的DDIM）与概率流ODE之间的联系。
