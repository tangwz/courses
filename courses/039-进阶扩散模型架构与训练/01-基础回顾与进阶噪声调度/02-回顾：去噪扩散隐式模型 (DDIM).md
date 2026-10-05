# 回顾：去噪扩散隐式模型 (DDIM)

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-1-diffusion-foundations-advanced-noise/ddim-recap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然去噪扩散概率模型 (DDPM) 为高质量图像生成提供了强大的架构，但其迭代采样过程通常需要数百或数千步，这带来了显著的计算瓶颈。去噪扩散隐式模型 (DDIM) 被引入作为 DDPM 的一个概括，专门用于通过实现更快的采样来解决这一限制。

DDPM 的一个主要区别在于逆向过程的性质。DDPM 假设一个马尔可夫逆向过程，即 $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$，其中每一步 $x_{t-1}$ 仅依赖于前一步 $x_t$。然而，DDIM 利用了更通用、*非马尔可夫*的推断过程。这个看似微小的改变具有深远的影响。重要的是，DDIM 使用*完全相同的神经网络 (neural network)* $\epsilon_\theta(x_t, t)$，该网络使用标准 DDPM 目标（通常是预测噪声的简化版本）进行训练。创新完全在于采样过程。

### DDIM 采样过程

DDIM 采样的核心思想源于对 DDPM 前向过程推导中使用的条件分布 $q(x_{t-1}|x_t, x_0)$ 的分析。回想一下，在 DDPM 中，逆向步骤 $p_\theta(x_{t-1}|x_t)$ 旨在近似 $q(x_{t-1}|x_t)$。DDIM 转而设计了一个直接使用从 $q(x_{t-1}|x_t, x_0)$ 导出的性质的采样过程。

首先，给定 $x_t$ 和预测噪声 $\epsilon_\theta(x_t, t)$，我们可以得到初始数据点 $x_0$ 的估计。使用前向过程定义 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$（其中 $\epsilon \sim \mathcal{N}(0, I)$ 且 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i = \prod_{i=1}^t (1-\beta_i)$），我们可以重新排列以预测 $x_0$：


$$
\hat{x}_0(x_t, t) = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t))
$$


这个 $\hat{x}_0$ 表示模型在时间步 $t$ 给定噪声图像 $x_t$ 时对原始清晰图像的最佳估计。

现在，DDIM 不再从近似后验 $p_\theta(x_{t-1}|x_t)$ 采样 $x_{t-1}$，而是使用 $\hat{x}_0$ 定义一个直接采样步骤。完整的 DDIM 更新步骤，考虑时间步长的一个子序列 $t_i, t_{i-1}$（其中 $i$ 从 $S$ 递减到 1，且 $t_S=T, t_0=0$），由下式给出：


$$
x_{t_{i-1}} = \underbrace{\sqrt{\bar{\alpha}_{t_{i-1}}} \hat{x}_0(x_{t_i}, t_i)}_{\text{方向到 } x_0} + \underbrace{\sqrt{1 - \bar{\alpha}_{t_{i-1}} - \sigma_{t_i}^2} \cdot \epsilon_\theta(x_{t_i}, t_i)}_{\text{噪声方向}} + \underbrace{\sigma_{t_i} z_t}_{\text{随机噪声}}
$$


这里，$z_t \sim \mathcal{N}(0, I)$ 是新的高斯噪声，$\sigma_{t_i}^2$ 控制过程的随机性。它通常由 $\eta$ 参数 (parameter)化：


$$
\sigma_{t_i}^2 = \eta \frac{(1-\bar{\alpha}_{t_{i-1}})}{(1-\bar{\alpha}_{t_i})} \left( 1 - \frac{\bar{\alpha}_{t_i}}{\bar{\alpha}_{t_{i-1}}} \right)
$$


### 确定性采样与加速

当设置超参数 (parameter) (hyperparameter) $\eta = 0$ 时，DDIM 的一个重要特性便会显现。这使得 $\sigma_{t_i} = 0$，消除了随机噪声项 $z_t$，从而得到一个*确定性*更新规则：


$$
x_{t_{i-1}} = \sqrt{\bar{\alpha}_{t_{i-1}}} \hat{x}_0(x_{t_i}, t_i) + \sqrt{1 - \bar{\alpha}_{t_{i-1}}} \cdot \epsilon_\theta(x_{t_i}, t_i)
$$


这种确定性意味着从相同的初始噪声 $x_T$ 开始，采样过程将始终生成完全相同的最终图像 $x_0$。此属性对于需要可复现性或潜在空间操控的任务很有价值。

此外，非马尔可夫公式允许 DDIM 在采样过程中跳过步骤。虽然 DDPM 通常需要对所有 $T$ 个时间步进行采样（例如 $T=1000$），但 DDIM 可以使用短得多的子序列 $\tau = \{t_1, t_2, ..., t_S\}$，其中 $S \ll T$（例如 $S=20, 50$ 或 $100$）。采样器直接从 $x_{t_i}$ 跳到 $x_{t_{i-1}}$，显著减少了通过网络 $\epsilon_\theta$ 所需的前向传播次数。

> DDPM（上，蓝色）和 DDIM（下，红色）采样路径的比较。与原始扩散步骤数 ($T$) 相比，DDIM 允许显著减少步骤数 ($S$)。

### 权衡与关联

DDIM 提供的加速伴随着权衡。虽然显著加快，但使用更少的采样步骤 ($S$) 有时会导致样本质量或多样性略有下降，相比于运行完整的 DDPM 过程或使用更大的 $S$。对 $S$ 和 $\eta$ 的选择允许调整速度和保真度之间的平衡。将 $\eta=1$ 设置为恢复与原始 DDPM 采样密切相关的过程（尽管仍在使用所选子序列上的非马尔可夫结构），重新引入了随机性。

从理论角度来看，确定性 DDIM ($\eta=0$) 过程可以解释为近似求解与扩散过程相关的特定概率流常微分方程 (ODE) 的轨迹。这种关联将扩散模型与连续时间生成模型联系起来，并为开发更先进的基于 ODE 的采样器提供了依据，我们将在第 6 章中讨论。

理解 DDIM 采样机制、其确定性变体以及通过跳过步骤加速生成的能力具有根本意义。它不仅为现有经过 DDPM 训练的模型提供了更快的采样实用方法，而且还作为扩散模型采样和蒸馏技术许多后续进展的构建模块，这些将在本课程后续部分介绍。

## 参考资料

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  本文介绍了去噪扩散隐式模型（DDIM），提出了一种DDPM的非马尔可夫泛化方法，以实现更快、确定性的采样。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  介绍去噪扩散概率模型（DDPM）的基础工作，DDIM在此基础上进行发展。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  本文通过随机微分方程统一了基于评分的生成模型，为DDIM的确定性采样及其与常微分方程的关联提供了理论框架。

---

[上一节](01-%E5%9B%9E%E9%A1%BE%EF%BC%9A%E5%8E%BB%E5%99%AA%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B%EF%BC%88DDPM%EF%BC%89.md) · [下一节](03-%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86%EF%BC%9A%E5%88%86%E6%95%B0%E5%8C%B9%E9%85%8D%E4%B8%8E%E5%B8%B8%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B.md)
