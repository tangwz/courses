---
course: "intro-diffusion-models"
chapter: "sampling-generation-process"
lesson: "ddim-sampling-algorithm"
sourceId: 5427
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process/ddim-sampling-algorithm"
title: "DDIM采样算法"
description: "比较DDIM更新规则与DDPM，并阐述其对更快、确定性采样的影响。"
order: 5
plots: []
sourceHash: "d8e7236b1efd427e7c8aa7f3f1950c65aacadf04308ba1f2680abc2b082b35a1"
sourceCorrections: []
---

DDPM采样算法提供了一套合理方法来反转扩散过程。然而，它通常需要模拟许多小步骤（等于正向步骤的数量，$T$），这可能在计算上代价高昂且耗时。试想一下，生成一张图片需要1000个步骤；这对许多应用来说很快就变得不实用。

幸运的是，由Song、Meng和Ermon（2020）提出的去噪扩散隐式模型（DDIM）提供了一种更灵活且通常明显更快的替代方案。DDIM的思路是，用于训练噪声预测网络$\epsilon_\theta$的目标函数并不严格依赖于DDPM假设的正向过程的马尔可夫性质。DDIM通过提出一种不同的*生成*过程（反向过程），该过程是非马尔可夫的，但仍然使用相同的已训练网络$\epsilon_\theta$来做到这一点。

### DDIM的核心思路

回顾DDPM的反向步骤旨在近似$p(x_{t-1}|x_t)$。DDIM采取了不同的方法。它从以下属性开始：给定原始数据$x_0$和用于生成$x_t$的噪声$\epsilon$，我们可以写出：


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$


我们可以重新排列此式，根据当前带噪样本$x_t$和预测噪声$\epsilon_\theta(x_t, t)$，得到原始数据$x_0$的估计值：


$$
\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$


这个$\hat{x}_0$表示从带噪中间样本$x_t$对最终干净数据点进行的预测。

DDIM接着通过将这个预测的$\hat{x}_0$与预测噪声$\epsilon_\theta(x_t, t)$结合来定义到$x_{t-1}$的步骤，但其方式使得能够进行确定性过渡和跳步。通用的DDIM更新规则是：


$$
x_{t-1} = \underbrace{\sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t, t)}_{\text{预测的 } x_0 \text{ 缩放后}} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t, t)}_{\text{指向 } x_t 	ext{ 的方向}} + \underbrace{\sigma_t z}_{\text{随机噪声}}
$$


其中$z \sim \mathcal{N}(0, \mathbf{I})$是标准高斯噪声，$\sigma_t$决定随机性程度。

### 确定性采样 ($\eta=0$)

DDIM的一个重要特点是我们可以选择标准差$\sigma_t$。一个常见选择是设置$\sigma_t = 0$。这使得更新步骤在给定$x_t$和预测$\epsilon_\theta(x_t, t)$的情况下完全确定：


$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t, t) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)
$$


代入$\hat{x}_0(x_t, t)$的表达式：


$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)
$$


这种确定性意味着从相同的初始噪声$x_T$开始，我们总是会生成相同的最终输出$x_0$。这与DDPM有很大不同，DDPM在每一步添加的噪声$\sigma_t z$会产生变化，即使从相同的$x_T$开始。

### 使用子序列进行更快采样

非马尔可夫性质和确定性更新规则（当$\sigma_t=0$时）使得DDIM即使只使用原始时间步$[1, ..., T]$的*子序列*也能有效工作。例如，我们可能不是取1000个步骤，而是定义一个只有50或100个时间步的采样序列$S$，比如$S = (T, T-\frac{T}{N_{steps}}, T-2\frac{T}{N_{steps}}, ..., 1)$，其中$N_{steps}$是所需的采样步骤数（例如50）。

DDIM更新接着通过在这个子序列中的连续时间步之间进行。如果$t$和$t'$是子序列$S$中两个连续的时间步（且$t' < t$），则更新根据$x_t$计算$x_{t'}$，使用相同的公式但将$t-1$替换为$t'$。这使得去噪过程可以有更大的“跳跃”，大幅加速生成。

### DDIM算法

以下是使用DDIM进行采样的步骤，常常采用确定性设置($\sigma_t=0$)：

1. **选择时间步:** 从$(1, ..., T)$中选择一个包含$N_{steps}$个时间步的子序列$S = (s_1, s_2, ..., s_{N_{steps}})$，通常按降序排列（例如，$s_1=T, s_2 \approx T-\frac{T}{N_{steps}}, ..., s_{N_{steps}} \approx 1$）。令$s_0=0$。
2. **初始噪声:** 采样起始噪声$x_{s_1} = x_T \sim \mathcal{N}(0, \mathbf{I})$。
3. **迭代去噪:** 对于$i = 1, ..., N_{steps}$：
   - 获取当前时间步$t = s_i$和前一个时间步$t' = s_{i+1}$（如果$i=N_{steps}$则使用$t'=0$）。
   - 使用训练好的模型预测噪声：$\epsilon_\theta(x_t, t)$。
   - 预测干净数据：$\hat{x}_0(x_t, t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$。
   - 必要时截断$\hat{x}_0$（例如，对于归一化 (normalization)图像截断到$[-1, 1]$）。
   - 计算方差$\sigma_t$。一种通用方式是使用参数 (parameter)$\eta \in [0, 1]$：
     
     $$
     \sigma_t = \eta \sqrt{\frac{1 - \bar{\alpha}_{t'}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t'}}}
     $$
     
     设置$\eta=0$得到$\sigma_t = 0$（确定性DDIM）。设置$\eta=1$得到随机版本。
   - 计算下一个样本$x_{t'}$：
     
     $$
     x_{t'} = \sqrt{\bar{\alpha}_{t'}} \hat{x}_0(x_t, t) + \sqrt{1 - \bar{\alpha}_{t'} - \sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t z
     $$
     
     其中，如果$\sigma_t > 0$，则$z \sim \mathcal{N}(0, \mathbf{I})$；如果$\sigma_t=0$，则$z=0$。请注意，如果$t'=0$，则$\bar{\alpha}_{t'}=1$并且公式简化为：$x_0 = \hat{x}_0(x_t, t)$。
4. **输出:** 返回最终样本$x_0$。

> DDPM和确定性DDIM($\eta=0$)采样的比较。DDPM基于马尔可夫假设使用小的随机步骤。DDIM在每一步计算一个预测的$x_0$，并根据选定的子序列$S$用它来执行更大的确定性步骤，大幅减少了所需的网络评估次数。

使用更少步骤（$N_{steps} \ll T$）的能力使得DDIM成为对生成速度有要求的实际使用中一个非常受欢迎的选择。在下一节中，我们将探讨这两种采样方法之间的权衡。

## 参考资料

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Jiaming Song, Chenlin Meng, Stefano Ermon (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2010.02502](https://doi.org/10.48550/arXiv.2010.02502)
  介绍DDIM算法的原始研究论文，详细阐述了其非马尔可夫生成过程、确定性采样和利用子序列加速生成的特性。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  通过将扩散模型（包括DDPM和DDIM）与SDE和ODE联系起来，提供了一个统一的理论框架。这有助于理解DDIM确定性及快速采样的理论基础。
