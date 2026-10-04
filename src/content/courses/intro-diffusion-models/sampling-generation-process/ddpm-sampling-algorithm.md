---
course: "intro-diffusion-models"
chapter: "sampling-generation-process"
lesson: "ddpm-sampling-algorithm"
sourceId: 5420
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-5-sampling-generation-process/ddpm-sampling-algorithm"
title: "DDPM采样算法"
description: "去噪扩散概率模型（DDPM）采样的分步流程。"
order: 2
plots: []
sourceHash: "0c4ec6c47539e72f127a8215bf5cbe1af9d543621e6ade18d6513be7b4ebc6b4"
sourceCorrections: []
---

扩散模型，特别是U-Net，被训练来预测在时间步$t$时含噪输入$x_t$中的噪声$\epsilon$。利用这个训练好的模型，可以生成新的数据。去噪扩散概率模型（DDPM）算法为此生成过程提供了原始的基本方法。

正如本章引言中所述，核心思想是从纯噪声开始，并使用学到的去噪步骤逐步对其进行精细化。我们首先从标准高斯分布中采样一个初始张量$x_T$，即$x_T \sim \mathcal{N}(0, \mathbf{I})$。这个$x_T$表示最大熵、纯噪声，对应于前向扩散过程的最终状态。

DDPM采样算法随后迭代应用学到的逆向过程，从时间步$t=T$逐步向后退到$t=1$。在每一步中，目标是给定当前状态$x_t$，采样得到一个噪声稍小的版本$x_{t-1}$。

### 去噪步骤

回顾第3章，逆向转换$p_\theta(x_{t-1} | x_t)$由一个高斯分布近似，其均值$\mu_\theta(x_t, t)$取决于$x_t$和预测噪声$\epsilon_\theta(x_t, t)$，其方差$\sigma_t^2$与噪声调度$\beta_t$相关。

模型$\epsilon_\theta(x_t, t)$接收当前含噪图像$x_t$和时间步$t$作为输入，并输出它对从$x_0$到$x_t$过程中添加的噪声成分的预测。使用这个预测，我们可以估计前一状态$x_{t-1}$的分布均值。均值$\mu_\theta(x_t, t)$的方程是从前向和逆向过程的属性推导出来的：


$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$


此处，$\alpha_t = 1 - \beta_t$和$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$是从前向过程中使用的噪声调度$\beta_t$派生出的参数 (parameter)。这个方程本质上是取当前含噪样本$x_t$，并减去按比例缩放的预测噪声$\epsilon_\theta$，以估计前一个噪声较小状态$x_{t-1}$的均值。

逆向步骤的方差$\sigma_t^2$也由噪声调度决定。一个常用的选择，为了匹配前向过程后验分布$q(x_{t-1} | x_t, x_0)$的方差而推导出的，是：


$$
\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$


注意，当$t=1$时，$\sigma_t^2 = 0$。为了执行一个去噪步骤并从$x_t$采样得到$x_{t-1}$，我们使用模型的预测计算均值$\mu_\theta(x_t, t)$，然后添加按标准差$\sigma_t$缩放的高斯噪声：


$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z
$$


其中$z \sim \mathcal{N}(0, \mathbf{I})$是标准高斯噪声。这个添加的噪声$z$为生成过程引入了随机性，使得模型即使从相同的初始$x_T$开始也能生成多样化的样本（尽管通常我们从不同的$x_T$样本开始）。然而，对于最后一步（$t=1$），我们通常将$z=0$以获得最终的确定性均值预测作为我们的输出$x_0$。

### 完整的DDPM算法

将这些步骤结合起来，完整的DDPM采样算法如下进行：

1. **初始化**：采样初始噪声张量$x_T \sim \mathcal{N}(0, \mathbf{I})$。这个张量应与期望的输出数据（例如，图像的高度、宽度、通道数）具有相同的维度。
2. **迭代去噪**：从时间步$t = T, T-1, \dots, 1$开始，向后循环。
   - 采样标准高斯噪声$z \sim \mathcal{N}(0, \mathbf{I})$。如果$t=1$，则设置$z=0$。
   - 使用训练好的神经网络 (neural network)$\epsilon_\theta$预测当前状态中存在的噪声：$\epsilon_{pred} = \epsilon_\theta(x_t, t)$。
   - 使用预测噪声计算逆向分布的均值：
     
     $$
     \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{pred} \right)
     $$
     
   - 计算方差$\sigma_t^2 = \tilde{\beta}_t$（从噪声调度中预先计算）。
   - 使用计算出的均值和方差采样下一个状态（噪声较小）：
     
     $$
     x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z
     $$
     
3. **输出**：循环结束后（在$t=1$时），得到的张量$x_0$就是生成的样本。

此过程可视化如下：

> 图表说明了迭代的DDPM采样过程。从噪声$x_T$开始，每一步都使用噪声预测器$\epsilon_\theta$来计算前一状态的均值$\mu_\theta$，然后通过添加缩放噪声$\sigma_t z$来采样$x_{t-1}$（除了最后一步$z=0$）。

最终样本$x_0$的质量在很大程度上取决于训练好的噪声预测器$\epsilon_\theta$的准确性以及所选的噪声调度（$\beta_t$）和扩散步数（$T$）。DDPM通常需要大量的步骤（例如$T=1000$）才能获得高质量结果，这可能导致采样相对缓慢。我们将在后续章节中讨论像DDIM这样更快的替代方法。

## 参考资料

- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Yang Song and Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 32; DOI: [10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600)
  介绍了基于分数的生成模型，该模型与DDPM在迭代去噪和噪声预测方面存在概念上的关联，提供了更广阔的理论背景。
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — Niels Rogge, Kashif Rasul (2022)
  Publisher: Hugging Face Blog
  对DDPM算法及其数学推导和实际实现提供了带有注释的详细解释，是一个教育资源。
