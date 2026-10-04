---
course: "intro-diffusion-models"
chapter: "reverse-diffusion-process"
lesson: "math-formulation-denoising-step"
sourceId: 5400
sourceUrl: "https://apxml.com/zh/courses/intro-diffusion-models/chapter-3-reverse-diffusion-process/math-formulation-denoising-step"
title: "去噪步骤的数学表达"
description: "控制逆向（去噪）过程单一步骤的方程。"
order: 5
plots: []
sourceHash: "9efad031688c546ea2d311fed26b1b91d8e400ef3754ce536e725d92619920c9"
sourceCorrections: []
---

逆向扩散过程旨在逆转加噪效果，从一个含噪样本$x_t$开始，并尝试估计含噪程度稍低的样本$x_{t-1}$。真实的逆向概率$q(x_{t-1}|x_t)$难以计算，但可以使用神经网络 (neural network)$p_\theta(x_{t-1}|x_t)$来近似。一个常用的方法是利用网络$\epsilon_\theta(x_t, t)$，它经过训练用于预测添加到原始数据$x_0$以得到$x_t$的噪声$\epsilon$。该预测噪声随后被用于定义去噪单步的数学运算。

目标是定义分布$p_\theta(x_{t-1}|x_t)$。结果表明，如果我们知道原始数据点$x_0$，真实的后验分布$q(x_{t-1}|x_t, x_0)$可以解析计算，并且是一个高斯分布。虽然我们在此不进行完整的推导（它涉及将贝叶斯定理应用于前向过程的高斯分布，如Ho等人2020年原始DDPM论文附录B中所述），但其结果很重要。后验分布$q(x_{t-1}|x_t, x_0)$由以下式子给出：


$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$


均值$\tilde{\mu}_t(x_t, x_0)$和方差$\tilde{\beta}_t$如下：


$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$


$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
$$


请记住，$\alpha_t = 1 - \beta_t$和$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$源自噪声调度$\beta_t$。

现在，在逆向（生成）过程中，我们无法获取原始的$x_0$。这时，我们训练好的神经网络$\epsilon_\theta(x_t, t)$就派上用场了。我们用它来近似$x_0$。回想一下允许直接从$x_0$采样$x_t$的前向过程方程：


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$


其中$\epsilon \sim \mathcal{N}(0, I)$。我们可以重新排列此式，用$x_t$和$\epsilon$表示$x_0$：


$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)
$$


由于我们的网络$\epsilon_\theta(x_t, t)$经过训练用于预测$\epsilon$，我们可以使用网络的输出得到$x_0$的一个估计，我们称之为$\hat{x}_0$：


$$
\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t))
$$


这基于当前含噪图像$x_t$和预测的噪声$\epsilon_\theta(x_t, t)$，为我们提供了原始图像的近似值。

现在，我们将这个估计值$\hat{x}_0$代回到后验均值$\tilde{\mu}_t(x_t, x_0)$的方程中。这为我们提供了*参数 (parameter)化*逆向步骤的均值$\mu_\theta(x_t, t)$：


$$
\mu_\theta(x_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \left( \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)) \right) + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t
$$


这看起来很复杂，但在经过一些代数简化后（使用关系式$\alpha_t = 1 - \beta_t$和$\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$），它简化为一个更简洁的形式：


$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$


这个方程是去噪步骤的核心。它告诉我们，给定当前状态$x_t$以及U-Net模型在该时间步预测的噪声$\epsilon_\theta(x_t, t)$，如何计算上一时间步$x_{t-1}$的分布均值。

对于逆向步骤$p_\theta(x_{t-1}|x_t)$的方差，我们需要选择一个值$\sigma_t^2$。DDPM论文建议使用源自后验分布$q(x_{t-1}|x_t, x_0)$的方差，即$\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$。后面讨论的另一个选项是简单地将$\sigma_t^2 = \beta_t$。这个方差项为生成过程引入了随机性；从这个分布中采样涉及添加按$\sigma_t$缩放的高斯噪声。

因此，逆向扩散过程中的单一步骤定义为从高斯分布中采样$x_{t-1}$：


$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$


如下：

- $x_t$是当前时间步的含噪数据。
- $\epsilon_\theta(x_t, t)$是神经网络预测的噪声。
- $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$是$x_{t-1}$的预测均值。
- $\sigma_t^2$是方差，通常选择为$\tilde{\beta}_t$或$\beta_t$。
- $\alpha_t$、$\beta_t$、$\bar{\alpha}_t$是噪声调度决定的常数。

通过反复应用此公式，从纯噪声$x_T \sim \mathcal{N}(0, I)$开始，并从$t=T$逆向步进到$t=1$，我们可以生成一个样本$x_0$，它应该类似于模型训练时的数据。这个迭代过程，在每一步都由神经网络的噪声预测引导，就是扩散模型生成新数据的方式。我们将在第5章介绍完整的采样算法。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇开创性论文介绍了去噪扩散概率模型（DDPM）框架，提供了前向和反向扩散过程的基础数学公式，包括本节中讨论的后验分布 q(x_{t-1}|x_t, x_0) 的推导。
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — Niels Rogge, Kashif Rasul (2022)
  Publisher: Hugging Face
  这篇详尽的教程是Hugging Face Diffusers文档的一部分，提供了DDPM数学步骤的注释实现和解释，与本节内容紧密相关。
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole (2020)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
  这篇论文将DDPM与基于分数的模型统一起来，并引入了生成扩散的连续时间框架，为反向过程提供了更深入的理论理解。
