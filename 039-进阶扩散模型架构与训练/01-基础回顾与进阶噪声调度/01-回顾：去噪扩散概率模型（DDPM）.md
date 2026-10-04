# 回顾：去噪扩散概率模型（DDPM）

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-1-diffusion-foundations-advanced-noise/ddpm-recap)

[返回章节目录](README.md) · [返回课程目录](../README.md)

去噪扩散概率模型（DDPM）是一个许多高阶技术所建立的底层框架。理解其主要运作方式，对于探索更复杂的架构和训练方法非常必要。DDPMs遵循一个简单而有效的工作原理：通过正向扩散过程系统地破坏数据结构，然后学习一个反向过程来恢复它，从而有效地生成新数据。

### 正向过程：添加噪声

正向过程，记为 $q$，在 $T$ 个离散时间步长上逐步向初始数据点 $x_0$（例如图像）添加高斯噪声。此过程定义为马尔可夫链，其中时间步 $t$ 的状态 $x_t$ 仅依赖于前一步的状态 $x_{t-1}$。

具体来说，转换由条件高斯分布定义：


$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
$$


这里，$\beta_t$ 表示在步长 $t$ 添加的噪声的方差。这些方差由预设的*方差调度* $\{\beta_t\}_{t=1}^T$ 决定，通常是随时间增加的小值（例如，从 $10^{-4}$ 到 $0.02$）。随着 $t$ 增加，数据逐步失去其独特特性，如果 $T$ 足够大，最终会接近各向同性高斯分布 $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

此过程的一个有用性质是，我们可以直接从 $x_0$ 采样 $x_t$，而无需迭代中间步骤。设 $\alpha_t = 1 - \beta_t$ 且 $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，则分布 $q(x_t | x_0)$ 也是高斯的：


$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$


这使得我们能够使用重参数 (parameter)化技巧将 $x_t$ 表示为：


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$


其中 $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是标准高斯噪声。这种表示方式对高效训练很要紧。

> 该图描绘了正向扩散过程，逐步从数据 $x_0$ 添加噪声直至近似噪声 $x_T$。

### 反向过程：学习去噪

生成能力源于反向过程 $p_\theta$，其目的是逆转扩散。从纯噪声 $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，目标是学习转换 $p_\theta(x_{t-1} | x_t)$ 以逐步去除噪声并最终采样一个真实数据点 $x_0$。

当以 $x_0$ 为条件时，真实的反向转换 $q(x_{t-1} | x_t, x_0)$ 是可处理的，并且也是高斯的。然而，在生成过程中 $x_0$ 是未知的。DDPM 使用神经网络 (neural network)学习的参数 (parameter)化高斯分布来近似真实的反向转换 $q(x_{t-1} | x_t)$：


$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$


神经网络，通常是U-Net架构（我们将在第二章中详细介绍），将带噪声数据 $x_t$ 和时间步长 $t$ 作为输入。它被训练来预测反向转换的参数。对于DDPM，方差 $\Sigma_\theta(x_t, t)$ 通常固定为与正向过程方差相关的值（$\sigma_t^2 \mathbf{I}$，其中 $\sigma_t^2$ 通常是 $\beta_t$ 或 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$）。该网络主要集中于学习均值 $\mu_\theta(x_t, t)$。

### 训练目标

训练涉及最大化数据的对数似然 $\log p_\theta(x_0)$。这通过优化对数似然的变分下界 (ELBO) 来实现。通过数学推导（涉及正向和反向过程的性质），目标函数可以得到大幅简化。实践中常用的一种有效目标函数是：


$$
L_{simple}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0 \sim \text{数据}, \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]
$$


其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，且 $\epsilon_\theta$ 是神经网络 (neural network)的输出。在这种表示中，网络被训练来预测添加到 $x_0$ 以获得 $x_t$ 的噪声 $\epsilon$。预测噪声 $\epsilon$ 已被发现比直接预测均值 $\mu_\theta(x_t, t)$ 或去噪图像 $x_0$ 更稳定和有效。

该目标函数将DDPM与基于分数的生成模型联系起来。最优的 $\epsilon_\theta(x_t, t)$ 与分数函数 $\nabla_{x_t} \log q(x_t)$ 相关，后者代表数据空间中指向更高密度的方向。实际上，神经网络学习估计这个按比例缩放的分数函数。我们将在本章稍后讨论分数匹配和ODE时再次提到这种联系。

### 采样

一旦模型 $p_\theta$ 训练完成，生成新样本包括：

1. 从先验分布中采样 $x_T$：$x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
2. 对于 $t = T, T-1, \dots, 1$，迭代地从 $p_\theta(x_{t-1} | x_t)$ 采样 $x_{t-1}$。这涉及在每个步骤运行神经网络 (neural network) $\epsilon_\theta$ 来计算 $\mu_\theta(x_t, t)$，然后从所得的高斯分布中采样。

这种迭代过程需要对神经网络进行 $T$ 次顺序评估，使得标准DDPM采样相对于其他生成模型（如GAN或VAE）而言相对较慢。像DDIM（我们接下来会回顾）以及在第六章中讨论的进阶采样器都旨在加速这一过程。

本次回顾提供了DDPM所需的背景。尽管原理简单，但噪声调度、网络架构和目标公式的选择提供了许多改进的方向，构成了后续章节中涵盖的高阶主题的要点。在我们继续分析噪声调度和研究更复杂的模型变体时，请记住这种基本结构。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 6840-6851; DOI: [10.55917/cb9f6785](https://doi.org/10.55917/cb9f6785)
  介绍并推广了去噪扩散概率模型（DDPM）的原始论文，详细阐述了其前向和反向过程、训练目标和生成能力。
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://proceedings.mlr.press/v37/sohl-dickstein15.pdf) — Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning; Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 37; Pages: 2252-2261; DOI: [10.5555/3045118.3045233](https://doi.org/10.5555/3045118.3045233)
  通过借鉴非平衡热力学原理，引入了扩散模型概念的奠基性论文，为后续DDPM的发展提供了理论基础。
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://proceedings.neurips.cc/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf) — Yang Song, Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 32; Pages: 11895-1907; DOI: [10.55917/cb0b8364](https://doi.org/10.55917/cb0b8364)
  介绍了基于分数的生成模型和分数匹配，为DDPM中使用的简化训练目标建立了强大的理论连接。

---

[下一节](02-%E5%9B%9E%E9%A1%BE%EF%BC%9A%E5%8E%BB%E5%99%AA%E6%89%A9%E6%95%A3%E9%9A%90%E5%BC%8F%E6%A8%A1%E5%9E%8B%20%28DDIM%29.md)
