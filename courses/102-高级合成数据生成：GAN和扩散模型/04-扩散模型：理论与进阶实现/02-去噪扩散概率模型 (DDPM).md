# 去噪扩散概率模型 (DDPM)

来源：[原文](https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-4-diffusion-models-theory-implementation/ddpm-theory-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

去噪扩散概率模型（DDPM）是扩散模型大家族中一种杰出且高效的框架。该模型由 Ho 等人于 2020 年提出，它采用一个固定的多步前向过程，逐步向数据添加高斯噪声，并结合一个学习到的逆向过程，通过迭代去除噪声来生成新的数据样本。

### 前向过程：添加噪声

前向过程，也称为扩散过程，定义为一个马尔可夫链。它在 $T$ 个离散时间步长中，通过添加少量高斯噪声，逐步破坏一个初始数据点 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$。每个时间步 $t$ 的噪声水平由预定义的方差调度 $\{\beta_t\}_{t=1}^T$ 控制，其中 $0 < \beta_1 < \beta_2 < ... < \beta_T < 1$。通常，$T$ 很大（例如 $T=1000$），并且 $\beta_t$ 值很小，以确保每一步的变化都很小。

噪声样本 $\mathbf{x}_t$ 在给定前一个样本 $\mathbf{x}_{t-1}$ 时的分布是高斯分布：


$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$


在此，$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \sigma^2\mathbf{I})$ 表示关于变量 $\mathbf{x}$、均值为 $\boldsymbol{\mu}$、对角协方差为 $\sigma^2\mathbf{I}$ 的高斯分布。

这个过程的一个重要属性是，我们可以通过闭合形式直接从原始数据 $\mathbf{x}_0$ 采样 $\mathbf{x}_t$，避免了迭代所有中间步骤。令 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。那么，$\mathbf{x}_t$ 在 $\mathbf{x}_0$ 为条件下的分布也是高斯分布：


$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t) \mathbf{I})
$$


这个方程可以解释为将原始图像 $\mathbf{x}_0$ 按 $\sqrt{\bar{\alpha}_t}$ 缩放，并添加按 $\sqrt{1-\bar{\alpha}_t}$ 缩放的高斯噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$：


$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}
$$


随着 $t$ 趋近于 $T$，$\bar{\alpha}_t$ 趋近于 0（因为 $0 < \alpha_t < 1$）。因此，$\mathbf{x}_T$ 几乎完全独立于 $\mathbf{x}_0$，并类似于一个各向同性高斯分布 $\mathcal{N}(0, \mathbf{I})$。方差调度 $\beta_t$ 通常选择为线性增长、二次增长或基于余弦调度。

### 逆向过程：学习去噪

DDPM 的生成能力源于学习逆向过程：从纯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 开始，逐步去除噪声以获得一个逼真的样本 $\mathbf{x}_0$。这涉及学习 $t = T, T-1, ..., 1$ 的转移概率 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$。

如果 $\beta_t$ 足够小，真实的逆向转移 $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 也会近似为高斯分布。难点在于计算真实的逆向转移 $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 需要对所有可能的数据点进行边际化，这难以处理。然而，后验 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ *可以* 处理，并且可以证明是高斯分布：


$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$


这里


$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \mathbf{x}_t
$$


和 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$。

由于我们在逆向采样过程中没有 $\mathbf{x}_0$（这正是我们想要生成的！），我们使用神经网络 (neural network)来近似这个分布。DDPM 将逆向转移 $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 参数 (parameter)化为高斯分布：


$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$


目标是训练参数 $\theta$，使该分布紧密匹配真实的（但未知的）$q(\mathbf{x}_{t-1} | \mathbf{x}_t)$。

### 网络参数 (parameter)化与简化目标函数

在最初的 DDPM 论文中，逆向过程的方差固定为未训练的常数 $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma_t^2 \mathbf{I}$。两个常见的选择是 $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde{\beta}_t$。因此，神经网络 (neural network)只需要预测均值 $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$。

并非直接预测均值 $\boldsymbol{\mu}_\theta$，DDPM 将网络重新参数化，以预测在步骤 $t$ 添加的噪声 $\boldsymbol{\epsilon}$。回想前向过程方程：$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。我们可以重新排列此方程，以 $\mathbf{x}_t$ 和 $\boldsymbol{\epsilon}$ 表示 $\mathbf{x}_0$：


$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon})
$$


将此 $\mathbf{x}_0$ 的表达式代入理想均值 $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)$ 的公式中，使我们能够以预测噪声 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 表示均值 $\boldsymbol{\mu}_\theta$：


$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)
$$


这种重新参数化建立了最小化预测均值 $\boldsymbol{\mu}_\theta$ 与真实均值 $\tilde{\boldsymbol{\mu}}_t$ 之间差异，以及最小化预测噪声 $\boldsymbol{\epsilon}_\theta$ 与实际噪声 $\boldsymbol{\epsilon}$ 之间差异的联系。

作者发现，使用源自数据对数似然变分下界（VLB）的简化目标函数来训练模型，在实践中效果非常好，并且更易于实现。这个简化目标旨在最小化前向过程中添加的真实噪声 $\boldsymbol{\epsilon}$ 与网络预测的噪声 $\boldsymbol{\epsilon}_\theta$ 之间的均方误差：


$$
L_{simple}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(1, T), \mathbf{x}_0 \sim q(\mathbf{x}_0), \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]
$$


噪声样本 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 是使用原始数据 $\mathbf{x}_0$ 和在随机选择的时间步 $t$ 处采样的噪声 $\boldsymbol{\epsilon}$ 生成的。

本质上，网络 $\boldsymbol{\epsilon}_\theta$ 以噪声样本 $\mathbf{x}_t$ 和时间步 $t$ 作为输入，并试图预测从 $\mathbf{x}_0$ 生成 $\mathbf{x}_t$ 所使用的噪声分量 $\boldsymbol{\epsilon}$。用此目标进行训练有效地教导网络在不同噪声水平下去噪样本。

> 图示固定的前向（加噪）过程，从数据 $\mathbf{x}_0$ 到噪声 $\mathbf{x}_T$；以及学习到的逆向（去噪）过程，通过在每个步骤预测噪声 $\boldsymbol{\epsilon}_\theta$ 来从噪声 $\mathbf{x}_T$ 生成数据 $\mathbf{x}_0$。

### DDPM 训练与采样算法

**训练：**

1. 采样一个真实数据点 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$。
2. 从 $\{1, ..., T\}$ 中均匀采样一个时间步 $t$。
3. 采样噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$。
4. 计算噪声样本 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$。
5. 使用梯度下降 (gradient descent)，通过最小化损失 $\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \|^2$ 来训练神经网络 (neural network) $\boldsymbol{\epsilon}_\theta$。在多次迭代和多个数据点上重复此过程。

**采样（生成）：**

1. 从纯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ 开始。
2. 从 $t=T$ 倒数到 $1$ 迭代：
   a. 如果 $t > 1$，则采样 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$，否则 $\mathbf{z} = 0$。
   b. 使用训练好的网络预测噪声 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$。
   c. 使用包含预测噪声的逆向转移公式计算去噪样本 $\mathbf{x}_{t-1}$：
   $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$
   （此处 $\sigma_t$ 是固定的标准差，例如 $\sqrt{\beta_t}$ 或 $\sqrt{\tilde{\beta}_t}$）。
3. 最终结果 $\mathbf{x}_0$ 即为生成的样本。

DDPM 为理解现代基于扩散的生成建模提供了坚实的基础。其生成高质量样本的能力，尤其是在图像合成方面，源于这种精心定义的加噪过程，以及通过直观的噪声预测目标训练出的去噪网络。后续部分将介绍基于分数的解释等变体，以及提高采样效率和控制的方法。

## 参考资料

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Volume: 37; Pages: 2252-2261; DOI: [10.48550/arXiv.1503.03585](https://doi.org/10.48550/arXiv.1503.03585)
  这项早期工作提出了扩散概率模型的理论基础，为DDPMs等后续发展奠定了基础。

---

[上一节](01-%E6%95%B0%E5%AD%A6%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%EF%BC%9A%E9%9A%8F%E6%9C%BA%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B.md) · [下一节](03-%E5%9F%BA%E4%BA%8E%E5%BE%97%E5%88%86%E7%9A%84%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B.md)
