# 高级损失函数形式 (v-预测, L_simple)

来源：[原文](https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/advanced-loss-functions)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管标准扩散模型训练目标——预测在时间步 $t$ 添加的噪声 $\epsilon$——表现出色，但研究人员已经对模型的预测目标和损失函数 (loss function)考察了其他形式。这些替代方案在样本质量、训练稳定性以及模型在不同噪声水平下的表现方面有优势。两种重要的方法是广泛使用的简化损失 $L_{simple}$ (基于 $\epsilon$-预测) 和 $v$-预测目标。

### 标准形式：$\epsilon$-预测和 $L_{simple}$

大多数扩散模型都训练来预测添加到原始数据 $x_0$ 中以生成含噪声样本 $x_t$ 的噪声 $\epsilon$。回想一下前向过程：


$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{这里} \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$


该模型，通常是 U-Net 或 Transformer，表示为 $\epsilon_\theta$，接收含噪声输入 $x_t$ 和时间步 $t$ (通常通过嵌入 (embedding))，并输出噪声的预测：$\epsilon_\theta(x_t, t)$。

从变分下界 (VLB) 推导出的理论损失包含依赖于时间步 $t$ 的加权项。然而，DDPM 论文发现，简化、未加权版本的损失在实践中通常会产生更好的结果：


$$
L_{simple} = \mathbb{E}_{t \sim \mathcal{U}(1, T), x_0 \sim q(x_0), \epsilon \sim \mathcal{N}(0, \mathbf{I})} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$


在此，$t$ 是从总时间步数 $T$ 中均匀采样的。这个 $L_{simple}$ 目标在计算上很简单，并构成了许多成功扩散模型实现的核心。它通过预测添加的噪声来直接优化模型以对输入进行去噪。

尽管取得了成功，但预测 $\epsilon$ 有时会面临挑战，特别是在扩散过程的边界附近 (非常小的 $t$ 或非常大的 $t$)，那里信噪比变化剧烈。

### 替代方案：$v$-预测

一种旨在解决 $\epsilon$-预测某些局限性的替代形式是 $v$-预测。模型不预测噪声 $\epsilon$，而是预测一个不同的目标 $v$，定义如下：


$$
v = \alpha_t \epsilon - \sigma_t x_0
$$


在这里，我们使用常见表示法，其中 $\alpha_t = \sqrt{\bar{\alpha}_t}$ (信号尺度) 和 $\sigma_t = \sqrt{1 - \bar{\alpha}_t}$ (噪声尺度)，这与前向过程方程 $x_t = \alpha_t x_0 + \sigma_t \epsilon$ 一致。

为什么预测 $v$？其直觉与扩散过程中信号的缩放有关。目标 $\epsilon$ 始终具有单位方差。然而，含噪声输入 $x_t$ 的方差随 $t$ 变化。预测 $v$ 可以被解释为预测一个在不同时间步长上具有更一致方差的目标，这可能使网络的学习任务更简单或更稳定。它有效地平衡了分数函数 ($\nabla_{x_t} \log p(x_t)$) 和数据 $x_0$ 的预测。

模型架构基本保持不变 (例如 U-Net)，但现在将其参数 (parameter)化以输出 $v_\theta(x_t, t)$。相应的损失函数 (loss function)与 $L_{simple}$ 类似，但以 $v$ 为目标：


$$
L_v = \mathbb{E}_{t, x_0, \epsilon} \left[ \| v - v_\theta(x_t, t) \|^2 \right]
$$


通常，类似于完整 VLB 中的加权项可能会被重新引入，或者损失可能基于分数匹配原理来制定，但核心思想是预测这个 $v$ 目标。

#### $v$-预测的优势

1. **样本质量提升：** 特别是在 Google 的 Imagen 等模型中，与 $\epsilon$-预测相比，$v$-预测被发现能产生更高质量的样本，尤其是在生成高分辨率图像或涉及复杂条件时。
2. **更好的缩放特性：** $v$ 目标的方差为 $\alpha_t^2 + \sigma_t^2 = 1$，使得预测目标在时间上比 $\epsilon$ 更均匀地缩放，这可以简化网络优化。
3. **处理边界条件：** $v$-预测可以简化处理 $t=0$ (纯数据) 和 $t=T$ (纯噪声) 的边界情况，这可能使训练更稳定。

#### 实现考量

使用 $v$-预测时，你需要调整模型输出在采样过程中的使用方式。如果模型 $v_\theta(x_t, t)$ 预测 $v$，你可以根据采样步骤 (如 DDIM 或 DDPM) 的需要恢复 $\epsilon$ 和 $x_0$ 的预测值：


$$
\text{预测的 } \epsilon = \alpha_t v_\theta(x_t, t) + \sigma_t x_t
$$


$$
\text{预测的 } x_0 = \alpha_t x_t - \sigma_t v_\theta(x_t, t)
$$


这些恢复的值随后可以插入到标准采样方程中。

### $\epsilon$-预测和 $v$-预测的比较

在 $\epsilon$-预测 ($L_{simple}$) 和 $v$-预测 ($L_v$) 之间的选择并非总是明确的，可能取决于具体的应用、数据集和模型架构。

- **$L_{simple}$ ($\epsilon$-预测):**

  - 更易于实现和理解。
  - 在大量模型中被证明有效 (包括原始 DDPM、Stable Diffusion 1.x/2.x)。
  - 预测目标 $\epsilon$ 具有恒定单位方差。
  - 可能需要仔细的噪声调度设计和网络调整，特别是对于极端信噪比情况。
- **$L_v$ ($v$-预测):**

  - 可带来样本质量的提升，特别是在大型模型中。
  - 预测目标 $v$ 具有的特性可能使跨时间步的训练更稳定。
  - 在 Imagen 等模型中使用。
  - 需要在采样期间修改模型输出的解释。

下图说明了网络在每种形式中旨在预测的不同目标：

> 该图比较了 $\epsilon$-预测和 $v$-预测形式的预测目标和损失计算。两者都以含噪声数据 $x_t$ 和时间步 $t$ 作为输入。

最终，$L_{simple}$ 和 $L_v$ 都是训练扩散模型的有效目标。虽然 $L_{simple}$ 仍是一个基准，但 $v$-预测提供了一个有价值的替代方案，特别是在追求样本质量或处理训练动态时。通常需要通过实验来确定给定项目的最佳选择。了解这些不同的形式为您提供了更多工具，可以在训练期间优化和控制扩散模型的行为。

## 参考资料

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  这篇基础性论文介绍了去噪扩散概率模型（DDPM）框架和简化的噪声预测损失（$L_{simple}$），这已成为训练扩散模型的标准方法。
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine (2022)
  Journal: Advances in Neural Information Processing Systems; DOI: [10.48550/arXiv.2206.00364](https://doi.org/10.48550/arXiv.2206.00364)
  这项研究系统地分析了扩散模型的各种架构和训练设计选择，包括不同的损失权重和预测目标（如v-prediction），提供了它们对性能影响的见解。

---

[上一节](03-%E5%AE%9E%E6%96%BD%E4%B8%8E%E8%B0%83%E6%95%B4CFG%E5%B0%BA%E5%BA%A6.md) · [下一节](05-%E6%A8%A1%E5%9E%8B%E5%8F%82%E6%95%B0%E5%8C%96%20%28epsilon%E9%A2%84%E6%B5%8B%20vs.%20x0%E9%A2%84%E6%B5%8B%29.md)
