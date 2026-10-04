---
course: "advanced-diffusion-architectures"
chapter: "consistency-models"
lesson: "consistency-training-distillation"
sourceId: 5452
sourceUrl: "https://apxml.com/zh/courses/advanced-diffusion-architectures/chapter-5-consistency-models/consistency-training-distillation"
title: "一致性模型训练：蒸馏方法"
description: "一致性模型如何通过从预训练扩散模型（教师）中蒸馏知识来训练。"
order: 3
plots: []
sourceHash: "7404bb28ba7eb4f72553d84652a7f58cb2385cce0364e1e0e48f36c7f0c81033"
sourceCorrections: []
---

借助预训练 (pre-training)扩散模型的强大能力，为训练一致性模型提供了一种高效的途径。这种方法，称为一致性蒸馏（CD），将现有扩散模型视为“教师”，指导“学生”一致性模型的训练。目标是将迭代教师模型学习到的生成能力，转移到能够快速（可能一步完成）生成的学生模型中。

### 师生框架

在此设置中：

1. **教师模型 ($\phi$)**：这是一个预训练 (pre-training)的、高性能扩散模型（例如经过DDPM或DDIM训练的模型）。它的作用是准确估计由扩散过程相关联的概率流ODE定义的解路径。它不直接生成最终输出，而是提供必要的中间步骤或分数估计。在一致性蒸馏过程中，教师模型的参数 (parameter)（$\phi$）是冻结的。
2. **学生模型 ($\theta$)**：这是我们旨在训练的一致性模型 $f_\theta(x, t)$。它接收一个噪声输入 $x_t$ 和一个时间步 $t$，并直接预测轨迹的估计起点 $\hat{x}_0$。
3. **目标模型 ($\theta^-$)**：为了稳定训练并提高性能，通常使用一个独立的目标网络 $f_{\theta^-}(x, t)$。这个网络的参数（$\theta^-$）是学生模型参数（$\theta$）的指数移动平均（EMA）。它在训练过程中为学生模型的预测提供目标值。

主要思想是训练学生模型 $f_\theta$，使其输出沿教师模型 $\phi$ 定义的轨迹保持一致。

### 一致性蒸馏目标

回想一致性特性：对于同一ODE轨迹上的任意点对 $(x_t, x_{t'})$，其中 $t' < t$，我们希望 $f(x_t, t) \approx f(x_{t'}, t')$。蒸馏通过最小化学生模型在较晚时间 $t$ 的输出与目标模型在较早时间 $t'$ （在同一轨迹上）的输出之间的差异来强制实现这一点，其中从 $x_t$ 到 $x_{t'}$ 的步进是使用教师模型估计的。

训练过程包含从离散化 $T=t_1 > t_2 > \dots > t_N = \epsilon > 0$ 中采样相邻时间步 $(t_{n+1}, t_n)$ 对。对于每一对：

1. 从 $p_{data}(x)$ 中采样一个数据点 $x_0$。
2. 采样高斯噪声 $z \sim \mathcal{N}(0, I)$。
3. 使用标准前向过程（例如，$x_{t_{n+1}} = \alpha_{t_{n+1}} x_0 + \sigma_{t_{n+1}} z$）生成对应于时间 $t_{n+1}$ 的噪声样本 $x_{t_{n+1}}$。
4. 使用**教师模型** $\phi$ 和一步ODE求解器（如欧拉或赫恩方法）来估计轨迹上先于 $x_{t_{n+1}}$ 的点 $x_{t_n}$。此步骤通常涉及使用教师的噪声预测 $\hat{\epsilon}_\phi(x_{t_{n+1}}, t_{n+1})$ 或分数估计 $\hat{s}_\phi(x_{t_{n+1}}, t_{n+1})$。例如，使用DDIM更新规则：
   
   $$
   \hat{x}_0 = \frac{x_{t_{n+1}} - \sigma_{t_{n+1}} \hat{\epsilon}_\phi(x_{t_{n+1}}, t_{n+1})}{\alpha_{t_{n+1}}}
   $$
   
   
   $$
   x_{t_n} = \alpha_{t_n} \hat{x}_0 + \sigma_{t_n} \hat{\epsilon}_\phi(x_{t_{n+1}}, t_{n+1})
   $$
   
   (注意：这里可以使用更复杂的ODE求解器以获得更好的精度)。
5. 计算一致性蒸馏损失：
   
   $$
   L_{CD}(\theta, \theta^-; \phi) = \mathbb{E}_{n, x_0, z} [ \lambda(t_n) d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(x_{t_n}, t_n)) ]
   $$
   
   这里：
   - $n$ 从 $\{1, \dots, N-1\}$ 中均匀采样。
   - $f_\theta(x_{t_{n+1}}, t_{n+1})$ 是学生模型使用“较晚”噪声样本进行的预测。
   - $f_{\theta^-}(x_{t_n}, t_n)$ 是目标模型使用通过教师模型估计的“较早”样本进行的预测。特别需要指出的是，梯度不通过目标网络 $f_{\theta^-}$ 或教师模型 $\phi$ 传播。
   - $d(\cdot, \cdot)$ 是衡量预测之间差异的距离函数。常见选择包括L2距离、L1距离或感知度量，如LPIPS。
   - $\lambda(t_n)$ 是一个可选的正加权函数，通常设为1。

### 目标网络更新

目标网络参数 (parameter) $\theta^-$ 使用学生参数 $\theta$ 的指数移动平均（EMA）定期更新：


$$
\theta^- \leftarrow \mu \theta^- + (1 - \mu) \theta
$$


动量参数 $\mu$ 通常接近1（例如0.99, 0.999）。这种缓慢的更新为学生模型提供了稳定的目标，防止振荡并提高收敛性，类似于强化学习 (reinforcement learning)和自监督学习 (supervised learning) (self-supervised learning)中使用的技术。

### 实现考量

- **时间步离散化 ($N$)**：训练期间使用的离散步数 $N$ 影响强制执行的一致性粒度。较大的 $N$ 提供更精细的控制，但会略微增加计算开销，因为它决定了可能的对 $(t_{n+1}, t_n)$。
- **ODE求解器**：使用教师模型从 $x_{t_{n+1}}$ 估计 $x_{t_n}$ 所选的ODE求解器，会影响目标的准确性。高阶求解器可能会以计算成本为代价带来更好的结果。
- **距离度量 ($d$)**：L2损失很常见，但L1对异常值更具鲁棒性。像LPIPS这样的损失有时可以产生与人类感知更一致的结果，特别是对于图像。
- **架构**：学生模型 $f_\theta$ 的架构通常仿照教师模型的架构（例如U-Net或DiT），但使用一致性目标进行训练。

> 图示一致性蒸馏训练过程。数据 $x_0$、噪声 $z$ 和时间步 $t_{n+1}$ 产生 $x_{t_{n+1}}$。教师模型 $\phi$ 帮助估计轨迹上的先前点 $x_{t_n}$。学生模型 $f_\theta$ 从 $x_{t_{n+1}}$ 预测起点，而目标模型 $f_{\theta^-}$ 从 $x_{t_n}$ 预测起点。损失函数 (loss function)最小化这些预测之间的距离，仅更新学生模型参数 (parameter) $\theta$。目标参数 $\theta^-$ 通过EMA从 $\theta$ 更新。

### 优点与缺点

**优点：**

- **使用强大的教师模型**：可以有效地转移最先进扩散模型的知识，而无需完全从头开始重新学习数据分布。
- **可能更快的收敛**：与从头训练（一致性训练）相比，蒸馏有时可以更快收敛，因为它有教师模型的强力指导。
- **高质量结果**：蒸馏得到的一致性模型已显示出在比其教师模型少得多的步骤中生成高保真样本的能力。

**缺点：**

- **依赖教师模型**：蒸馏得到的一致性模型的性能本质上受限于教师扩散模型的质量。教师模型中的任何缺陷或偏差都可能被转移。
- **需要预训练 (pre-training)模型**：这种方法需要有一个训练良好的扩散模型可用，这本身就需要大量的计算资源和数据。

一致性蒸馏提供了一种实用且高效的方法，通过在成熟扩散模型的成功基础上，获得快速生成模型。它代表了缓解扩散模型生成方法中常见的采样速度慢、从而限制其适用性的重要一步。

## 参考资料

- [Consistency Models](https://arxiv.org/abs/2303.01469) — Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever (2023)
  Journal: ICML 2023; DOI: [10.48550/arXiv.2303.01469](https://doi.org/10.48550/arXiv.2303.01469)
  介绍了一致性模型及其一致性蒸馏训练方法，可实现快速一步生成。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  提出了去噪扩散概率模型的核心架构和训练方法，该模型常作为一致性蒸馏中的教师模型。
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — Prafulla Dhariwal, Alex Nichol (2021)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2021); DOI: [10.48550/arXiv.2105.05233](https://doi.org/10.48550/arXiv.2105.05233)
  详细说明了如何训练高质量扩散模型并提高采样效率，使其成为蒸馏技术的有效教师模型。
- [Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378) — Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, Hang Zhao (2023)
  Journal: arXiv preprint arXiv:2310.04378; DOI: [10.48550/arXiv.2310.04378](https://doi.org/10.48550/arXiv.2310.04378)
  将一致性蒸馏应用于潜在空间扩散模型，以更少的推理步骤实现高分辨率图像生成。
