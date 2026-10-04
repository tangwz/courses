---
course: "bayesian-machine-learning"
chapter: "bayesian-deep-learning"
lesson: "mcmc-methods-bnns"
sourceId: 3626
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-6-bayesian-deep-learning/mcmc-methods-bnns"
title: "用于贝叶斯神经网络的MCMC方法（例如，随机梯度哈密顿蒙特卡洛）"
description: "了解先进的MCMC技术（例如SGHMC）在训练贝叶斯神经网络中的应用。"
order: 4
plots: []
sourceHash: "7822eec6ff288edb022c7df94f50eda20d50779ad46394b9b8015f1de9c1b386"
sourceCorrections: []
---

贝叶斯神经网络 (neural network)（BNN）提供了一种原则性方法，通过将网络权重 (weight) $w$ 视为具有后验分布 $p(w | \mathcal{D})$ 的随机变量，从而将不确定性纳入深度学习 (deep learning)。主要的难题在于描述这种高维后验。MCMC方法，例如哈密顿蒙特卡洛（HMC），是用于从复杂分布中采样的高效技术。然而，将这些方法直接应用于大型神经网络时，会遇到严重的计算难题。

### 标准MCMC的扩展性问题

标准HMC需要计算对所有参数 (parameter) $w$ 的对数后验梯度。对数后验由贝叶斯定理给出：


$$
\log p(w | \mathcal{D}) = \log p(\mathcal{D} | w) + \log p(w) - \log p(\mathcal{D})
$$


梯度项 $\nabla_w \log p(w | \mathcal{D})$ 包含对数似然的梯度 $\nabla_w \log p(\mathcal{D} | w)$ 和对数先验的梯度 $\nabla_w \log p(w)$。假设数据点 $\mathcal{D} = \{x_i, y_i\}_{i=1}^N$ 是独立同分布的，则对数似然是整个数据集上的总和：


$$
\log p(\mathcal{D} | w) = \sum_{i=1}^N \log p(y_i | x_i, w)
$$


计算 $\nabla_w \log p(\mathcal{D} | w)$ 需要对所有 $N$ 个数据点进行一次完整遍历。对于 $N$ 可能达到数百万或数十亿的深度学习 (deep learning)数据集，在HMC蛙跳积分的每一步计算此梯度会变得成本过高。这使得标准HMC无法适用于典型的深度学习情况。

### MCMC的改进：随机梯度方法

解决方案与训练标准深度神经网络 (neural network)的方法相似：使用小批量进行随机梯度估计。我们不使用完整数据集计算梯度，而是在每次迭代 $t$ 时，使用一个大小为 $M \ll N$ 的小而随机采样的子集（一个小批量）$\mathcal{D}_t = \{x_i, y_i\}_{i \in \mathcal{I}_t}$ 来近似它。

对数后验的梯度近似为：


$$
\nabla_w \log p(w | \mathcal{D}) \approx \frac{N}{M} \sum_{i \in \mathcal{I}_t} \nabla_w \log p(y_i | x_i, w) + \nabla_w \log p(w)
$$


这会给梯度估计引入噪声。简单地将这些带噪声的梯度代入标准HMC动力学中，会导致不正确的采样行为；模拟轨迹会发散，并且采样器不会收敛到真实的后验分布。

### 随机梯度哈密顿蒙特卡洛（SGHMC）

随机梯度哈密顿蒙特卡洛（SGHMC）是HMC的一种改进，专门用于处理来自小批量的噪声梯度。由Chen、Fox和Guestrin（2014）提出，SGHMC修改了哈密顿动力学方程，以考虑梯度噪声，从而在特定条件下确保收敛到正确的预期分布。

回顾HMC模拟一个具有位置 $w$ （权重 (weight)）和动量 $p$ 的物理系统。其动力学由哈密顿量 $H(w, p) = U(w) + K(p)$ 控制，其中 $U(w) = -\log p(w|\mathcal{D})$ 是势能（负对数后验），$K(p)$ 是动能。

SGHMC修改了动量更新步骤。SGHMC不直接使用带噪声的梯度估计 $\nabla_w \tilde{U}(w)$ （其中 $\tilde{U}(w)$ 是使用小批量估计的势能），而是引入了一个 *摩擦项* $C$。对于步长 $\epsilon$ 的离散更新方程如下：

1. **位置更新：**

   
   $$
   w_{t+1} = w_t + \epsilon M^{-1} p_t
   $$
   

   （此处 $M^{-1}$ 是逆质量矩阵，通常取为单位矩阵）。
2. **动量更新：**

   
   $$
   p_{t+1} = p_t - \epsilon \nabla_w \tilde{U}(w_t) - \epsilon C M^{-1} p_t + \mathcal{N}(0, 2\epsilon(C - \hat{B}))
   $$
   

让我们分解一下动量更新：

- $-\epsilon \nabla_w \tilde{U}(w_t)$：标准力项，但使用了势能的*随机*梯度。
- $-\epsilon C M^{-1} p_t$：新增的摩擦项。此项会抑制动量，有助于抵消随机梯度注入的噪声。$C$ 是用户指定的正定矩阵（通常为对角矩阵，$C = \alpha I$）。它的作用类似于带有动量的SGD等优化算法中的动量衰减。
- $\mathcal{N}(0, 2\epsilon(C - \hat{B}))$：添加到动量更新中的高斯噪声项。这是必需的。$\hat{B}$ 是随机梯度估计中噪声协方差的估计。这种注入的噪声精确地补偿了离散化和摩擦项的影响，确保采样器的平稳分布保持为预期后验 $p(w|\mathcal{D})$。在实际中，估计 $\hat{B}$ 可能很复杂，通常假定它为零或一个简单的标量，这需要调整 $C$。

引入摩擦项 $C$ 是主要的修改。它有助于控制噪声梯度引入的方差，稳定模拟。附加的噪声项纠正了动力学，以确保收敛到正确后验分布的理论保证。

### SGHMC的实际考虑

为贝叶斯神经网络 (neural network)实现SGHMC涉及多项选择：

- **小批量大小 ($M$)：** 较大的批量可以减少梯度噪声，但会增加每一步的计算量。需要找到一个平衡点。
- **步长 ($\epsilon$)：** 类似于SGD中的学习率。需要仔细调整。过大会导致不稳定；过小会导致采样慢。循环步长等技术有时会有帮助。
- **摩擦项 ($C$)：** 控制阻尼。较高的摩擦力会带来更高的稳定性，但可能导致状态空间采样速度变慢。它需要与估计的噪声协方差 $\hat{B}$（如果使用）和步长进行平衡。通常通过经验调整。
- **积分步数：** 类似于HMC，SGHMC在生成样本之前，会模拟一定数量的步（蛙跳步）的动力学。步数影响计算成本和采样效率。
- **预热与稀疏化：** 标准MCMC实践适用。初始样本（预热期）会被丢弃，样本可能进行稀疏化以减少自相关。
- **收敛诊断：** 相较于标准MCMC，使用随机梯度评估收敛更具挑战性。像 $\hat{R}$ 这样的标准诊断可能不太可靠。观察重要参数 (parameter)或模型性能指标随迭代变化的轨迹图是很重要的。

### 其他随机梯度MCMC方法

SGHMC是一种重要方法，但也有其他方法：

- **随机梯度朗之万动力学（SGLD）：** 一种更简单的方法，主要是在SGD更新中添加高斯噪声。当摩擦项 $C$ 较高时（过阻尼极限），它对应于SGHMC的动力学。它更容易实现，但在某些情况下可能比SGHMC混合速度慢。
  
  $$
  \Delta w_t = \frac{\epsilon_t}{2} (\nabla_w \log p(w_t) + \frac{N}{M} \sum_{i \in \mathcal{I}_t} \nabla_w \log p(y_i | x_i, w_t)) + \eta_t
  $$
  
  其中 $\eta_t \sim \mathcal{N}(0, \epsilon_t)$。步长 $\epsilon_t$ 必须随时间衰减才能收敛。

### 优点和缺点

**SGMCMC用于贝叶斯神经网络 (neural network)的优点：**

- **可扩展性：** 使得MCMC采样可以应用于标准MCMC不可行的大型数据集和复杂模型。
- **完整后验描述（理论上）：** 与变分推断不同，MCMC方法旨在从真实后验中采样，在时间充足的情况下，可能更好地捕捉复杂形状和多峰性。
- **理论依据：** 构建在MCMC成熟的理论依据之上。

**缺点：**

- **调优复杂性：** 需要仔细调整步长、摩擦、小批量大小，以及可能的噪声估计。不良的调优可能导致收敛缓慢或发散。
- **计算成本：** 尽管比标准MCMC更具可扩展性，但SGMCMC方法在训练深度网络方面通常仍比VI或标准基于梯度的优化慢得多。生成许多独立样本可能需要大量时间。
- **收敛诊断：** 可靠地评估收敛仍然是研究的一个开放问题，在实际中可能很困难。

总之，像SGHMC和SGLD这样的随机梯度MCMC方法提供了一种将MCMC采样原理应用于贝叶斯深度学习 (deep learning)这一具有挑战性方面的方式。它们提供了一条从BNN后验分布获取样本的途径，实现了比点估计更丰富的量化 (quantization)不确定性，但由于小批量处理中固有的噪声梯度估计，它们需要仔细的实现和调整。它们是计算密集型但可能更准确的替代方法，与接下来讨论的变分方法形成对比。

## 参考资料

- [Stochastic Gradient Hamiltonian Monte Carlo](https://proceedings.mlr.press/v32/chenc14.html) — Chen, Chuan, Fox, Emily, Guestrin, Carlos (2014)
  Journal: Proceedings of the 31st International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 32; Pages: 1-9; DOI: [10.5555/3044806.3045056](https://doi.org/10.5555/3044806.3045056)
  提出了用于处理带噪声梯度的大规模贝叶斯推断的随机梯度哈密顿蒙特卡洛（SGHMC）方法。
- [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://icml.cc/2011/papers/398_icmlpaper.pdf) — Max Welling, Yee Whye Teh (2011)
  Journal: Proceedings of the 28th International Conference on Machine Learning (ICML); Publisher: Omnipress; Pages: 681-688
  介绍了随机梯度朗之万动力学（SGLD），作为一种利用随机梯度进行大规模贝叶斯采样的早期方法。
- [MCMC using Hamiltonian Dynamics](http://arxiv.org/abs/1206.1901) — Neal, Radford M. (2011)
  Journal: Handbook of Markov Chain Monte Carlo; Publisher: Chapman & Hall/CRC; Pages: 113-162; DOI: [10.1201/b10905-6](https://doi.org/10.1201/b10905-6)
  《马尔可夫链蒙特卡洛手册》中的第5章。详细介绍了哈密顿蒙特卡洛方法。
