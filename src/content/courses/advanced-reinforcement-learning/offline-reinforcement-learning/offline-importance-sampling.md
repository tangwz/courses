---
course: "advanced-reinforcement-learning"
chapter: "offline-reinforcement-learning"
lesson: "offline-importance-sampling"
sourceId: 3487
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/offline-importance-sampling"
title: "重要性采样及其局限性"
description: "使用重要性采样进行离策略评估及其挑战，特别是高方差。"
order: 5
plots: ["plots/3487-0.json"]
sourceHash: "4dad3c0de28ea1a671ffa22bafd783efbca766c66f7a519c04d2ed75d1298970"
sourceCorrections: []
---

重要性采样（IS）是一种基本的统计方法，用于仅使用由可能不同的行为策略 $\pi_b$ 收集的数据来评估新策略 $\pi_e$。其核心思想精妙：对离线数据集中观测到的回报进行重新加权，以纠正策略之间的不匹配。如果一条轨迹在评估策略 $\pi_e$ 下比在行为策略 $\pi_b$ 下更可能出现，那么它的回报应被赋予更高的权重 (weight)，反之亦然。

### 重要性采样的工作原理

设想你有一个数据集 $\mathcal{D} = \{ \tau_1, \tau_2, ..., \tau_N \}$，包含 $N$ 条轨迹，其中每条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$ 都是通过遵循行为策略 $\pi_b$ 生成的。我们的目标是估计评估策略 $\pi_e$ 的预期回报，表示为 $J(\pi_e) = \mathbb{E}_{\tau \sim \pi_e} [R(\tau)]$，其中 $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ 是轨迹 $\tau$ 的累积折扣回报。

重要性采样允许我们使用从 $\pi_b$ 获得的样本来估计这个期望，方法是将每条轨迹 $R(\tau_i)$ 的回报乘以一个“重要性比率”，该比率量化 (quantization)了该轨迹在 $\pi_e$ 下相对于 $\pi_b$ 发生的相对概率。

### 数学公式

对于给定轨迹 $\tau$，在策略 $\pi$ 下观测到该状态和动作序列的概率（为简化起见，假设环境转移是确定的，尽管它可以扩展到随机情况）为：
$P(\tau | \pi) = p(s_0) \prod_{t=0}^{T} \pi(a_t | s_t) p(s_{t+1} | s_t, a_t)$
整个轨迹的重要性比率是这些概率在 $\pi_e$ 和 $\pi_b$ 下的比率：
$\rho_{0:T}(\tau) = \frac{P(\tau | \pi_e)}{P(\tau | \pi_b)} = \frac{p(s_0) \prod_{t=0}^{T} \pi_e(a_t | s_t) p(s_{t+1} | s_t, a_t)}{p(s_0) \prod_{t=0}^{T} \pi_b(a_t | s_t) p(s_{t+1} | s_t, a_t)} = \prod_{t=0}^{T} \frac{\pi_e(a_t | s_t)}{\pi_b(a_t | s_t)}$
设 $\rho_t = \frac{\pi_e(a_t | s_t)}{\pi_b(a_t | s_t)}$ 为每步重要性比率。那么完整轨迹比率就是 $\rho_{0:T} = \prod_{t=0}^{T} \rho_t$。

使用从 $\pi_b$ 获得的 $N$ 条轨迹的标准（或“普通”）重要性采样估计器 $J(\pi_e)$ 为：
$\hat{J}_{IS}(\pi_e) = \frac{1}{N} \sum_{i=1}^{N} \rho_{0:T}^{(i)} R(\tau_i)$
该估计器是无偏的，这意味着 $\mathbb{E}[\hat{J}_{IS}(\pi_e)] = J(\pi_e)$，这在理论上很吸引人。然而，其在离线强化学习 (reinforcement learning)中的实际效用受到严重限制，主要是因为其可能巨大的方差。

### 方差问题：重要性采样的致命弱点

重要性采样估计器的无偏性代价巨大：方差。$\hat{J}_{IS}(\pi_e)$ 的方差取决于加权回报的二阶矩，尤其涉及 $\mathbb{E}_{\tau \sim \pi_b}[(\rho_{0:T}(\tau))^2 R(\tau)^2]$。这里的核心项是平方轨迹重要性比率，即 $(\rho_{0:T})^2$。

考虑 $\rho_{0:T} = \prod_{t=0}^{T} \rho_t$ 的结构。这是可能由许多每步比率相乘得到的。

- 如果 $\pi_e$ 和 $\pi_b$ 即使在几步上略有不同，乘积 $\rho_{0:T}$ 也可能变得非常大或趋近于零。
- 特别地，方差通常随轨迹长度 $T$ 呈指数级增长。即使每步概率的微小差异也会以乘法方式累积，导致权重 (weight)相差许多数量级。
- 少数具有特别大重要性比率 $\rho_{0:T}^{(i)}$ 的轨迹可以主导估计器中的求和，使得估计结果非常不稳定和不可靠，尤其是在数据集大小 $N$ 有限的情况下。

方差随时间长度呈指数级增长是有效应用基本重要性采样在典型强化学习 (reinforcement learning)问题中的一个主要障碍，因为这些问题通常涉及长序列决策。



![IS权重偏度](plots/3487-0.json)



> 随着时间长度增加，重要性权重的分布趋于严重偏斜，许多权重接近零，而少数权重极其大，导致重要性采样估计值的高方差。（示意图）

### 对行为策略的敏感性和缺乏支持度

方差问题因行为策略 $\pi_b$ 的性质及其与评估策略 $\pi_e$ 的关系而加剧。

- **$\pi_b$ 中的小概率：** 如果行为策略将非常小的概率 $\pi_b(a_t | s_t)$ 分配给某个动作 $a_t$，而评估策略 $\pi_e$ 以非零概率 $\pi_e(a_t | s_t)$ 采取该动作，则相应的每步比率 $\rho_t$ 将非常大。这个单一的大比率可以极大地夸大轨迹比率 $\rho_{0:T}$。
- **缺乏重叠（支持度）：** 如果存在状态-动作对 $(s, a)$ 使得 $\pi_e(a|s) > 0$ 但 $\pi_b(a|s) = 0$，则重要性采样估计器将从根本上失效。如果评估策略可能采取了数据集中行为策略在该状态下从未采取过（或被采取的概率为零）的动作，那么重要性比率就会变为无穷大 ($\frac{>0}{0}$)，并且期望值将是未定义的。这是分布偏移的直接后果——离线数据可能根本不包含关于评估策略将采取的动作的信息。

### 已知行为策略的要求

“一个实际限制是，标准重要性采样要求明确知道数据集中遇到的所有状态-动作对的概率 $\pi_b(a_t | s_t)$。在许多收集离线数据的场景中（例如，来自已部署系统的日志、人类演示），生成数据的确切策略是未知的。人们可能会尝试从数据中估计 $\pi_b$（例如，使用行为克隆），但这会给OPE过程引入另一层近似和潜在误差，可能导致重要性采样估计产生偏差。”

### 缓解尝试：加权重 (weight)要性采样和每步重要性采样

已提出重要性采样的一些变体以缓解方差问题，尽管没有一个能完全解决：

1. **加权重要性采样（WIS）：** 加权重要性采样不是简单平均，而是通过权重的和来归一化 (normalization)加权回报：
   $\hat{J}_{WIS}(\pi_e) = \frac{\sum_{i=1}^{N} \rho_{0:T}^{(i)} R(\tau_i)}{\sum_{i=1}^{N} \rho_{0:T}^{(i)}}$
   加权重要性采样是一个有偏估计器（对于有限的 $N$，其期望并非精确等于 $J(\pi_e)$，但它是一致的），但它通常表现出显著更低的方差，尤其当权重差异很大时。它有效降低了具有巨大比率的异常轨迹的影响。
2. **每步重要性采样（PDIS）：** 每步重要性采样旨在通过仅在估计状态或状态-动作值时对感兴趣的时间步应用重要性修正来减少方差。为了估计预期回报，一种常见形式是考虑折扣重要性比率：
   $\hat{J}_{PDIS}(\pi_e) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \gamma^t \rho_{0:t}^{(i)} r_t^{(i)}$
   虽然在某些设置下可能提供方差减少，但与标准重要性采样相比，每步重要性采样仍然受到比率累积乘积的影响。

总而言之，尽管重要性采样提供了一个有理论依据的方法来执行离策略评估，但其极高的方差以及对策略不匹配和数据覆盖的敏感性使其在实际离线强化学习 (reinforcement learning)中常常不可靠。这些局限性突显了分布偏移带来的重大挑战，并促使了替代OPE方法和离线学习算法的开发，例如我们接下来将讨论的策略约束和值正则化 (regularization)技术，这些方法明确旨在缓解由评估在离线数据集中表示不足的动作或状态所引起的问题。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  提供强化学习的全面介绍，包括第七章中关于带重要性采样的异策略预测的基础解释。
- [High-Confidence Off-Policy Evaluation](https://doi.org/10.1609/aaai.v29i1.9541) — P. S. Thomas, G. Theocharous, M. Ghavamzadeh (2015)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 29(1); Pages: 3104-3110; DOI: [10.1609/aaai.v29i1.9541](https://doi.org/10.1609/aaai.v29i1.9541)
  解决异策略评估中高方差的重大挑战，提出了为重要性采样估计器提供置信区间的方法。
