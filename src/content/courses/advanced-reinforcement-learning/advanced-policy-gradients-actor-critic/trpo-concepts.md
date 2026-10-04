---
course: "advanced-reinforcement-learning"
chapter: "advanced-policy-gradients-actor-critic"
lesson: "trpo-concepts"
sourceId: 3396
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/trpo-concepts"
title: "信任区域策略优化 (TRPO)"
description: "TRPO 的理论基础，确保策略更新保持在信任区域内以实现稳定学习。"
order: 7
plots: []
sourceHash: "dc361c552aa16b982f3dbe163b772c5bea6f0dd703a637576ff5aa356b174628"
sourceCorrections: []
---

虽然基本的策略梯度方法通过沿着估计梯度的方向迈步来更新策略，但选择合适的步长（学习率）具有挑战性。步长过小会导致进展缓慢，而步长过大则可能大幅改变策略，这可能导致性能灾难性下降，智能体可能永远无法恢复。Actor-Critic 方法使用评论员来获得更好的梯度估计，但步长问题依然存在。

信任区域策略优化 (TRPO) 直接解决了这个稳定性问题。TRPO 不仅仅是遵循梯度，它旨在在每一步中最大化策略的预期性能提升，但关键在于，它增加了一个约束：新策略不能与旧策略偏离太远。当前策略周围的这个受限区域就是“信任区域”，这意味着我们只在这个局部范围内信任我们的性能估计。

### 核心思想：受限策略更新

想象您正在调整策略 $\pi_{\theta}$ 的参数 (parameter) $\theta$。标准策略梯度方法会告诉您哪个方向 $g$（梯度）在局部增加了性能。它们随后更新 $\theta_{k+1} = \theta_k + \alpha g$，其中 $\alpha$ 是学习率。TRPO 采用了一种不同的方法。它旨在找到最佳的参数 $\theta$，以最大化代表性能提升的目标函数，同时确保新策略 $\pi_{\theta}$ 与旧策略 $\pi_{\theta_k}$ 之间的“距离”保持较小。

这个“距离”通常使用 Kullback-Leibler (KL) 散度 $D_{KL}(\pi_{\theta_k} || \pi_{\theta})$ 来衡量，它量化 (quantization)了新策略定义的概率分布与旧策略的不同程度。

### TRPO 优化问题

TRPO 旨在最大化一个替代目标函数，该函数近似于新策略 $\pi_{\theta}$ 相对于旧策略 $\pi_{\theta_k}$ 的预期优势。这个目标函数使用重要性采样来修正我们正在使用通过 $\pi_{\theta_k}$ 收集的数据来评估 $\pi_{\theta}$ 的事实：


$$
L_{\theta_k}(\theta) = \mathbb{E}_{s \sim \rho_{\theta_k}, a \sim \pi_{\theta_k}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a) \right]
$$


这里，$\rho_{\theta_k}$ 是运行策略 $\pi_{\theta_k}$ 时遇到的状态分布，$A^{\pi_{\theta_k}}(s,a)$ 是旧策略下的优势估计（例如，来自 GAE）。

TRPO 的创新在于约束更新步长。我们希望新旧策略之间的平均 KL 散度，在旧策略访问的状态上，小于某个小常数 $\delta$：


$$
\mathbb{E}_{s \sim \rho_{\theta_k}} [D_{KL}(\pi_{\theta_k}(\cdot|s) || \pi_{\theta}(\cdot|s))] \le \delta
$$


因此，每次迭代 $k$ 的完整 TRPO 优化问题是：


$$
\theta_{k+1} = \arg \max_{\theta} L_{\theta_k}(\theta)
$$


$$
\text{受限于 } \bar{D}_{KL}(\pi_{\theta_k} || \pi_{\theta}) \le \delta
$$


其中 $\bar{D}_{KL}$ 表示上述平均 KL 散度。超参数 (parameter) (hyperparameter) $\delta$ 定义了信任区域的大小。

### 通过近似进行实际实现

精确求解这个受限优化问题是困难的。TRPO 通过使用近似使其变得可行：

1. **目标近似：** 目标函数 $L_{\theta_k}(\theta)$ 通过其在 $\theta_k$ 附近的一阶泰勒展开式进行近似：
   $L_{\theta_k}(\theta) \approx L_{\theta_k}(\theta_k) + g^T (\theta - \theta_k)$，其中 $g = \nabla_{\theta} L_{\theta_k}(\theta)|_{\theta=\theta_k}$ 是在当前参数 (parameter) $\theta_k$ 处评估的策略梯度。由于 $L_{\theta_k}(\theta_k)$ 是常数，最大化它等价于最大化 $g^T (\theta - \theta_k)$。
2. **约束近似：** KL 散度约束通过其在 $\theta_k$ 附近的二阶泰勒展开式进行近似：
   $\bar{D}_{KL}(\pi_{\theta_k} || \pi_{\theta}) \approx \frac{1}{2} (\theta - \theta_k)^T F (\theta - \theta_k)$，其中 $F$ 是费雪信息矩阵 (FIM)。FIM 像策略参数空间上的一个度量，衡量策略的输出分布对于参数的微小变化改变了多少。它被定义为 $F = \mathbb{E}_{\pi_{\theta_k}}[ \nabla_{\theta} \log \pi_{\theta}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s)^T ]|_{\theta=\theta_k}$。

替换这些近似后，问题变为找到求解更新步长 $\Delta \theta = \theta - \theta_k$ 的方法：


$$
\max_{\Delta \theta} g^T \Delta \theta
$$


$$
\text{受限于 } \frac{1}{2} \Delta \theta^T F \Delta \theta \le \delta
$$


这是一个带有二次约束的标准二次规划问题。

### 共轭梯度法

直接计算、存储和求 FIM $F$ 的逆在计算上成本很高，特别是对于大型神经网络 (neural network)。TRPO 巧妙地通过使用**共轭梯度 (CG)** 算法来避免这一点。CG 是一种求解 $Fx = g$ 形式的线性方程组的迭代方法。重要的是，CG 只需要能够计算矩阵向量 (vector)积，特别是对于任意向量 $v$ 计算 $Fv$。这种 FIM 向量积可以不显式形成 $F$ 而高效地计算，使用类似于计算梯度本身的技术（通常涉及自动微分）。

CG 近似找到最佳步长方向 $\Delta \theta \propto F^{-1} g$。随后，步长的幅度会进行缩放，以满足 KL 约束 $\frac{1}{2} \Delta \theta^T F \Delta \theta = \delta$。

### 线搜索

因为泰勒近似只在局部精确，通过 CG 计算的步长可能实际上不满足 KL 约束，或者不能改善*真实的*（未近似的）目标函数。因此，TRPO 执行**线搜索**：它沿着 CG 找到的方向尝试指数式缩小的步长（$\alpha^j \Delta \theta$ 对于 $j=0, 1, 2, ...$，其中 $\alpha \in (0, 1)$ 是回溯系数），直到找到一个既满足平均 KL 约束又为替代目标函数 $L_{\theta_k}$ 产生正值的步长。

### TRPO 更新流程

以下图表展示了单个 TRPO 策略更新中的操作序列：

> 此图表展示了 TRPO 中一次更新步长的典型工作流程。它涉及数据收集、优势估计、梯度计算、使用共轭梯度法和费雪向量 (vector)积来找到搜索方向，最后进行线搜索以确保在更新策略之前满足约束。

### TRPO 的优点与缺点

**优点：**

- **稳定性：** 与标准策略梯度方法相比，通常提供更稳定和单调的性能提升。
- **理论基础：** 基于优化一个定义明确的目标，并受限于与策略变化相关的有意义的约束。
- **强大性能：** 在复杂连续控制基准测试上通常能取得不错的结果。

**缺点：**

- **计算复杂度：** 由于 CG 迭代以及对二阶信息（FIM 向量 (vector)积）的需求，每次更新的计算成本远高于 A2C 或普通策略梯度等一阶方法。
- **实现复杂度：** 与更简单的策略梯度变体相比，正确实现它更复杂。
- **兼容性问题：** TRPO 中使用的推导和近似不轻易适应某些神经网络 (neural network)技术，例如 Actor 和 Critic 之间的参数 (parameter)共享、Dropout 或批量归一化 (normalization)，因为这些技术可能会干扰 KL 散度及其二次近似的精确计算。

TRPO 在开发可靠的策略优化算法方面代表着一个重要的进步。然而，它的复杂性促使人们寻求能够实现类似稳定性保证的更简单方法。这直接促成了近端策略优化 (PPO) 的发展，我们将在接下来进行讨论。

## 参考资料

- [Trust Region Policy Optimization](http://proceedings.mlr.press/v37/schulman15.pdf) — John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 37; Pages: 1889-1897
  这篇基础论文介绍了信任区域策略优化（TRPO），详述了其理论基础、约束优化和实际近似方法。
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438)
  这篇论文介绍了广义优势估计（GAE），一种用于减少优势估计方差的方法，TRPO及其他策略梯度算法均有使用。
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  一本关于强化学习的教科书，涵盖策略梯度和Actor-Critic方法，为策略优化中的稳定性问题提供了背景。
- [A Natural Policy Gradient](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGTveVmE5Pi33juiC0TEqZ9ow3F-mhSAQL1oyFhaHJUkatky0l0m98kqIzzv1VBOCVIoPIUXD5d8BG3nsv3o0sU4PZB7A55-A6qouA8K9w-XX6V7VI_o44t0U_0FVzptTbmeSLng90Y7_cQmgRiQ9MBlwHOKQ=) — Sham M. Kakade (2001)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 14; Pages: 1339-1346
  这篇论文提出了使用Fisher信息矩阵的自然策略梯度的理论基础，对TRPO的约束优化方法有影响。
