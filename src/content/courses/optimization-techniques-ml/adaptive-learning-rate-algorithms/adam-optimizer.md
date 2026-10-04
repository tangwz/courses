---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "adam-optimizer"
sourceId: 1265
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/adam-optimizer"
title: "Adam：结合动量与RMSprop"
description: "详细了解自适应矩估计（Adam）优化器，其中包含一阶和二阶矩估计。"
order: 4
plots: []
sourceHash: "a26fda22e2b048a522742759b30c955aabef0c46488034e6f3084f19ba49c600"
sourceCorrections: []
---

机器学习 (machine learning)中的优化算法通常根据梯度的矩来调整学习率。一些方法，如RMSprop，仅使用梯度的二阶矩来调整学习率。另一些方法，如动量（Momentum），则使用一阶矩来构建速度。由此引出一个自然的问题：能否将两者的优点结合起来？我们能否拥有一种算法，它既能利用动量在梯度方向一致时加速收敛，又能根据每个参数 (parameter)梯度的历史大小调整步长？Adam优化器对此给出了肯定的答案，并已成为深度学习 (deep learning)中最常用的优化算法之一。

Adam，即自适应矩估计（Adaptive Moment Estimation）的简称，通过跟踪过去梯度（一阶矩）和过去梯度平方（二阶矩）的指数衰减平均值，为每个参数计算自适应学习率。

### Adam算法

我们来细看Adam的运作机制。在每个时间步 $t$，当参数 (parameter)为 $\theta_t$ 且损失函数 (loss function)为 $J(\theta)$ 时，Adam执行以下步骤：

1. **计算梯度：** 计算损失函数对当前参数的梯度：
   $g_t = \nabla_{\theta} J(\theta_t)$
2. **更新有偏一阶矩估计：** 更新梯度的移动平均值（一阶矩估计）。这类似于动量更新，但使用由超参数 (hyperparameter) $\beta_1$ 控制的指数移动平均。
   $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   这里，$m_t$ 是梯度平均值的估计。$\beta_1$ 的常用值通常接近1，例如0.9。
3. **更新有偏二阶矩估计：** 更新梯度平方的移动平均值（二阶原点矩估计）。这类似于RMSprop中的更新，使用由超参数 $\beta_2$ 控制的指数移动平均。
   $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   这里，$v_t$ 估计梯度的未中心化方差（$g_t^2$ 使用逐元素平方）。$\beta_2$ 通常也接近1，例如0.999。
4. **计算偏差校正后的一阶矩估计：** 矩估计 $m_t$ 和 $v_t$ 被初始化为零向量 (vector)。这种初始化会引入对零的偏置 (bias)，尤其是在初始时间步。Adam通过计算偏差校正后的估计来纠正此偏置：
   $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
   随着 $t$ 增加，偏差校正项 $1 - \beta_1^t$ 接近1，使得校正作用减弱。
5. **计算偏差校正后的二阶矩估计：** 类似地，校正二阶矩估计中的偏置：
   $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
   项 $1 - \beta_2^t$ 用于校正二阶矩估计的偏置。
6. **更新参数：** 最后，更新参数。更新规则使用偏差校正后的一阶矩估计 $\hat{m}_t$（类似于动量），并将其除以偏差校正后的二阶矩估计 $\hat{v}_t$ 的平方根（类似于RMSprop），从而实现了每个参数的学习率缩放。
   $\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
   这里，$\eta$ 是学习率（步长），$\epsilon$ 是一个很小的常数（例如 $10^{-8}$），用于数值稳定性，以防止当 $\hat{v}_t$ 非常接近零时出现除以零的情况。

### 超参数 (parameter) (hyperparameter)

Adam引入了几个超参数：

- $\eta$：学习率或步长。控制参数更新的整体幅度。例如0.001的值通常是一个不错的起始点。
- $\beta_1$：一阶矩估计的指数衰减率。建议默认值为0.9。
- $\beta_2$：二阶矩估计的指数衰减率。建议默认值为0.999。
- $\epsilon$：一个防止除以零的小值。建议默认值为 $10^{-8}$。

Adam的一个吸引人之处在于，其默认超参数值通常在多种问题上表现良好，与一些其他优化器相比，减少了大量调优的需求。

### 原理与优点

Adam有效地结合了在RMSprop和动量中观察到的优点：

- **自适应学习率：** 像RMSprop一样，除以 $\sqrt{\hat{v}_t}$ 提供了每个参数 (parameter)的学习率。接收到大或频繁梯度的参数将具有较小的有效学习率，而梯度小或不频繁的参数将具有较大的有效学习率。这在稀疏梯度场景中尤其有益。
- **动量：** 使用一阶矩估计 $\hat{m}_t$ 引入了动量，帮助优化器在低曲率方向加速，并在高曲率方向抑制震荡，从而实现更快的收敛。
- **偏差校正：** 偏差校正步骤在训练初期尤其重要，因为此时矩估计仍然受到其零初始化的严重影响。若无校正，初始步长可能会远小于预期。

Adam计算效率高，内存开销小（只需存储与参数大小相同的一阶和二阶矩向量 (vector)），并且通常很适合处理具有大数据集和高维参数空间的问题，这在深度学习 (deep learning)中很常见。

### 与其他方法的关系

了解Adam与其他已讨论的算法的关系会很有帮助：

- 如果我们将 $\beta_1 = 0$，则一阶矩估计 $m_t$ 仅取决于当前梯度 $g_t$（按 $1-\beta_1$ 缩放），从而去除了动量作用。
- 二阶矩的更新和缩放与RMSprop非常相似，主要区别在于指数移动平均的具体形式和是否包含偏差校正。

Adam可以看作是RMSprop（根据二阶矩处理自适应缩放）和动量（利用一阶矩指导方向）的一种精巧结合，并增加了偏差校正机制，以提高优化初期阶段的稳定性。其广泛应用源于其在各种机器学习 (machine learning)任务中的强大经验表现，尤其是在训练深度神经网络 (neural network)方面。尽管后续章节将讨论像AMSGrad这样解决Adam可能存在的特定收敛问题的变体，但原始的Adam算法仍然是一种基本且高效的优化技术。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器的原始研究论文，详细阐述了其算法、偏差校正和实证性能。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习理论基础和实践的综合教材，其中第8章详细解释了Adam及其他优化算法。
- [CS231n: Deep Learning for Computer Vision, Lecture 7: Training Neural Networks I](https://cs231n.github.io/optimization-1/) — Stanford University (2023)
  Publisher: Stanford University
  斯坦福大学备受推崇的深度学习课程讲义，提供了包括Adam在内的各种优化技术的易懂且实用的解释。
