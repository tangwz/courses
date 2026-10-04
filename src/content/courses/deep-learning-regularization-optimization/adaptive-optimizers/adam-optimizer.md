---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "adam-optimizer"
sourceId: 4992
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/adam-optimizer"
title: "Adam：自适应矩估计"
description: "讲解 Adam 优化算法，其整合了动量法和 RMSprop 的思路。"
order: 5
plots: []
sourceHash: "470ad1021c0cec24f20c80a92ab8dd46b366e9cbf83e3f8e7ffd42c4354260c8"
sourceCorrections: []
---

Adam 优化器是深度学习 (deep learning)中应用广泛且高效的优化算法之一。它融合了自适应学习率和动量策略的原理。Adam 代表“自适应矩估计”，它为每个参数 (parameter)计算自适应学习率。实现这一目标的方式是，存储过去*梯度平方*的指数衰减平均值（一种常用于自适应学习率的技术）和过去梯度本身的指数衰减平均值（一种融入动量的方法）。

可以把 Adam 看作是集合了 RMSprop 的优点（处理非平稳目标和逐参数学习率）以及动量法的优势（帮助加速沿一致梯度方向的进展并抑制震荡）。

### Adam 的工作方式：动量与尺度调整的结合

对于每个正在优化的参数 (parameter) $\theta$，Adam 维护两个主要内部状态变量。这些本质上是移动平均值：

1. **一阶矩向量 (vector) ($m$)：** 这是过去梯度的指数衰减平均值。其作用类似于带动量 SGD 中的动量项。它有助于加快收敛并应对崎岖区域。
2. **二阶矩向量 ($v$)：** 这是过去*梯度平方*的指数衰减平均值。其作用类似于 RMSprop 中的分母项，根据近期梯度的幅度，提供逐参数的自适应学习率。

设 $g_t$ 为在时间步 $t$ 时，目标函数相对于参数 $\theta$ 的梯度。移动平均值 $m_t$ 和 $v_t$ 的更新按以下方式计算：


$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$


$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$


此处：

- $m_0$ 和 $v_0$ 初始化为零向量。
- $\beta_1$ 和 $\beta_2$ 是超参数 (hyperparameter)，分别控制一阶和二阶矩估计的指数衰减率。它们通常接近 1（常见的默认值是 $\beta_1 = 0.9$ 和 $\beta_2 = 0.999$）。
- $g_t^2$ 表示梯度向量 $g_t$ 的元素级平方。

### 针对初始化偏差进行修正

由于 $m_0$ 和 $v_0$ 初始化为零，可能会出现一个问题。尤其是在最初的时间步，当 $\beta_1$ 和 $\beta_2$ 接近 1 时，矩估计 $m_t$ 和 $v_t$ 将偏向零。Adam 引入了一个偏差修正步骤来抵消这种初始化偏差，这在训练早期作用显著。

经过偏差修正的估计值 $\hat{m}_t$ 和 $\hat{v}_t$ 计算方式如下：


$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$


$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$


请注意，随着时间步 $t$ 的增加，分母 $(1 - \beta_1^t)$ 和 $(1 - \beta_2^t)$ 趋近于 1。这表示偏差修正初期作用更大，随后会随时间逐渐减弱。

### Adam 更新规则

最后，参数 (parameter) $\theta$ 使用偏差修正后的矩估计进行更新。更新规则类似于 RMSprop，但使用的是偏差修正后的一阶矩估计 $\hat{m}_t$，而非原始梯度 $g_t$：


$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$


这里：

- $\alpha$ 是学习率（步长），这是另一个超参数 (hyperparameter)。
- $\epsilon$ 是一个很小的常数（例如 $10^{-8}$），添加到分母中以增强数值稳定性，防止在 $\hat{v}_t$ 可能非常接近零时出现除以零的情况。

实际上，Adam 根据类似动量项的 $\hat{m}_t$ 计算更新，但凭借存储在 $\sqrt{\hat{v}_t}$ 中的梯度平方信息，对每个参数的学习率进行单独调整。对于近期梯度较大（方差较高）的参数，其有效学习率会相对基础学习率 $\alpha$ 降低；而对于近期梯度较小（方差较低）的参数，其有效学习率则会相对增加。

### Adam 的优势与考量

Adam 在实践中通常在各类深度学习 (deep learning)任务上表现良好。其主要优势包括：

- **优点整合：** 将自适应学习率与动量法整合。
- **计算效率高：** 仅需一阶梯度，内存开销小。
- **适用场景：** 适用于大数据集、高维参数 (parameter)空间以及存在噪声或稀疏梯度的问题。
- **良好的默认设置：** 推荐的 $\beta_1$、$\beta_2$ 和 $\epsilon$ 默认值通常表现良好，尽管学习率 $\alpha$ 通常仍需调整。

然而，一些研究表明，在某些情况下，与经过精细调整的带动量 SGD 相比，Adam 可能会收敛到次优解，尽管它在初期通常收敛更快。它仍然是一个非常强大的基准方法，并且常是许多实践者的默认选项。

Adam 是优化算法发展中的一个重要进展，凭借根据梯度历史调整学习过程的能力，为训练深度神经网络 (neural network)提供了有效方法。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化器的原始研究论文，详细阐述了其算法机制、偏差校正和经验性能。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 299-304
  一本广受认可的深度学习教材，提供了深度学习概念的全面概述，包括对Adam优化算法的详细解释。
- [Optimizers in Deep Learning](http://cs231n.github.io/neural-networks-3/#adam) — Andrej Karpathy, Justin Johnson, Serena Yeung, et al. (2023)
  Journal: Stanford University CS231n: Convolutional Neural Networks for Visual Recognition, Lecture Notes; Publisher: Stanford University
  一门广受欢迎的斯坦福课程的教学笔记，提供了对Adam及其他深度学习优化算法清晰易懂的解释。
