# 适用于Transformer的优化器 (Adam, AdamW)

来源：[原文](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/optimizers-for-transformers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

优化器在训练深度神经网络 (neural network)时处理复杂的损失情况方面发挥着重要作用。虽然标准随机梯度下降 (gradient descent)（SGD）提供了初始方法，但其固定的学习率对于Transformer模型的规模和复杂性来说往往不够用。自适应学习率算法能够动态调整每个参数 (parameter)的步长，已成为不可或缺的工具。在这些算法中，Adam及其改进版AdamW是训练Transformer时普遍采用的优化器。

## Adam: 自适应矩估计

Adam是自适应矩估计的简称，它结合了动量方法和RMSprop的思路。它根据梯度的一阶和二阶矩的估算值，为不同的参数 (parameter)计算各自的自适应学习率。

1. **一阶矩（动量）：** Adam维持着过去梯度的一个指数衰减平均值。这起到了动量的作用，帮助优化器在损失曲面浅平的山谷中更快地前进，并减缓高曲率方向的振荡。时间步 $t$ 时的一阶矩估算值 $m_t$ 使用梯度 $g_t$ 计算：
   $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   超参数 (hyperparameter) $\beta_1$ 决定了指数衰减率，通常设为0.9这样的值。
2. **二阶矩（方差自适应）：** Adam还维持着过去梯度平方的一个指数衰减平均值。这作为每个参数梯度变化程度的估算（更精确地说，是未中心化的二阶矩）。该估算值被用来反向调整学习率；梯度历史值波动较大的参数更新步长较小，而梯度较小或稀疏的参数更新步长较大。二阶矩估算值 $v_t$ 计算方式如下：
   $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   超参数 $\beta_2$ 决定此衰减率，通常设为0.999。
3. **偏差修正：** 由于矩估算值 $m_t$ 和 $v_t$ 初始化为零，它们在训练初期存在偏向零的偏差。Adam引入了偏差修正项来抵消这种影响：
   $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
   $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
   这些修正后的估算值 $\hat{m}_t$ 和 $\hat{v}_t$ 提供了对真实矩更准确的估算，尤其是在训练初期。
4. **参数更新：** 最后，使用偏差修正后的矩估算值来更新参数 $w$。更新规则通过修正后的一阶矩与修正后的二阶矩的平方根的比值，来调整基础学习率 $\alpha$：
   $w_{t+1} = w_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
   小常数 $\epsilon$ (例如 $10^{-8}$) 被添加到分母中，以防止数值异常，从而避免除以零。

Adam在多种深度学习 (deep learning)任务中常表现出快速的初期收敛和良好性能。然而，当Adam与标准L2正则化 (regularization)（权重 (weight)衰减）结合使用时，会浮现一个不明显的问题。L2正则化旨在通过在损失函数 (loss function)中添加惩罚项 $\frac{\lambda}{2} ||w||^2$ 来减少过拟合 (overfitting)。这导致梯度 $g_t$ 中增加一个额外的项 $-\lambda w_t$。在传统的Adam实现方式中，这个梯度组成部分 $g_t = \nabla \mathcal{L}(w_t) + \lambda w_t$ 被用于计算 $m_t$ 和 $v_t$。这使得权重衰减作用与自适应学习率机制耦合在一起。具体来说，更新过程中施加的实际权重衰减 ($\alpha \frac{\lambda w_t}{\sqrt{\hat{v}_t} + \epsilon}$) 会受到自适应分母 $\sqrt{\hat{v}_t}$ 的调整。这意味着与历史梯度大小（较大的 $\hat{v}_t$）相关的权重，其实际衰减作用会减弱，可能阻碍正则化作用。

## AdamW: 带有解耦权重 (weight)衰减的Adam

AdamW被提出是为了解决原始Adam算法中权重衰减与自适应学习率之间不理想的配合问题。Loshchilov & Hutter (2019) 提出的主要思路简单而有效：将权重衰减更新与Adam执行的基于梯度的更新分离开来。

AdamW没有将L2惩罚纳入用于矩估算的梯度 $g_t$ 中，而是仅使用来自主要损失函数 (loss function) $\nabla \mathcal{L}(w_t)$ 的梯度来计算矩估算值和主要的自适应更新步骤。权重衰减随后在自适应步骤 *之后* 单独且直接施加到权重上。

该过程可概括为：

1. 仅根据任务损失计算梯度 $g_t = \nabla \mathcal{L}(w_t)$。
2. 使用 $g_t$ 更新一阶和二阶矩估算值（$m_t$, $v_t$），并计算它们的偏差修正版本（$\hat{m}_t$, $\hat{v}_t$），与标准Adam中相同。
3. 计算主要的Adam更新步长（尚未考虑学习率）：
   $\Delta w'_t = \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
4. 通过施加缩放后的自适应步长和解耦的权重衰减来更新权重。使用 $\eta_t$ 作为时间步 $t$ 的调度学习率：
   $w_{t+1} = w_t - \eta_t (\Delta w'_t + \lambda w_t)$
   或者，更接近原始论文的表达，衰减可以在更新前以乘法形式施加：$w_{t+1} = (1 - \eta_t \lambda) w_t - \eta_t \Delta w'_t$。主要观察点是，衰减项 $\lambda w_t$ 不再受 $1 / (\sqrt{\hat{v}_t} + \epsilon)$ 的影响。

通过解耦，AdamW保证了权重衰减在所有参数 (parameter)上施加得更一致，仅与权重大小 $w_t$ 和学习率 $\eta_t$ 成比例，不受 $\hat{v}_t$ 中捕获的梯度历史影响。这一修改常能带来更好的模型泛化能力和最终表现，相比使用L2正则化 (regularization)的标准Adam而言，特别是对于Transformer这类有效正则化非常有益的复杂模型。因此，AdamW已成为训练现代Transformer的事实标准优化器。

## 超参数 (parameter) (hyperparameter)与考量

有效运用Adam或AdamW需要设定多个超参数：

- **学习率 ($\alpha$ 或 $\eta_t$):** 这依然是一个非常敏感的超参数。对于Transformer而言，它几乎从不固定；相反，包含一个初始“预热”阶段，随后是衰减阶段的学习率调度十分重要（将在后续章节中讨论）。通常的峰值学习率处于 $10^{-5}$ 到 $10^{-3}$ 的范围，受模型大小、批量大小以及所用具体调度等因素影响。
- **$\beta_1$:** 一阶矩估算值的指数衰减率。默认值0.9在多数情况下效果良好。
- **$\beta_2$:** 二阶矩估算值的指数衰减率。虽然默认值0.999普遍使用，但一些研究表明略低的值（例如0.98、0.99）偶尔可能带来一些好处。
- **$\epsilon$:** 一个小值，用于防止数值异常。像 $10^{-8}$ 或 $10^{-6}$ 这样的默认值通常足够。
- **权重 (weight)衰减 ($\lambda$):** 可直接用于AdamW，或者在标准Adam中作为L2惩罚系数。这是一个重要的正则化 (regularization)超参数，需要通过实验调整。常见的起始值在0.01或0.1左右，但最佳值取决于数据集和模型。将 $\lambda=0$ 设为0实际等于关闭权重衰减。

总而言之，尽管Adam在优化方面取得了显著进展，但AdamW提供了一种更优化的方法来纳入权重衰减，在训练大型语言模型时带来了明显的改进。对于Transformer架构，它通常是推荐的选择，总是与仔细调整的学习率调度和正则化参数配合使用。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2014)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍 Adam 优化器的原始研究论文，概述其算法和理论基础。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR 2019); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  这篇论文介绍了 AdamW，它是 Adam 的一个改进，正确应用了权重衰减，从而增强了正则化效果。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 281-322 (Chapter 8: Optimization for Training Deep Models)
  一本基础教材，全面概述了深度学习，包括对优化算法的详细解释。
- [CS231n Convolutional Neural Networks for Visual Recognition: Neural Network Training Part 2](http://cs231n.github.io/neural-networks-3/) — Stanford University (2023)
  Journal: Online Course Notes; Publisher: Stanford University
  作为一门广受认可课程的一部分，这些笔记提供了神经网络训练优化器的实用且直观的解释。

---

[上一节](02-%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E7%AD%96%E7%95%A5.md) · [下一节](04-%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E5%BA%A6%20%28%E7%83%AD%E8%BA%AB%2C%20%E8%A1%B0%E5%87%8F%29.md)
