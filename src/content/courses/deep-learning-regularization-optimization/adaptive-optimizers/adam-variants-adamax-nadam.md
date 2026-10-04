---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "adam-variants-adamax-nadam"
sourceId: 4994
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/adam-variants-adamax-nadam"
title: "Adamax 和 Nadam 变体（简要概述）"
description: "简要提及 Adam 的其他变体，如 Adamax 和 Nadam。"
order: 7
plots: []
sourceHash: "49803a37bcdcc26676192dc1d25a48f6330a253454990e178897f6ffe72a6d8d"
sourceCorrections: []
---

尽管 Adam 优化器在各类深度学习 (deep learning)任务中表现良好，但为提升稳定性和收敛速度，优化器变种仍在不断涌现 (emergence)。Adamax 和 Nadam 是其中两个值得关注的例子。它们被视为对 Adam 方法的优化，而非根本性的改变。

### Adamax

Adamax 与 Adam 在同一篇论文中提出，在学习率缩放方面提供了不同的思路。回想一下，Adam 对每个参数 (parameter)的学习率进行缩放，使其与过去*平方*梯度（$v_t$，表示 $L_2$ 范数）的指数衰减平均值的平方根成反比。Adamax 通过使用基于过去梯度*无穷范数*（$L_\infty$）的指数衰减平均值来调整此方法。

Adamax 不在 $v_t$ 中累积平方梯度，而是跟踪 $u_t$，其递归定义为：


$$

$$


这里，$u_t$ 主要记录了迄今为止遇到的梯度的最大绝对值（受 $\beta_2$ 控制的指数衰减）。然后，参数更新使用 $u_t$ 进行缩放，取代了 Adam 中的 $\sqrt{\hat{v}_t} + \epsilon$ 项：


$$
\theta_{t+1} = \theta_t - \frac{\alpha}{u_t} \hat{m}_t
$$


(其中 $\hat{m}_t$ 是偏差校正后的第一阶矩估计，计算方式与 Adam 类似)。

为何使用无穷范数？基于 $u_t$ 的更新步骤，在梯度非常稀疏或不常出现大梯度的情形下，与 Adam 中的 $L_2$ 范数缩放相比，可以认为更为稳定。因为 $u_t$ 取决于最大值而非平方和，所以即使出现一些异常大的梯度，它也不会增长过快，可能避免学习率过早变得过小。尽管 Adamax 通常表现与 Adam 类似，但如果您发现 Adam 存在不稳定或进展缓慢的问题，Adamax 有时会是一个有益的替代方案。

### Nadam

Nadam，是 Nesterov 加速自适应矩估计的简称，结合了 Adam 与 Nesterov 加速梯度（NAG）的思想。您可能回想第 5 章中提到，NAG 通过在应用当前速度*之后*计算梯度，改进了标准动量，有效地“向前看”以预测参数 (parameter)的未来位置。

Nadam 将这种前瞻能力直接融入 Adam 的更新规则中。它改变了第一阶矩估计（$m_t$）的使用方式。Nadam 不再基于*先前*的估计值（$m_{t-1}$）应用动量步骤，而是在计算更新时纳入*当前*的动量估计值（$m_t$），从而达到类似于 NAG 前瞻的效果。

推导涉及用经 Nesterov 改进的版本替换偏差校正后的估计值 $\hat{m}_t$。虽然具体公式稍复杂，但核心思路是在计算参数步骤时，通过考虑动量更新本身的效果，更有效地应用动量更新。

这种结合 Nesterov 动量的方法，常使 Nadam 比 Adam 收敛稍快，尤其在那些 NAG 比标准动量提供了明显好处的任务上。

### 实际考量

由于其稳固性和良好的通用性能，Adam 仍然是许多实践者的首选自适应优化器。然而，Adamax 和 Nadam 在您的优化工具包中提供了有价值的备选方案。

- **何时尝试 Adamax？** 如果怀疑存在梯度爆炸导致 Adam 的 $v_t$ 累加器不稳定，或者正在处理非常稀疏的更新，可以考虑使用它。
- **何时尝试 Nadam？** 如果您寻求可能比 Adam 更快的收敛，尤其是在初步测试显示基于动量的方法对您的问题效果良好时，值得尝试。

Adamax 和 Nadam 在 PyTorch 和 TensorFlow 等流行的深度学习 (deep learning)库中都易于获取。实现它们通常只需更改优化器类名即可。

```python
import torch
import torch.optim as optim

# 假设 'model_parameters' 包含您的网络参数
learning_rate = 1e-3
beta1 = 0.9
beta2_adamax = 0.999 # Adamax 的标准 beta2
beta2_nadam = 0.999 # Nadam 的标准 beta2

# 使用 Adam (用于比较)
# optimizer_adam = optim.Adam(model_parameters, lr=learning_rate, betas=(beta1, 0.999))

# 使用 Adamax
optimizer_adamax = optim.Adamax(model_parameters, lr=learning_rate, betas=(beta1, beta2_adamax))

# 使用 Nadam
optimizer_nadam = optim.Nadam(model_parameters, lr=learning_rate, betas=(beta1, beta2_nadam))

# 为您的训练循环选择一个优化器
# optimizer = optimizer_nadam
```

与其他优化器选择一样，表现最佳者常取决于具体数据集、模型架构和超参数 (parameter) (hyperparameter)设置。通过仔细实验进行经验评估，是判断 Adamax 或 Nadam 是否比 Adam 更具优势来应对您的特定深度学习难题的最可靠方法。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR) 2015; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam和Adamax优化器的原始论文，详细阐述了它们的算法和理论基础。
- [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=ryW0_yWp) — Timothy Dozat (2016)
  Journal: ICLR 2016 Workshop; Publisher: Stanford University
  该报告通过将Nesterov动量与Adam优化器结合，介绍了Nadam（Nesterov加速自适应矩估计）。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖各种优化算法（包括Adam等自适应方法）的综合性教材，提供了理论背景和实践见解。
- [torch.optim.Adamax and torch.optim.Nadam](https://pytorch.org/docs/stable/optim.html) — PyTorch Authors (2025)
  Publisher: PyTorch Foundation
  PyTorch深度学习框架中Adamax和Nadam优化器的官方文档，包含参数详情和使用示例。
