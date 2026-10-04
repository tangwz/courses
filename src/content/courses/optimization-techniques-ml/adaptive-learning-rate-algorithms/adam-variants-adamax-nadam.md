---
course: "optimization-techniques-ml"
chapter: "adaptive-learning-rate-algorithms"
lesson: "adam-variants-adamax-nadam"
sourceId: 1268
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/adam-variants-adamax-nadam"
title: "Adamax 和 Nadam 变体"
description: "介绍 Adam 的变体，例如 Adamax（使用无穷范数）和 Nadam（引入 Nesterov 动量）。"
order: 5
plots: []
sourceHash: "4a05d7a66a392a03732bba70b323c234d5b5fc2f579f9a28d029ac42f055125d"
sourceCorrections: []
---

Adam 优化器结合了动量和基于梯度二阶矩的自适应缩放。研究人员提出了一些变体，旨在进一步提升性能或处理特定行为。两个重要的变体是 Adamax 和 Nadam。这些方法修改了 Adam 的核心组成部分，具体来说是它如何调整学习率或引入动量，为某些优化场景提供了不同的途径。

### Adamax：使用无穷范数的自适应学习率

Adam 根据过去梯度平方的衰减平均值（即 $v_t$）为每个参数 (parameter)调整学习率，这实际上使用了 $L_2$ 范数。Adamax 由 Kingma 和 Ba 在提出 Adam 的同一篇论文中引入，它转而使用 $L_\infty$ 范数（无穷范数或最大范数）进行缩放。

这样做的理由源于对 Adam 更新规则的推广。Adam 的逐参数更新涉及将一阶矩估计 $\hat{m}_t$ 除以 $\sqrt{\hat{v}_t}$。表达式 $\sqrt{v_t}$ 大致与当前和过去梯度的 $L_2$ 范数成比例。如果我们将此推广到 $L_p$ 范数，会发生什么？更新分量 $v_t$ 将涉及梯度的 $p$ 次幂：


$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) |g_t|^p
$$


参数更新将涉及除以 $v_t^{1/p}$。Adam 对应于 $p=2$。Adamax 考虑 $p \to \infty$ 的情况。随着 $p \to \infty$， $L_p$ 范数收敛到 $L_\infty$ 范数（最大绝对值）。这使得缩放项的更新异常简单，通常用 $u_t$ 表示以区别于 Adam 的 $v_t$：


$$

$$


注意，此更新不需要元素级的平方或平方根运算。它依赖于 `max`（最大值）操作。因为 $u_t$ 依赖于 `max` 操作，所以在处理非常大但稀疏的梯度时，它可能比 Adam 的 $v_t$ 更为稳定，因为与平方和相比，`max` 受不常出现的大值的影响较小。

完整的 Adamax 参数更新规则为：


$$
\theta_{t+1} = \theta_t - \frac{\eta}{u_t} \hat{m}_t
$$


这里，$\hat{m}_t$ 是经过偏差校正的一阶矩估计，计算方式与 Adam 完全相同：


$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$


$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$


关于 Adamax 的重要点：

- **缩放：** 使用 $L_\infty$ 范数（最大值）进行缩放，其来源于 $L_p$ 范数的极限。
- **更新规则 ($u_t$)：** 与 Adam 的 $v_t$ 相比，计算更简单（$u_t = \max(\beta_2 u_{t-1}, |g_t|)$）。
- **稳定性：** 可能比 Adam 更为稳定，尤其是在稀疏梯度或突然出现的大梯度情况下，因为 `max` 操作比平方和更不敏感。
- **超参数 (hyperparameter)：** 使用与 Adam 相同的超参数（$\eta$， $\beta_1$， $\beta_2$），且默认值通常相似（$\beta_1 \approx 0.9$， $\beta_2 \approx 0.999$）。

Adamax 提供了一个计算上稍简单的 Adam 变体，在某些情况下可能提供更好的稳定性。

### Nadam：将 Nesterov 动量引入 Adam

Nesterov 加速梯度 (NAG) 方法通常能通过在应用初步动量步骤（预见）之后计算梯度，从而提升标准动量方法的收敛速度。由 Dozat 提出的 Nadam（Nesterov 加速自适应矩估计）优化器，旨在将 Nesterov 动量原理整合到 Adam 框架中。

回顾一下，Adam 使用梯度的一阶矩 ($m_t$) 和二阶矩 ($v_t$) 的偏差校正估计值来更新参数 (parameter)。Nadam 修改了一阶矩估计影响更新步骤的方式，以纳入 NAG 的“预见”特点。

NAG 的核心思想是在 $\theta_t + \mu m_{t-1}$ 处（其中 $\mu$ 是动量系数）而不是在 $\theta_t$ 处计算梯度。Nadam 在 Adam 的更新中应用了类似的校正。Nadam 没有仅使用前一个动量项 $\beta_1 \hat{m}_{t-1}$ 来预测下一个位置，而是将*当前*梯度信息更直接地纳入更新的动量部分。

我们来看看 Nadam 的更新步骤：

1. 计算梯度：$g_t = \nabla f(\theta_{t-1})$
2. 更新有偏一阶矩估计：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. 更新有偏二阶矩估计：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. 计算偏差校正的一阶矩：$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
5. 计算偏差校正的二阶矩：$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
6. 计算 Nadam 参数更新：
   
   $$
   \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \beta_1 \hat{m}_t + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t} \right)
   $$
   

将 Nadam 更新（步骤 6）与标准 Adam 更新进行比较：

$\text{Adam 更新: } \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \frac{\beta_1 m_{t-1}}{1 - \beta_1^t} + \frac{(1 - \beta_1) g_t}{1 - \beta_1^t} \right)$

主要区别在于动量项的应用方式。Nadam 使用 $\beta_1 \hat{m}_t$，其中包含通过 $m_t$ 传递的*当前*梯度 $g_t$ 的影响，并结合了另一个包含当前梯度 $\frac{(1 - \beta_1) g_t}{1 - \beta_1^t}$ 的项。这种结构有效地使用更新后的动量 $m_t$ 应用了动量步骤，并以模仿 NAG 预见行为的方式引入梯度校正。

关于 Nadam 的重要点：

- **动量：** 将 Nesterov 动量与 Adam 的自适应学习率相结合。
- **更新规则：** 修改 Adam 更新，以采用 Nesterov 风格的动量计算。
- **收敛：** 通常比 Adam 收敛更快，尤其是在 NAG 优于标准动量的问题上。它对于训练循环神经网络 (neural network) (RNN)或深度 CNN 等复杂模型可能特别有效。
- **超参数 (hyperparameter)：** 使用与 Adam 相同的超参数（$\eta$， $\beta_1$， $\beta_2$），默认值通常相似。

当需要更快收敛且稍复杂的更新计算开销可接受时，Nadam 通常是一个不错的选择。

### 实际考量

Adamax 和 Nadam 在流行的机器学习 (machine learning)库中随处可用，例如 TensorFlow (`tf.keras.optimizers.Adamax`, `tf.keras.optimizers.Nadam`) 和 PyTorch (`torch.optim.Adamax`, `torch.optim.Nadam`)。

- **何时选择：** 尽管 Adam 仍然是非常流行且通常有效的默认选项，但如果更快的收敛是优先考虑的，可以尝试 Nadam。如果你在使用 Adam 时观察到可能与大梯度或稀疏梯度有关的不稳定性，那么 Adamax 值得一试。
- **超参数 (parameter) (hyperparameter)调优：** 尽管默认值通常效果不错，但调整学习率 $\eta$ 和动量衰减参数 $\beta_1, \beta_2$ 仍能带来性能提升，就像 Adam 一样。$\beta_1$ 通常影响动量方面，而 $\beta_2$ 则控制基于过去梯度幅度的自适应性。
- **经验评估：** 与所有优化算法一样，最佳选择通常取决于特定的问题、数据集和模型架构。建议在你的特定任务上，通过实际比较 Adam、Adamax 和 Nadam（以及可能包括 RMSprop 或带动量的 SGD 等其他优化器）的性能来做出选择。

通过理解 Adamax 和 Nadam 引入的细微但重要的修改，你将获得更多优化工具，有可能为你的机器学习模型实现更稳定或更快的训练。它们代表了在 Adam 算法之上构建的宝贵改进。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍 Adam 优化器及其变体 Adamax 的原始论文，详细阐述了它们的推导过程和特性。
- [Incorporating Nesterov Momentum into Adam](https://cs229.stanford.edu/proj2016/report/Dozat-IncorporatingNesterovMomentumIntoAdam-Report.pdf) — Timothy Dozat (2016)
  Journal: Proceedings of the 4th International Conference on Learning Representations; Pages: 1-4
  引入了 Nadam 优化器的报告，该优化器将 Nesterov 动量与 Adam 的自适应学习率机制相结合。
- [tf.keras.optimizers.Adamax](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow Keras Adamax 优化器的官方文档，提供了使用示例和实现细节。
- [tf.keras.optimizers.Nadam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow Keras Nadam 优化器的官方文档，概述了其参数和实际应用。
