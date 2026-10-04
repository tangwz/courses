# AMSGrad：提升 Adam 的收敛性

来源：[原文](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-3-adaptive-learning-rate-algorithms/amsgrad-optimizer)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管前面讨论过的 Adam 优化器因其高效和有效性已成为许多深度学习 (deep learning)任务的常用选择，但进一步分析显示其收敛性证明存在潜在问题。具体来说，使用过去梯度平方 ($v_t$) 的指数衰减平均值不能保证此项保持非递减。在某些情况下，尤其是在训练后期，或当信息量较大的梯度出现在信息量较小的梯度之后时，$v_t$ 可能会明显减少。当此减少的值被用于参数 (parameter)更新步骤的分母（$\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$）时，可能导致不合需要的大步长，从而可能使优化器发散或未能收敛到最优解。

AMSGrad（改进收敛保证的自适应矩估计）被提出，旨在直接处理 Adam 收敛性分析中的这一理论缺陷。其核心思想简单而有效：确保用于缩放梯度的自适应学习率项不会随时间增加。

## AMSGrad 的改进

AMSGrad 不直接使用第二动量的当前估计值 ($v_t$)，而是保持该估计值到目前为止的*最大*值。我们来分解一下更新步骤：

1. **计算梯度：** 在时间步 $t$ 计算损失函数 (loss function)对参数 (parameter) $\theta_t$ 的梯度 $g_t$。

   
   $$
   g_t = \nabla_{\theta} L(\theta_t)
   $$
   
2. **更新有偏一阶矩估计：** 这与 Adam 中相同，使用超参数 (hyperparameter) $\beta_1$。

   
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   
3. **更新有偏二阶矩估计：** 这也与 Adam 中相同，使用超参数 $\beta_2$。

   
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   

   （注意：$g_t^2$ 表示元素级别的平方）。
4. **保持二阶矩估计的最大值：** 这是 AMSGrad 引入的**主要区别**。一个新变量 $\hat{v}_t$ 记录了到时间步 $t$ 为止遇到的 $v$ 的最大值。

   
   $$
   \hat{v}_t = \max(\hat{v}_{t-1}, v_t)
   $$
   

   初始化 $\hat{v}_0 = 0$。 $\max$ 操作是按元素进行的。
5. **计算偏差校正后的一阶矩估计：** 与 Adam 相同。

   
   $$
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
   $$
   
6. **更新参数：** 在分母中使用最大二阶矩估计值 $\hat{v}_t$，而不是 Adam 更新中使用的偏差校正后的 $\hat{v}_t$。请注意，在用于更新步骤的标准 AMSGrad 公式中，$\hat{v}_t$ 本身*没有*进行偏差校正。

   
   $$
   \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
   $$
   

   在这里，$\eta$ 是步长（学习率），$\epsilon$ 是一个用于数值稳定的小常数。

通过确保分母项 $\sqrt{\hat{v}_t} + \epsilon$ 是非递减的，AMSGrad 防止了当 $v_t$ 意外减小时在 Adam 中可能出现的大步长，这些步长可能导致不稳定。此改进提供了更强的理论收敛保证，特别是在在线和非凸环境下。

## 实际考量与使用

理论上的改进能否转化为更好的实际性能？结果不一，通常取决于具体问题。

- **稳定性：** AMSGrad 有时能提供更稳定的训练和收敛，尤其是在 Adam 可能表现出不稳定或收敛问题的数据集或架构上。
- **性能：** 在许多标准深度学习 (deep learning)应用（例如图像分类）中，尽管存在理论担忧，标准 Adam 通常收敛更快，并能达到与 AMSGrad 相当甚至略好的最终性能。Adam 收敛失败的条件在这些常见基准测试中可能不常遇到。
- **内存：** AMSGrad 需要略多一点内存，因为它需要存储额外的 $\hat{v}_t$ 向量 (vector)。

**何时考虑使用 AMSGrad：**

1. 你在你的特定任务中观察到 Adam 存在收敛问题或不稳定。
2. 你正在处理某种情况（可能是强化学习 (reinforcement learning)或在线学习），其中长期行为和收敛保证更重要。
3. 相较于在简单情况下可能更快的收敛速度，你更看重稳健性和理论合理性。

大多数深度学习框架（如 TensorFlow 和 PyTorch）在其 Adam 优化器实现中提供了 AMSGrad 作为简单的布尔标志（例如，`tf.keras.optimizers.Adam(amsgrad=True)` 或 `torch.optim.Adam(amsgrad=True)`）。这使得实验变得简单。

尽管 Adam 仍然是一个非常强的基准，并且通常是首选，但 AMSGrad 提供了一个有价值的替代方案，其立足于处理 Adam 的特定理论缺陷。了解其机制和动机有助于你在调整复杂模型的优化策略时做出明智的决定。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍Adam优化器的原始论文，为AMSGrad的改进奠定了基础。
- [Optimization for Deep Learning](https://cs231n.github.io/optimization-2/) — Stanford University CS231n Course Staff (2023)
  Publisher: Stanford University
  斯坦福大学CS231n课程的综合讲义，涵盖深度学习中的各种优化算法，包括Adam和对AMSGrad的讨论。
- [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) — PyTorch Authors (2024)
  Publisher: PyTorch
  PyTorch Adam优化器的官方文档，包含启用和使用AMSGrad变体的实用细节。

---

[上一节](05-Adamax%20%E5%92%8C%20Nadam%20%E5%8F%98%E4%BD%93.md) · [下一节](07-%E4%BA%86%E8%A7%A3%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.md)
