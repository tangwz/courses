# Adam算法细致分析

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/adam-algorithm-breakdown)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Adam（自适应矩估计）是一种优化器，它融合了动量和RMSprop等自适应缩放方法的优势。Adam的更新机制通过维护两个独立的指数衰减移动平均值来实现优化：一个用于过去的梯度（第一矩估计），另一个用于过去的平方梯度（第二矩估计）。

### 第一矩和第二矩估计

在每个时间步$t$时，在计算了关于参数 (parameter)$\theta$的梯度$g_t = \nabla_\theta J(\theta_{t-1})$之后，Adam会更新这两个移动平均值：

1. **第一矩估计（均值）：** 这类似于动量项。它累积了过去梯度的指数衰减平均值。

   
   $$
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   $$
   

   这里，$m_t$是第一矩向量 (vector)，$g_t$是当前时间步的梯度，$\beta_1$是第一矩估计的指数衰减率（通常接近1，例如0.9）。$m_0$初始化为全零向量。
2. **第二矩估计（非中心方差）：** 这类似于RMSprop中使用的项。它累积了过去平方梯度的指数衰减平均值。

   
   $$
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   $$
   

   这里，$v_t$是第二矩向量，$g_t^2$表示梯度向量的逐元素平方，$\beta_2$是第二矩估计的指数衰减率（通常设置得更高，例如0.999）。$v_0$也初始化为全零向量。

项$m_t$和$v_t$分别是梯度均值和非中心方差的估计值。超参数 (hyperparameter)$\beta_1$和$\beta_2$控制着这些移动平均值的衰减率。接近1的值表示过去的梯度影响时间更长。

### 偏差校正

一个可能出现的问题是，$m_t$和$v_t$被初始化为零向量 (vector)。特别是在训练的初始时间步，当$t$较小时，这些估计值会偏向于零。试想如果$\beta_1 = 0.9$；那么$m_1 = 0.1 g_1$。这明显小于实际梯度。

Adam通过计算偏差校正后的第一矩和第二矩估计（分别表示为$\hat{m}_t$和$\hat{v}_t$）来处理这个初始化偏差：


$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$


$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$


请注意分母中的项$(1 - \beta_1^t)$和$(1 - \beta_2^t)$。在训练开始时（$t$较小），$\beta_1^t$和$\beta_2^t$接近1，这使得分母变小。这种除法有效地抵消了初始的零偏差。随着训练的进行和$t$的增加，$\beta_1^t$和$\beta_2^t$趋近于零（因为$\beta_1, \beta_2 < 1$），所以校正项$(1 - \beta_1^t)$和$(1 - \beta_2^t)$趋近于1，偏差校正的作用减小。这确保了在整个训练过程中估计值更准确。

### 参数 (parameter)更新规则

最后，Adam使用这些偏差校正后的估计值来更新模型参数$\theta$。这个更新规则与RMSprop非常相似，但它使用了校正后的动量估计$\hat{m}_t$，而不是原始梯度$g_t$：


$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$


这里：

- $\theta_t$是时间步$t$时的参数。
- $\alpha$是学习率（或步长）。
- $\hat{m}_t$是偏差校正后的第一矩估计。
- $\sqrt{\hat{v}_t}$是偏差校正后的第二矩估计的逐元素平方根。此项自适应地缩放每个参数的更新，类似于RMSprop。过去平方梯度较大（即$\hat{v}_t$较大）的参数将获得较小的更新。
- $\epsilon$是一个小常数（例如$10^{-8}$），用于数值稳定性，以防止在$\hat{v}_t$可能非常接近零时出现除以零的情况。

### Adam算法概览

总而言之，Adam在每个时间步$t$的更新过程包含以下步骤：

1. 计算梯度：$g_t = \nabla_\theta J(\theta_{t-1})$
2. 更新有偏差的第一矩估计：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. 更新有偏差的第二矩估计：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. 计算偏差校正后的第一矩估计：$\hat{m}_t = m_t / (1 - \beta_1^t)$
5. 计算偏差校正后的第二矩估计：$\hat{v}_t = v_t / (1 - \beta_2^t)$
6. 更新参数 (parameter)：$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

最初Adam论文的作者建议默认值设定为$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。学习率$\alpha$（通常建议约为$0.001$）仍然是一个超参数 (hyperparameter)，通常需要进行调整。

这个循序渐进的过程，结合了基于第二矩的自适应缩放和基于第一矩的动量，以及重要的偏差校正步骤，使得Adam成为深度学习 (deep learning)模型中广泛使用且通常有效的默认优化器。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam算法、其数学公式和实验结果，是该算法的开创性论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 282-301
  全面解释了深度学习中的优化算法，包含对Adam的详细讨论。
- [Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/) — Stanford University (2024)
  提供对神经网络中使用的各种优化技术的易懂概述，并清晰地解释了Adam。

---

[上一节](05-Adam%EF%BC%9A%E8%87%AA%E9%80%82%E5%BA%94%E7%9F%A9%E4%BC%B0%E8%AE%A1.md) · [下一节](07-Adamax%20%E5%92%8C%20Nadam%20%E5%8F%98%E4%BD%93%EF%BC%88%E7%AE%80%E8%A6%81%E6%A6%82%E8%BF%B0%EF%BC%89.md)
