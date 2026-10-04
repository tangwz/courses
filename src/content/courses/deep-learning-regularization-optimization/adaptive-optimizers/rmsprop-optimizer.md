---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "rmsprop-optimizer"
sourceId: 4991
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/rmsprop-optimizer"
title: "RMSprop：处理AdaGrad的局限性"
description: "引入RMSprop，作为AdaGrad的一种优化，通过使用梯度平方的移动平均。"
order: 4
plots: ["plots/4991-0.json"]
sourceHash: "48f74294a7e9e713ccdf3749acc08ce655f827f90b40259808064bbff2d7569b"
sourceCorrections: []
---

AdaGrad引入了根据每个参数 (parameter)的梯度历史调整学习率的强大思想。然而，AdaGrad的一个主要缺点是在分母中不断累积梯度的平方。随着时间的推移，这个和会持续增大，导致有效学习率单调减小，并最终变得无限小。这种过快的衰减可能会提前停止学习过程，在深度学习 (deep learning)中常见的非凸优化场景下，即使经过许多次迭代，可能仍需要持续优化。

RMSprop（均方根传播）与AdaGrad大约在同一时间独立开发，旨在解决这种快速衰减问题。RMSprop的核心在于使用梯度的平方的*指数衰减移动平均*，而不是累加所有过去的梯度平方。这种方法赋予近期梯度信息更大的权重 (weight)，并能有效地“遗忘”久远的过去。

### RMSprop的更新机制

相对于为每个参数 (parameter)保留所有过去梯度平方的总和 $G_t$，RMSprop保留了梯度*均方*的估计值，表示为 $E[g^2]_t$。这个值在每个时间步 $t$ 使用衰减率超参数 (hyperparameter) $\beta$（通常在0.9到0.99之间）进行更新：


$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2
$$


这里：

- $g_t$ 是在时间步 $t$ 时损失函数 (loss function)对参数的梯度。
- $E[g^2]_t$ 是在时间步 $t$ 时梯度平方的移动平均。
- $\beta$ 是衰减率，用于控制移动平均的记忆周期。值越接近1表示对过去的记忆越久。

这种计算方式确保 $E[g^2]_t$ 主要受近期梯度平方的影响。如果近期梯度值较大，$E[g^2]_t$ 也会较大；如果梯度值较小，它就会减小。

RMSprop的参数更新规则随后在分母中使用该移动平均的平方根，这与AdaGrad类似，但重要的是，它使用了自适应的 $E[g^2]_t$，而不是不断增大的总和：


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$


其中：

- $\theta_t$ 表示时间步 $t$ 的模型参数。
- $\eta$ 是全局学习率。
- $\epsilon$ 是一个很小的常数（例如，$10^{-8}$），用于增加数值稳定性，避免除以零。

### RMSprop为何优于AdaGrad

通过使用指数衰减平均 $E[g^2]_t$，RMSprop阻止了分母项 $\sqrt{E[g^2]_t + \epsilon}$ 在整个训练过程中单调增长。它会适应每个参数 (parameter)近期梯度的量级。如果一个参数的梯度在最初较大后变小，$E[g^2]_t$ 会随时间减小，这使得该参数的有效学习率有可能再次提高，这与AdaGrad中学习率会保持受抑制的情况不同。这种适应性使得RMSprop在长时间训练深度神经网络 (neural network)时更为有效。

思考分母可能如何演变：



![分母行为 (示意)](plots/4991-0.json)



> AdaGrad的分母项因累加所有过去的梯度平方而倾向于持续增长。RMSprop的分母则基于移动平均进行调整，如果近期梯度变小，它会随之减小，从而避免学习率过快衰减。（注意：这是一个假设梯度序列开始时较大，然后变小的示意图）。

### 实际应用

RMSprop是一种广泛使用的优化器，在所有主要的深度学习 (deep learning)框架中均可使用。使用它时，您通常需要指定学习率 $\eta$，并且常常需要指定衰减因子 $\beta$（在库中通常命名为 `alpha` 或 `rho`，默认值约为0.9或0.99）以及 $\epsilon$。尽管与SGD相比，它通常对学习率的调整要求较低，但找到合适的 $\eta$ 和 $\beta$ 值仍然会影响性能。

```python
import torch
import torch.optim as optim
import torch.nn as nn

# 假设 'model' 是您定义的神经网络 (nn.Module)
# 示例: model = nn.Linear(10, 2)

# 初始化RMSprop优化器
# 常用参数：要优化的参数、学习率 (lr)、alpha (衰减因子)、epsilon (eps)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08)

# --- 在您的训练循环内部 ---
# loss.backward() # 计算梯度
# optimizer.step() # 使用RMSprop更新参数
# optimizer.zero_grad() # 为下一次迭代重置梯度
# --- 训练循环结束 ---
```

RMSprop为AdaGrad提供了一种有益的替代方案，为Adam等优化器中出现的后续改进奠定了基础，我们将在接下来讨论Adam。

## 参考资料

- [Neural Networks for Machine Learning - Lecture 6a: RMSProp](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Geoffrey Hinton, Tijmen Tieleman (2012)
  Journal: Coursera: Neural Networks for Machine Learning; Publisher: University of Toronto
  这份讲义是RMSprop算法创建者Geoffrey Hinton对其进行的最初的非正式介绍。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8, Section 8.5.2
  在深度学习优化背景下，本书对包括RMSprop在内的自适应学习率方法进行了全面的学术阐述。
- [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/abs/1609.04747) — Sebastian Ruder (2016)
  Journal: arXiv preprint arXiv:1609.04747; DOI: [10.48550/arXiv.1609.04747](https://doi.org/10.48550/arXiv.1609.04747)
  一篇被广泛引用且易于理解的综述论文，解释了包括RMSprop在内的各种梯度下降优化算法、它们的公式和优点。
- [torch.optim.RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) — PyTorch Documentation (2024)
  Journal: PyTorch Documentation
  PyTorch中RMSprop优化器的官方文档，涵盖其参数和用法，补充了实际使用部分。
