---
course: "autoencoders-representation-learning"
chapter: "applications-training-strategies"
lesson: "advanced-optimization-algorithms"
sourceId: 2928
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-7-applications-training-strategies/advanced-optimization-algorithms"
title: "高级优化算法"
description: "讨论适用于训练自编码器的优化器（Adam, RMSprop）及其调整方法。"
order: 6
plots: ["plots/2928-0.json"]
sourceHash: "af6afcac99e7fbeda6eba85d544aa1e56ef931fa5829de5481436f7602afdc24"
sourceCorrections: []
---

训练自编码器，特别是复杂的架构或具有复杂损失函数 (loss function)（如VAEs中的证据下界 (ELBO)）的模型，涉及最小化一个可能高维、非凸的损失函数 $L$。虽然基本的随机梯度下降 (gradient descent) (SGD) 提供了一个基本方法，但其收敛可能很慢，容易陷入次优局部最小值或鞍点，并且对学习率的选择高度敏感。为了有效地训练深度自编码器模型并找到有意义的表示，我们通常会采用更精密的优化算法来调整学习过程。

这些高级优化器旨在更智能地引导损失区域，加速收敛并提高最终模型的质量。它们通常通过结合当前梯度信息来实现这一点，比如过去的梯度或它们的大小。

### 在梯度下降 (gradient descent)基础上：动量

标准SGD仅根据当前小批量计算的梯度来更新参数 (parameter)。这可能导致更新噪声大和波动，尤其在损失曲面在一个方向上急剧弯曲而在另一个方向上平缓的区域（在深度网络中很常见）。

动量通过将前一个更新向量 (vector)的一部分添加到当前更新向量来解决这个问题。想象一个球滚下山坡：它会积累动量，平滑路径并加速在一致方向上的移动，同时抑制梯度快速变化方向上的波动。

更新包含一个“速度”项 $v_t$，它是过去梯度的指数衰减移动平均值：

$v_t = \beta v_{t-1} + (1 - \beta) \nabla_{\theta} L(\theta_t)$
$\theta_{t+1} = \theta_t - \alpha v_t$

这里，$\theta$ 表示模型参数，$\alpha$ 是学习率，$\nabla_{\theta} L(\theta_t)$ 是损失函数 (loss function) $L$ 在步骤 $t$ 时相对于参数 $\theta$ 的梯度，$\beta$ 是动量系数（通常设为0.9左右）。这使得优化器能够“记住”过去的梯度方向，并继续沿这些方向移动，即使当前梯度很小或有噪声。

### 自适应学习率：AdaGrad, RMSprop, Adam

固定的学习率 $\alpha$ 带来挑战。如果它太小，训练会很慢。如果它太大，优化器可能会越过最小值或发散。自适应学习率算法在训练期间调整学习率，通常是针对每个参数 (parameter)单独调整。

#### AdaGrad (自适应梯度算法)

AdaGrad 针对每个参数单独调整学习率，对于与频繁出现的特征相关的参数执行较小的更新，对于与不频繁特征相关的参数执行较大的更新。它通过将学习率除以该参数所有过去平方梯度的和的平方根来实现这一点。

尽管在某些情况下有效，AdaGrad 的主要缺点是分母中累积的平方梯度和在训练期间持续增长。这导致学习率单调减小，有时在真正达到收敛之前变得无限小，有效地过早停止学习。

#### RMSprop (均方根传播)

RMSprop 修改了 AdaGrad，以解决其学习率单调递减的问题。RMSprop 不再累积所有过去的平方梯度，而是使用平方梯度的指数衰减平均值。这意味着最近的梯度信息被赋予更高的权重 (weight)，从而防止分母无限增长，并使学习率能够更动态地调整。

更新涉及维护平方梯度的移动平均值 $E[g^2]_t$：

$E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) (\nabla_{\theta} L(\theta_t))^2$
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \nabla_{\theta} L(\theta_t)$

这里，$\gamma$ 是衰减率（类似于动量中的 $\beta$，通常在0.9左右），$\epsilon$ 是一个用于数值稳定的小常数（例如 $10^{-8}$）。

#### Adam (自适应矩估计)

Adam 可以说是当今深度学习 (deep learning)中使用最广泛且常作为默认选择的优化算法之一。它结合了动量（使用一阶矩估计，类似于动量）和自适应学习率（使用二阶矩估计，类似于RMSprop）的思想。

Adam 维护两个指数衰减的移动平均值：

1. 梯度的第一矩（均值）（$m_t$），类似于动量。
2. 梯度的第二矩（非中心方差）（$v_t$），类似于RMSprop。

$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{\theta} L(\theta_t)$
$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_{\theta} L(\theta_t))^2$

由于 $m_t$ 和 $v_t$ 初始化为零，它们会偏向零，尤其是在初始步骤中。Adam 执行偏差校正来抵消这种情况：

$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

最终的参数更新使用这些经过偏差校正的估计值：

$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

常见的默认值是 $\beta_1 = 0.9$，$\beta_2 = 0.999$，和 $\epsilon = 10^{-8}$。Adam 通常在各种问题中表现良好，只需少量调整，包括训练复杂的自编码器。

#### AdamW

AdamW 是实践中常见的改进版本。它通过将权重衰减（L2正则化 (regularization)）从梯度更新步骤中分离出来，从而修改了标准Adam算法。在标准Adam中，权重衰减与自适应学习率隐式关联，这有时可能导致次优正则化。AdamW 在 Adam 更新步骤*之后*直接将权重衰减应用于权重，这通常会带来更好的泛化性能。许多深度学习库现在都提供AdamW 作为一种独立的优化器。



![优化器收敛情况](plots/2928-0.json)



> 示例说明不同优化器如何在损失空间中调整。SGD可能波动，动量平滑路径，而Adam等自适应方法通常收敛更快且更可靠。

### 优化器的选择与调整

对于训练自编码器，Adam 或 AdamW 通常是出色的起始选择，因为它们具有鲁棒性，并且在各种架构和数据集上表现良好。RMSprop 也能有效。虽然带有动量的标准SGD可以工作，但它通常需要更仔细地调整学习率和动量参数 (parameter)。

在现代深度学习 (deep learning)框架（如TensorFlow和PyTorch）中，实现这些优化器很简单：

```python
# PyTorch 示例
import torch.optim as optim

# model 是您定义的自编码器架构
# learning_rate 是您选择的学习率

# 使用 AdamW
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# 使用 RMSprop
# optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)

# 使用带动量的 SGD
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# --- 训练循环 ---
# for data in dataloader:
#     inputs, _ = data
#     optimizer.zero_grad()
#     outputs, latent_vars = model(inputs) # 或者您的模型前向传播工作方式
#     loss = calculate_loss(outputs, inputs, ...) # 计算合适的损失
#     loss.backward()
#     optimizer.step()
# ---------------------
```

```python
# TensorFlow/Keras 示例
import tensorflow as tf

# model 是您定义的 Keras 自编码器模型
# learning_rate 是您选择的学习率

# 使用 AdamW (可通过 tensorflow_addons 或在新版 TF 中内置获得)
# from tensorflow.keras.optimizers import AdamW # 检查 TF 版本
# optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.01)

# 使用 Adam (更常见)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 使用 RMSprop
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9)

# 使用带动量的 SGD
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

# model.compile(optimizer=optimizer, loss=your_loss_function)
# model.fit(x_train, x_train, epochs=num_epochs, batch_size=batch_size, ...)
```

尽管 Adam 等优化器的默认超参数 (hyperparameter)（$\beta_1=0.9, \beta_2=0.999$）通常表现良好，学习率（$\alpha$）仍然是一个重要的超参数，通常需要调整。优化器的效果也与训练的其他方面密切相关，例如批量大小、正则化 (regularization)强度，以及重要的学习率调度，我们将在后续讨论。选择合适的优化器并进行适当调整是成功训练高性能自编码器模型的重要步骤。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam，一种广泛使用的优化算法，结合了自适应学习率和类似动量的梯度平均。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov and Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出了AdamW，是Adam的一个改进版本，将权重衰减从自适应梯度更新中解耦，从而改进了正则化效果。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本教材，涵盖了深度学习中的优化算法，包括SGD、动量、AdaGrad、RMSprop和Adam。
- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html) — John Duchi, Elad Hazan, and Yoram Singer (2011)
  Journal: Journal of Machine Learning Research; Volume: 12; Pages: 2121-2159; DOI: [10.1162/jmlr.2011.12.1.849](https://doi.org/10.1162/jmlr.2011.12.1.849)
  介绍了AdaGrad算法，这是最早的自适应学习率优化方法之一。
