---
course: "how-to-build-a-large-language-model"
chapter: "optimization-algorithms-llms"
lesson: "adaptive-optimizers-adam-adamw"
sourceId: 6034
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-17-optimization-algorithms-llms/adaptive-optimizers-adam-adamw"
title: "自适应优化器：Adam和AdamW"
description: "阐述Adam优化器及带有解耦权重衰减的AdamW形式。"
order: 2
plots: []
sourceHash: "c22e233bccbbe881da4de84f58116500f10f6c323e72f31cb218bf60515362a8"
sourceCorrections: []
---

标准的随机梯度下降 (gradient descent)（SGD）及其动量变体是优化方法的基础，但训练像大型语言模型（LLMs）这样庞大的模型，通常能从自适应学习率方法中获得更好的表现。这些算法会为每个参数 (parameter)单独调整学习率，可能加快收敛速度，特别是在梯度稀疏或参数间梯度大小不一的情形下，这在深度神经网络 (neural network)中很普遍。

### Adam：自适应矩估计

最受青睐的自适应优化器之一是Adam（自适应矩估计）。Adam通过梯度的第一和第二矩估计，为不同参数 (parameter)计算各自的自适应学习率。它主要结合了动量（使用第一矩估计，即过去梯度的指数衰减平均）和RMSprop（使用第二矩估计，即过去梯度平方的指数衰减平均）的理念。

设$\mathbf{g}_t$是目标函数在时间步$t$时对参数$\theta$的梯度。Adam维护两个移动平均值：

1. **第一矩估计（动量）：**

   
   $$
   \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t
   $$
   

   这是梯度的均值估计。$\beta_1$是指数衰减率，通常接近1（例如，0.9）。
2. **第二矩估计（未中心化方差）：**

   
   $$
   \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2
   $$
   

   这是梯度未中心化方差的估计（逐元素平方）。$\beta_2$是指数衰减率，通常也接近1（例如，0.999）。

由于$\mathbf{m}_t$和$\mathbf{v}_t$被初始化为零向量 (vector)，它们会偏向于零，特别是在最初的时间步。Adam对这种偏差进行了校正：


$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}
$$


$$
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$


其中$t$是当前时间步索引（从1开始）。

最后，参数更新规则是：


$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$


这里，$\eta$是基础学习率，$\epsilon$是一个小常数（例如，$10^{-8}$），为了数值稳定性而添加，主要用于避免除以零。项$\frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$起到了实际的、针对参数的学习率的作用。过去梯度较大的参数（$\hat{\mathbf{v}}_t$较大）获得较小的更新，而过去梯度较小的参数则获得较大的更新。

在PyTorch中，使用Adam很简单：

```python
import torch
import torch.optim as optim

# 假设模型是一个已定义的torch.nn.Module
# learning_rate, beta1, beta2, epsilon 是超参数
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=epsilon
)

# 在训练循环中：
# 损失 = 计算损失(...)
# 优化器.梯度清零()
# 损失.反向传播()
# 优化器.步进()
```

### AdamW：解耦权重 (weight)衰减

尽管Adam在多种场景下表现良好，但它对L2正则化 (regularization)（权重衰减）的处理可能不尽理想。标准的L2正则化会向损失函数 (loss function)添加项$\frac{\lambda}{2} \|\theta\|^2$，导致梯度项$\lambda \theta$被加到$\mathbf{g}_t$中。在Adam中，这个权重衰减项$\lambda \theta$通过$\mathbf{m}_t$和$\mathbf{v}_t$成为自适应学习率计算的一部分。这表明应用于参数 (parameter)的实际权重衰减取决于其梯度历史大小（通过$\sqrt{\hat{\mathbf{v}}_t}$）。梯度大的参数所受到的实际权重衰减小于预期，而梯度小的参数所受到的实际权重衰减则大于预期。

AdamW提出了一个简单修正：将权重衰减与梯度更新解耦。AdamW不再将$\lambda \theta$加到梯度$\mathbf{g}_t$中，而是仅使用来自主要损失函数的梯度进行标准的Adam更新，然后在Adam步骤*之后*直接对参数应用权重衰减。

AdamW的更新规则如下：

1. 按照Adam的方式计算$\mathbf{m}_t$、$\mathbf{v}_t$、$\hat{\mathbf{m}}_t$、$\hat{\mathbf{v}}_t$，只使用来自损失函数的梯度$\mathbf{g}_t$（不包含$\lambda \theta$项）。
2. 执行自适应更新：
   
   $$
   \mathbf{u}_t = \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
   $$
   
3. 应用权重衰减并更新参数：
   
   $$
   \theta_t = \theta_{t-1} - \mathbf{u}_t - \eta \lambda \theta_{t-1}
   $$
   

请注意最后一项：$-\eta \lambda \theta_{t-1}$。权重衰减直接作用于先前的权重值$\theta_{t-1}$，并且只按全局学习率$\eta$进行缩放，而非自适应学习率。这使得权重衰减的行为更接近于标准SGD加上动量时的表现，从而在很多情况下，特别是对于Transformer这类深度模型，能够带来更好的泛化性能。

在PyTorch中使用AdamW与Adam类似，只额外需要`weight_decay`参数：

```python
import torch
import torch.optim as optim

# 假设模型是一个已定义的torch.nn.Module
# learning_rate, beta1, beta2, epsilon, weight_decay_lambda 是超参数
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=epsilon,
    weight_decay=weight_decay_lambda # 注意 weight_decay 参数
)

# 在训练循环中：
# 损失 = 计算损失(...)
# 优化器.梯度清零()
# 损失.反向传播()
# 优化器.步进()
```

由于其对权重衰减的改进处理和出色的经验表现，AdamW已成为训练大型语言模型的一个非常常见的选择。Adam和AdamW之间的选择，以及它们超参数 (hyperparameter)（$\eta, \beta_1, \beta_2, \epsilon, \lambda$）的设置，通常取决于模型架构、数据集和训练配置，这需要仔细调整，本章后面会对此进行讨论。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2014)
  Journal: Proceedings of the 3rd International Conference for Learning Representations (ICLR), 2015; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍 Adam 优化算法的原始研究论文。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出 AdamW 优化器及其解耦权重衰减的原始研究论文。
- [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch 中 Adam 优化器的官方文档。
- [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch 中 AdamW 优化器的官方文档。
