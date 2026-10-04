---
course: "intro-normalizing-flows"
chapter: "autoregressive-flows"
lesson: "inverse-autoregressive-flow"
sourceId: 8266
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-3-autoregressive-flows/inverse-autoregressive-flow"
title: "逆自回归流"
description: "分析逆自回归流模型及其如何优化采样速度。"
order: 4
plots: []
sourceHash: "ce451c085327d9f92eabf0977c8c17d3871e821fbe7e67c76e0ec749761e4c5d"
sourceCorrections: []
---

掩码自回归 (autoregressive)流 (MAF) 在密度估计方面表现优异，因为计算数据点的精确似然值仅需通过网络进行单次前向传播。然而，使用 MAF 生成新数据时，需要对所有维度进行顺序循环。在处理音频或图像等高维数据时，这种顺序采样过程的计算成本很高。逆自回归流 (IAF) 通过重构自回归变换来解决这个问题，从而实现了高效的并行采样。

让我们来看看这种变换的数学原理。在逆自回归流中，目标变量 $x$ 是从简单分布（通常是标准高斯分布）中提取的基础变量 $z$ 生成的。自回归变换定义为：

$x_i = \mu_i(z_{1:i-1}) + \sigma_i(z_{1:i-1}) \cdot z_i$

请注意 IAF 和 MAF 之间的显著区别。在 MAF 中，缩放和平移参数 (parameter)取决于之前生成的目标变量 $x_{1:i-1}$。而在 IAF 中，参数 $\mu_i$ 和 $\sigma_i$ 取决于之前采样的基础变量 $z_{1:i-1}$。

由于整个向量 (vector) $z$ 是从基础分布中一次性独立采样的，因此在计算 $x$ 之前，每个元素 $z_{1:i-1}$ 都是已知的。这意味着我们可以通过神经网络 (neural network)进行单次并行传播来处理整个向量 $z$，并同时输出所有的 $\mu$ 和 $\sigma$ 值。

一旦网络输出了这些参数，最终的变量 $x$ 就可以通过简单的逐元素操作计算出来。无论数据维度如何，生成一个完整的样本都只需要一个步骤。

> 逆自回归流的采样过程从基础样本中并行计算所有平移和缩放参数，消除了顺序循环。

这种快速采样架构的代价是密度估计较慢。假设你有一个数据点 $x$ 并希望评估其精确似然值。为此，你必须反转变换以找到相应的基础变量 $z$。反向操作为：

$z_i = \frac{x_i - \mu_i(z_{1:i-1})}{\sigma_i(z_{1:i-1})}$

这就是计算瓶颈所在。为了计算 $z_i$，你必须已经知道 $z_{1:i-1}$ 才能计算出 $\mu_i$ 和 $\sigma_i$。你无法并行计算这些参数，因为神经网络的输入正是你正在尝试求解的 $z$ 值。你必须先计算 $z_1$，将其传入网络以获取 $\mu_2$ 和 $\sigma_2$，再用它们计算 $z_2$，并对所有维度重复此过程。评估密度需要顺序循环，这使得通过精确最大似然估计进行训练变得效率低下。

为了清晰展示采样过程的机制，让我们看看如何在 PyTorch 中实现前向生成步骤。假设我们有一个初始化为 `autoregressive_net` 的自回归网络，它接收 $z$ 作为输入并输出拼接后的平移和缩放参数。

```python
import torch
import torch.nn as nn

class IAFLayer(nn.Module):
    def __init__(self, autoregressive_net):
        super().__init__()
        # 该网络可以采用 MADE 架构
        self.net = autoregressive_net

    def forward_sample(self, z):
        # z 是从基础分布中采样的：z ~ N(0, I)
        # 通过网络单步处理 z
        params = self.net(z)

        # 将参数拆分为平移 (mu) 和对数缩放 (log_sigma)
        mu, log_sigma = torch.chunk(params, 2, dim=-1)

        # 约束缩放参数为正值
        sigma = torch.exp(log_sigma)

        # 并行计算最终样本 x
        x = mu + sigma * z

        # 计算雅可比行列式的对数 (log determinant)
        log_det_jacobian = log_sigma.sum(dim=-1)

        return x, log_det_jacobian
```

在此代码片段中，`forward_sample` 方法执行时没有任何循环。我们对 `log_sigma` 使用指数函数，以确保缩放参数始终为正，这是变换可逆的硬性要求。雅可比行列式的对数仅仅是对数缩放参数的总和，这与 MAF 中的情况一致。

MAF 和 IAF 体现了一种独特的架构对偶性。如果你使用精确最大似然估计来训练生成模型，MAF 是常规选择。如果你的主要目标是快速生成合成数据，IAF 则是更优的模型。在实际应用中，研究人员有时会结合这两种架构的优点。例如，某些训练流程会先训练一个评估速度快的 MAF 模型。然后，利用训练好的 MAF 将知识“蒸馏”到 IAF 模型中。这种被称为概率密度蒸馏的技术，使 IAF 模型能够在训练过程中不依赖缓慢的顺序循环就能学习目标分布。最终得到的是一个既能高效训练又能即时采样的部署模型。
