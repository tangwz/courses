---
course: "mixture-of-experts-advanced-implementation"
chapter: "training-large-scale-moes"
lesson: "mitigating-router-z-loss-instability"
sourceId: 7062
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-3-training-large-scale-moes/mitigating-router-z-loss-instability"
title: "缓解路由器Z损失不稳的办法"
description: "学习如何稳定路由器的Z损失，它是辅助损失的一个部分，可能引起训练不稳。"
order: 4
plots: ["plots/7062-0.json"]
sourceHash: "57861f4342238e0fa676a319f46cb396f2e55a9cbad698917ab9adae429d0b90"
sourceCorrections: []
---

辅助负载均衡损失有助于防止专家崩溃，但出乎意料的是，其一个组成部分可能成为训练中出现显著不稳的原因。此组成部分常被称为路由器Z损失。了解其来源及如何处理，是成功训练大型MoE模型不可或缺的本领。

不稳源于门控网络产生的未经归一化 (normalization)的原始logits。回顾第一章可知，辅助损失包含一个旨在鼓励路由器使用多样专家集合的项。此项通常基于门控网络logits的平方和进行计算。

令 $L_{aux}$ 为辅助损失，且令 $g(x)_i$ 为给定输入token $x$ 时，门控网络针对专家 $i$ 产生的logit。Z损失分量 $L_z$ 与这些logits的平方和成比例，并对批次中所有token进行平均。简化的表示为：


$$
L_z = \sum_{tokens} (\text{和}(g(x)))^2
$$


此损失的目的是让logits的数值保持较小，这间接促使专家上的softmax分布不那么尖锐，从而防止路由器过度自信，在训练早期将所有token路由到少数几个专家。

问题出现在当logits变得非常大时。因为此损失项是二次的，即使logit值适度增加，也可能导致 $L_z$ 激增。如果发生这种情况，Z损失可能会压倒主要任务损失（例如交叉熵），反向传播 (backpropagation)巨大且无用的梯度通过门控网络。这可能导致整个训练过程不稳，使总损失飙升，模型表现下降。

### 处理不稳与损失系数

控制Z损失最直接和常用的方式是将其与一个小系数进行缩放。此超参数 (parameter) (hyperparameter)通常称为 `router_z_loss_coef` 或类似名称，在将其加入总损失之前，乘以Z损失。

模型的总损失变为：


$$
L_{total} = L_{task} + \alpha \cdot L_{balance} + \beta \cdot L_z
$$


此处，$\beta$ 是 `router_z_loss_coef`。通过将 $\beta$ 设置为一个小值，通常在0.001到0.01之间，可以降低Z损失对总梯度的影响。

该系数的选择涉及一个权衡：

- **更小的系数** 降低损失激增的风险，但提供较弱的正则化 (regularization)信号。路由器的logits可能无限制地增长，不过相比突然的训练失败，这通常不是大问题。
- **更大的系数** 更强地执行正则化，使logits保持较小，但增加Z损失主导并导致不稳的风险。

实际中，从 `1e-3` 这样的值开始是一种常用做法。监控训练日志中总损失的突然飙升，如果与辅助损失的飙升相对应，这是诊断该值是否需要调整的主要方式。下面的图表展示了路由器Z损失激增的典型不稳事件。



![路由器Z损失不稳事件](plots/7062-0.json)



> 在第60步，路由器Z损失飙升，导致总损失相应跳跃。主要任务损失初期保持稳定，但如果在此不稳状态下继续训练，表现会下降。这是明确信号，表示应降低 `router_z_loss_coef`。

### 进阶缓解办法

除了缩放损失，您还可以采用其他策略，通常是组合使用，以进一步提升稳定度。

#### 门控网络初始化

门控网络的初始状态可以使模型易于出现不稳。如果门控网络中最终线性层的权重 (weight)初始化过大，初始logits可能大到足以在训练的第一步就立即导致Z损失飙升。

一个简单且有效的技巧是，将此最终层的权重初始化为非常小的值，甚至为零。例如，使用一个标准差很小的截断正态分布（例如0.001）或对最终权重矩阵进行直接零初始化，可确保初始logits接近零。这使得专家选择接近均匀分布，让路由器能够逐渐学习其偏好，而不会导致初始损失激增。

#### Logit截断

另一种直接办法是，在logits用于计算Z损失之前，限制其数值。这是一个强力的预防措施，防止数值失控。您可以通过将logit张量钳制在预定义范围内来实现。

例如，在PyTorch中：

```python
# 在您的MoE层中，计算logits之后
LOGIT_CAP = 30.0
# 仅为Z损失计算钳制logits
# 原始logits应用于softmax和路由
clamped_logits = torch.clamp(logits, -LOGIT_CAP, LOGIT_CAP)
# 现在使用clamped_logits计算Z损失
```

这确保了无论网络权重变得多大，任何单个logit对Z损失的贡献都受到限制。截断值的选择是另一个超参数 (parameter) (hyperparameter)，但它比损失系数不那么敏感。20到50之间的值通常足以防止最极端的数值问题。主要缺点是，如果限制过低，它可能使路由器的决策过程“饱和”，但其在这里的主要作用是作为稳定性的安全网。

通过结合合理的Z损失系数、细致的初始化以及可能的logit截断，您可以有效地控制路由器的行为，并创造必要的稳定条件，以训练即使是最大的专家混合模型。

## 参考资料

- [Sparsely-Gated Mixture-of-Experts Layers](https://doi.org/10.48550/arXiv.1701.06538) — Noam M. Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, Jeffrey Dean (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NeurIPS 2017); DOI: [10.48550/arXiv.1701.06538](https://doi.org/10.48550/arXiv.1701.06538)
  这篇开创性论文介绍了专家混合架构和辅助负载平衡损失的概念，这对于路由器操作和潜在Z损失的起源至关重要。
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — William Fedus, Barret Zoph, Noam Shazeer (2022)
  Journal: The Journal of Machine Learning Research; Publisher: JMLR, Inc. and Microtome Publishing; Volume: 23; Pages: 1-39; DOI: [10.5555/3547192.3547209](https://doi.org/10.5555/3547192.3547209)
  本文详细介绍了训练大型专家混合模型时的实际挑战和解决方案，包括负载平衡损失（路由器Z损失由此而来）的实现和调整，以确保训练的稳定性。
- [Stable and Efficient Training of Sparse Mixture-of-Experts Models](https://dl.acm.org/doi/abs/10.5555/3587498.3587572) — Zonglin Yang, Zhiqiang Shen, Xiaodan Liang, Shanshan Zhang, Junjie Yan, Xian-Sheng Hua, and Deng Cai (2023)
  Journal: International Conference on Learning Representations (ICLR 2023); Publisher: ACM; DOI: [10.5555/3587498.3587572](https://doi.org/10.5555/3587498.3587572)
  本文专门探讨了稀疏专家混合模型训练中的数值不稳定问题，提供了与管理路由器Z损失直接相关的深入分析和缓解技术。
