---
course: "graph-neural-networks-gnns"
chapter: "gnn-training-complexities"
lesson: "gnn-optimization-strategies"
sourceId: 2714
sourceUrl: "https://apxml.com/zh/courses/graph-neural-networks-gnns/chapter-3-gnn-training-complexities/gnn-optimization-strategies"
title: "图神经网络的优化策略"
description: "讨论训练复杂的图神经网络模型时适用的优化器和学习率调度方法。"
order: 8
plots: ["plots/2714-0.json"]
sourceHash: "2b93bd0f50bbd1b3e68cd876cf55ed18fdcd39173275cf4252e92d26118399a4"
sourceCorrections: []
---

选择正确的架构并采用方法来管理规模和表示质量是重要步骤，但优化过程本身是在训练中实现这些选择的环节。标准的深度学习 (deep learning)优化技术构成了基础，但图数据独特的结构以及在图神经网络 (neural network)训练中遇到的具体难题，使得需要仔细考量优化器、学习率调度和相关策略。未能有效优化可能导致收敛缓慢、不稳定或模型性能不佳，从而抵消了先进架构或扩展方法的益处。

## 图神经网络 (neural network)的优化器

优化器的选择决定了模型参数 (parameter)如何根据计算出的梯度进行更新。虽然带有动量的随机梯度下降 (gradient descent)（SGD）仍然是一个可行选项，但自适应学习率优化器已成为训练大多数深度学习 (deep learning)模型（包括图神经网络）的标准选择。

### Adam及其变体

Adam（自适应矩估计）优化器可以说是训练图神经网络 (neural network)最常见的起点。它通过存储过去平方梯度（如RMSprop）和过去梯度（如动量）的指数衰减平均值，为每个参数 (parameter)计算自适应学习率。

其公式包含：

1. 更新有偏的一阶矩估计：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. 更新有偏的二阶原始矩估计：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. 计算偏差修正后的一阶矩估计：$\hat{m}_t = m_t / (1 - \beta_1^t)$
4. 计算偏差修正后的二阶原始矩估计：$\hat{v}_t = v_t / (1 - \beta_2^t)$
5. 更新参数：$\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

此处$g_t$是时间步$t$处的梯度，$\beta_1$和$\beta_2$是衰减率（通常为0.9和0.999），$\eta$是学习率，$\epsilon$是用于数值稳定的小常数（例如，$10^{-8}$）。

与SGD相比，Adam通常能带来更快的初期收敛。AdamW是Adam的一个变体，它将权重 (weight)衰减与自适应学习率机制解耦，常受青睐，因为它通过在参数更新步骤之前直接将权重衰减应用于权重，而不是将其纳入梯度计算中（梯度计算会被自适应项缩放），从而可以带来更好的泛化能力。

尽管功能强大，Adam并非没有自身的考量。它可能对学习率和beta参数的选择敏感，有时会收敛到更尖锐的极小值，其泛化能力可能略差于带有动量的SGD所找到的极小值，虽然这常有争议且依赖于具体情况。

### 其他优化器

RMSprop是另一种有时在图神经网络 (neural network)上表现良好的自适应优化器。新优化器偶尔出现，但Adam/AdamW仍是标准且被广泛采用的默认选择。在面对特定难题或架构时，尝试不同的方法可能是合理的。

## 学习率调度

固定的学习率，特别是较大的学习率，在整个训练过程中很少能达到最优。学习率调度随时间调整学习率，通常旨在初期实现更快的收敛，并在训练进行时进行更精细的调整。

### 常见调度策略

- **步长衰减：** 在预设的周期（epoch）处将学习率乘以一个衰减因子进行降低。方法简单，但需要调整周期和衰减因子。
- **指数衰减：** 在每个周期甚至每个批次后，将学习率乘以一个略小于1的衰减因子。提供比步长衰减更平滑的降低。
- **余弦退火：** 按照余弦曲线逐渐降低学习率，通常在经过一定数量的周期后，从初始学习率降至零或一个较小的值。它在寻找损失曲面上的优良极小值方面可能有效。常与重启结合使用（带重启的余弦退火或SGDR），在此情况下，学习率会周期性地重置为初始值并再次退火。
- **验证集指标停滞时降低学习率（ReduceLROnPlateau）：** 监控一个验证指标（例如，验证损失或准确率），当指标在指定数量的周期（'patience'）内停止提升时，将学习率乘以一个因子进行降低。这会适应实际的训练进程。

### 学习率热身

特别是在使用Adam等自适应优化器和大数据量批次（在可扩展的图神经网络 (neural network)训练中很常见）时，直接使用目标学习率可能导致训练初期不稳定。一个*热身*阶段通过从一个很小的学习率开始，并在指定数量的初始步长或周期内逐渐将其提高到目标学习率来解决此问题。这使得Adam中的自适应矩在进行大幅更新之前得以稳定。线性或余弦热身调度是常见的。



![学习率调度曲线](plots/2714-0.json)



> 不同学习率调度策略随训练步数的变化比较，包括常与余弦退火结合使用的热身阶段。

选择正确的调度策略常涉及实验。带线性热身阶段的余弦退火是训练图神经网络和Transformer等复杂模型时常使用的强大组合。

## 梯度裁剪

在深度网络或处理序列（或图中路径）的模型中，梯度有时会变得过大，导致训练不稳定，称为梯度爆炸。梯度裁剪通过重新缩放梯度来缓解此问题，即当梯度的幅值超过特定阈值时进行操作。

- **按范数裁剪：** 这是最常见的方法。如果整个梯度向量 (vector)（所有参数 (parameter)的梯度）的L2范数（欧几里得长度）超过阈值$C$，梯度向量$g$会被重新缩放：
  
  $$
  g = g \cdot \frac{C}{\|g\|_2} \quad \text{如果 } \|g\|_2 > C
  $$
  
  这保留了梯度的方向但限制了其幅值。典型的裁剪阈值可能在1.0到5.0之间，但最优值取决于模型和数据。
- **按值裁剪：** 将每个单独的梯度分量裁剪到特定范围$[-C, C]$内。这在一般深度学习 (deep learning)训练中较不常见。

梯度裁剪作为一种安全机制，在训练初期或使用高学习率时特别有用。

## 正则化 (regularization)考量

虽然权重 (weight)衰减（L2正则化）和dropout等正则化技术常被视为模型架构的一部分，但它们直接影响优化过程。

- **权重衰减：** 惩罚大权重，鼓励更简单的模型。如前所述，AdamW以不同于标准Adam的方式实现权重衰减，这可能是有益的。最优权重衰减因子是一个需要调整的超参数 (parameter) (hyperparameter)。
- **Dropout：** 标准dropout、DropEdge或DropNode等技术在训练期间随机移除单元、边或节点。这引入了噪声，这有助于防止模型过度依赖特定特征或连接。这种噪声本质上会影响优化器使用的梯度。

## 可扩展训练环境下的优化

前面介绍的采样和聚类技术（如邻域采样、GraphSAINT、Cluster-GCN）使得在大图上进行训练成为可能，这通过使用从子图派生的小批量数据实现。与全批量梯度下降 (gradient descent)相比，这给梯度估计带来了方差。

- **优化器选择：** Adam/AdamW因其对噪声梯度的处理能力而仍是标准选择。
- **学习率：** 有时，可能需要稍小的学习率或更长的热身期以应对增加的方差。反之，当使用非常大的有效批量大小（例如，通过梯度累积或使用大簇）时，线性增加学习率（带热身）是一种常见启发式方法，尽管其有效性可能有所不同。
- **训练时长：** 由于梯度估计带有噪声，使用采样进行训练可能比全批量训练需要更多周期才能收敛。

## 超参数 (parameter) (hyperparameter)调整

找到优化器、学习率、调度参数、权重 (weight)衰减和裁剪阈值的最优组合对于达到最佳性能很重要。这通常涉及超参数调整：

- **网格搜索/随机搜索：** 系统地或随机地检查超参数组合。
- **贝叶斯优化：** 更高级的方法，它们构建目标函数（例如，验证性能）的概率模型，并用其选择接下来要评估的有前景的超参数。
- **自动化工具：** Optuna、Ray Tune或Weights & Biases Sweeps等库可以自动化调整过程，使其更高效。

从合理的默认值开始（例如，AdamW，使用$\eta=10^{-3}$，$\beta_1=0.9$，$\beta_2=0.999$，带热身的余弦退火，以及$10^{-4}$或$10^{-5}$等中等权重衰减），然后系统地进行调整，初期主要关注学习率和权重衰减。

总而言之，虽然AdamW等标准优化器构成了基础，但有效的图神经网络 (neural network)训练常需要仔细调整学习率调度（特别是热身和衰减）、可能需要使用梯度裁剪以增强稳定性，并认识到可扩展训练技术如何影响优化动态。系统性的超参数调整几乎总是必要的，以使复杂图神经网络模型取得最佳结果。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化器，一种广泛应用于训练深度神经网络的自适应学习率方法。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Ilya Loshchilov, Frank Hutter (2019)
  Journal: International Conference on Learning Representations (ICLR) 2019; DOI: [10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
  提出了AdamW，一种Adam的变体，它正确地将权重衰减与梯度更新分离，从而实现更好的泛化性能。
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) — Ilya Loshchilov, Frank Hutter (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1608.03983](https://doi.org/10.48550/arXiv.1608.03983)
  提出了带热重启的余弦退火（SGDR）作为一种有效的学习率调度策略，以实现更快的收敛和更好的泛化。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习优化基础技术的综合性教科书，包括优化器、学习率策略和梯度裁剪。
