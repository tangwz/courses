---
course: "introduction-to-deep-learning"
chapter: "building-training-dnns"
lesson: "weight-initialization"
sourceId: 5056
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/weight-initialization"
title: "权重初始化方法"
description: "讨论适当权重初始化（例如 Xavier/Glorot、He 初始化）的重要性。"
order: 5
plots: []
sourceHash: "2c7d87c4515410b3f67f8ea4d1db3e9f98ac707c43f2d9fea1772ae4a788cc9e"
sourceCorrections: []
---

定义好神经网络 (neural network)的结构，确定各层及其连接后，训练开始前的下一个重要步骤是为每层的权重 (weight)（和偏置 (bias)）设定初始值。你可能会想，为什么不全部设为零呢？或者随机数？事实证明，参数 (parameter)的初始化方式对网络的训练效果有很大影响。不当的初始化可能导致收敛缓慢，甚至完全阻碍网络学习。

### 简单初始化的弊端

让我们考虑几种直接的方法，以及它们为何常常不足：

1. **零初始化：** 将所有权重 (weight)初始化为零看似是一种干净的开始，但这会引入一个严重问题：对称性。如果输入某一层的所有权重都为零，那么在前向传播时，该层的所有神经元将产生相同的输出。结果是，在反向传播 (backpropagation)时，它们都会接收到相同的梯度信号，并以相同的方式更新权重。实际上，该层中的所有神经元都表现得像一个，抵消了拥有多个神经元的好处。网络无法打破对称性并学习不同特征。
2. **大随机值：** 好的，零不好。那将权重初始化为大随机值呢？这可以打破对称性，但会带来自身问题。如果权重过大，激活函数 (activation function)（如 Sigmoid 或 Tanh）的输入可能很快落入其饱和区域。在这些区域中，梯度接近于零。这会导致“梯度消失”问题，即梯度变得非常小，使得权重更新微不足道，并阻碍学习过程。大权重有时也可能导致“梯度爆炸”，即梯度变得过大，引起不稳定更新和发散。
3. **小随机值：** 将权重初始化为小随机数（例如，从具有小标准差的高斯分布中采样）更好。这打破了对称性，并减少了立即饱和的可能性。然而，如果值*过*小，输出的方差可能会随着激活通过网络层传播而逐渐缩小，这可能导致训练后期出现梯度消失，特别是在深度网络中。

挑战在于找到一个“最佳点”，一个既能打破对称性又能保持信号（激活和梯度）有效流经网络层的初始化方案。

### 目标：保持方差

现代权重 (weight)初始化方法在设计时都考虑了一个特定目标：在激活和梯度通过网络传播时，保持其方差不变。如果在前向传播（激活）和反向传播 (backpropagation)（梯度）过程中，方差在各层之间大致保持不变，网络就不太可能出现信号消失或爆炸的问题。这有助于确保所有层都以合理的速度学习。

Xavier/Glorot 初始化和 He 初始化是两种主要采用的初始化方法。

### Xavier (Glorot) 初始化

该方法由 Xavier Glorot 和 Yoshua Bengio 于 2010 年提出，旨在很好地适用于那些关于零对称且输出在一定范围内有界的激活函数 (activation function)，如 Sigmoid 和 Tanh。

它通过从一个均值为零且方差经过精心选择的分布中抽取权重 (weight)来设定它们。方差取决于层的输入单元数量（$n_{in}$ 或 $fan_{in}$）和输出单元数量（$n_{out}$ 或 $fan_{out}$）。

- **正态分布：** 权重从均值为 0、方差为 $\sigma^2$ 的正态分布中抽取：

  
  $$
  \sigma^2 = \frac{2}{n_{in} + n_{out}}
  $$
  
- **均匀分布：** 权重从范围 $[-r, r]$ 的均匀分布中抽取，此处的 $r$ 为：

  
  $$
  r = \sqrt{\frac{6}{n_{in} + n_{out}}}
  $$
  

其原因在于，这种缩放有助于使层输出的方差大致等于其输入的方差，反向传播 (backpropagation)时梯度的情况也类似。

### He 初始化

尽管 Xavier 初始化适用于 Sigmoid 和 Tanh，但对于修正线性单元 (ReLU) 激活函数 (activation function)及其变体（Leaky ReLU、ELU 等）来说，它不那么理想。ReLU 将所有负输入设为零，这与 Tanh 等对称函数相比改变了方差动态。

Kaiming He 等人于 2015 年提出了一种专门为基于 ReLU 的网络定制的初始化方法。它通过调整方差缩放来考虑 ReLU 的非线性。

- **正态分布：** 权重 (weight)从均值为 0、方差为 $\sigma^2$ 的正态分布中抽取：

  
  $$
  \sigma^2 = \frac{2}{n_{in}}
  $$
  
- **均匀分布：** 权重从范围 $[-r, r]$ 的均匀分布中抽取，此处的 $r$ 为：

  
  $$
  r = \sqrt{\frac{6}{n_{in}}}
  $$
  

请注意，He 初始化主要考虑输入单元的数量（$n_{in}$）。这种调整有助于防止信号方差在通过 ReLU 单元层时下降过快。

### 实际操作与框架支持

像 PyTorch 和 TensorFlow/Keras 这样的深度学习 (deep learning)框架使这些方法的实现变得直接。在定义层时，你通常有选项可以指定所需的权重 (weight)初始化方法。

例如，在 PyTorch 中，你可以定义一个线性层，然后应用 He 初始化：

```python
import torch
import torch.nn as nn
import math

# 示例层维度
in_features = 128
out_features = 64

# 定义一个线性层
linear_layer = nn.Linear(in_features, out_features)

# 应用 He 初始化（正态分布）
nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')

# 初始化偏置（通常设为零或小常数）
if linear_layer.bias is not None:
    nn.init.constant_(linear_layer.bias, 0)

print(f"已初始化权重形状: {linear_layer.weight.shape}")
print(f"样本权重（前5个）:\n{linear_layer.weight[0, :5]}")
print(f"\n已初始化偏置形状: {linear_layer.bias.shape}")
print(f"偏置值:\n{linear_layer.bias[:5]}")
```

许多框架会根据上下文 (context)或层类型默认使用 He 或 Xavier 初始化，但了解如何明确设置它们对于微调 (fine-tuning)或实现自定义层非常有价值。

### 选择合适的方法

- **对于 ReLU 及其变体（Leaky ReLU, PReLU, ELU）：** He 初始化通常是推荐的选择。
- **对于 Sigmoid 和 Tanh：** Xavier/Glorot 初始化通常首选。

偏置 (bias)呢？它们通常初始化为零。然而，对于 ReLU 单元，有时会将偏置初始化为一个小的正值（例如 0.01），以确保大多数 ReLU 单元在初始时是激活的，但零初始化仍然常见且有效。

尽管这些初始化方法提供了优秀的起点，但它们并不能消除对其他技术的需要，如批量归一化 (normalization)（稍后会介绍），这些技术进一步帮助稳定训练动态。尽管如此，适当的权重 (weight)初始化是构建高效训练深度网络的重要环节。它为梯度下降 (gradient descent)能有效地优化损失并找到好的解创造了条件。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 9; Pages: 249-256
  介绍了Xavier（又称Glorot）初始化方法，一种通过保持跨层信号方差来初始化神经网络权重的方法，特别适用于Sigmoid和Tanh激活函数。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://doi.org/10.48550/arXiv.1502.01852) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Pages: 1026-1034; DOI: [10.48550/arXiv.1502.01852](https://doi.org/10.48550/arXiv.1502.01852)
  提出了He初始化方法，专为使用ReLU及其变体的深度神经网络设计，解决了这些非线性激活函数所带来的方差缩放问题。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习广泛主题的著作，其中详细讨论了权重初始化策略及其理论基础。
