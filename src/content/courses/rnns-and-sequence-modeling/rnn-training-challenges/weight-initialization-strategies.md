---
course: "rnns-and-sequence-modeling"
chapter: "rnn-training-challenges"
lesson: "weight-initialization-strategies"
sourceId: 2560
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-4-rnn-training-challenges/weight-initialization-strategies"
title: "权重初始化策略"
description: "讨论适用于循环层的权重初始化方法。"
order: 5
plots: []
sourceHash: "56072ad2999b125e67e676afe6783ff268c6f84760f0008ddff016057fe7d383"
sourceCorrections: []
---

梯度消失和梯度爆炸问题在训练循环神经网络 (neural network) (RNN)时带来显著的难题，特别是在许多时间步上。时间反向传播 (backpropagation)（BPTT）涉及对相同的循环权重 (weight)矩阵（在我们简单的RNN公式中为$W_{hh}$）进行重复乘法。如果权重的大小（或更准确地说，权重矩阵的特征值或奇异值）持续小于1，梯度在反向传播时会呈指数级缩小。反之，如果它们持续大于1，梯度会呈指数级增长。在训练开始时初始化网络权重的方式，对于为稳定的梯度流动做好准备起着重要作用。

随机初始化权重是标准做法，但这些随机值的*尺度*和*分布*非常重要。简单的初始化策略，例如从非常小范围的均匀分布（如$U[-0.01, 0.01]$）或具有微小标准差的高斯分布中提取权重，通常直接导致梯度消失，特别是在深度网络或展开许多步骤的RNN中。同样，使用过大的值进行初始化会从一开始就引发梯度爆炸。

现代深度学习 (deep learning)依赖于更符合原理的初始化方法，这些方法旨在保持信息通过层向前传播时的信号方差，以及误差向后传播时的梯度方差。

### Glorot (Xavier) 初始化

由Glorot和Bengio（2010年）提出，此方法旨在使激活和梯度的方差在各层之间保持近似不变。它假设使用对称激活函数 (activation function)，如双曲正切（`tanh`），这在简单RNN和LSTM/GRU中很常见。其核心思想是根据给定层（或权重 (weight)矩阵）的输入单元数（$n_{in}$）和输出单元数（$n_{out}$）来缩放初始权重。

- **Glorot 均匀分布：** 权重从均匀分布中抽取：
  
  $$
  W \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]
  $$
  
- **Glorot 正态分布：** 权重从均值为零、方差为以下值的正态分布中抽取：
  
  $$
  \sigma^2 = \frac{2}{n_{in} + n_{out}}
  $$
  

对于一个RNN单元，$n_{in}$通常是输入特征向量 (vector)的大小加上隐藏状态的大小，$n_{out}$则是隐藏状态的大小。许多深度学习 (deep learning)框架将Glorot初始化作为全连接层和循环层的默认设置。

### He 初始化

由He等人（2015年）开发，此初始化方案专门为使用修正线性单元（ReLU）激活函数 (activation function)及其变体（Leaky ReLU、ELU）的层设计。由于ReLU不对称且会将负输入置为零，因此其方差动态与`tanh`不同。He初始化考虑了这一点。

- **He 均匀分布：** 权重 (weight)从均匀分布中抽取：
  
  $$
  W \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]
  $$
  
- **He 正态分布：** 权重从均值为零、方差为以下值的正态分布中抽取：
  
  $$
  \sigma^2 = \frac{2}{n_{in}}
  $$
  

请注意，He初始化仅考虑输入单元数（$n_{in}$）。虽然ReLU作为单元*内*的主要循环激活函数较不常见（因为无界激活可能导致不稳定），但它可能用于更复杂单元内的前馈连接或输出层。如果大量使用类似ReLU的激活函数，He初始化通常更受青睐。

### 正交初始化

针对RNN而言，循环权重 (weight)矩阵（$W_{hh}$）在每个时间步都被重复应用。正交矩阵具有保持其所乘向量 (vector)的范数（长度）不变的属性。将$W_{hh}$初始化为（或接近于）一个正交矩阵，理论上可以帮助在BPTT期间保持梯度范数，从而减轻梯度消失和梯度爆炸问题。

在初始化时实现完美的正交性通常涉及对初始随机矩阵进行奇异值分解（SVD），将奇异值设置为1，然后重构矩阵等方法。虽然计算上比Glorot或He稍微复杂一些，但它对于普通RNN或LSTM/GRU中的循环权重特别有效，尤其是在处理非常长的序列时。框架可能提供此选项作为特定的初始化器（例如，`tf.keras.initializers.Orthogonal` 或 `torch.nn.init.orthogonal_`）。

### 实际考量

- **框架默认设置：** 了解你所选深度学习 (deep learning)库（TensorFlow/Keras、PyTorch）使用的默认初始化器。它们通常默认为Glorot/Xavier，这是一个合理的起点。
- **自定义：** 框架允许你为循环层内的不同权重 (weight)矩阵（例如，输入权重、循环权重、偏置 (bias)）指定不同的初始化器。你可以尝试将正交初始化专门用于循环权重（$W_{hh}$），同时将Glorot或He用于输入权重（$W_{xh}$）。
- **偏置初始化：** 偏置通常初始化为零，尽管有时建议将LSTM遗忘门的偏置初始化为一个小的正值（例如1.0），以鼓励在训练开始时默认记住信息。

尽管精密的初始化有助于建立一个良好的训练起点，但它并不能完全消除梯度问题，特别是对于非常深的网络或长序列。它作为一项重要的预防措施，通常与梯度裁剪（将在下文讨论）以及LSTM和GRU等架构（将在后续章节介绍）结合使用，这些架构本身对这些问题更具鲁棒性。选择合适的初始化策略是配置RNN模型以实现成功训练的基本步骤。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 9; Pages: 249-256; DOI: [10.5598/AISTATS.2010.53](https://doi.org/10.5598/AISTATS.2010.53)
  介绍了Xavier（Glorot）初始化方法，用于在具有对称激活函数的深度网络中保持稳定的信号方差。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://ieeexplore.ieee.org/document/7410543) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 1026-1034; DOI: [10.1109/ICCV.2015.122](https://doi.org/10.1109/ICCV.2015.122)
  提出了适用于ReLU激活函数的He初始化方法，确保深度网络中恰当的方差缩放。
- [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/pdf/1312.6120.pdf) — Andrew M. Saxe, James L. McClelland, Stephen Ganguli (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6120](https://doi.org/10.48550/arXiv.1312.6120)
  为正交矩阵如何在深度网络中保持信号和梯度范数提供了理论基础，与循环权重初始化有关。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖各种神经网络主题的教科书，包括初始化策略和RNN面临的挑战。
- [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Authors (2023)
  Publisher: PyTorch
  PyTorch中权重初始化函数的官方文档，展示了所讨论策略的实际实现。
