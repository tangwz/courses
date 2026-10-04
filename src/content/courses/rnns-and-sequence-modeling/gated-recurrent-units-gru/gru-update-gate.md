---
course: "rnns-and-sequence-modeling"
chapter: "gated-recurrent-units-gru"
lesson: "gru-update-gate"
sourceId: 2584
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru/gru-update-gate"
title: "更新门"
description: "说明更新门在平衡上一个和候选隐藏状态中的作用。"
order: 3
plots: []
sourceHash: "a9776da4a313ada56a12538e56ecf0963bbdf74998ba68adf5392b96e8c28a11"
sourceCorrections: []
---

GRU单元中的**更新门**（表示为$z_t$）对管理信息流和形成下一个隐藏状态$h_t$起着重要作用。它的主要作用是决定来自*上一个*隐藏状态$h_{t-1}$的信息应保留并传递多少，以及*新计算的候选状态*$ilde{h}_t$应有多少影响。

### 计算更新门的激活值

更新门根据当前输入向量 (vector)$x_t$和上一个隐藏状态$h_{t-1}$计算其激活值$z_t$。计算涉及学习到的权重 (weight)矩阵（输入$x_t$的$W_z$、上一个隐藏状态$h_{t-1}$的$U_z$）和偏置 (bias)向量$b_z$。这些值通过一个Sigmoid激活函数 (activation function)($\sigma$)处理，生成0到1之间的值。

在时间步$t$更新门激活值的公式是：


$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$


我们来分解一下这些组成部分：

- $x_t$: 当前时间步$t$的输入向量。
- $h_{t-1}$: 上一个时间步$t-1$的隐藏状态向量。
- $W_z$: 与输入$x_t$关联的权重矩阵。
- $U_z$: 与上一个隐藏状态$h_{t-1}$关联的权重矩阵。
- $b_z$: 更新门的偏置向量。
- $\sigma$: Sigmoid激活函数，它将组合后的加权输入压缩到$(0, 1)$的范围内。

输出$z_t$是一个与隐藏状态维度相同的向量。$z_t$中的每个元素都对应隐藏状态向量中的一个维度。

### 解释更新门的输出

向量 (vector)$z_t$中的值充当门或过滤器。由于Sigmoid函数，每个元素都在0到1之间：

- **值接近1：** 这表明*上一个*隐藏状态($h_{t-1}$)相应维度中的信息是重要的，应大部分保留。它表示“保留旧记忆”。
- **值接近0：** 这表明上一个状态相应维度中的信息对当前时间步不太相关，应主要由*新候选*隐藏状态($\tilde{h}_t$)的信息替换。它表示“用新信息更新”。

这种门控机制是按元素进行的，这意味着GRU可以根据当前输入决定保留过去的某些特征，同时更新其他特征。

### 在更新隐藏状态中的作用

更新门$z_t$直接参与最终隐藏状态$h_t$的计算。GRU通过对上一个隐藏状态$h_{t-1}$和候选隐藏状态$\tilde{h}_t$进行插值来计算$h_t$。更新门$z_t$控制这种插值。

最终隐藏状态$h_t$的公式是：


$$
h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t
$$


其中：

- $\tilde{h}_t$: 候选隐藏状态，使用当前输入和*重置后*的上一个隐藏状态计算（我们将单独讨论重置门和候选状态的计算）。它代表为当前时间步提出的“新”信息。
- $\odot$: 表示按元素乘法（Hadamard积）。

这个等式表明了$z_t$如何平衡影响：

- 项$z_t \odot h_{t-1}$保留了上一个状态$h_{t-1}$的部分。当$z_t$接近1时，$h_{t-1}$的贡献很大。
- 项$(1 - z_t) \odot \tilde{h}_t$结合了候选状态$\tilde{h}_t$的部分。当$z_t$接近0时，$(1 - z_t)$接近1，$\tilde{h}_t$的贡献很大。

本质上，$z_t$为隐藏状态的每个维度动态决定是复制上一个时间步的值，还是用新计算的候选值进行更新。

> 更新门($z_t$)的计算流程及其在结合上一个隐藏状态($h_{t-1}$)和候选状态($\tilde{h}_t$)以生成新的隐藏状态($h_t$)中的作用。

### 促进长期依赖

更新门能够让来自$h_{t-1}$的信息在很大程度上保持不变地通过（当$z_t \approx 1$时），这是GRU与简单RNN相比能捕获更长距离依赖的重要原因。如果网络学习到某些信息在许多时间步中都相关，它就可以将$z_t$的相应元素设置为接近1，从而有效地为这些信息在时间上传递创建了一个捷径。

这种机制也有助于缓解梯度消失问题。在反向传播 (backpropagation)过程中，梯度可以通过$z_t \odot h_{t-1}$项流回。如果$z_t$接近1，与$h_{t-1}$相关的梯度可以相对畅通地向后传递，从而避免它在许多时间步中过快衰减。

总而言之，更新门$z_t$在GRU单元内提供了一种灵活自适应的机制。它学习控制存储在$h_{t-1}$中的过去上下文 (context)应保留多少，以及来自候选状态$\tilde{h}_t$的新信息应整合多少，从而使网络能够有效建模具有不同依赖长度的序列数据。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  介绍了门控循环单元（GRU）架构，详细阐述了包括更新门在内的各种门控机制，及其在序列到序列学习中的应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第10章提供了循环神经网络的全面理论背景，包括GRU架构及其门控机制的详细解释。
- [CS230: Deep Learning - Lecture Notes for Recurrent Neural Networks](https://github.com/afshinea/stanford-cs-230-deep-learning/blob/master/cheatsheet-recurrent-neural-networks.pdf) — Afshine Amidi, Shervine Amidi (2018)
  Publisher: Stanford University
  提供了循环神经网络、长短期记忆网络和门控循环单元的清晰教学性解释，包括数学公式和门控机制的直观理解。
