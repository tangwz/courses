---
course: "rnns-and-sequence-modeling"
chapter: "rnn-training-challenges"
lesson: "problem-vanishing-gradients"
sourceId: 2553
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-4-rnn-training-challenges/problem-vanishing-gradients"
title: "梯度消失问题"
description: "说明为什么在BPTT过程中梯度会变得非常小，从而阻碍长距离依赖的学习。"
order: 1
plots: []
sourceHash: "f321e9c37af3883db612b82b779fd27cdebc05a8ad4991ff60fc410b853294dc"
sourceCorrections: []
---

训练循环神经网络 (neural network)（RNN）涉及计算损失函数 (loss function)相对于网络权重 (weight)的梯度。这个梯度表明了如何调整权重以提升性能。时间反向传播 (backpropagation)（BPTT）通过在时间维度上展开网络，并从最后一个时间步向前应用链式法则来实现这一点。

梯度消失问题直接源于反向传播的机制，尤其是在许多时间步长上重复应用链式法则。为说明原因，我们来考虑梯度信息是如何流动的。损失$L$相对于早期时间步$k$的隐藏状态的梯度，记作$\frac{\partial L}{\partial h_k}$，依赖于晚期时间步$t$的梯度$\frac{\partial L}{\partial h_t}$。这种联系通过链式法则建立，其中包含时间步$k$和$t$之间隐藏状态转换的导数：


$$
\frac{\partial L}{\partial h_k} = \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial h_k}
$$


表达式$\frac{\partial h_t}{\partial h_k}$表示时间步$k$的隐藏状态变化如何影响时间步$t$的隐藏状态。这本身是连接$k+1$到$t$的每个时间步长的中间雅可比矩阵（偏导数矩阵）的乘积：


$$
\frac{\partial h_t}{\partial h_k} = \frac{\partial h_t}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial h_{t-2}} \dots \frac{\partial h_{k+1}}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}
$$


每个中间雅可比矩阵$\frac{\partial h_i}{\partial h_{i-1}}$都依赖于循环权重矩阵$W_{hh}$以及RNN单元中使用的激活函数 (activation function)的导数。对于使用激活函数$f$的简单RNN，隐藏状态更新为$h_i = f(W_{hh}h_{i-1} + W_{xh}x_i + b_h)$。导数$\frac{\partial h_i}{\partial h_{i-1}}$涉及$W_{hh}^T$乘以激活函数$f'$的逐元素导数。

现在，考虑如果这些雅可比矩阵的“大小”（更正式地说，最大奇异值或谱半径）始终小于1，会发生什么。当时间间隔$t-k$很大时，需要多次将这些矩阵相乘，结果乘积$\frac{\partial h_t}{\partial h_k}$的整体大小会呈指数级缩小。

想象一下将一个数字重复乘以0.9。

- $0.9^1 = 0.9$
- $0.9^{10} \approx 0.35$
- $0.9^{50} \approx 0.005$
- $0.9^{100} \approx 0.000027$

这个值迅速趋近于零。

早期RNN中常用的激活函数，例如双曲正切（tanh）或S型函数（sigmoid），它们的导数对于大多数输入都严格小于1（tanh的导数始终$\leq 1$，sigmoid的导数始终$\leq 0.25$）。当这些小的导数在反向传播经过许多时间步时被重复相乘，并可能与范数也小于1的循环权重矩阵$W_{hh}$（可能是由于初始化或正则化 (regularization)）结合时，雅可比矩阵$\frac{\partial h_i}{\partial h_{i-1}}$的大小通常小于1。

> BPTT过程中梯度流动的示意图。梯度信号（红色箭头，粗细表示大小）在时间步长上向后传播时（从右到左）减弱。这种衰减是由于雅可比项的重复乘法造成的，这些项由于激活函数导数（$f'$）和权重矩阵（$W_{hh}$）的影响，其大小通常小于1。

实际结果很严重：来自晚期时间步错误的梯度信号，当它到达对应早期时间步的网络层时，变得非常微小，或者说“消失”了。这些接近零的梯度意味着，负责处理序列早期时间步的权重（$W_{hh}$、$W_{xh}$、$b_h$）几乎没有收到基于长期结果的更新信号。

本质上，网络无法学习序列中相隔长时间的事件之间的关联。如果正确预测长段落末尾的情感严重依赖于开头附近的一个词，那么受梯度消失问题影响的简单RNN很可能无法捕捉这种依赖关系。随着梯度信号在反向传播过程中逐渐消失，初始词对最终预测误差的影响也随之丧失。这严重削弱了使用循环网络的主要目的：建模时间依赖性，包括长距离依赖。

理解梯度消失现象非常重要，因为它直接推动了更复杂的循环架构的发展，例如长短期记忆网络（LSTM）和门控循环单元（GRU），我们将在后续章节中学习它们。这些架构包含特定的机制，通常称为“门”，旨在调节信息流动并允许梯度在较长时间内更有效地传播，从而缓解梯度消失问题。

## 参考资料

- [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181) — Yoshua Bengio, Patrice Simard, Paolo Frasconi (1994)
  Journal: IEEE Transactions on Neural Networks; Publisher: IEEE; Volume: 5; Pages: 157-160; DOI: [10.1109/72.279181](https://doi.org/10.1109/72.279181)
  这篇基础性论文严谨地分析了循环神经网络中的梯度消失和梯度爆炸问题，为理解梯度下降法难以学习长期依赖关系提供了数学基础。
- [Long Short-Term Memory](https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇开创性论文介绍了长短期记忆 (LSTM) 网络，它通过引入门控机制，旨在解决循环神经网络中的梯度消失问题，从而使梯度在长序列中更稳定地传播。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  一本权威教科书，全面解释了循环神经网络，包括对梯度消失问题的详细阐述及其数学原理。
