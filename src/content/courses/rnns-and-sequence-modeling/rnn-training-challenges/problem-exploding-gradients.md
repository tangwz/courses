---
course: "rnns-and-sequence-modeling"
chapter: "rnn-training-challenges"
lesson: "problem-exploding-gradients"
sourceId: 2555
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-4-rnn-training-challenges/problem-exploding-gradients"
title: "梯度爆炸问题"
description: "解释梯度为何会变得过大，从而导致训练不稳定。"
order: 2
plots: ["plots/2555-0.json"]
sourceHash: "072eed45baffcf6559aa806a8d160983b91787ea5bfb628d9db9c89fd38d8244"
sourceCorrections: []
---

虽然梯度消失问题导致学习信号在长序列中减弱，但**梯度爆炸问题**在循环神经网络 (neural network)（RNN）训练期间呈现出另一种数值不稳定现象。梯度非但没有趋近于零，反而可能变得异常巨大，有时会导致数值超出标准表示范围，产生 `NaN`（非数字）输出，从而使训练过程完全中断。

### 为什么会出现梯度爆炸？

梯度爆炸的机制与通过时间反向传播 (backpropagation)（BPTT）过程以及循环权重 (weight)矩阵的重复应用密切相关。回想一下，BPTT通过将误差信号逐步反向传播到展开的网络中来计算梯度。

考虑简单RNN中隐藏状态 $h_t$ 的更新规则：


$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$


当计算损失 $L$ 相对于时间上很早的隐藏状态 $h_k$ 的梯度时，BPTT会计算诸如 $\frac{\partial h_t}{\partial h_{t-1}}$、$\frac{\partial h_{t-1}}{\partial h_{t-2}}$ 等项，一直回溯到 $\frac{\partial h_{k+1}}{\partial h_k}$。每个偏导数都涉及与循环权重矩阵 $W_{hh}$（以及激活函数 (activation function)的导数）相乘。

从较晚时间步 $t$ 到较早时间步 $k$ ($k < t$) 的梯度贡献涉及一系列雅可比矩阵的乘积：


$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}
$$


每个项 $\frac{\partial h_i}{\partial h_{i-1}}$ 都近似地与 $W_{hh}$ 成比例。如果循环权重矩阵 $W_{hh}$ 的主导特征值（或更一般地说，奇异值）的幅值始终大于1，那么这种跨越许多时间步 ($t-k$) 的重复乘法会导致最终的梯度值呈指数级增长。

将其想象为复利：如果你反复将一个值乘以一个略大于1的因子，它会增长得非常快。在BPTT中，与 $W_{hh}$ 相关的梯度分量会重复相乘。如果这些分量倾向于放大信号（幅值 > 1），则整体梯度可能会爆炸。

### 对训练的影响

梯度爆炸以几种破坏性的方式表现出来：

1. **数值溢出：** 梯度值变得非常大，以至于超出了浮点数的表示范围，通常导致损失或权重 (weight)中出现 `NaN` 值。
2. **剧烈权重更新：** 即使没有发生数值溢出，巨大的梯度也会导致在梯度下降 (gradient descent)过程中对网络权重进行过大的更新。想象一下，试图通过迈出巨大步子下山；你很可能会严重越过最小值，最终离目标更远。这使得优化过程不稳定，权重可能会剧烈振荡或发散。
3. **训练失败：** 模型无法有效学习。成本函数可能会不规则地波动或无限增长，从而阻碍收敛到一个有用的解。



![训练过程中的梯度范数](plots/2555-0.json)



> 梯度范数在初始训练迭代中保持相对稳定，但随后突然急剧增加，表明发生了梯度爆炸事件。请注意，y轴上使用了对数刻度，以适应这种大幅度的尖峰。

与梯度消失（它悄然阻碍长距离依赖的学习）不同，梯度爆炸通常会导致更明显和灾难性的训练失败。幸运的是，它们显著的影响使其更容易被发现（例如，通过监测梯度范数或观察突发的 `NaN` 损失）。我们将在下一节讨论梯度裁剪等方法，它们能有效处理这种不稳定情况。

## 参考资料

- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf) — Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 28; Pages: 1310-1318; DOI: [10.55982/pascanu13.1310](https://doi.org/10.55982/pascanu13.1310)
  这篇基础性论文严谨地分析了RNN中的梯度消失和梯度爆炸问题，解释了其原因并提出了梯度裁剪作为解决方案。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  一本权威的教科书，提供了关于循环神经网络及其训练困难的全面理论背景，包括对梯度爆炸的详细讨论。
- [CS224N: Natural Language Processing with Deep Learning, Lecture Notes 06: Recurrent Neural Networks](http://web.stanford.edu/class/cs224n/2023/lectures/lecture6-rnn.pdf) — Tatsunori Hashimoto (slides mostly from Christopher Manning's 2023 version) (2023)
  Publisher: Stanford University
  来自顶尖大学课程的高质量讲义，清晰地教学式地解释了循环神经网络，包括对梯度爆炸问题及其实际影响的讨论。
