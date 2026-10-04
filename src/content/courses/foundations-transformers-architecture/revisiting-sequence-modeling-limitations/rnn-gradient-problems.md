---
course: "foundations-transformers-architecture"
chapter: "revisiting-sequence-modeling-limitations"
lesson: "rnn-gradient-problems"
sourceId: 1916
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations/rnn-gradient-problems"
title: "梯度消失与梯度爆炸问题"
description: "审视深度循环网络中梯度消失与梯度爆炸的数学基础。"
order: 2
plots: ["plots/1916-0.json"]
sourceHash: "65c7a90cfe247ab0243b920cd8bec76f2e58a30a4738de178ffb764034b25a72"
sourceCorrections: []
---

递归神经网络 (neural network) (RNN) 的序列特性使其能够逐步处理序列，但在训练时会造成明显的障碍，尤其是对于深度网络或长序列。训练 RNN 的主要机制是时间反向传播 (backpropagation) (BPTT)，它本质上将递归网络沿序列长度展开，并应用标准的反向传播算法。这个过程涉及到通过对每个时间步重复应用链式法则反向传播，来计算损失函数 (loss function)相对于网络参数 (parameter)的梯度。

考虑计算最终时间步 $T$ 处损失 $L$ 相对于某个较早时间步 $k$ 处隐藏状态 $h_k$ 的梯度。使用链式法则，这涉及将中间时间步的雅可比矩阵相乘：


$$
\frac{\partial L}{\partial h_k} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial h_{T-2}} \dots \frac{\partial h_{k+1}}{\partial h_k}
$$


这个公式表明了核心问题。项 $\frac{\partial h_t}{\partial h_{t-1}}$ 代表了时间 $t$ 的隐藏状态相对于时间 $t-1$ 的隐藏状态的变化方式。这个雅可比矩阵取决于循环权重 (weight)矩阵 $W_{hh}$ 和循环转换中使用的激活函数 (activation function)的导数。来自最终损失的梯度信号必须通过这些雅可比矩阵的乘积反向传播。

### 梯度消失

如果这些雅可比矩阵 $\frac{\partial h_t}{\partial h_{t-1}}$ 的范数（或更正式地说，奇异值）持续小于 1，它们的乘积在反向时间传播 ($T-k$ 步) 时会呈指数级缩小。


$$
\frac{\partial L}{\partial h_k} \approx \frac{\partial L}{\partial h_T} \prod_{t=k+1}^{T} J_t \quad \text{这里 } J_t = \frac{\partial h_t}{\partial h_{t-1}}
$$


如果 $||J_t|| < 1$ 平均而言，那么当 $T-k$ 增加时，$|| \prod_{t=k+1}^{T} J_t ||$ 会非常迅速地趋近于零。这意味着梯度信号在到达较早的时间步之前就有效地消失了。

**影响：**

- **难以学习长程依赖：** 网络难以学习序列中相距较远的输入和输出之间的关联。负责捕获这些长期效应（主要受来自早期时间步如 $k$ 的梯度影响）的权重 (weight)获得的更新微乎其微。
- **训练缓慢：** 与序列开头相关的参数 (parameter)学习速度远慢于接近末尾的参数。

双曲正切 (tanh) 或 Sigmoid 等激活函数 (activation function)的选择（这些在旧式 RNN 中常用）使这个问题更为严重，因为它们的导数严格小于 1（tanh 在单一点除外）。

### 梯度爆炸

相反，如果雅可比矩阵 $\frac{\partial h_t}{\partial h_{t-1}}$ 的范数持续大于 1，它们的乘积在反向传播 (backpropagation)时会呈指数级增大。

**影响：**

- **训练不稳定：** 梯度变得过大，导致权重 (weight)的更新量巨大。这可能导致优化过程发散，常在计算中导致数值溢出（NaN，即“非数字”值）。
- **破坏性更新：** 即使训练没有完全发散，大的梯度更新也可能抹去先前学到的知识。

虽然梯度爆炸通常更容易检测和缓解（例如，使用梯度裁剪，即当梯度超过某个阈值时将其缩小），但它们仍然对稳定训练构成明显挑战。



![梯度随时间传播示意图](plots/1916-0.json)



> 该图显示了当梯度通过时间步反向传播时，重复乘以略小于 1 的值（蓝线，范数 0.8）会导致指数衰减（消失），而重复乘以略大于 1 的值（红线，范数 1.2）会导致指数增长（爆炸）。Y 轴被截断以呈现这两种趋势。

这些梯度问题从根本上限制了简单 RNN 架构有效建模序列的能力，尤其是在存在长期模式时。这一限制是推动 LSTM 和 GRU 等更复杂循环单元发展的主要动力，我们将在接下来审视它们。

## 参考资料

- [Learning Long-Term Dependencies with Gradient Descent Is Difficult](https://ieeexplore.ieee.org/document/279181) — Yoshua Bengio, Patrice Simard, Paolo Frasconi (1994)
  Journal: IEEE Transactions on Neural Networks; Publisher: IEEE; Volume: 5; Pages: 157-160; DOI: [10.1109/72.279181](https://doi.org/10.1109/72.279181)
  本文是开创性工作，它从数学角度分析了循环神经网络中梯度消失和梯度爆炸问题，证明了它们在学习长期依赖方面的固有难度。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 370-381 (Chapter 10)
  本书第10章对循环神经网络进行了全面解释，包括对时间反向传播以及梯度消失和梯度爆炸问题的详细分析。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  本文介绍了长短期记忆（LSTM）网络，这是一种专门的循环神经网络架构，旨在缓解梯度消失问题并有效学习长期依赖。
