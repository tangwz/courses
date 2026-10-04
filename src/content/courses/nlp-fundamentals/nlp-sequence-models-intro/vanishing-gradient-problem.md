---
course: "nlp-fundamentals"
chapter: "nlp-sequence-models-intro"
lesson: "vanishing-gradient-problem"
sourceId: 2505
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro/vanishing-gradient-problem"
title: "理解梯度消失问题"
description: "了解由于梯度消失而训练深度 RNN 所面临的挑战。"
order: 3
plots: []
sourceHash: "f85cc27fa31a75e3d1f31ea593088f7e4567ab2739bd65be1347e0e3335f03c9"
sourceCorrections: []
---

循环神经网络 (neural network)（RNN）被设计来处理序列，方法是维护一个隐藏状态，该状态将信息从一个时间步传递到下一个时间步。这种“记忆”使它们理论上能够连接序列中不同位置的信息。但它们如何学习 *哪些* 过去的信息是相关的呢？像其他神经网络一样，RNN 通过根据其输出计算出的误差来调整权重 (weight)进行学习，通常使用反向传播 (backpropagation)的一种变体。

### 随时间反向传播 (backpropagation)（BPTT）

训练 RNN 需要计算网络误差相对于其权重 (weight)的变化方式。因为给定时间步 $t$ 的输出依赖于所有先前时间步（$t-1, t-2, \ldots, 1$）的计算，我们需要将误差信号向后传播通过整个序列。这个过程被称为随时间反向传播（BPTT）。

BPTT 的工作方式是将 RNN 按时间步“展开”。设想为序列中的每个时间步创建一个单独的层，并且权重在所有这些“层”之间共享。然后，标准的反向传播算法应用于这个展开的网络。共享权重（例如循环权重矩阵 $W_{hh}$）的总梯度是在每个时间步计算出的梯度的总和。

### 长序列的挑战

尽管优雅，BPTT 在处理长序列时面临一个明显的实际问题。核心问题源于将梯度反向传播 (backpropagation)通过许多时间步时所需的链式法则的重复应用。

为了计算损失函数 (loss function)相对于过去较早时间步 $h_k$（例如 $k \ll t$）的梯度，我们需要将许多雅可比矩阵（偏导数矩阵）相乘：


$$
\frac{\partial L}{\partial h_k} = \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial h_{t-2}} \cdots \frac{\partial h_{k+1}}{\partial h_k}
$$


每个项 $\frac{\partial h_i}{\partial h_{i-1}}$ 都涉及循环权重 (weight)矩阵 $W_{hh}$ 以及 RNN 单元中使用的激活函数 (activation function)的导数。

现在，思考在这种重复乘法过程中会发生什么。如果这些雅可比矩阵中的相关值持续小于 1，它们的乘积将随着我们向时间回溯而指数级缩小。将 0.9 自身相乘 10 次大约得到 0.35，但相乘 50 次大约得到 0.005。梯度信号变得微乎其微。

这就是**梯度消失问题**：早期时间步的信息对梯度的贡献在后续时间步中几乎变为零。

> 随时间反向传播（BPTT）过程中梯度流的简化图示。误差信号（梯度，由虚线红色箭头表示）从损失附近开始，但随着它向后传播通过许多时间步，会明显减弱（由颜色变浅和线条变细表示），使得很难根据早期输入更新权重。

### 对学习的影响

梯度消失的实际影响很明显：RNN 难以学习**长距离依赖关系**。如果一个重要信息出现在序列的早期，但当它从稍后的输出反向传播 (backpropagation)回来时，与其相关的误差信号已经消失，网络就无法有效调整其权重 (weight)来捕捉这种依赖关系。

考虑对一份长篇产品评论进行情感分析：“我最初怀着很高的期望订购了这款产品，因为它有出色的描述和照片。产品很快送达，包装也很棒。然而，使用了一周后，我发现主要部件完全失灵了。非常令人失望。”

为了正确将整体情感分类为负面，模型需要将最后的“非常令人失望”的陈述（以及故障描述）与评论的开头联系起来，可能需要忽略最初的积极评价。如果评论末尾的梯度信号在到达处理评论开头的网络部分之前就消失了，模型可能会过分重视早期的积极信号，或者无法学习评论中部描述的部件故障的重要性。它本质上会形成一种短期记忆，无法关联长时间间隔的事件。

### 梯度爆炸：问题的另一面

尽管不那么常见但仍可能发生，相反的问题也可能出现：**梯度爆炸**。如果在 BPTT 期间重复相乘的雅可比矩阵中的值持续大于 1，梯度信号会指数级增长，变得非常巨大。这会导致训练不稳定，权重 (weight)更新巨大且不稳定，常常导致数值溢出（NaN 值）。

幸运的是，梯度爆炸通常比梯度消失更容易检测和处理。一种常见的技术，称为**梯度裁剪**，涉及为梯度的范数（大小）设置一个最大阈值。如果在反向传播 (backpropagation)期间梯度的范数超过此阈值，它会在应用权重更新之前按比例缩小，从而避免过大的步长。

### 后续发展

然而，梯度消失问题仍然是学习简单 RNN 中长距离依赖关系的一个更根本的障碍。它明显限制了它们在需要理解扩展序列上下文 (context)的任务上的有效性。这种困难是开发更复杂的循环架构的主要动力，特别是长短期记忆（LSTM）网络和门控循环单元（GRU），我们将在下一步进行研究。这些架构包含了明确设计用于调节信息流并缓解梯度消失问题的机制。

## 参考资料

- [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181/) — Yoshua Bengio, Patrice Simard, Paolo Frasconi (1994)
  Journal: IEEE Transactions on Neural Networks; Publisher: IEEE; Volume: 5; Pages: 157-160; DOI: [10.1109/72.279181](https://doi.org/10.1109/72.279181)
  这篇开创性论文正式识别并分析了循环神经网络中的梯度消失和梯度爆炸问题，为理解这些挑战提供了理论基础。
- [Long Short-Term Memory](https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇原始论文介绍了长短期记忆（LSTM）网络，该网络专门设计用于缓解循环神经网络中的梯度消失问题。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习概念的权威教科书，包括循环神经网络、时间反向传播以及梯度消失/爆炸问题。
