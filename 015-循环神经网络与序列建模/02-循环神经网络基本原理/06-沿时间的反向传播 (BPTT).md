# 沿时间的反向传播 (BPTT)

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-2-rnn-fundamentals/backpropagation-through-time-bptt)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network)（RNN）一步步处理序列，并使用其隐藏状态传递信息。那么网络是怎样*学习*的呢？就像前馈网络一样，RNN通过根据其产生的误差调整权重 (weight)来进行学习。对此的标准算法是反向传播 (backpropagation)。然而，RNN中的循环连接和共享权重需要进行修改：**沿时间的反向传播（BPTT）**。

设想您向RNN输入一个长度为$T$的序列。如前所述，网络在每一步执行相同的计算，使用相同的权重集合（$W_{hh}, W_{xh}, W_{hy}, b_h, b_y$）。为训练网络，我们首先需要计算一个损失，它衡量网络预测值（$y_1, y_2, ..., y_T$）与该序列真实目标值之间的距离。此损失通常在整个序列上计算，常为每个时间步损失的总和或平均值。

BPTT的主要思路是应用微积分的链式法则，与标准反向传播一样，但要跨*时间*操作序列进行。为形象地说明这一点，可以想象将RNN针对特定输入序列进行“展开”。这会形成一个大型的、类似前馈的网络，其中每个时间步对应一个层。重要的一点是，权重在这些“层”之间是*共享*的。

让我们考虑梯度计算。最终时间步$T$的误差直接取决于隐藏状态$h_T$以及权重$W_{hy}$和$b_y$。时间步$T-1$的误差取决于$h_{T-1}$，并可能通过循环连接（$h_T$取决于$h_{T-1}$）影响时间步$T$的误差。BPTT通过计算总损失相对于每个时间步的输出和隐藏状态的梯度来运行，从最后一步$T$开始，并向后移动到第一步$t=1$。

损失$L$相对于时间$t$隐藏状态的梯度，表示为$\frac{\partial L}{\partial h_t}$，依赖于两点：

1. $h_t$如何直接影响输出$y_t$（从而影响时间步$t$的损失）。
2. $h_t$如何影响*下一个*隐藏状态$h_{t+1}$（从而影响所有后续步骤$t+1, ..., T$的损失）。

第二点是“沿时间”的部分。梯度信号通过循环权重矩阵$W_{hh}$从$h_{t+1}$向后流向$h_t$。在数学上，这涉及到如下项：

$\frac{\partial h_{t+1}}{\partial h_t} = \frac{\partial f(W_{hh}h_{t} + W_{xh}x_{t+1} + b_h)}{\partial h_t}$

这种传播一步步向后进行。

> BPTT的反向传播过程。梯度（虚线红色）从每个时间步的损失流回，通过网络输出（$y_t$）和隐藏状态（$h_t$）。重要的是，梯度也通过循环连接（通过$W_{hh}$）向后流动，影响早期时间步的梯度计算。相对于共享权重（紫色线）的梯度在所有时间步累积。

BPTT的一个重要方面源于共享权重。由于在每个时间步都使用*相同*的权重矩阵（$W_{hh}, W_{xh}, W_{hy}$）和偏置 (bias)向量 (vector)（$b_h, b_y$），因此为特定权重计算的梯度需要考虑其在整个序列中的作用。因此，共享参数 (parameter)（例如$W_{hh}$）的最终梯度是相对于其在每个时间步$t=1, ..., T$的使用所计算的梯度之和。

$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}} \quad \text{(在反向传播过程中计算)}$

$W_{xh}$、$W_{hy}$、$b_h$和$b_y$的计算方式类似。一旦这些总梯度计算完毕，就会执行标准的梯度下降 (gradient descent)更新（或其变体之一，如Adam或RMSprop）来调整网络参数。

$W \leftarrow W - \eta \frac{\partial L}{\partial W}$

这里$W$代表任何共享参数，而$\eta$是学习率。

尽管BPTT允许我们训练RNN，但这种在可能很长的序列上传播梯度的过程并非没有困难。随着梯度信号在许多时间步中向后传播，它可能呈指数级缩小至零（梯度消失）或呈指数级增大（梯度爆炸）。我们将在第4章检验这些训练上的难题及其影响。目前，主要观点是BPTT扩展了反向传播，以处理循环架构中固有的时间依赖性和共享参数。下一节将更仔细地讨论展开网络以促进此过程的实际方法。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  权威教科书，其中有一章解释了循环神经网络和时间反向传播算法。
- [Recurrent Neural Networks (RNNs)](http://web.stanford.edu/class/cs224n-2023/slides/cs224n_2023_lecture06.pdf) — Christopher Manning (2023)
  斯坦福大学课程讲义，全面解释了循环神经网络和 BPTT。

---

[上一节](05-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E4%BF%A1%E6%81%AF%E6%B5%81%E5%8A%A8.md) · [下一节](07-%E5%B1%95%E5%BC%80%E7%BD%91%E7%BB%9C%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83.md)
