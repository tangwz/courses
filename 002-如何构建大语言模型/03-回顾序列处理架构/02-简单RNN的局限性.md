# 简单RNN的局限性

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures/limitations-simple-rnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管简单的循环神经网络 (neural network)（RNN）通过维护一个隐藏状态来处理序列，但在处理更长序列时会遇到显著困难。这些局限性是开发LSTM、GRU等更复杂架构，并最终催生Transformer的主要推动力。其主要问题源于训练时梯度在网络中经过多个时间步的传播方式。

### 梯度消失问题

训练RNN通常涉及时间反向传播 (backpropagation)（BPTT）。该算法将RNN沿序列长度展开并应用标准反向传播。为了计算损失函数 (loss function)相对于序列早期参数 (parameter)（例如，影响隐藏状态 $h_k$ 的权重 (weight)）的梯度，链式法则要求将对应循环状态转换的多个雅可比矩阵相乘。

设 $L$ 为序列上的总损失。损失相对于早期隐藏状态 $h_k$ 的梯度取决于所有后续状态 $h_t$ （$t > k$）的梯度：


$$
\frac{\partial L}{\partial h_k} = \sum_{t=k+1}^{T} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k}
$$


其中 $T$ 是序列长度，$L_t$ 是时间步 $t$ 的损失。项 $\frac{\partial h_t}{\partial h_k}$ 表示 $h_k$ 对 $h_t$ 的影响。这种影响通过链式计算状态转换的雅可比矩阵得到：


$$
\frac{\partial h_t}{\partial h_k} = \frac{\partial h_t}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial h_{t-2}} \dots \frac{\partial h_{k+1}}{\partial h_k}
$$


每个雅可比矩阵 $\frac{\partial h_i}{\partial h_{i-1}}$ 都依赖于循环权重矩阵 $W_{hh}$ 和激活函数 (activation function)的导数（例如，$\tanh'$）。如果这些雅可比矩阵的幅值（特别是奇异值）持续小于1，它们的乘积会随着距离 $t-k$ 的增加呈指数级减小。

> 时间反向传播（BPTT）示意图。后续时间步（如 $T$）的梯度必须通过循环连接（受 $W_{hh}$ 影响）反向流动，以更新影响早期状态（如 $h_k$）的参数。在此反向传播过程中，雅可比矩阵的重复相乘是梯度消失/爆炸问题的根源。

这种现象被称为**梯度消失问题**。结果是，来自后续时间步的误差信号变得过于微弱，无法有效更新负责获取早期信息的权重。实际上，RNN难以学习数据中的长距离依赖关系。模型很难连接序列中相隔多个时间步的事件或信息。

### 梯度爆炸问题

反之，如果雅可比矩阵 $\frac{\partial h_i}{\partial h_{i-1}}$ 的幅值持续大于1，它们的乘积会呈指数级增大。这会导致**梯度爆炸问题**。

当梯度爆炸时，权重 (weight)更新会变得过大，可能导致模型的参数 (parameter)发散至无穷大或NaN（非数值）。这会使训练过程不稳定，常导致损失函数 (loss function)突然急剧增加或训练完全失败。

梯度爆炸问题通常更容易发现，有时可以通过**梯度裁剪**（当梯度范数超过一定阈值时将其按比例缩小）等方法来减轻，但裁剪更多是一种权宜之计，而非根本解决方法。它能阻止灾难性发散，但并未从根本上解决在长时序上传播有效梯度信号的问题，这正是梯度消失问题所针对的主要局限性。

```python
import torch
import torch.nn as nn

# 假设模型参数为'params'，已计算出'loss'
# PyTorch中的梯度裁剪示例

# 1. 计算梯度
# loss.backward()

# 2. 定义最大梯度范数阈值
max_grad_norm = 1.0

# 3. 就地裁剪梯度
nn.utils.clip_grad_norm_(params, max_grad_norm)

# 4. 应用优化器步骤
# optimizer.step()
```

> PyTorch中演示梯度裁剪的简单示例。计算梯度后，如果总范数超过 `max_grad_norm`，`clip_grad_norm_` 会按比例缩小梯度。

### 对性能的影响

这些梯度问题严重限制了简单RNN在需要对长序列建模或获取跨越多个时间步的依赖关系的任务上的实际效果。例如：

- **机器翻译：** 翻译长句时，其含义可能依赖于相隔较远的词语。
- **文档摘要：** 理解长篇文章的主旨需要整合跨段落的信息。
- **语言建模：** 预测下一个词通常依赖于文本中很早之前建立的上下文 (context)。

无法可靠学习这些长距离依赖关系意味着，与旨在解决这些梯度传播问题的模型相比，简单RNN通常表现不佳。这促成了LSTM和GRU的出现，它们包含了专门设计用于更好控制信息和梯度随时间流动的门控机制，我们将在后续章节中看到这一点。

## 参考资料

- [Long Short-Term Memory](https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）架构，旨在解决RNN中的梯度消失问题。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教科书，详细解释了循环神经网络、时间反向传播和梯度问题。请参阅第10章了解序列建模。

---

[上一节](01-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28RNN%29%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%86%85%E5%AE%B9.md) · [下一节](03-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E7%BD%91%E7%BB%9C.md)
