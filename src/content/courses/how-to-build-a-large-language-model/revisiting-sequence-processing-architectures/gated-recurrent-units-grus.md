---
course: "how-to-build-a-large-language-model"
chapter: "revisiting-sequence-processing-architectures"
lesson: "gated-recurrent-units-grus"
sourceId: 5960
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures/gated-recurrent-units-grus"
title: "门控循环单元 (GRU)"
description: "描述GRU的结构和功能，作为LSTM的一种替代方案。"
order: 4
plots: []
sourceHash: "c3cc0ddfe7bedb7f0a0ec1f110559af9a230fc6310e2465b61d7e7336ca2abf9"
sourceCorrections: []
---

长短期记忆（LSTM）网络以有效解决梯度消失问题并捕获长距离依赖关系而闻名。然而，这些网络通常由于其架构（包含三个独立的门：输入、遗忘、输出以及一个独立的细胞状态）而引入了相当大的计算复杂性。2014年，Cho等人提出了一种名为门控循环单元（GRU）的变体，它在许多任务上实现了相似的表现，但结构更简单。GRU将遗忘门和输入门合并成一个“更新门”，并合并了细胞状态和隐藏状态。

让我们看看GRU单元的结构。与其他RNN一样，它接收当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 来生成下一个隐藏状态 $h_t$。其作用通过两个门来实现：重置门 ($r_t$) 和更新门 ($z_t$)。

### 重置门

重置门决定了如何将新输入与前一个隐藏状态结合。具体来说，它控制在计算候选隐藏状态时，前一个隐藏状态 ($h_{t-1}$) 应该“遗忘”多少。其计算方式如下：


$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$


这里，$W_r$、$U_r$ 和 $b_r$ 是重置门的可学习权重 (weight)矩阵和偏置 (bias)向量 (vector)。Sigmoid函数 $\sigma$ 将输出压缩在0到1之间。值接近0表示前一个隐藏状态大部分被忽略，而值接近1表示它大部分被保留。

### 更新门

更新门决定了前一个隐藏状态 ($h_{t-1}$) 有多少应该传递到新的隐藏状态 ($h_t$)，以及有多少*新的*候选隐藏状态应该被使用。这个门基本结合了LSTM的遗忘门和输入门的作用。它的计算方式与重置门类似：


$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$


同样，$W_z$、$U_z$ 和 $b_z$ 是可学习参数 (parameter)，$\sigma$ 是Sigmoid函数。$z_t$ 的值接近1表示前一个状态 $h_{t-1}$ 大部分被保留，而值接近0表示新的候选状态被主要使用。

### 候选隐藏状态

在计算最终隐藏状态之前，GRU会计算一个候选隐藏状态 ($\tilde{h}_t$)。这个计算受到重置门的影响，重置门决定了前一个隐藏状态 $h_{t-1}$ 的贡献程度：


$$
\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)
$$


这里，$\odot$ 表示按元素相乘（哈达玛积）。如果重置门 $r_t$ 的值接近0，那么 $h_{t-1}$ 的贡献将被有效清除，使候选状态主要基于当前输入 $x_t$。$W_h$、$U_h$ 和 $b_h$ 是另一组可学习的权重 (weight)和偏置 (bias)。$\tanh$ 函数有助于调节网络中的值，通常将其压缩在-1到1之间。

### 最终隐藏状态

最后，更新门 $z_t$ 在前一个隐藏状态 $h_{t-1}$ 和候选隐藏状态 $\tilde{h}_t$ 之间进行调节，以生成当前时间步的最终隐藏状态 $h_t$：


$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$


这个方程的作用类似于加权平均。如果 $z_t$ 接近1，候选状态 $\tilde{h}_t$ 贡献更多，从而有效地用新信息更新隐藏状态。如果 $z_t$ 接近0，前一个隐藏状态 $h_{t-1}$ 被保留更多，允许信息在多个时间步中不变地传递。这个机制是GRU如何维护长距离依赖关系的方式。

> GRU单元内信息流的简化视图。$x_t$ 是输入，$h_{t-1}$ 是前一个隐藏状态。重置门 ($r_t$) 影响候选状态 ($\tilde{h}_t$)，更新门 ($z_t$) 将候选状态与前一个状态结合，生成最终隐藏状态 $h_t$。

### GRU与LSTM的比较

GRU通常被视为LSTM的更精简替代方案。

- **更少的门：** GRU有两个门（重置、更新），而LSTM有三个（输入、遗忘、输出）。
- **无独立细胞状态：** GRU不像LSTM那样维护独立的细胞状态 ($c_t$)；它们直接修改隐藏状态 ($h_t$)。
- **效率：** 参数 (parameter)和操作的减少通常意味着GRU比同等大小的LSTM训练速度稍快，并且内存需求更少。

在实践中，LSTM和GRU之间的选择通常取决于具体的数据集和任务。两者在所有场景下都没有哪个能持续胜过另一个，尽管GRU因其相对简单和相近的表现而获得欢迎。

以下是一个PyTorch代码片段，呈现了单个GRU步骤的核心计算（假设输入为`x_t`、`h_tm1`以及预定义的权重 (weight)/偏置 (bias)张量）：

```python
import torch
import torch.nn.functional as F

# 示例张量 (batch_size, input_size/hidden_size)
# 替换为实际维度和已初始化的权重/偏置
batch_size = 1
input_size = 10
hidden_size = 20

x_t = torch.randn(batch_size, input_size)
h_tm1 = torch.randn(batch_size, hidden_size) # h_{t-1}

# --- 假设权重矩阵 (W_*, U_*) 和偏置 (b_*) 已定义 ---
# 示例初始化（替换为实际学习到的参数）
W_r = torch.randn(input_size, hidden_size)
U_r = torch.randn(hidden_size, hidden_size)
b_r = torch.randn(hidden_size)

W_z = torch.randn(input_size, hidden_size)
U_z = torch.randn(hidden_size, hidden_size)
b_z = torch.randn(hidden_size)

W_h = torch.randn(input_size, hidden_size)
U_h = torch.randn(hidden_size, hidden_size)
b_h = torch.randn(hidden_size)
# ---------------------------------------------------------------------

# 重置门计算
r_t = torch.sigmoid(x_t @ W_r + h_tm1 @ U_r + b_r)

# 更新门计算
z_t = torch.sigmoid(x_t @ W_z + h_tm1 @ U_z + b_z)

# 候选隐藏状态计算
h_tilde_t = torch.tanh(x_t @ W_h + (r_t * h_tm1) @ U_h + b_h)

# 最终隐藏状态计算
h_t = (1 - z_t) * h_tm1 + z_t * h_tilde_t

print("前一个隐藏状态的形状:", h_tm1.shape)
print("当前隐藏状态的形状:", h_t.shape)
```

这种简化结构虽然有效，但仍然依赖于顺序处理。时间步 $t$ 的计算依赖于时间步 $t-1$ 的结果。这种固有的顺序依赖性限制了训练期间的并行化，并且在处理非常长的序列时仍然是一个瓶颈，这为Transformer中使用的非循环注意力机制 (attention mechanism)奠定了基础。

## 参考资料

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  介绍了门控循环单元（GRU）架构及其在序列到序列模型中的应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10: Sequence Modeling: Recurrent and Recursive Networks
  提供了循环神经网络的理论说明，包括GRU，并包含数学和概念基础。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Stanford University (2023)
  包含课程材料，包括讲座视频和笔记，解释了循环神经网络和GRU。
