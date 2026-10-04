---
course: "how-to-build-a-large-language-model"
chapter: "revisiting-sequence-processing-architectures"
lesson: "fundamentals-rnns"
sourceId: 5957
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures/fundamentals-rnns"
title: "循环神经网络 (RNN) 的基本内容"
description: "阐述基本 RNN 中隐状态和序列处理的原理。"
order: 1
plots: []
sourceHash: "f4636cd46472a88e86dc719619942e83fd96b52764306a3bd344ab9dbe244f67"
sourceCorrections: []
---

在 Transformer 出现之前，循环神经网络 (neural network) (RNN) 是处理序列数据（例如文本或时间序列）的标准架构。与独立处理输入的普通前馈网络不同，RNN 具有一种记忆形式，使得序列中先前步骤的信息能够影响当前步骤的处理。这使得它们天生适合处理那些注重上下文 (context)和顺序的任务。

### 核心思想：隐状态和循环

RNN 的核心思想是**隐状态**，在时间步 $t$ 通常表示为 $h_t$。这个隐状态充当了截至该时间点序列中已见到信息的压缩摘要。在每个时间步 $t$，RNN 接收两个输入：序列中的当前输入元素 $x_t$ 和前一时间步的隐状态 $h_{t-1}$。然后，它计算一个新的隐状态 $h_t$，并可选地生成一个输出 $y_t$。

想象阅读一个句子：“The cat sat on the \_\_\_”。要预测下一个词，你需要记住“The cat sat on the”。RNN 通过在处理每个词时更新其隐状态来模仿这一点，从而传递相关上下文 (context)。

这个过程包含一个循环：在每个时间步都应用相同的操作和权重 (weight)集，并使用前一个隐状态作为输入。这种共享权重结构使得 RNN 在参数 (parameter)上很高效，因为它们不需要为序列中的每个位置设置单独的参数。

### 数学表达式

我们来看一下在时间步 $t$ 简单 RNN 单元内部的计算：

1. **计算新的隐状态 $h_t$**：这通常通过使用权重 (weight)矩阵和激活函数 (activation function)（通常是双曲正切函数 $\tanh$）将当前输入 $x_t$ 和前一个隐状态 $h_{t-1}$ 组合起来完成。

   
   $$
   h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
   $$
   

   此处：

   - $x_t$ 是时间步 $t$ 的输入向量 (vector)。
   - $h_{t-1}$ 是前一时间步的隐状态向量（$h_0$ 通常初始化为零）。
   - $W_{xh}$ 是连接输入 $x_t$ 到隐状态的权重矩阵。
   - $W_{hh}$ 是连接前一个隐状态 $h_{t-1}$ 到当前隐状态的权重矩阵。
   - $b_h$ 是隐状态计算的偏置 (bias)向量。
   - $\tanh$ 是双曲正切激活函数，它将值压缩到 -1 和 1 之间。
2. **计算输出 $y_t$（可选）**：根据任务不同，可能会在每个时间步基于当前隐状态生成一个输出。

   
   $$
   y_t = W_{hy} h_t + b_y
   $$
   

   此处：

   - $h_t$ 是当前隐状态向量。
   - $W_{hy}$ 是连接隐状态到输出的权重矩阵。
   - $b_y$ 是输出计算的偏置向量。
   - 根据具体应用，可能会对 $y_t$ 应用一个激活函数（如用于分类的 softmax 或用于回归的线性函数）。

重要的是，权重矩阵 ($W_{xh}, W_{hh}, W_{hy}$) 和偏置 ($b_h, b_y$) 在所有时间步中都是*相同*的。网络学习一个单一的转换函数，并重复应用它。

### 网络在时间上的展开

虽然我们经常用循环来绘制 RNN 单元，但将其在序列长度上“展开”来直观理解很有用。这展示了计算如何从一个时间步流向下一个时间步。

> 一个在三个时间步上展开的 RNN。相同的 RNN 单元（表示共享权重 (weight) $W_{xh}, W_{hh}, W_{hy}$）处理输入 $x_t$ 和前一个隐状态 $h_{t-1}$，以生成当前隐状态 $h_t$ 和输出 $y_t$。

### 简单的 PyTorch 实现

PyTorch 为 RNN 提供了方便的模块。以下是定义和使用单层 RNN 的一个基本示例：

```python
import torch
import torch.nn as nn

# 定义参数
input_size = 10  # 输入向量 x_t 的维度
hidden_size = 20 # 隐状态 h_t 的维度
sequence_length = 5
batch_size = 3

# 创建一个 RNN 层
# batch_first=True 表示输入/输出张量的批次维度在前
# (批次, 序列, 特征)
rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True)

# 创建一些虚拟输入数据
# 形状：(批次大小, 序列长度, 输入大小)
input_sequence = torch.randn(batch_size, sequence_length, input_size)

# 初始化隐状态（可选，默认为零）
# 形状：(层数 * 方向数, 批次大小, 隐状态大小)
# -> 在本例中为 (1, 3, 20)
initial_hidden_state = torch.zeros(1, batch_size, hidden_size)

# 将输入序列和初始隐状态通过 RNN
# output 包含*每个*时间步的隐状态
# final_hidden_state 只包含*最后*的隐状态
output, final_hidden_state = rnn_layer(input_sequence, initial_hidden_state)

print("Input shape:", input_sequence.shape)
# 输出形状：(批次大小, 序列长度, 隐状态大小)
print("Output shape:", output.shape)
# 最终隐状态形状：(层数 * 方向数, 批次大小,
#                            隐状态大小)
print("Final hidden state shape:", final_hidden_state.shape)

# 示例：从输出中获取最后一个时间步的隐状态
last_time_step_output = output[:, -1, :]
print("Last time step hidden state from output shape:",
      last_time_step_output.shape)

# 验证其与最终隐状态是否匹配（挤压掉第一个维度）
print(
    "最终隐状态和最后一个输出步是否相等？",
    torch.allclose(
        final_hidden_state.squeeze(0),
        last_time_step_output
    )
)
```

这种简单的结构使得 RNN 能够对序列依赖进行建模。然而，正如我们将在下一节看到的，基本的 RNN 在学习序列中相距较远元素之间的关系时存在困难。这个局限性促成了 LSTM 和 GRU 等更复杂的架构的出现。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  一本基础教科书，全面讨论了循环神经网络，涵盖其架构、训练方法和各种高级形式。
- [Finding structure in time](https://doi.org/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  这篇开创性论文介绍了最早且最具影响力的简单循环网络架构之一，常被称为Elman网络，展示了其学习序列依赖性的能力。
- [RNN - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch Development Team (2024)
  Publisher: PyTorch Foundation
  PyTorch `torch.nn.RNN` 模块的官方文档，提供了其构造函数、参数、输入/输出形状和实用示例的详细信息。
- [CS224N: Natural Language Processing with Deep Learning - Lecture Notes](https://web.stanford.edu/class/cs224n/) — Christopher Manning and Stanford CS224N Staff (2023)
  Publisher: Stanford University; Pages: Lecture 6
  斯坦福大学这门备受推崇的课程的讲义，清晰而透彻地解释了RNN的基础知识、其展开过程以及在自然语言处理中的初步应用。
