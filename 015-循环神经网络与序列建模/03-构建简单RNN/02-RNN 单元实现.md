# RNN 单元实现

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns/conceptual-rnn-cell-implementation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在查看深度学习 (deep learning)框架（如 TensorFlow 或 PyTorch）提供的便捷 API 之前，了解一个简单 RNN 单元的主要逻辑如何使用基础操作实现是很有益的。这有助于巩固 RNN 数学公式的理解，并填补理论与高层库使用之间的空白。

回顾一个简单 RNN 单元在单个时间步 $t$ 的基本方程式：

1. **隐藏状态计算：** 新的隐藏状态 $h_t$ 是根据当前输入 $x_t$ 和前一个隐藏状态 $h_{t-1}$ 计算的。
   
   $$
   h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
   $$
   
2. **输出计算：** 输出 $y_t$（或后续层的预激活值）通常是从当前隐藏状态 $h_t$ 计算的。
   
   $$
   y_t = W_{hy} h_t + b_y
   $$
   

这里，$W_{xh}$、$W_{hh}$ 和 $W_{hy}$ 表示权重 (weight)矩阵，而 $b_h$ 和 $b_y$ 是偏置 (bias)向量 (vector)。$\tanh$ 函数是隐藏状态常用的激活函数 (activation function)。

让我们将其转化为一个简化的 Python 实现，使用 NumPy，NumPy 在机器学习 (machine learning)中常用于数值运算。

### NumPy 中的简单 RNN 单元

设想我们想在一个函数中表示这个计算。这个函数会把当前输入和前一个隐藏状态作为参数 (parameter)，并返回新的隐藏状态和该时间步的输出。它还需要访问网络的参数（权重 (weight)和偏置 (bias)）。

```python
import numpy as np

def simple_rnn_cell_forward(xt, h_prev, parameters):
    """
    为一个简单 RNN 单元执行一个前向步骤。

    参数:
    xt -- 当前时间步的输入数据，形状为 (n_features,)
    h_prev -- 前一时间步的隐藏状态，形状为 (n_hidden,)
    parameters -- 包含以下内容的 Python 字典：
                    Wxh -- 乘以输入的权重矩阵，形状为 (n_hidden, n_features)
                    Whh -- 乘以隐藏状态的权重矩阵，形状为 (n_hidden, n_hidden)
                    Why -- 连接隐藏状态到输出的权重矩阵，形状为 (n_output, n_hidden)
                    bh  -- 隐藏状态的偏置，形状为 (n_hidden,)
                    by  -- 输出的偏置，形状为 (n_output,)

    返回值:
    h_next -- 下一个隐藏状态，形状为 (n_hidden,)
    yt_pred -- 此时间步的预测，形状为 (n_output,)
    """

    # 获取参数
    Wxh = parameters['Wxh']
    Whh = parameters['Whh']
    Why = parameters['Why']
    bh  = parameters['bh']
    by  = parameters['by']

    # 如果输入是一维数组，确保它们是列向量
    xt = xt.reshape(-1, 1) # Shape (n_features, 1)
    h_prev = h_prev.reshape(-1, 1) # Shape (n_hidden, 1)

    # 计算新的隐藏状态
    # 注意: np.dot 执行矩阵乘法
    h_next = np.tanh(np.dot(Wxh, xt) + np.dot(Whh, h_prev) + bh.reshape(-1, 1))

    # 计算输出
    yt_pred = np.dot(Why, h_next) + by.reshape(-1, 1)

    # 返回展平的数组，以在其他地方需要时保持一致性
    return h_next.flatten(), yt_pred.flatten()

# 示例用法（说明性 - 需要定义参数和数据）
# n_features = 10 # 输入特征的数量
# n_hidden = 20   # 隐藏单元的数量
# n_output = 5    # 输出单元的数量

# # 初始化参数（随机用于演示）
# parameters = {
#     'Wxh': np.random.randn(n_hidden, n_features) * 0.01,
#     'Whh': np.random.randn(n_hidden, n_hidden) * 0.01,
#     'Why': np.random.randn(n_output, n_hidden) * 0.01,
#     'bh': np.zeros((n_hidden, 1)),
#     'by': np.zeros((n_output, 1))
# }

# # 示例输入和前一个隐藏状态
# xt_sample = np.random.randn(n_features)
# h_prev_sample = np.zeros((n_hidden,)) # 初始隐藏状态通常从零开始

# # 执行一个步骤
# h_next_sample, yt_pred_sample = simple_rnn_cell_forward(xt_sample, h_prev_sample, parameters)

# print("下一个隐藏状态的形状:", h_next_sample.shape)
# print("输出预测的形状:", yt_pred_sample.shape)
```

### 处理序列

为了处理整个序列，你会在每个时间步上迭代调用 `simple_rnn_cell_forward` 函数。在时间步 $t$ 计算出的隐藏状态 `h_next` 会成为时间步 $t+1$ 的 `h_prev`。

```python
def simple_rnn_forward(x_sequence, h0, parameters):
    """
    使用简单 RNN 执行序列的前向传播。

    参数:
    x_sequence -- 输入序列，形状为 (n_features, sequence_length)
    h0 -- 初始隐藏状态，形状为 (n_hidden,)
    parameters -- 参数字典 (Wxh, Whh, Why, bh, by)

    返回值:
    h -- 所有时间步的隐藏状态，形状为 (n_hidden, sequence_length)
    y_pred -- 所有时间步的预测，形状为 (n_output, sequence_length)
    """
    # 获取维度
    n_features, sequence_length = x_sequence.shape
    n_hidden = parameters['Whh'].shape[0]
    n_output = parameters['Why'].shape[0]

    # 初始化隐藏状态和预测数组
    h = np.zeros((n_hidden, sequence_length))
    y_pred = np.zeros((n_output, sequence_length))

    # 初始化第一个隐藏状态
    h_next = h0.copy() # 从初始隐藏状态开始

    # 遍历时间步
    for t in range(sequence_length):
        # 获取当前时间步的输入
        xt = x_sequence[:, t]

        # 更新隐藏状态并获取预测
        h_next, yt = simple_rnn_cell_forward(xt, h_next, parameters)

        # 存储结果
        h[:, t] = h_next
        y_pred[:, t] = yt

    return h, y_pred

# 示例用法（说明性）
# sequence_length = 15
# x_seq_sample = np.random.randn(n_features, sequence_length)
# h0_sample = np.zeros((n_hidden,))

# h_all, y_pred_all = simple_rnn_forward(x_seq_sample, h0_sample, parameters)

# print("所有隐藏状态的形状:", h_all.shape)
# print("所有预测的形状:", y_pred_all.shape)
```

这个实现突出了几个方面：

1. **顺序处理：** 主要计算在每个时间步上迭代应用。
2. **状态传播：** 隐藏状态 $h_t$ 作为一个记忆，将信息从前一个步骤带到下一个步骤。
3. **参数 (parameter)共享：** 相同的权重 (weight)矩阵（$W_{xh}$、$W_{hh}$、$W_{hy}$）和偏置 (bias)（$b_h$、$b_y$）在所有时间步中都被使用。这种参数共享是 RNN 的一个基本特点，并使它们适用于可变长度的序列。

"请记住，这是一个简化的视角。框架的实现能同时处理批量的序列，包含性能优化，管理参数初始化，并提供反向传播 (backpropagation)（BPTT）的机制。然而，了解这个基本的前向传播为使用这些更抽象的工具打下了良好的根基，我们将在接下来介绍它们。"

## 参考资料

- [Finding Structure in Time](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: Wiley; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  描述了简单循环网络（SRN），即本节中实现的简单RNN单元的基础架构。这篇论文引入了上下文层来维持随时间变化的状态。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  为循环神经网络提供了严谨的数学和概念基础，包括基本的RNN单元、其方程以及梯度消失/爆炸等挑战，详见第10章。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  第4版。提供了循环神经网络的实际实现和解释，包括简单RNN，通常从基本概念到特定框架代码逐步构建。它为读者从NumPy实现过渡到高级API提供了一个很好的桥梁。
- [Lecture Notes on Recurrent Neural Networks (Winter 2023, Lecture 4)](https://web.stanford.edu/class/cs224n/lectures/cs224n-2023-lecture04-rnn-lm.pdf) — Chris Chute, Jenny Hong, Kevin Du (2023)
  Publisher: Stanford University
  提供了RNN架构的详细讲义和解释，包括基本的RNN单元、其前向传播以及其背后的动机，通常附有说明性图表和理论见解。来自斯坦福大学CS224N 2023年冬季课程。

---

[上一节](01-%E6%90%AD%E5%BB%BA%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83.md) · [下一节](03-%E4%BD%BF%E7%94%A8%E6%A1%86%E6%9E%B6API%E6%9E%84%E5%BB%BA%E7%AE%80%E5%8D%95RNN%E5%B1%82.md)
