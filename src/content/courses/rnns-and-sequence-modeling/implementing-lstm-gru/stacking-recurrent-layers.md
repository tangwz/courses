---
course: "rnns-and-sequence-modeling"
chapter: "implementing-lstm-gru"
lesson: "stacking-recurrent-layers"
sourceId: 2604
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/stacking-recurrent-layers"
title: "堆叠循环层"
description: "通过堆叠多个LSTM或GRU层来实现深度RNN。"
order: 4
plots: []
sourceHash: "a8b22c8e3e8243ef9f5105def3306d9005cbbb47eb8faf5bdb2e9878e120a691"
sourceCorrections: []
---

堆叠循环层，例如长短期记忆（LSTM）或门控循环单元（GRU）层，可以形成多层网络。这种技术使网络能够学习更复杂的表示和分层的时间特征。第一个循环层可能处理原始输入序列并捕获较低级别的模式，而后续层则在下方层的隐藏状态序列上运行，有可能学习较高级别或更长范围的时间抽象。

### 为何堆叠循环层？

堆叠循环层能增加模型的表示能力。每增加一层都会增加参数 (parameter)和计算步骤，使得网络能够对序列数据中更复杂的关联进行建模。

1. **分层特征学习：** 较低层可以学习基本的序列模式（例如文本中的局部依赖或时间序列中的短期趋势），而较高层可以使用较低层的输出，在更长的时间范围内整合信息。
2. **增加模型深度：** 与深度前馈网络类似，RNN的深度有时可以在复杂任务上带来更好的表现，前提是数据充足且训练方法得当。

### `return_sequences` 参数 (parameter)

堆叠循环层时，最重要的考虑是控制每一层的输出。在TensorFlow/Keras和PyTorch等框架中，循环层通常有一个选项，在Keras中常命名为`return_sequences`，在PyTorch中则由你如何使用输出来隐式控制，它决定了该层是输出：

1. **仅最终隐藏状态：** 当`return_sequences=False`时（Keras默认设置）发生这种情况。输出形状通常是`(batch_size, units)`，代表处理完最后一个时间步后的隐藏状态。如果任务涉及对整个序列进行概括，例如在序列分类中，这适用于*最终*的循环层。
2. **完整隐藏状态序列：** 当`return_sequences=True`时发生这种情况。输出形状是`(batch_size, time_steps, units)`，为输入序列中的*每个*时间步提供隐藏状态。对于其输出需要输入到*另一个*循环层的所有循环层来说，这是必需的，因为下一层期望以序列作为输入。当需要每个时间步的输出时，例如在序列到序列任务中或稍后应用注意力机制 (attention mechanism)时，也会使用它。

### 框架中的实现

我们来看看堆叠在常用框架中是如何运作的。

#### TensorFlow (Keras API)

在Keras中，使用`Sequential` API或Functional API进行堆叠非常直接。你只需依次添加循环层，并确保除了可能最后一层之外的所有层都设置`return_sequences=True`。

```python
import tensorflow as tf

# 假设 input_shape = (时间步, 特征)
# 对于变长序列，时间步可以为 None: (None, features)
num_features = 10
num_units_l1 = 64
num_units_l2 = 32
output_dim = 5 # 示例用于分类

model = tf.keras.Sequential([
    # 输入形状仅在第一层需要
    tf.keras.layers.LSTM(num_units_l1, return_sequences=True, input_shape=(None, num_features)),
    # 第1层输出形状: (批大小, 时间步, 第一层单元数)

    # 这一层接收来自前一层的完整序列
    tf.keras.layers.GRU(num_units_l2, return_sequences=False), # 或者如果后续需要，设置为 True
    # 第2层 (return_sequences=False) 输出形状: (批大小, 第二层单元数)

    # 添加一个全连接层用于分类/回归
    tf.keras.layers.Dense(output_dim, activation='softmax') # 示例激活函数
])

model.summary()
```

> 一个简单的两层堆叠循环网络。第一个LSTM层必须返回序列以供第二个GRU层使用。最终的GRU层只返回最后一个隐藏状态，适合后续的Dense层进行分类。

#### PyTorch

在PyTorch中，你在`__init__`方法中定义层，并在`forward`方法中指定连接。你需要手动将一层的输出序列作为下一层的输入传递。PyTorch的`nn.LSTM`和`nn.GRU`模块会返回完整的输出序列以及最终的隐藏/细胞状态。你通常会使用输出序列张量进行堆叠。

值得注意的是，PyTorch的`nn.LSTM`和`nn.GRU`也有一个`num_layers`参数 (parameter)，允许你在单个模块实例内部创建堆叠循环层。这通常比手动堆叠独立的层实例在计算上更高效，尤其是在GPU上，因为有优化的内核（如cuDNN）。然而，如果你想要不同类型的层（例如LSTM后面跟GRU）或每层有不同的配置，手动堆叠会提供更大的灵活性。

以下是手动堆叠的一个示例：

```python
import torch
import torch.nn as nn

class StackedRNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        # batch_first=True 使输入/输出张量形状为 (批, 序列长度, 特征)
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        # 第二层的输入大小是第一层的隐藏大小
        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # x 形状: (批大小, 序列长度, 输入大小)

        # output_seq1 形状: (批大小, 序列长度, 隐藏大小1)
        # final_states1 是 LSTM 的元组 (h_n, c_n)
        output_seq1, final_states1 = self.lstm1(x)

        # 将 lstm1 的输出序列馈送到 gru2
        # output_seq2 形状: (批大小, 序列长度, 隐藏大小2)
        # final_state2 是 GRU 的 h_n
        output_seq2, final_state2 = self.gru2(output_seq1)

        # 如果我们需要最后一个时间步的输出来进行分类：
        # output_seq2[:, -1, :] 选择最后一个时间步的输出
        # 形状变为: (批大小, 隐藏大小2)
        last_time_step_output = output_seq2[:, -1, :]

        # 将最终输出通过全连接层
        out = self.fc(last_time_step_output)
        # out 形状: (批大小, 输出大小)
        return out

# 示例用法：
# input_features = 10
# seq_len = 20
# batch_size = 4
# model = StackedRNN(input_size=10, hidden_size1=64, hidden_size2=32, output_size=5)
# dummy_input = torch.randn(batch_size, seq_len, input_features)
# output = model(dummy_input)
# print(output.shape) # 应为 torch.Size([4, 5])
```

在PyTorch中使用`num_layers`参数：

```python
import torch
import torch.nn as nn

class StackedRNNInternal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        # 在内部创建一个堆叠LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True) # 如果需要，在层之间添加 dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # output_seq 包含*最后一层*每个时间步的隐藏状态
        # h_n 和 c_n 包含*所有层*的最终隐藏/细胞状态
        output_seq, (h_n, c_n) = self.lstm(x)

        # 使用最后一层最后一个时间步的输出
        last_time_step_output = output_seq[:, -1, :]
        out = self.fc(last_time_step_output)
        return out

# 示例用法：
# model_internal = StackedRNNInternal(input_size=10, hidden_size=64, num_layers=2, output_size=5)
# dummy_input = torch.randn(4, 20, 10)
# output = model_internal(dummy_input)
# print(output.shape) # 应为 torch.Size([4, 5])
```

### 可视化数据流

我们可以可视化一个简单的堆叠架构：

> 一个两层堆叠RNN的数据流。最终输出取自第二个循环层的最后一个时间步，供全连接层处理。请注意第一层需要设置`return_sequences=True`。

### 考量

虽然堆叠可以提升模型性能，但请记住以下几点：

- **计算成本：** 每增加一层都会增加训练和推断时的计算时间。
- **梯度消失/爆炸：** 尽管LSTM和GRU在减轻这些问题方面优于简单RNN，但非常深的堆叠有时仍可能遇到梯度问题，尽管程度较轻。梯度裁剪等方法仍然适用。
- **过拟合 (overfitting)：** 更深的模型具有更多参数 (parameter)，更容易过拟合，尤其是在数据有限的情况下。正则化 (regularization)方法，例如 dropout（特别是为RNN设计的循环 dropout 变体），变得更为重要。我们将在第10章讨论正则化。
- **收益递减：** 增加越来越多的层并不总是能保证更好的性能。通常，两到三个循环层就足够了，此时性能可能会趋于平稳甚至下降。需要通过实践来寻找特定任务的最佳深度。

在接下来的部分中，我们将介绍另一种常见的架构模式：双向RNN，它会处理两个方向的序列。之后，我们将通过一个情绪分析的实践示例来应用这些实现想法。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇原始论文介绍了长短期记忆（LSTM）网络，它是构建循环神经网络（包括堆叠架构）的核心组件。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书为深度学习提供了基础背景，涵盖了循环神经网络、LSTM、GRU以及深度架构的原理。第10章和第11章尤其相关。
- [tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) — TensorFlow Developers (2024)
  Keras LSTM层的官方文档，包含了关于`return_sequences`参数的详细信息，这对于在TensorFlow/Keras模型中堆叠循环层很重要。
- [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch LSTM模块的官方文档，解释了`num_layers`等参数以及输出序列的结构，这与在PyTorch中构建堆叠式RNN相关。
