# 使用框架API构建简单RNN层

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns/framework-apis-simple-rnn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用流行的深度学习 (deep learning)框架实现循环神经网络 (neural network)（RNN）是一种常见做法。虽然理解简单循环神经网络的机制以及它如何通过随时间反向传播 (backpropagation)进行学习很有益处，但仅使用基本矩阵运算从头开始编写RNN，尽管具有指导意义，但在实际中很少这样做。像TensorFlow（使用Keras API）和PyTorch这样的框架提供了高度优化、预构建的层，处理了状态管理和梯度计算的复杂性。

这些高级API使我们能够仅用几行代码定义循环层，指定必要的配置参数 (parameter)，而无需手动实现底层循环关系或随时间反向传播。这种抽象显著加快了开发速度，使我们能够专注于模型架构和应用。

### 使用TensorFlow/Keras：`tf.keras.layers.SimpleRNN`

TensorFlow通过其高级Keras API提供了`SimpleRNN`层。该层实现了我们之前讨论过的基本循环单元结构。

要使用它，通常需要导入它并在Keras模型中（通常是`Sequential`模型或使用函数式API）实例化它。

```python
import tensorflow as tf

# 定义一个SimpleRNN层
# 'units' 指定了隐藏状态和输出空间的维度。
rnn_layer = tf.keras.layers.SimpleRNN(units=64)

# 示例：将其作为Sequential模型中的第一层添加
# 需要指定输入形状：(时间步长, 特征)
# 注意：批次大小通常在此处省略，隐含处理。
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(units=32, input_shape=(10, 5)) # 10个时间步长，每个时间步长5个特征
    # ... 可能添加更多层，例如用于输出的Dense层
])

# 你可以查看层的配置
# print(rnn_layer.get_config())
```

`tf.keras.layers.SimpleRNN`的重要参数 (parameter)：

- **`units`**：这是一个必需的正整数，定义了RNN单元中隐藏单元的数量。这也决定了层在每个时间步（如果`return_sequences=True`）或最终时间步（如果`return_sequences=False`）生成的输出向量 (vector)的维度。你可以将其视为网络“内存”的大小。
- **`activation`**：应用于隐藏状态的激活函数 (activation function)。默认是`'tanh'`，这在简单RNN中很常用，因为其输出范围（-1到1）在一定程度上可以帮助减轻梯度爆炸，尽管相较于ReLU，它仍然容易受到梯度消失的影响。也可以使用其他选项，例如`'relu'`。
- **`return_sequences`**：一个布尔值（默认：`False`）。
  - 如果为`False`，层只返回*最后*一个时间步的输出，形状为`(batch_size, units)`。当你只需要序列的最终摘要时适用，例如在序列分类中。
  - 如果为`True`，层返回*每个*时间步的完整序列输出，形状为`(batch_size, time_steps, units)`。在堆叠多个循环层或进行序列到序列任务（其中每一步都需要输出）时是必需的。
- **`return_state`**：一个布尔值（默认：`False`）。如果为`True`，层除了主要输出外，还返回两个额外输出：来自最终时间步的隐藏状态。对于`SimpleRNN`，这只是一个最终隐藏状态张量。这对于简单RNN来说不那么常用，但在更复杂的模型中（例如编码器-解码器架构）很有用。
- **`input_shape`**：仅对`Sequential`模型中的*第一个*层是必需的。它是一个元组，指定输入序列的形状，不包括批次大小。格式通常是`(时间步长, 特征)`。对于后续层，Keras会自动推断输入形状。

### 使用PyTorch：`torch.nn.RNN`

PyTorch提供了`torch.nn.RNN`模块来创建简单的循环层。

```python
import torch
import torch.nn as nn

# 定义一个RNN层
# input_size: 在每个时间步中输入x的特征数量
# hidden_size: 隐藏状态h中的特征数量（等同于Keras中的'units'）
# batch_first=True使输入/输出张量形状为(批次, 序列长度, 特征)
rnn_layer = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

# 示例使用虚拟输入：
# 输入形状：(批次大小, 序列长度, 输入特征)
batch_size = 5
sequence_length = 15
input_features = 10
dummy_input = torch.randn(batch_size, sequence_length, input_features)

# 初始化隐藏状态（可选，如果未提供则默认为零）
# 形状：(层数 * 方向数, 批次大小, 隐藏大小) -> 对于此层为(1, 5, 20)
initial_hidden_state = torch.randn(1, batch_size, 20)

# 将输入和初始隐藏状态通过层
# output: 包含每个时间步的输出特征
# final_hidden_state: 包含最后一个时间步的隐藏状态
output, final_hidden_state = rnn_layer(dummy_input, initial_hidden_state)

print("Input shape:", dummy_input.shape)         # torch.Size([5, 15, 10])
print("Output shape:", output.shape)             # torch.Size([5, 15, 20])
print("Final hidden state shape:", final_hidden_state.shape) # torch.Size([1, 5, 20])
```

`torch.nn.RNN`的重要参数 (parameter)：

- **`input_size`**：每个时间步中输入$x$的预期特征数量。
- **`hidden_size`**：隐藏状态$h$中的特征数量。这定义了层内部记忆及其输出的维度。
- **`num_layers`**：垂直堆叠的循环层数量（默认：1）。我们稍后会讨论堆叠，但这使得创建更深的RNN变得容易。
- **`nonlinearity`**：激活函数 (activation function)。`'tanh'`（默认）或`'relu'`。适用与Keras的`activation`类似的考虑。
- **`batch_first`**：一个布尔值（默认：`False`）。这是一个重要的参数，影响输入和输出张量的预期形状。
  - 如果为`False`，输入和输出的形状预期为`(sequence_length, batch_size, features)`。
  - 如果为`True`，输入和输出使用形状`(batch_size, sequence_length, features)`。这通常更方便，因为它与数据通常被其他层类型加载和处理的方式一致。**通常建议将`batch_first`设置为`True`以保持一致性。**
- **`bias`**：是否在计算中包含偏置 (bias)项（默认：`True`）。

当你将输入张量通过PyTorch `RNN`层时，它会返回两个输出：

1. **`output`**：一个张量，包含来自*最终*循环层的*每个*时间步的输出隐藏状态$h_t$。如果`batch_first=True`，其形状为`(batch_size, sequence_length, hidden_size)`。
2. **`h_n`**：一个张量，包含堆叠中每一层在*t = 序列长度*（最后一个时间步）时的最终隐藏状态。其形状为`(num_layers * num_directions, batch_size, hidden_size)`。即使是单层（`num_layers=1`），它也具有这种3D形状。

请注意，与Keras中`return_sequences`控制输出形状不同，PyTorch的`nn.RNN`在`output`张量中*总是*提供完整的序列输出。如果你只需要最后一个时间步的输出（等同于Keras的`return_sequences=False`），你需要手动从`output`张量中选择它（例如，如果`batch_first=True`，则为`output[:, -1, :]`）。`h_n`输出直接为你提供最终隐藏状态。

使用TensorFlow/Keras和PyTorch中的这些高级API，我们能够轻松地将简单RNN功能整合到我们的模型中。下一步涉及理解如何正确地塑造我们的输入数据以及管理这些层产生的输出形状，我们将在下一节中涵盖。

## 参考资料

- [tf.keras.layers.SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN) — TensorFlow Developers (2024)
  Keras SimpleRNN层的官方文档，详细说明了TensorFlow实现中的参数和用法。
- [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch RNN模块的官方文档，描述了其构造函数、输入、输出和可配置选项。
- [Deep Learning](https://www.deeplearningbook.org/contents/rnn.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 367
  第10章“序列建模：循环和递归网络”提供了关于RNN机制和通过时间反向传播的基础理论理解。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 451-460
  实用指南，包含使用TensorFlow和Keras实现简单RNN的代码示例和解释，为API用法提供了背景信息。

---

[上一节](02-RNN%20%E5%8D%95%E5%85%83%E5%AE%9E%E7%8E%B0.md) · [下一节](04-%E5%A4%84%E7%90%86%E8%BE%93%E5%85%A5%E4%B8%8E%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md)
