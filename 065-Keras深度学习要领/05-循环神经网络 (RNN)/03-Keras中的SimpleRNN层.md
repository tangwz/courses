# Keras中的SimpleRNN层

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/simplernn-layer-keras)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network)（RNN）逐步处理序列并保持记忆，以捕捉时间上的依赖关系。在Keras中，`SimpleRNN` 层是这种架构的一个主要实现。这个层为理解循环操作提供了一个直接的起点，之后可以学习更复杂的版本。

### SimpleRNN的运作方式

`SimpleRNN` 层通过迭代输入序列的时间步来处理序列。在每个时间步 $t$，它接收该步的输入 $x_t$ 以及前一步的隐藏状态 $h_{t-1}$，从而计算当前的隐藏状态 $h_t$，并可选择性地计算输出 $y_t$。

核心操作可以通过以下隐藏状态 $h_t$ 的更新规则表示：


$$
h_t = \text{激活函数}(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$


如果每一步都产生输出（由我们稍后讨论的一个参数 (parameter)控制），则通常按以下方式计算：


$$
y_t = \text{输出激活函数}(W_{hy} h_t + b_y)
$$


其中：

- $x_t$: 时间步 $t$ 的输入向量 (vector)。
- $h_t$: 时间步 $t$ 的隐藏状态（或输出，取决于配置）。
- $h_{t-1}$: 前一个时间步的隐藏状态。初始隐藏状态 $h_0$ 通常是一个全零向量。
- $W_{xh}$: 连接输入 $x_t$ 到隐藏状态计算的权重 (weight)矩阵。
- $W_{hh}$: 连接前一个隐藏状态 $h_{t-1}$ 到当前隐藏状态计算的权重矩阵（这是“循环”权重）。
- $b_h$: 隐藏状态计算的偏置 (bias)项。
- $W_{hy}$: 连接隐藏状态 $h_t$ 到输出 $y_t$ 的权重矩阵。
- $b_y$: 输出计算的偏置项。
- $\text{激活函数}$ 和 $\text{输出激活函数}$: 逐元素应用的激活函数 (activation function)。对于 `SimpleRNN`，默认的隐藏状态激活函数是双曲正切（'tanh'）。

主要思想是在所有时间步中重复使用权重矩阵 ($W_{xh}$, $W_{hh}$, $W_{hy}$) 和偏置 ($b_h$, $b_y$)。这种参数共享使得RNN能高效处理不同长度的序列，并能够将被序列中某一点学到的模式推广到其他点。

可以将隐藏状态 $h_t$ 看作网络在时间步 $t$ 的记忆。它捕获了网络认为对处理当前时间步 $t$ 和未来时间步重要的所有先前步骤（$0$ 到 $t-1$）的信息。

### 在Keras中实现`SimpleRNN`

您可以像添加任何其他层一样，将 `SimpleRNN` 层添加到您的Keras模型中。它位于 `keras.layers.SimpleRNN`。

```python
import keras
from keras import layers

# 示例：向Sequential模型添加SimpleRNN层
model = keras.Sequential()

# 定义输入形状：(时间步数, 每个时间步的特征数)
# 例如，一个包含10个时间步的序列，每个时间步有8个特征。
model.add(keras.Input(shape=(10, 8))) 

# 添加一个具有32个单元（隐藏状态/输出的维度）的SimpleRNN层
# 默认情况下，它只返回*最后*一个时间步的输出。
model.add(layers.SimpleRNN(units=32)) 

# 您可以在RNN之后添加Dense层用于分类/回归
model.add(layers.Dense(units=1, activation='sigmoid')) # 例如用于二元分类

model.summary()
```

#### 参数 (parameter)

- `units`: 这是最主要的参数。它定义了隐藏状态的维度，如果 `return_sequences=False`，也随之定义了输出空间维度。选择合适数量的单元取决于具体问题，通常需要通过尝试来确定。数量越多，网络可能存储更复杂的模式，但会增加计算成本和过拟合 (overfitting)的风险。
- `activation`: 指定用于隐藏状态计算的激活函数 (activation function)。默认是 `'tanh'`（双曲正切），它输出介于-1和1之间的值。虽然有时可以使用其他激活函数，如 `'relu'`，但 `'tanh'` 是简单RNN的传统选择。
- `input_shape`: 与其他Keras层一样，您需要为模型中的第一个层指定输入形状。对于RNN，此形状通常是一个元组 `(时间步数, 特征数)`，其中 `时间步数` 是序列的长度（可为 `None` 表示可变长度序列），`特征数` 是每个时间步的特征数量。例如，文本数据可能是 `(序列长度, 嵌入维度)`，而时间序列可能是 `(时间段数, 传感器读数数量)`。
- `return_sequences`: 这个布尔参数控制层的输出内容：
  - `False` (默认): 该层只输出 *最后* 一个时间步的隐藏状态。输出形状将是 `(批大小, 单元数)`。当您只需要整个序列的摘要时（例如，在最终的 `Dense` 层进行分类之前），这很常见。
  - `True`: 该层输出序列中 *每个* 时间步的隐藏状态。输出形状将是 `(批大小, 时间步数, 单元数)`。如果您想堆叠多个RNN层（因为下一个RNN层需要序列作为输入）或者您正在构建序列到序列模型（如机器翻译或时间序列预测，其中您在每个时间步预测值），则此项是必需的。

#### 示例：堆叠RNN层

要堆叠 `SimpleRNN` 层，所有前面的RNN层必须返回其完整的序列输出。

```python
import keras
from keras import layers

# 示例：堆叠SimpleRNN层
model = keras.Sequential(name="Stacked_SimpleRNN")
model.add(keras.Input(shape=(None, 10))) # 可变长度序列，每个时间步10个特征

# 第一个SimpleRNN层：必须返回序列以供下一个RNN层使用
model.add(layers.SimpleRNN(units=64, return_sequences=True)) 

# 第二个SimpleRNN层：如果它是Dense层之前的最后一个RNN层，可以只返回最后一个输出；
# 如果后面跟着另一个RNN，则返回序列。
model.add(layers.SimpleRNN(units=32, return_sequences=False)) # 只返回最后一个输出

# 添加一个Dense层用于分类
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
```

### 何时使用`SimpleRNN`

`SimpleRNN` 层简单明了，提供了一种处理序列信息的机制。它在序列中相关信息包含在相对短期依赖关系的任务中，可以表现得相当不错。

然而，`SimpleRNN` 有一个明显的局限性，即所谓的**梯度消失问题**。在反向传播 (backpropagation)过程中，梯度在时间上传播时会呈指数级减小。这使得网络很难学习序列中相距很远事件之间的连接。实际上，网络难以“记住”很久以前的信息。反之，梯度也可能爆炸（变得非常大），尽管这通常通过梯度裁剪等技术更容易管理。

由于梯度消失问题，对于需要捕获长距离依赖的任务，`SimpleRNN` 通常不如更高级的循环层，如LSTM和GRU有效，这些任务在自然语言处理、复杂时间序列分析等领域很常见。我们将在后续章节中介绍这些更强大的层。

## 参考资料

- [Learning long-term dependencies in recurrent nets](https://doi.org/10.1109/72.279181) — Yoshua Bengio, Patrice Simard, and Paolo Frasconi (1994)
  Journal: IEEE Transactions on Neural Networks; Publisher: IEEE; Volume: 5; Pages: 157-166; DOI: [10.1109/72.279181](https://doi.org/10.1109/72.279181)
  这篇论文介绍了循环神经网络中的梯度消失问题，这是本节解释的一个限制。
- [SimpleRNN layer - Keras documentation](https://keras.io/api/layers/recurrent_layers/simple_rnn/) — Keras team (2023)
  Keras中`SimpleRNN`层的官方文档，提供了API规范和使用示例。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，提供了循环神经网络的理论背景，包括简单RNN的工作原理和梯度消失问题。

---

[上一节](02-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%A6%81%E7%82%B9.md) · [下一节](04-%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98.md)
