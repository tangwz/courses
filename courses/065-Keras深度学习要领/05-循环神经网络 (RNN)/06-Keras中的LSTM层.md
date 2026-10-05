# Keras中的LSTM层

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/lstm-layer-keras)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network)（RNN）如`SimpleRNN`层，经常面临梯度消失问题，这使得网络难以学习长序列中的依赖关系。为了解决这一局限性，研究人员开发了更精巧的循环架构。其中，长短期记忆（LSTM）网络是最成功且广泛使用的一种。

LSTM引入了一种机制，可以明确地管理信息随时间流动，使其能够选择性地记忆或遗忘信息。这是通过一个门控系统实现的，该系统控制着一个专门的*细胞状态*（$c_t$），该状态就像一条贯穿整个序列的传送带，以最少处理传输信息。

### LSTM单元结构

一个LSTM单元处理当前时间步的输入（$x_t$）以及前一时间步的隐藏状态（$h_{t-1}$）。与`SimpleRNN`不同，它使用三个主要的门，并同时更新隐藏状态（$h_t$）和细胞状态（$c_t$）。

1. **遗忘门（$f_t$）：** 决定要从细胞状态中丢弃哪些信息。它查看$h_{t-1}$和$x_t$，对细胞状态$c_{t-1}$中的每个数字输出一个0到1之间的值。1表示“完全保留此信息”，而0表示“完全丢弃此信息”。它通常使用sigmoid激活函数 (activation function)。
   $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. **输入门（$i_t$）：** 决定我们将要存储到细胞状态中的新信息。这包含两个部分：
   - 一个sigmoid层（$i_t$）决定要更新哪些值。
   - 一个`tanh`层（$\tilde{c}_t$）创建一个新的候选值向量 (vector)，以添加到状态中。
     $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
     $\tilde{c}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
3. **细胞状态更新：** 旧细胞状态$c_{t-1}$被更新为新细胞状态$c_t$。遗忘门$f_t$乘以旧状态，输入门$i_t$乘以候选值$\tilde{c}_t$。
   $c_t = f_t * c_{t-1} + i_t * \tilde{c}_t$
4. **输出门（$o_t$）：** 决定细胞状态的哪一部分将作为输出。
   - 一个sigmoid层（$o_t$）决定细胞状态的哪些部分要输出。
   - 细胞状态通过`tanh`函数（将值推到-1到1之间），并乘以sigmoid门的输出。这个过滤后的版本成为新的隐藏状态$h_t$。
     $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
     $h_t = o_t * \tanh(c_t)$

隐藏状态$h_t$是当前时间步LSTM单元的输出。细胞状态和门的结合使得LSTM能够处理比`SimpleRNN`长得多的序列并保持相关信息，从而缓解了梯度消失问题。

> 一个LSTM单元的图示，呈现了信息如何通过遗忘门、输入门和输出门流动，并联系着细胞状态和隐藏状态。

### 在Keras中实现LSTM

在Keras中使用LSTM简单直接，这多亏了`keras.layers.LSTM`层。它的功能类似于`SimpleRNN`，但融入了上面描述的更复杂的内部逻辑。

```python
import keras
from keras import layers

# 定义一个包含64个单元的LSTM层
# 假设输入形状为 (批量大小, 时间步, 特征)
# 例如，(32, 10, 8) 表示32个序列，每个序列10个时间步，每个时间步8个特征
lstm_layer = layers.LSTM(units=64)

# 您可以将其添加到Sequential模型中：
model = keras.Sequential([
    # 第一层需要输入形状
    layers.Input(shape=(None, 8)), # (时间步, 特征) - None允许变长的序列
    layers.LSTM(units=64, return_sequences=True), # 返回完整的序列输出
    layers.LSTM(units=32), # 只返回最后一个输出
    layers.Dense(units=10) # 示例最终分类层
])

model.summary()
```

`keras.layers.LSTM`的重要参数 (parameter)：

- `units`：这是输出空间的维度，也对应着隐藏状态$h_t$和细胞状态$c_t$的维度。这是一个必需的参数。
- `activation`：应用于候选细胞状态（$\tilde{c}_t$）和最终隐藏状态输出计算（$h_t$）的激活函数 (activation function)。默认是`'tanh'`。
- `recurrent_activation`：用于三个门（遗忘、输入、输出）的激活函数。默认是`'sigmoid'`。
- `return_sequences`：一个布尔值。
  - 如果为`False`（默认），该层只返回输入序列中*最后*一个时间步的隐藏状态（$h_T$）。当LSTM层是`Dense`层之前的最终循环层，用于序列分类等任务时，这非常适合。
  - 如果为`True`，该层返回*每个*时间步的隐藏状态（$h_1, h_2, ..., h_T$）。这在堆叠LSTM层（以便下一个LSTM层接收序列作为输入）或在需要每个时间步都有输出的序列到序列任务中是必需的。
- `input_shape`：与其他Keras层一样，您需要为模型中的第一层指定输入的形状。对于循环层，这通常是`(时间步, 特征)`。如果您的序列长度可变，可以将`时间步`维度设置为`None`。

默认情况下，当在兼容的GPU上运行时，`LSTM`层使用优化的CuDNN核，从而大幅加快训练速度。

与`SimpleRNN`相比，`LSTM`层由于其内部门控机制，每个时间步涉及更多计算。然而，正是这种复杂性使其能够有效学习长距离依赖关系，从而使其成为许多序列建模任务的强大工具。在本章后面的实践部分，您将实现一个用于文本分类的LSTM模型。

## 参考资料

- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）架构的原始论文，详细阐述了其门控和单元状态机制。
- [\`keras.layers.LSTM\`](https://keras.io/api/layers/recurrent_layers/lstm/) — Keras Team (2024)
  Keras LSTM层的官方文档，提供其参数和实现用法的具体说明。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本关于循环神经网络的学术教材，包含对LSTM及其优势的理论说明。
- [Sequence Models (Course 5 of Deep Learning Specialization)](https://www.coursera.org/learn/nlp-sequence-models/) — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri (2023)
  Publisher: Coursera / DeepLearning.AI
  一个在线课程模块，提供序列模型的直观解释和实用示例，侧重于LSTM及其应用。

---

[上一节](05-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E7%BD%91%E7%BB%9C.md) · [下一节](07-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29.md)
