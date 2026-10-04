# 构建一个简单RNN模型

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-3-building-simple-rnns/constructing-basic-rnn-model)

[返回章节目录](README.md) · [返回课程目录](../README.md)

单个`SimpleRNN`层通过框架API定义，输入数据需准备成正确形状（通常为`batch_size`、`time_steps`、`features`）。将这些组件结合，可构建一个完整、简单的RNN模型。

大多数序列模型都遵循一个通用模式：即通过一个或多个循环层处理输入序列，然后使用标准前馈层将最终的循环状态（或一系列状态）映射到期望的输出。

### 构建模型结构

一个典型的用于序列任务的简单RNN模型可能包含以下层：

1. **（可选）嵌入 (embedding)层：** 如果处理的是整数编码序列（例如用ID表示的单词），嵌入层通常是第一步。它将这些整数转换为固定大小的稠密向量 (vector)。这在自然语言处理中很常见。如果你的输入特征已经是数值向量（例如归一化 (normalization)的时间序列值），则可以跳过这一层。
2. **SimpleRNN层：** 这是核心循环层，它逐步处理序列并维持其隐藏状态。它将前一层的输出（可以是嵌入层或原始输入）作为其输入。
3. **全连接层：** 一个或多个全连接（Dense）层通常在RNN层之后添加。这些层接收最终的隐藏状态（或有时是每个时间步的输出，取决于配置和任务）并将其转换为最终的预测格式。例如：
   - 单个带 sigmoid 激活的 Dense 层，用于二分类。
   - 带 `num_classes` 个单元和 softmax 激活的 Dense 层，用于多分类。
   - 带一个单元和线性激活的 Dense 层，用于回归（例如预测时间序列中的一个值）。

让我们看看这在常用框架中如何呈现。

**使用TensorFlow (Keras API)**

Keras 使用 `Sequential` 模型API使层堆叠变得简单。

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

# --- 假设这些变量已定义 ---
vocab_size = 10000  # 文本的示例词汇量大小
embedding_dim = 16 # 嵌入向量的维度
rnn_units = 32     # SimpleRNN层中的单元数量
num_classes = 2    # 示例：用于二分类
max_sequence_length = 100 # 示例最大序列长度
# -----------------------------------------

# 带嵌入层的示例
model_text = keras.Sequential(name="SimpleRNN_Text_Classifier")
model_text.add(layers.Input(shape=(max_sequence_length,), dtype='int32', name="Input_Sequence")) # 显式定义输入形状
model_text.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="Token_Embedding"))
model_text.add(layers.SimpleRNN(rnn_units, name="Simple_RNN")) # 默认情况下，只返回最后一个隐藏状态
model_text.add(layers.Dense(num_classes, activation='softmax', name="Output_Classifier")) # 用于分类的输出层

# 不带嵌入层的示例（例如，用于数值时间序列）
# 输入形状：(时间步长, 每个时间步的特征数量)
input_features = 5 # 每个时间步的示例特征数量
model_numeric = keras.Sequential(name="SimpleRNN_Numeric_Regressor")
model_numeric.add(layers.Input(shape=(max_sequence_length, input_features), name="Input_TimeSeries")) # 显式定义输入形状
model_numeric.add(layers.SimpleRNN(rnn_units, name="Simple_RNN"))
model_numeric.add(layers.Dense(1, name="Output_Regressor")) # 用于回归的输出层

# 你可以打印模型摘要以查看架构
print("文本模型摘要:")
model_text.summary()

print("\n数值模型摘要:")
model_numeric.summary()
```

**使用PyTorch**

在PyTorch中，你通常会定义一个继承自`torch.nn.Module`的自定义类。

```python
import torch
import torch.nn as nn

# --- 假设这些变量已定义 ---
vocab_size = 10000
embedding_dim = 16
rnn_units = 32
num_classes = 2
input_features = 5 # 用于数值示例
# -----------------------------------------

# 带嵌入层的示例
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first=True使输入形状变为 (批量大小, 序列长度, 特征数量)
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, num_classes)

    def forward(self, x):
        # x 形状: (批量大小, 序列长度)
        embedded = self.embedding(x)
        # embedded 形状: (批量大小, 序列长度, 嵌入维度)
        # output 包含每个时间步的隐藏状态
        # hidden 包含最终的隐藏状态
        output, hidden = self.rnn(embedded)
        # 我们通常使用最终的隐藏状态进行分类
        # hidden 形状: (层数 * 方向数, 批量大小, RNN单元数)
        # 如果层数和方向数均为1，则压缩第一个维度
        final_hidden = hidden.squeeze(0)
        # final_hidden 形状: (批量大小, RNN单元数)
        out = self.fc(final_hidden)
        # out 形状: (批量大小, 类别数)
        # Softmax 通常在损失函数 (CrossEntropyLoss) 中应用
        # 如果需要显式应用: return torch.softmax(out, dim=1)
        return out

# 不带嵌入层的示例（例如，用于数值时间序列）
class SimpleRNNRegressor(nn.Module):
    def __init__(self, input_features, rnn_units):
        super().__init__()
        self.rnn = nn.RNN(input_features, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, 1) # 回归输出

    def forward(self, x):
        # x 形状: (批量大小, 序列长度, 输入特征数)
        output, hidden = self.rnn(x)
        final_hidden = hidden.squeeze(0)
        out = self.fc(final_hidden)
        # out 形状: (批量大小, 1)
        return out

# 实例化模型
model_text_pt = SimpleRNNClassifier(vocab_size, embedding_dim, rnn_units, num_classes)
model_numeric_pt = SimpleRNNRegressor(input_features, rnn_units)

# 打印模型结构
print("PyTorch 文本模型结构:")
print(model_text_pt)

print("\nPyTorch 数值模型结构:")
print(model_numeric_pt)
```

### 可视化模型流程

我们可以将这种结构可视化为数据流经各层。对于文本分类任务，它可能看起来像这样：

> 一个用于序列分类的简单RNN模型的典型数据流，从输入序列开始，以预测结果结束。

在TensorFlow/Keras和PyTorch中，创建模型都涉及定义层的顺序，并确保一层的输出形状与下一层的预期输入形状匹配。框架会处理权重 (weight)初始化，并提供访问层参数 (parameter)的机制。在继续之前，使用`summary()`方法（Keras）或打印模型对象（PyTorch）有助于验证层的连接、输出形状以及可训练参数的总数。

一旦模型结构定义并实例化，下一步是配置训练过程。这包括选择适合任务的损失函数 (loss function)（例如，分类任务的`CategoricalCrossentropy`或`SparseCategoricalCrossentropy`，回归任务的`MeanSquaredError`），选择优化器（如Adam或RMSprop），并可能指定在训练期间要监控的指标。我们将在下一节中详细阐述训练循环。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 10
  全面介绍了深度学习，包括循环神经网络和序列建模的详细解释。
- [\`SimpleRNN\` layer](https://keras.io/api/layers/recurrent_layers/simple_rnn/) — Keras Team (2024)
  Keras `SimpleRNN` 层的官方文档，提供其配置和在模型构建中使用的具体细节。
- [\`torch.nn.RNN\`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) — PyTorch Team (2024)
  Publisher: PyTorch Foundation
  PyTorch `nn.RNN` 模块的官方文档，解释其构建循环模型的功能和参数。

---

[上一节](04-%E5%A4%84%E7%90%86%E8%BE%93%E5%85%A5%E4%B8%8E%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md) · [下一节](06-RNN%20%E7%9A%84%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md)
