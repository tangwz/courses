# 实践：构建RNN/LSTM用于文本分类

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/practice-rnn-lstm-text-classification)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 Keras 构建并训练 LSTM 模型以执行文本分类，这是自然语言处理 (NLP) 中的一项常见任务。内容涵盖如何使用流行的 IMDB 电影评论数据集，目标是根据评论的文本内容将其分类为正面或负面。这展示了序列数据准备、嵌入 (embedding)层和循环层的实现方法。

### 准备工作：IMDB数据集

IMDB数据集包含50,000条电影评论，预先分为25,000条用于训练，25,000条用于测试。每条评论都被标记 (token)为正面（1）或负面（0）。Keras提供了方便地访问此数据集的方式，数据已预处理成单词索引（整数）序列。每个整数代表字典中的一个特定单词。

我们首先加载数据。为了使输入数据易于处理，我们将词汇量限制为前10,000个最常出现的词。

```python
import keras
from keras.datasets import imdb
from keras import utils

# 加载数据集，只保留前10,000个最常出现的词
vocabulary_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocabulary_size)

print(f"训练样本数量: {len(train_data)}")
print(f"测试样本数量: {len(test_data)}")

# 示例：查看第一条评论（单词索引序列）
print(f"第一条训练评论（索引）: {train_data[0][:15]}...")
print(f"第一条评论的标签: {train_labels[0]}")
```

你会注意到每个`train_data`和`test_data`样本都是一个整数列表。此外，评论的长度也不同。

### 准备序列数据

循环神经网络 (neural network) (RNN)要求输入具有一致的序列长度。我们需要对评论进行填充（或截断），以便每个序列都具有相同数量的元素。我们将使用Keras的`pad_sequences`实用工具。让我们选择一个最大序列长度，例如250个词。短于此长度的评论将在开头用零填充（`padding='pre'`），长于此长度的评论将被截断。

```python
# 设置每条评论序列的最大长度
max_sequence_length = 250

# 填充序列
padded_train_data = utils.pad_sequences(train_data, maxlen=max_sequence_length, padding='pre')
padded_test_data = utils.pad_sequences(test_data, maxlen=max_sequence_length, padding='pre')

print(f"填充后的训练数据形状: {padded_train_data.shape}")
print(f"填充后的测试数据形状: {padded_test_data.shape}")

# 示例：查看第一条填充后的评论
print(f"第一条填充后的训练评论: {padded_train_data[0]}")
```

现在，`padded_train_data`和`padded_test_data`是形状为`(num_samples, max_sequence_length)`的二维张量。

### 构建LSTM模型

我们将构建一个简单的序贯模型：

1. **嵌入 (embedding)层：** 此层接收整数编码的词汇表 (vocabulary)，并学习每个词的嵌入向量 (vector)。它将每个词索引映射到一个固定大小（`embedding_dim`）的稠密向量。此层需要`input_dim`（词汇量大小）和`output_dim`（嵌入维度）。
2. **LSTM层：** 这是核心循环层，用于处理嵌入向量序列。我们在此层中指定LSTM单元（神经元）的数量。这些单元获取序列中的上下文 (context)信息。
3. **密集层：** 最后一个全连接层，包含一个单元和一个sigmoid激活函数 (activation function)。它输出一个介于0和1之间的概率，表示评论为正面的可能性。

让我们使用Keras序贯API定义此模型。

```python
from keras import layers
from keras import models

embedding_dim = 32  # 词嵌入向量的维度
lstm_units = 32     # LSTM层中的单元数量

model = models.Sequential(name="imdb_lstm_classifier")
model.add(layers.Embedding(input_dim=vocabulary_size,
                           output_dim=embedding_dim,
                           input_length=max_sequence_length,
                           name="word_embedding"))
model.add(layers.LSTM(units=lstm_units, name="lstm_layer"))
model.add(layers.Dense(units=1, activation='sigmoid', name="output_classifier"))

model.summary()
```

`model.summary()`的输出显示了各层、它们的输出形状以及参数 (parameter)数量。请注意嵌入层（`vocabulary_size * embedding_dim`）和LSTM层中的大量参数。

### 编译模型

在训练之前，我们需要使用`compile()`方法配置学习过程。我们指定：

- **优化器：** `adam`是一种广泛使用且有效的优化算法。
- **损失函数 (loss function)：** `binary_crossentropy`适用于输出是概率的二元分类问题。
- **度量：** 我们将在训练和评估期间监控`accuracy`。

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("模型编译成功。")
```

### 训练模型

现在我们使用`fit()`方法训练模型。我们提供填充后的训练数据和标签。我们还设置：

- `epochs`：遍历整个训练数据集的次数。
- `batch_size`：每次梯度更新的样本数量。
- `validation_split`：训练数据的一部分用作验证数据。模型在此集合上的性能在每个周期结束时受到监控，有助于我们检测过拟合 (overfitting)。

```python
num_epochs = 10
batch_size = 128
validation_fraction = 0.2

print("开始训练...")
history = model.fit(padded_train_data,
                    train_labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_split=validation_fraction,
                    verbose=1) # 将verbose设置为1或2以查看每个周期的进度

print("训练完成。")
```

`fit()`方法返回一个`History`对象，其中包含每个周期的训练和验证损失以及度量。我们可以使用它来可视化训练过程。



[交互图表：模型训练历史](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/practice-rnn-lstm-text-classification#plot-1gvt0pd)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "模型训练历史",
    "xaxis": {
      "title": "周期"
    },
    "yaxis": {
      "title": "准确率"
    },
    "legend": {
      "title": "度量"
    },
    "width": 700,
    "height": 400
  },
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        0.75,
        0.88,
        0.92,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
        0.99,
        0.99
      ],
      "mode": "lines+markers",
      "name": "训练准确率",
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        0.85,
        0.87,
        0.86,
        0.88,
        0.87,
        0.86,
        0.85,
        0.86,
        0.85,
        0.85
      ],
      "mode": "lines+markers",
      "name": "验证准确率",
      "line": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 训练和验证准确率随周期变化。请注意，验证准确率在训练准确率持续上升的同时，常常趋于平稳甚至下降，这表示可能出现过拟合。



[交互图表：模型训练历史](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/practice-rnn-lstm-text-classification#plot-nmx23p)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "模型训练历史",
    "xaxis": {
      "title": "周期"
    },
    "yaxis": {
      "title": "损失"
    },
    "legend": {
      "title": "度量"
    },
    "width": 700,
    "height": 400
  },
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        0.5,
        0.3,
        0.22,
        0.17,
        0.13,
        0.1,
        0.08,
        0.06,
        0.05,
        0.04
      ],
      "mode": "lines+markers",
      "name": "训练损失",
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        0.35,
        0.31,
        0.33,
        0.3,
        0.34,
        0.38,
        0.42,
        0.45,
        0.49,
        0.53
      ],
      "mode": "lines+markers",
      "name": "验证损失",
      "line": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 训练和验证损失随周期变化。验证损失增加，同时训练损失减少，是过拟合的明显迹象。

### 评估模型

最后，让我们使用`evaluate()`方法评估我们训练好的模型在未见过测试数据上的性能。

```python
loss, accuracy = model.evaluate(padded_test_data, test_labels, verbose=0)

print(f"\n测试损失: {loss:.4f}")
print(f"测试准确率: {accuracy:.4f}")
```

这为我们提供了模型在训练期间从未遇到过的数据上的最终性能度量。对于此任务，您通常会看到准确率明显优于随机猜测（50%）。

### 总结和后续步骤

在此实践部分，您成功地完成了以下内容：

1. 加载并预处理了用于序列建模的文本数据（IMDB评论）。
2. 应用了填充以确保序列长度一致。
3. 使用Keras构建了一个带有`Embedding`层和`LSTM`层的文本分类模型。
4. 编译、训练并使用验证数据监控了模型性能。
5. 评估了测试集上的最终模型。

这为将RNN和LSTM应用于基于序列的问题提供了坚实的基础。在此基础上，您可以尝试进行以下实验：

- 使用`GRU`层而不是`LSTM`。
- 堆叠多个循环层。
- 使用双向LSTM（`keras.layers.Bidirectional`）以正向和反向处理序列。
- 调整超参数 (parameter) (hyperparameter)，如嵌入 (embedding)维度、LSTM单元、批量大小和优化器。
- 应用下一章讨论的技术，例如dropout正则化 (regularization)或提前停止，以对抗过拟合 (overfitting)并可能提高测试准确率。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍长短期记忆（LSTM）架构的原始论文，其对循环神经网络至关重要。
- [Keras Documentation: Layers API](https://keras.io/api/layers/) — Keras team (2024)
  Keras层API的官方文档，提供构建文本分类模型中使用的`Embedding`、`LSTM`和`Dense`层的详细信息。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教科书，涵盖深度学习的理论基础，包括循环神经网络、序列建模和嵌入技术。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  本书提供Keras在不同深度学习任务中的实践范例，包括使用LSTM和词嵌入进行文本分类，与课程内容直接相关。

---

[上一节](08-%E7%94%A8%E4%BA%8ERNN%E7%9A%84%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87.md) · [下一节](../06-%E6%94%B9%E8%BF%9B%E6%A8%A1%E5%9E%8B%E8%A1%A8%E7%8E%B0%E5%92%8C%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B/01-%E7%90%86%E8%A7%A3%E8%BF%87%E6%8B%9F%E5%90%88%E5%92%8C%E6%AC%A0%E6%8B%9F%E5%90%88.md)
