---
course: "rnns-and-sequence-modeling"
chapter: "implementing-lstm-gru"
lesson: "practice-sentiment-analysis"
sourceId: 2610
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-7-implementing-lstm-gru/practice-sentiment-analysis"
title: "动手实践：情感分析"
description: "使用代码示例实现LSTM或GRU模型，用于文本情感分类任务。"
order: 7
plots: ["plots/2610-0.json"]
sourceHash: "71c02a7880d658f295ab57fc838c8157cc79684bc998f6f1b10ec87cd3d79e5a"
sourceCorrections: []
---

情感分析是一种常见的序列建模任务，它涉及构建和训练一个模型。该模型使用LSTM或GRU层将文本评论分类为正面或负面。开发过程展示了如何应用框架API、处理序列数据以及构建一个完整的模型。

我们假定您对文本预处理步骤（如分词 (tokenization)和填充）有基本了解，这些内容在第8章中有详细介绍。在这里，我们将侧重于将这些步骤与LSTM/GRU模型实现相结合。

### 准备工作：IMDB数据集

我们将使用常用的IMDB数据集，它包含50,000条电影评论，标记 (token)为正面(1)或负面(0)。这个数据集通常直接包含在深度学习 (deep learning)框架中，使其便于获取。

```python
# 使用TensorFlow/Keras的示例
import tensorflow as tf
from tensorflow import keras

# 加载数据集，只保留最常见的N个词
VOCAB_SIZE = 10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

print(f"训练样本数: {len(train_data)}, 标签数: {len(train_labels)}")
print(f"示例评论（整数编码）: {train_data[0][:20]}...")
```

数据已进行整数编码，其中每个整数代表数据集词汇表 (vocabulary)中的一个特定词。

### 准备数据

循环网络需要长度一致的输入。由于电影评论的长度不一，我们需要将其填充或截断到固定大小。我们将使用后填充，这意味着在较短序列的末尾添加零。掩码（通常由框架层自动处理）将确保这些填充值在计算过程中被忽略。

```python
# 将序列填充到最大长度
MAX_SEQUENCE_LENGTH = 256

train_data_padded = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=0, # 填充值
    padding='post', # 在末尾填充
    maxlen=MAX_SEQUENCE_LENGTH
)

test_data_padded = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=0,
    padding='post',
    maxlen=MAX_SEQUENCE_LENGTH
)

print(f"填充后的评论示例长度: {len(train_data_padded[0])}")
print(f"填充后的评论示例: {train_data_padded[0][:30]}...")
```

### 构建情感分析模型

现在，我们使用Keras Sequential API定义模型架构。

1. **嵌入 (embedding)层：** 此层接收整数编码的词汇表 (vocabulary)索引，并为每个词查找对应的密集向量 (vector) (dense vector)表示（嵌入）。它在训练过程中学习这些嵌入。它期望输入形状为`(batch_size, sequence_length)`，输出形状为`(batch_size, sequence_length, embedding_dim)`。
2. **LSTM或GRU层：** 这是核心循环层。它处理嵌入序列。我们将从LSTM层开始。
3. **全连接输出层：** 一个带有sigmoid激活函数 (activation function)的单个神经元输出一个介于0和1之间的值，表示评论为正面的概率。

```python
EMBEDDING_DIM = 16
RNN_UNITS = 32 # LSTM/GRU层中的单元数量

model = keras.Sequential([
    keras.layers.Embedding(input_dim=VOCAB_SIZE,
                           output_dim=EMBEDDING_DIM,
                           mask_zero=True, # 重要：为填充值启用掩码
                           input_length=MAX_SEQUENCE_LENGTH),
    keras.layers.LSTM(RNN_UNITS), # 你可以在这里将LSTM替换为GRU
    keras.layers.Dense(1, activation='sigmoid') # 用于二元分类的输出层
])

model.summary()
```

在`Embedding`层中`mask_zero=True`参数 (parameter)很重要。它告诉下游层（如LSTM）忽略输入为0（我们的填充值）的时间步。

#### 模型架构可视化

> 基本模型结构：输入 -> 嵌入 -> LSTM -> 全连接输出。

### 编译模型

在训练之前，我们需要使用`compile`来配置学习过程。我们指定优化器、损失函数 (loss function)以及要监测的指标。

- **优化器：** `adam`通常是一个不错的初始选择。
- **损失函数：** `binary_crossentropy`适用于具有sigmoid输出的二元（0/1）分类问题。
- **指标：** `accuracy`使我们能够跟踪训练和评估期间正确分类评论的百分比。

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 训练模型

我们现在可以使用`fit`方法训练模型，提供填充后的训练数据和标签。我们还划出部分训练数据用于训练期间的验证，以监测模型在未见过数据上的表现并检查是否出现过拟合 (overfitting)。

```python
EPOCHS = 10
BATCH_SIZE = 512

# 从训练数据中创建验证集
validation_split = 0.2
num_validation_samples = int(validation_split * len(train_data_padded))

x_val = train_data_padded[:num_validation_samples]
partial_x_train = train_data_padded[num_validation_samples:]

y_val = train_labels[:num_validation_samples]
partial_y_train = train_labels[num_validation_samples:]

print("正在训练模型...")
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_val, y_val),
                    verbose=1) # 设置 verbose=1 或 2 以查看每个epoch的进度
print("训练完成。")
```

### 评估表现

训练完成后，我们评估模型在预留的测试集上的表现。我们还可以可视化训练和验证集的准确率和损失随周期变化的情况，以了解学习过程。

```python
print("\n正在评估测试数据...")
results = model.evaluate(test_data_padded, test_labels, verbose=0)
print(f"测试损失: {results[0]:.4f}")
print(f"测试准确率: {results[1]:.4f}")

# 绘制训练历史（需要Plotly）
import plotly.graph_objects as go
from plotly.subplots import make_subplots

history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_range = range(1, EPOCHS + 1)

fig = make_subplots(rows=1, cols=2, subplot_titles=("训练和验证损失", "训练和验证准确率"))

fig.add_trace(go.Scatter(x=list(epochs_range), y=loss, name='训练损失', mode='lines+markers', line=dict(color='#4263eb')), row=1, col=1)
fig.add_trace(go.Scatter(x=list(epochs_range), y=val_loss, name='验证损失', mode='lines+markers', line=dict(color='#f76707')), row=1, col=1)
fig.add_trace(go.Scatter(x=list(epochs_range), y=acc, name='训练准确率', mode='lines+markers', line=dict(color='#12b886')), row=1, col=2)
fig.add_trace(go.Scatter(x=list(epochs_range), y=val_acc, name='验证准确率', mode='lines+markers', line=dict(color='#ae3ec9')), row=1, col=2)

fig.update_layout(height=400, width=800, title_text="模型训练历史")
fig.update_xaxes(title_text="周期", row=1, col=1)
fig.update_xaxes(title_text="周期", row=1, col=2)
fig.update_yaxes(title_text="损失", row=1, col=1)
fig.update_yaxes(title_text="准确率", row=1, col=2)

# 以下代码生成Plotly JSON输出
print(fig.to_json())
```



![模型训练历史](plots/2610-0.json)



> 示例训练历史，展示了训练集和验证集的损失和准确率曲线随周期变化的情况。（注意：实际曲线值仅供参考，取决于具体的训练运行）。

该图有助于识别潜在的过拟合 (overfitting)（即训练准确率持续提高，但验证准确率趋于平稳或下降），并确定是否需要更多训练周期。

### 变体与后续步骤

- **GRU：** 尝试将`keras.layers.LSTM(RNN_UNITS)`替换为`keras.layers.GRU(RNN_UNITS)`并重新训练。比较性能和训练时间。
- **堆叠循环网络：** 要堆叠层，请确保中间循环层返回完整的序列输出，而不仅仅是最终输出。

  ```python
  model_stacked = keras.Sequential([
      keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, input_length=MAX_SEQUENCE_LENGTH),
      keras.layers.LSTM(RNN_UNITS, return_sequences=True), # 返回每个时间步的隐藏状态
      keras.layers.LSTM(RNN_UNITS), # 此层接收序列
      keras.layers.Dense(1, activation='sigmoid')
  ])
  ```
- **双向循环网络：** 用`keras.layers.Bidirectional`封装一个循环层，以在正向和反向处理输入序列，可能更有效地捕获上下文 (context)信息。

  ```python
  model_bidirectional = keras.Sequential([
      keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, input_length=MAX_SEQUENCE_LENGTH),
      keras.layers.Bidirectional(keras.layers.LSTM(RNN_UNITS)), # 封装LSTM层
      keras.layers.Dense(1, activation='sigmoid')
  ])
  ```

  请注意，`Bidirectional`层通常会使输出特征维度加倍（一组用于正向，一组用于反向），除非另行配置。
- **超参数 (parameter) (hyperparameter)调整：** 尝试调整`EMBEDDING_DIM`、`RNN_UNITS`、`optimizer`选择、学习率和`BATCH_SIZE`。添加`Dropout`用于正则化 (regularization)（稍后会介绍）。

这个实践示例为实现LSTM和GRU模型进行序列分类提供了扎实的基础。通过修改输入数据准备和模型的最终输出层，可以将此结构应用于各种其他基于序列的任务。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇基础论文介绍了长短期记忆（LSTM）循环神经网络架构，旨在解决传统RNN中梯度消失的问题。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio (2014)
  Journal: EMNLP 2014; DOI: [10.48550/arXiv.1406.1078](https://doi.org/10.48550/arXiv.1406.1078)
  这篇论文介绍了门控循环单元（GRU），这是一种循环神经网络的变体，它提供了比LSTM更简单的结构，同时通常能达到相似的性能。
- [Keras RNN layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN) — TensorFlow Authors (2023)
  TensorFlow Keras官方文档提供了循环层（如LSTM、GRU和Bidirectional）的详细信息和使用示例，这些是本实践练习的核心内容。
