---
course: "nlp-fundamentals"
chapter: "nlp-sequence-models-intro"
lesson: "practice-simple-sequence-model"
sourceId: 2509
sourceUrl: "https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro/practice-simple-sequence-model"
title: "动手实践：构建一个简单的序列模型"
description: "实践实现用于简单文本任务的基本RNN或LSTM模型。"
order: 7
plots: ["plots/2509-0.json"]
sourceHash: "e25ee54f6a158e8db89857f3ad710193a660df3fe0132741df5dad5960eb6066"
sourceCorrections: []
---

使用常用的深度学习 (deep learning)库TensorFlow及其Keras API构建一个简单的序列模型。此实践方法演示了循环神经网络 (neural network)（RNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）等概念的应用。本次练习将巩固你对如何准备序列文本数据以及如何为典型任务构建基本循环模型的理解。

“我们将解决一个简化的情感分析问题：将短文本片段归类为正面或负面。尽管情感分析通常涉及更复杂的数据集和模型，但本例纯粹侧重于设置和训练序列模型的机制。”

### 准备工作：库和数据

首先，请确保已安装TensorFlow。如果未安装，通常可以使用pip进行安装：
`pip install tensorflow`

我们将使用TensorFlow自带的Keras来构建模型。让我们定义一个小型合成数据集用于演示。

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam

# 固定随机种子以确保可重复性
tf.keras.utils.set_random_seed(42)

# 示例数据：(文本，标签) -> 0 表示负面，1 表示正面
# 每个句子至少包含一个明确的情感词，以便模型能够在训练/验证分割中学会泛化。
positives = [
    "this is a great movie",
    "i really enjoyed the film",
    "what a fantastic performance",
    "loved every minute of it",
    "truly amazing storytelling",
    "absolutely wonderful experience",
    "a brilliant and captivating film",
    "the acting was superb",
    "an outstanding piece of cinema",
    "excellent direction and great writing",
    "i enjoyed this film immensely",
    "wonderful and deeply moving",
    "a great story told brilliantly",
    "fantastic visuals and amazing score",
    "loved the characters and the plot",
    "superb cinematography and great acting",
    "an excellent and enjoyable watch",
    "brilliant performances throughout",
    "outstanding film highly recommended",
    "i loved the wonderful atmosphere",
    "a fantastic journey from start to end",
    "amazing how great this film is",
    "enjoyed the brilliant script",
    "truly wonderful and moving experience",
    "great film with excellent characters",
    "superb acting and a fantastic story",
    "an absolutely wonderful movie",
    "loved the outstanding direction",
    "brilliant film i really enjoyed it",
    "a great and amazing experience",
    "i found this film truly excellent",
    "wonderful performances and great pacing",
    "a fantastic and superb achievement",
    "loved it brilliant from beginning to end",
    "amazing story and excellent execution",
    "great script and wonderful acting",
    "enjoyed every scene it was brilliant",
    "outstanding and amazing in every way",
    "a superb film loved every moment",
    "excellent and wonderful in equal measure",
]
negatives = [
    "this is a terrible movie",
    "i really hated the film",
    "what a dreadful performance",
    "boring from the very first scene",
    "truly awful storytelling",
    "absolutely disappointing experience",
    "a bad and tedious film",
    "the acting was terrible",
    "a poor piece of cinema",
    "awful direction and bad writing",
    "i hated this film entirely",
    "dull and deeply boring",
    "a terrible story told badly",
    "dreadful visuals and awful score",
    "hated the characters and the plot",
    "poor cinematography and terrible acting",
    "a bad and disappointing watch",
    "awful performances throughout",
    "worst film do not recommend",
    "i hated the dull atmosphere",
    "a terrible journey from start to end",
    "disappointing how bad this film is",
    "hated the awful script",
    "truly dreadful and boring experience",
    "bad film with terrible characters",
    "poor acting and a dreadful story",
    "an absolutely awful movie",
    "hated the disappointing direction",
    "terrible film i really hated it",
    "a bad and dull experience",
    "i found this film truly awful",
    "disappointing performances and poor pacing",
    "a dreadful and terrible achievement",
    "hated it boring from beginning to end",
    "awful story and bad execution",
    "terrible script and dull acting",
    "hated every scene it was awful",
    "worst and disappointing in every way",
    "a terrible film hated every moment",
    "bad and dreadful in equal measure",
]

texts = positives + negatives
labels = np.array([1] * len(positives) + [0] * len(negatives))

# 打乱顺序，使验证集包含两个类别的均衡混合
idx = np.random.permutation(len(texts))
texts = np.array(texts)[idx]
labels = labels[idx]

print(f"样本数量: {len(texts)}（{labels.sum()} 个正面，{(1-labels).sum()} 个负面）")
print(f"示例文本: '{texts[0]}', 标签: {labels[0]}")
```

### 准备文本数据

序列模型不直接处理原始文本。我们需要将句子转换为模型可以处理的数值表示。这涉及两个主要步骤：词元 (token)化和填充。

1. **词元化：** 我们为数据集（即我们的词汇表 (vocabulary)）中的每个不同词汇分配一个唯一的整数索引。
2. **填充：** 由于RNN逐步处理序列，它们通常要求输入序列具有统一的长度。我们通过向较短序列添加特殊的“填充”词元（通常用0表示）来实现这一点，直到它们匹配最长序列的长度（或预定义的最大长度）。

```python
# --- 词元化 ---
vocab_size = 500 # 根据词频保留的最大词汇数量
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>") # <OOV> 用于表示词汇表外词汇
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)

print("\n词汇索引示例:", list(word_index.items())[:10])
print("原始文本:", texts[0])
print("序列表示:", sequences[0])

# --- 填充 ---
max_length = 10 # 定义最大序列长度（可推断或设定）
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

print("\n填充序列示例（后置填充）:")
print(padded_sequences[0])
print("填充序列的形状:", padded_sequences.shape)
```

注意 `pad_sequences` 如何在末尾添加零（`padding='post'`）以使所有序列长度为10。如果序列长度超过 `max_length`，则会被截断（`truncating='post'`）。

### 构建序列模型

现在，让我们构建模型。我们将使用Keras的`Sequential` API，它允许我们线性堆叠层。

1. **嵌入 (embedding)层：** 这是第一层。它接收整数编码的词汇表 (vocabulary)，并为每个词查找对应的嵌入向量 (vector)。这些嵌入在训练期间被学习。它需要 `input_dim`（词汇表大小）和 `output_dim`（嵌入向量的维度）。我们还指定了 `input_length`，它对应于我们填充操作中的 `max_length`。
2. **循环层：** 这是我们序列模型的核心。我们将从 `SimpleRNN` 开始。主要参数 (parameter)是 `units`，它定义了隐藏状态（和输出空间）的维度。其他循环层，如 `LSTM` 或 `GRU`，可以在此处替换使用。
3. **全连接输出层：** 由于这是一个二分类问题（正面/负面），我们需要一个带有单个单元和 `sigmoid` 激活函数 (activation function)的最终 `Dense` 层。`sigmoid` 函数输出一个介于0和1之间的值，表示正面类别的概率。

```python
embedding_dim = 16 # 词嵌入的维度
rnn_units = 32    # RNN层中的单元数量

model = Sequential([
    # 1. 嵌入层
    Embedding(input_dim=vocab_size,
              output_dim=embedding_dim),

    # 2. 循环层 (SimpleRNN)
    # 稍后尝试将 SimpleRNN 替换为 LSTM 或 GRU！
    SimpleRNN(units=rnn_units),
    # 如果堆叠RNN层，在中间层使用 return_sequences=True:
    # SimpleRNN(units=rnn_units, return_sequences=True),
    # SimpleRNN(units=rnn_units), # 最后一层RNN不需要 return_sequences=True

    # 3. 输出层
    Dense(units=1, activation='sigmoid')
])

# 显示模型的架构
model.summary()
```

摘要显示了层、它们的输出形状以及可训练参数的数量。注意 `SimpleRNN` 层如何输出形状为 `(None, 32)` 的单个向量，其中32是 `rnn_units`。如果设置了 `return_sequences=True`，则输出形状将是 `(None, max_length, rnn_units)`。

### 编译模型

在训练之前，我们需要使用 `model.compile()` 配置学习过程。这涉及指定：

- **优化器：** 用于更新模型权重 (weight)的算法（例如，Adam、RMSprop、SGD）。Adam通常是一个好的默认选择。
- **损失函数 (loss function)：** 衡量模型在训练数据上的表现。对于带有 sigmoid 输出的二分类，`binary_crossentropy` 是合适的。
- **指标：** 用于监控训练和测试步骤。对于分类任务，`accuracy` 是一个常用指标。

```python
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n模型编译成功。")
```

### 训练模型

现在，我们使用准备好的数据训练模型。我们将填充后的序列作为输入 (`X`)，并提供相应的标签 (`y`)。

- **epochs（周期）：** 模型将遍历整个训练数据集的次数。
- **batch\_size（批大小）：** 在模型权重 (weight)更新之前处理的样本数量。
- **validation\_split（验证分割）：** 可选地，留出一部分训练数据，在每个周期结束时评估损失和指标。这有助于监控过拟合 (overfitting)。

```python
num_epochs = 30
batch_size = 8
validation_fraction = 0.2 # 使用20%的数据进行验证

print(f"\n开始训练，共 {num_epochs} 个周期...")

history = model.fit(padded_sequences,
                    labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_split=validation_fraction,
                    verbose=1) # 设置 verbose=0 可隐藏周期进度

print("\n训练完成。")
```

在训练期间，Keras会在每个周期之后打印训练集和验证集（如果提供）的损失和准确率。

### 可视化训练历史

绘制训练和验证损失以及准确率随周期的变化曲线，是评估模型学习进展和检查过拟合 (overfitting)的标准方式。当模型在训练数据上表现良好，但在未见的验证数据上表现不佳时（即训练损失降低而验证损失增加），就会发生过拟合。

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 提取历史数据
acc = history.history['accuracy']
val_acc = history.history.get('val_accuracy') # 使用 .get() 以防 validation_split 为 0
loss = history.history['loss']
val_loss = history.history.get('val_loss')
epochs_range = range(1, num_epochs + 1)

# 创建带子图的图形
fig = make_subplots(rows=1, cols=2, subplot_titles=('训练和验证准确率', '训练和验证损失'))

# 添加准确率轨迹
fig.add_trace(go.Scatter(x=list(epochs_range), y=acc, name='训练准确率', mode='lines+markers', marker_color='#1f77b4'), row=1, col=1)
if val_acc:
    fig.add_trace(go.Scatter(x=list(epochs_range), y=val_acc, name='验证准确率', mode='lines+markers', marker_color='#ff7f0e'), row=1, col=1)

# 添加损失轨迹
fig.add_trace(go.Scatter(x=list(epochs_range), y=loss, name='训练损失', mode='lines+markers', marker_color='#1f77b4'), row=1, col=2)
if val_loss:
    fig.add_trace(go.Scatter(x=list(epochs_range), y=val_loss, name='验证损失', mode='lines+markers', marker_color='#ff7f0e'), row=1, col=2)

# 更新布局
fig.update_layout(
    height=400,
    width=800,
    xaxis_title='周期',
    yaxis_title='准确率',
    xaxis2_title='周期',
    yaxis2_title='损失',
    legend_title_text='指标',
    margin=dict(l=20, r=20, t=50, b=20) # 调整边距
)
# 显示图表（在支持 Plotly 渲染的环境中）
# fig.show() # 如果配置了 Plotly，取消注释以在本地显示

# 或者提供用于网页嵌入的 JSON 表示
plotly_json = fig.to_json()
```



![Interactive chart](plots/2509-0.json)



> 训练和验证准确率及损失曲线随训练周期变化的图示。

在这个数据易于分离的简单例子中，训练准确率迅速达到1.0（100%），而验证准确率稳定在93–94%左右，且损失曲线的明显分叉显示出明显的过拟合迹象。在更真实的数据集上，你预期会看到更平缓的准确率提升和不那么显著的过拟合现象。

### 进行预测

最后，让我们看看如何使用训练好的模型来预测新的、未见文本的情感。请记住对新数据应用相同的预处理步骤（词元 (token)化和填充）。

```python
new_texts = [
    "it was truly great",
    "a complete waste of time",
    "amazing film loved it"
]

# 预处理新文本
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding='post', truncating='post')

print("\n新的填充序列:")
print(new_padded)

# 获取预测结果（概率）
predictions = model.predict(new_padded)
print("\n原始预测结果（概率）:")
print(predictions)

# 解释预测结果（阈值为0.5）
predicted_labels = (predictions > 0.5).astype(int).flatten() # flatten 将 [[0],[1]] 转换为 [0,1]

print("\n预测标签（0=负面, 1=正面）:")
for text, label in zip(new_texts, predicted_labels):
    sentiment = "正面" if label == 1 else "负面"
    print(f"'{text}' -> {sentiment}")
```

输出显示了模型分配给正面类别的概率（值越接近1表示正面情感，越接近0表示负面情感）以及基于0.5阈值的最终预测标签。

### 实验与后续步骤

本例提供了一个基本框架。建议你进行实验：

1. **替换循环层：** 在模型定义中将 `SimpleRNN` 替换为 `LSTM` 或 `GRU`。观察训练速度或最终性能是否存在差异（尽管此数据集过于简单，难以看出与梯度消失相关的显著差异）。

   ```python
   # 使用LSTM的示例
   # model = Sequential([
   #     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
   #     LSTM(units=rnn_units), # 将 SimpleRNN 替换为 LSTM
   #     Dense(units=1, activation='sigmoid')
   # ])
   ```
2. **调整超参数 (parameter) (hyperparameter)：** 更改 `embedding_dim`、`rnn_units`、`learning_rate`、`batch_size` 或 `num_epochs` 并重新训练模型。
3. **堆叠层：** 尝试堆叠多个循环层（记住在除最后一个循环层之外的所有层上设置 `return_sequences=True`）。
4. **尝试不同的任务：** 调整结构以适应不同的序列任务，或许是多类别分类，甚至是简单的字符级生成模型（尽管那需要更重大的更改）。
5. **使用真实数据：** 将此过程应用于更大、更真实的数据集，例如 `tensorflow_datasets` 中提供的IMDB电影评论数据集。

本次实践练习展示了构建和训练用于文本分类的简单序列模型的端到端过程。你现在已经掌握了基础代码结构，可以处理使用RNN、LSTM或GRU进行更复杂的序列处理任务。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖了深度学习的数学和概念基础，包括循环神经网络、词嵌入和训练方法。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）网络架构的原始论文，这是解决循环神经网络中梯度消失问题的重要进展。
- [CS224n: Natural Language Processing with Deep Learning (Course Materials)](https://web.stanford.edu/class/cs224n/) — Stanford University (2025)
  斯坦福大学的公开课程材料，提供了深度学习方法应用于自然语言处理的全面讲座和作业，包括序列模型及其实际实现。
