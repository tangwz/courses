# 用于RNN的序列数据准备

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/preparing-sequence-data-rnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

循环神经网络 (neural network) (RNN)按顺序处理信息，并保持一个内部状态或记忆。这种设计使它们非常适合处理涉及序列的任务，例如文本分析或时间序列预测。然而，与其它神经网络一样，RNN对数值数据进行操作，特别是张量。原始序列数据，无论是充满词语的句子，还是随时间变化的温度读数列表，都需要在RNN处理之前转换为这种数值格式。这一准备步骤对于模型训练成功来说极其重要。

让我们来看看准备两种主要序列数据类型的常见方法：文本和时间序列。

## 准备文本数据

文本数据需要经过几个步骤，才能将句子或文档转换为适合`SimpleRNN`、`LSTM`或`GRU`等RNN层的数值序列。

### 分词 (tokenization)

第一步是**分词**，它涉及将文本分解成称为词元 (token)的较小单元。这些词元通常是单词，但根据任务和所需的细粒度，它们也可以是字符或子词 (subword)单元。

例如，句子“Keras makes deep learning easy.”可以被分词为单词：`["Keras", "makes", "deep", "learning", "easy", "."]`。

Keras 提供了一些实用工具，例如在`keras.preprocessing.text`中（或者在与TensorFlow/PyTorch后端集成的较新Keras版本中可能是`keras.layers.TextVectorization`）找到的`Tokenizer`类，用于高效处理此过程。

```python
from keras.preprocessing.text import Tokenizer

sentences = [
    "Keras makes deep learning easy.",
    "RNNs are great for sequences."
]

# 初始化分词器
# num_words 限制词汇表大小为最常出现的词
tokenizer = Tokenizer(num_words=100)

# 根据句子构建词汇表
tokenizer.fit_on_texts(sentences)

# 分词器现在拥有一个 word_index 字典
print(tokenizer.word_index)
# 输出（示例）：{'keras': 1, 'makes': 2, 'deep': 3, 'learning': 4, 'easy': 5,
#                   'rnns': 6, 'are': 7, 'great': 8, 'for': 9, 'sequences': 10}
```

### 词汇表 (vocabulary)构建和整数编码

分词 (tokenization)后，我们需要构建一个**词汇表**：将每个词元 (token)（单词）唯一映射到整数索引。`Tokenizer` 在 `fit_on_texts` 步骤中自动处理此项。`word_index` 属性存储此映射。

一个常见做法是保留索引0用于填充（接下来讨论），并可能保留另一个索引用于“词汇表外”（OOV）词元。这些是在处理过程中遇到的、未出现在用于构建词汇表的训练文本中的词语。

拟合分词器 (tokenizer)后，您可以使用 `texts_to_sequences` 方法将文本序列转换为整数序列。

```python
# 将句子转换为整数序列
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# 输出（基于之前的示例）：[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
# 注意：标点符号的处理方式可能因分词器设置而异。
```

### 填充序列

“循环神经网络 (neural network) (RNN)，尤其是在为了提高效率而批量处理时，要求输入序列具有统一的长度。然而，句子或文档很少具有相同数量的单词。我们通过使用**填充**来解决这个问题。”

填充涉及向较短的序列添加一个特殊的填充词元 (token)（通常用整数0表示），直到它们达到指定的`最大长度（maxlen）`。您可以在序列的开头（`pre`）或结尾（`post`）添加填充。后置填充通常更受青睐，因为它允许RNN首先处理序列的初始部分，而不会被填充值所主导。

Keras 为此提供了 `pad_sequences` 实用工具。

```python
from keras.preprocessing.sequence import pad_sequences

# 假设 'sequences' 是上一步得到的整数序列列表
# [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

# 定义最大序列长度（可以从数据中确定或手动设置）
maxlen = 8

# 填充序列
padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')
print(padded_sequences)
# 输出：
# [[ 1  2  3  4  5  0  0  0]
#  [ 6  7  8  9 10  0  0  0]]
```

这些填充后的整数序列现在已接近准备完成。通常，模型架构内的下一步是使用 `Embedding` 层。该层以这些整数序列作为输入，并将每个整数索引转换为固定大小（嵌入 (embedding)维度）的密集向量 (vector) (dense vector)。这些密集向量捕捉词语之间的语义关系，并在模型训练期间进行学习。我们在此不详细说明 `Embedding` 层，但了解这些填充后的整数序列是大多数使用RNN的自然语言处理模型中此类层的直接输入，这是很重要的。

## 准备时间序列数据

时间序列数据，例如股票价格、传感器读数或随时间变化的 L气象测量值，需要不同的准备方法。

### 归一化 (normalization)或标准化

神经网络 (neural network)通常在输入数值数据特征被缩放到标准范围时表现更好。特征之间的大幅波动或不同尺度会阻碍学习过程。常见方法包含：

1. **归一化（最小-最大缩放）**：将特征重新缩放到固定范围，通常是 [0, 1] 或 [-1, 1]。公式为：
   
   $$
   X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}
   $$
   
2. **标准化（Z-分数归一化）**：将特征重新缩放以使其具有零均值 ($\mu=0$) 和单位方差 ($\sigma=1$)。公式为：
   
   $$
   X_{std} = \frac{X - \mu}{\sigma}
   $$
   

您应该仅在*训练*数据上拟合缩放器（例如，Scikit-learn 中的 `MinMaxScaler` 或 `StandardScaler`），然后使用拟合好的缩放器转换训练和验证/测试数据。这可以防止信息从验证/测试集泄漏到训练过程中。

### 窗口化（创建序列）

循环神经网络 (neural network) (RNN)从序列中学习。对于时间序列，我们需要将连续数据重构为固定长度的输入序列（窗口）和相应的目标值。

设想一个时间序列 $[x_1, x_2, x_3, x_4, x_5, x_6, ...]$。我们可以通过使用一个过去值的窗口来预测未来值，从而创建监督学习 (supervised learning)示例。如果我们选择3个时间步的窗口大小（或回溯期）来预测下一个时间步的值，我们将生成如下对：

- 输入序列：$[x_1, x_2, x_3]$，目标：$x_4$
- 输入序列：$[x_2, x_3, x_4]$，目标：$x_5$
- 输入序列：$[x_3, x_4, x_5]$，目标：$x_6$
- ……依此类推。

这一过程本质上将时间序列预测问题转换为一个监督学习问题，在该问题中，模型学习将一系列过去的观测值映射到未来的观测值。

> 使用大小为3的回溯窗口，从时间序列创建输入序列（窗口）和相应目标。

您可以使用简单的Python循环或更高效地使用NumPy或Pandas等库来实现此窗口化逻辑。

## 为Keras RNN层构造数据

准备好后（对于文本数据：分词 (tokenization)/编码/填充；对于时间序列数据：归一化 (normalization)/窗口化），数据需要正确整形以适应Keras RNN层。这些层通常期望输入数据采用3D张量格式：

`(batch_size, timesteps, features)`

让我们分解一下：

- `batch_size`：在训练或推理 (inference)期间一次性处理的序列数量。在使用`fit`等方法时，Keras通常会隐式处理此维度。
- `timesteps`：每个序列的长度。
  - 对于填充的文本数据，这是填充时使用的`最大长度（maxlen）`。
  - 对于窗口化的时间序列数据，这是回溯窗口的大小。
- `features`：每个时间步表示输入的特征数量。
  - 对于进入`Embedding`层的整数编码文本数据，这通常是1（整数索引本身）。`Embedding`层随后将其扩展到嵌入 (embedding)维度。
  - 对于时间序列数据，这是在每个时间点测量的不同变量的数量（例如，对于像温度这样的单变量序列是1，对于像温度、压力和湿度这样的多变量序列是多个）。

例如：

- 一个包含32个填充文本序列的批次，每个序列长50个词元 (token)，准备用于`Embedding`层，其形状将是`(32, 50, 1)`。（尽管通常会省略最后一个维度，Keras会推断它，或者`Embedding`层直接处理`(32, 50)`的形状）。
- 一个包含64个时间序列窗口的批次，每个窗口回溯10个时间步，每个时间步测量3个特征，其形状将是`(64, 10, 3)`。

理解这种预期的输入形状在构建Keras模型和确保数据管道正确馈送RNN层时非常重要。通过分词、编码、填充、归一化和窗口化正确准备序列数据，为训练有效的循环神经网络 (neural network) (RNN)奠定基础。

## 参考资料

- [Text processing with the TextVectorization layer](https://www.tensorflow.org/text/tutorials/basic_text_classification) — TensorFlow Developers (2024)
  Keras官方文档，解释了使用TextVectorization层进行文本分词、词汇表构建和整数编码。
- [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras作者撰写的实用指南，涵盖RNN的文本和时间序列数据准备，包括分词、填充和窗口化。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本综合性学术教科书，提供文本分词、词汇表创建和语言数值表示的基础知识。
- [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/) — Rob J Hyndman and George Athanasopoulos (2021)
  Publisher: OTexts
  一本关于时间序列分析的权威指南，涵盖数据转换、缩放和预测建模的数据结构，与窗口化技术相关。

---

[上一节](07-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BARNN-LSTM%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.md)
