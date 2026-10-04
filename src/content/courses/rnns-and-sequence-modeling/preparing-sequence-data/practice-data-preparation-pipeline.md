---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "practice-data-preparation-pipeline"
sourceId: 2628
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/practice-data-preparation-pipeline"
title: "实践：数据准备流程"
description: "为文本数据集构建一个可复用的数据准备流程。"
order: 10
plots: []
sourceHash: "3a5556b9c2857e0e99d9abfa895dc2396bad829bc792d77493f7ee27d842ea71"
sourceCorrections: []
---

文本预处理步骤被整合为一个可复用的流程。构建一个结构化的过程对于在训练、验证和新预测数据上一致地应用相同的转换非常有益。为文本数据创建一个流程是重点，它将原始句子转换为带填充的整数序列，使其适用于嵌入 (embedding)层以及随后的LSTM或GRU等循环层。

### 定义流程目标

我们的目标是创建一个函数或类，它接收一个原始文本文档列表（例如，句子或段落），并输出一个数值张量，其中每个文档都表示为一个整数序列，并填充到统一长度。

### 主要工具

我们将主要使用常见深度学习 (deep learning)框架中可用的工具。例如，TensorFlow/Keras 提供了 `tensorflow.keras.preprocessing.text.Tokenizer` 类和 `tensorflow.keras.preprocessing.sequence.pad_sequences` 函数，它们能处理大部分繁重工作。如果您使用 PyTorch，`torchtext` 等库提供类似功能。在此示例中，我们将展示如何使用 TensorFlow/Keras 工具。

### 流程步骤

我们的文本准备流程包含的主要步骤是：

1. **拟合分词 (tokenization)器 (tokenizer)：** 分析文本语料库以构建词汇表 (vocabulary)，并将词语映射到唯一的整数索引。此步骤通常仅在训练数据上执行，以避免验证集或测试集的数据泄露。
2. **整数编码：** 基于已拟合的词汇表，将每个文本文档转换为其对应的整数索引序列。
3. **填充序列：** 通过在较短序列的开头（'pre'）或结尾（'post'）添加填充标记 (token)（通常为0），确保所有整数序列具有相同长度。较长的序列可能会被截断。

以下是流程图：

> 文本数据准备流程将原始文本转换为固定大小的数值张量，适用于RNN输入。

### 实现流程

我们来逐步实现使用 TensorFlow/Keras 的过程。

#### 1. 示例数据

首先，我们定义一些要处理的示例文本数据。

```python
# 文本文档示例语料库
corpus = [
    "Recurrent networks process sequences.",
    "LSTMs handle long dependencies.",
    "Padding makes sequences uniform.",
    "Embeddings represent words numerically."
]

# 稍后使用*相同*的已拟合流程处理的新数据
new_data = [
    "GRUs are another type of recurrent network.",
    "Uniform sequence length is needed for batching."
]
```

#### 2. 分词 (tokenization)与词汇表 (vocabulary)构建

我们初始化一个 `Tokenizer` 并在 `corpus` 上拟合它。`num_words` 参数 (parameter)将词汇表大小限制为最常用的词。设置 `oov_token` 确保稍后遇到的、不在初始拟合语料库中的词语被分配一个特殊的“词汇外”标记 (token)。

```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 配置
vocab_size = 20  # 要保留的最大词数
oov_tok = "<OOV>" # 词汇外词语的标记

# 初始化并拟合分词器
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(corpus)

# 显示学习到的词语索引（词汇表）
word_index = tokenizer.word_index
print("词语索引:")
# 为简洁起见，显示一个子集
print({k: word_index[k] for k in list(word_index)[:15]})
```

此拟合过程构建内部词汇表。`word_index` 包含词语到整数的映射。请注意，`num_words` 控制编码期间使用的词汇表大小，而不是`word_index`本身的长度。分词器 (tokenizer)保留索引0并将OOV标记分配给索引1。

#### 3. 整数编码

现在，我们使用已拟合的 `tokenizer` 将文本句子转换为整数序列。

```python
# 将语料库转换为整数序列
sequences = tokenizer.texts_to_sequences(corpus)
print("\n原始语料库整数序列:")
for seq in sequences:
    print(seq)
```

每个整数列表都对应 `corpus` 中的一个句子，使用上一步学习到的索引。

#### 4. 填充序列

序列目前长度不同。我们使用 `pad_sequences` 使其统一。`maxlen` 定义目标长度。`padding='post'` 在末尾添加填充（0），而 `truncating='post'` 在序列过长时从末尾移除元素。'pre' 也是一个常见选择，特别是对于LSTMs/GRUs，因为它将最新信息保留在序列的末尾。

```python
# 填充配置
max_sequence_len = 10 # 定义最大长度
padding_type = 'post'
truncation_type = 'post'

# 填充序列
padded_sequences = pad_sequences(sequences,
                                 maxlen=max_sequence_len,
                                 padding=padding_type,
                                 truncating=truncation_type)

print("\n填充后的语料库序列（张量形状：{}）:".format(padded_sequences.shape))
print(padded_sequences)
```

输出 `padded_sequences` 现在是一个形状为 `(number_of_samples, max_sequence_len)` 的 NumPy 数组（或 TensorFlow 张量）。这是您的 RNN 模型中嵌入 (embedding)层期望的格式。索引 0 是默认的填充值。

#### 5. 处理新数据

此流程的一个重要优点是，可以使用*已拟合*的分词器，对新的、未见过的数据应用*相同*的转换。

```python
# 使用*相同*的分词器将新数据转换为整数序列
new_sequences = tokenizer.texts_to_sequences(new_data)
print("\n新数据整数序列:")
for seq in new_sequences:
    print(seq) # 请注意原始拟合中不存在的词语的 OOV 标记（索引 1）

# 使用*相同*的参数填充新序列
new_padded_sequences = pad_sequences(new_sequences,
                                     maxlen=max_sequence_len,
                                     padding=padding_type,
                                     truncating=truncation_type)

print("\n填充后的新数据序列（张量形状：{}）:".format(new_padded_sequences.shape))
print(new_padded_sequences)
```

请注意，“grus”或“batching”等词语，如果不在原始 `corpus` 中，会被映射到 `oov_token` 索引（1）。这确保了处理的一致性。

### 封装流程

为了更好的可复用性，您可以将这些步骤封装到一个函数或类中。以下是函数式方法：

```python
def create_text_pipeline(training_texts, max_vocab_size=10000, oov_token="<OOV>"):
    """在训练文本上拟合分词器。"""
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_texts)
    return tokenizer

def preprocess_texts(texts, tokenizer, max_len, padding='post', truncating='post'):
    """使用已拟合的分词器应用分词和填充。"""
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding=padding, truncating=truncating)
    return padded

# --- 用法示例 ---
# 1. 在训练数据上创建流程（拟合分词器）
train_corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
pipeline_tokenizer = create_text_pipeline(train_corpus, max_vocab_size=100)
pipeline_maxlen = 10 # Set based on analysis or requirement

# 2. 处理训练数据
train_padded = preprocess_texts(train_corpus, pipeline_tokenizer, pipeline_maxlen)
print("\n--- 流程使用示例 ---")
print("已处理训练数据形状:", train_padded.shape)
# print(train_padded) # （可选）打印数组

# 3. 处理新/验证/测试数据
validation_data = ["This is another document to process."]
validation_padded = preprocess_texts(validation_data, pipeline_tokenizer, pipeline_maxlen)
print("已处理验证数据形状:", validation_padded.shape)
print(validation_padded)
```

### 最终考量

- **模型输入：** `padded_sequences` 张量现在已准备就绪。模型中通常的下一步是 `Embedding` 层。此层将接收这些整数序列，并将每个整数转换为一个密集向量 (vector) (dense vector)表示。如果您使用填充值 0（默认值），请在 Keras 中配置 `Embedding` 层时使用 `mask_zero=True`，或确保随后的 RNN 层能适当地处理掩码，以便模型学会忽略这些填充步骤。
- **词汇表 (vocabulary)大小和 `maxlen`：** 选择 `vocab_size` 和 `max_sequence_len` 涉及权衡。较大的值能捕获更多信息，但会增加模型大小、内存使用和计算量。请分析您的数据（例如，序列长度分布）以做出明智选择。
- **填充/截断策略：** 'pre' 与 'post' 填充/截断有时会影响性能，特别是对于近期信息更重要的任务。可能需要进行实验。
- **集成：** 无论何时向模型输入数据——在训练、评估和预测期间——都应一致地应用此预处理逻辑。`tf.data` 或 PyTorch `DataLoader` 等框架有助于将此类预处理步骤高效地集成到您的数据加载流程中。

此实用流程提供了一种标准且可重复的方式来准备您的文本数据，在训练 LSTM 和 GRU 等复杂序列模型之前，这是一个基本步骤。

## 参考资料

- [Keras preprocessing layers and utilities](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing) — TensorFlow Developers (2024)
  Publisher: Google
  关于使用 Keras 工具（如 `Tokenizer` 和 `pad_sequences`）进行文本数据准备的指南。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一份实用指南，展示了如何使用 Keras/TensorFlow 进行文本预处理、分词和填充技术。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  广泛涵盖了基础文本处理组件，包括分词、词汇表创建和序列表示。
