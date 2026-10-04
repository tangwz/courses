---
course: "julia-deep-learning"
chapter: "constructing-nn-architectures"
lesson: "embeddings-sequential-data-flux"
sourceId: 6907
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/embeddings-sequential-data-flux"
title: "处理序列数据的嵌入"
description: "学习在 Flux.jl 中使用嵌入层，以在神经网络中表示分类数据或文本数据。"
order: 6
plots: []
sourceHash: "93f34da03ad7fd134d6e4211f3bcc28a75e3183a6e2010327fb90eb830186b54"
sourceCorrections: []
---

处理序列数据时，例如自然语言处理中的文本或时间序列预测，原始数据元素（如词语或离散时间步）不能直接被神经网络 (neural network)使用。这些网络需要数值输入。虽然独热编码是将分类数据转换为向量 (vector)的直接方法，但它经常导致非常高维和稀疏的表示，特别是当词汇量很大时（例如，数万个不重复的词）。嵌入 (embedding)层提供了一种更高效、更强大的替代方法，它能将这些离散项转换为紧密的、低维的、并且更为重要的——*学习到的*向量表示。

这些学习到的向量，即嵌入，可以捕获词汇表 (vocabulary)中各项之间的语义关系。例如，在一个训练得好的语言模型中，含义相近的词可能在向量空间中具有彼此接近的嵌入向量。这种以紧凑形式表示含义的能力，对于处理序列数据的模型的性能来说很重要。

### `Flux.Embedding` 层

Flux.jl 提供 `Embedding` 层（通过 `Flux.Embedding` 调用），它是将离散项目转换为紧密、低维向量 (vector)表示的主要工具。初始化 `Embedding` 层时，通常需要指定两个主要参数 (parameter)：

1. `vocab_size`：你的词汇表 (vocabulary)中不重复项的总数（例如，如果你正在处理文本，就是不重复词语的数量）。
2. `embed_dim`：嵌入 (embedding)向量的目标维度。这是一个你选择的超参数 (hyperparameter)；常见的值在 50 到 300 之间，对于非常大的词汇量和复杂的任务，甚至可能更高。

在内部，`Embedding` 层本质上是一个查找表。它维护一个大小为 `vocab_size × embed_dim` 的权重 (weight)矩阵。当你向此层传入一个整数（表示词汇表中某一项的索引）时，它会查找该矩阵中对应的行，该行即成为输入项的嵌入向量。如果你传入一个整数序列，它会返回一个对应的嵌入向量序列。

让我们看一个简单的例子：

```julia
using Flux

# 假设词汇表中有 100 个不重复的词
# 我们希望将每个词表示为 10 维向量
vocab_size = 100
embed_dim = 10

# 创建嵌入层
embedding_layer = Flux.Embedding(vocab_size, embed_dim)

# 例子：获取索引为 5、12 和 1 的词的嵌入
# 输入应该是整数或整数数组
input_indices = [5, 12, 1]
output_vectors = embedding_layer(input_indices)

println("输出形状: ", size(output_vectors))
# 预期输出：输出形状：(10, 3)
# 这表示我们获得了三个 10 维向量，每个对应一个输入索引。
# Flux 通常在第一维输出特征。
```

这个查找表（嵌入矩阵）的权重是随机初始化的，就像其他神经网络 (neural network)层中的权重一样。在训练过程中，这些权重通过反向传播 (backpropagation)进行调整，以使整个模型的损失函数 (loss function)最小化。这使得网络能够根据当前任务，为词汇表中的项学习有意义的表示。

### 为嵌入 (embedding)层准备数据

在使用 `Embedding` 层之前，你的序列数据（例如句子）需要转换为整数索引序列。这个过程通常包含：

1. **分词 (tokenization)**: 将输入序列分解为独立的单元或“词元 (token)”。对于文本，这通常意味着将句子拆分为词语或子词 (subword)单元。
2. **词汇表 (vocabulary)创建**: 构建一个字典（或映射），为训练数据集中的每个词元分配一个不重复的整数索引。通常做法是为未知词元（词汇表外或 OOV 词元）预留一个索引，并可能为填充预留另一个索引。
3. **整数编码**: 将输入序列中的每个词元替换为其在词汇表中对应的整数索引。

例如，句子“julia is fast”可能被分词为 `["julia", "is", "fast"]`。如果你的词汇表将“julia”映射到 7，“is”映射到 4，“fast”映射到 22，那么整数编码序列将是 `[7, 4, 22]`。这是 `Embedding` 层所期望的输入类型。

### 将嵌入 (embedding)整合到网络架构中

嵌入层最常被用作处理序列数据的网络中的第一层，特别是循环神经网络 (neural network)（RNN）、长短期记忆网络（LSTM）或门控循环单元（GRU）。`Embedding` 层将整数编码序列转换为紧密向量 (vector)序列，然后作为后续循环层的输入。

这是一个说明此流程的图表：

> 输入整数通过 `Embedding` 层转换为紧密向量，然后由循环层处理。

在 Flux.jl 中，你可以使用 `Chain` 来组合这些层：

```julia
using Flux

vocab_size = 10000 # 示例词汇量大小
embed_dim = 128   # 嵌入维度
hidden_dim = 256  # LSTM 隐藏层维度
output_dim = 10   # 示例输出维度（例如，用于 10 类分类）

model = Chain(
    Flux.Embedding(vocab_size, embed_dim),          # 输出：(嵌入维度, 序列长度, 批次大小)
    Flux.LSTM(embed_dim, hidden_dim),               # 输出：(隐藏层维度, 序列长度, 批次大小)
    # 对于每个序列获得一个输出向量用于分类，
    # 通常会取 LSTM 的最后一个输出。
    # 这可以通过自定义层或函数完成，例如 x -> x[:, end, :]
    # 为简化起见，假设我们希望对每个时间步进行分类：
    # 如果不是，你可能需要一个层来选择最后一个时间步的输出
    # 例如，data -> data[:, end, :]
    # 之后是全连接层
    x -> reshape(x, hidden_dim, :), # 如果 LSTM 为单个序列步输出 3D，则进行展平
    Flux.Dense(hidden_dim, output_dim)
)

# 例子：一个包含 3 个序列的批次，每个序列长度为 5
sample_input_indices = rand(1:vocab_size, 5, 3) # (序列长度, 批次大小)
output = model(sample_input_indices)
println("模型输出形状: ", size(output))
# 根据 LSTM 输出的处理方式，这可能是 (输出维度, 元素数量)
# 或者 (输出维度, 批次大小)，如果只处理最后一步。
# 对于当前模型，如果 seq_len*batch_size 元素由 Dense 层独立处理：
# 模型输出形状：(10, 15)，如果使用 reshape(x, hidden_dim, :) 且 LSTM 输出所有步
# 或者 (10, 3)，如果 LSTM 只输出最后一个隐藏状态且是 2D。
# 如果输入是矩阵（特征 x 批次），Flux.LSTM 默认只输出最后一个隐藏状态
# 如果输入是向量序列，它输出向量序列。
# 对于序列批次 (Matrix{Int})，嵌入输出是 (嵌入维度, 序列长度, 批次大小)
# 那么 LSTM 输出是 (隐藏层维度, 序列长度, 批次大小)
# Reshape(x, hidden_dim, :) 将其展平为 (隐藏层维度, 序列长度*批次大小)
# 所以输出将是 (输出维度, 序列长度*批次大小)
```

在上面的例子中，`Flux.LSTM` 可以直接处理嵌入序列。LSTM 序列输出用于下游任务的具体处理方式（例如，取最后一个输出用于序列分类，或使用所有输出用于序列到序列任务）取决于具体问题，并且是你将遇到的一个常见模式。对于基于整个序列的分类任务，你可以在将其传递给 `Dense` 层之前，提取 LSTM 在最后时间步的输出。

### 优势与考量

使用嵌入 (embedding)提供多项优势：

- **降维**: 嵌入向量 (vector)是紧密的，并且维度远低于稀疏的独热编码向量，这使得后续计算更易于管理。
- **计算效率**: 下游层更小的输入表示意味着更少的参数 (parameter)和更快的训练。
- **语义表示**: 学习到的嵌入可以捕获词汇表 (vocabulary)中各项之间有意义的关系（如相似性、类比），这通常能提升模型的泛化能力。
- **参数共享**: 词汇表中的每个不重复项都有自己的嵌入向量，但该向量在输入中每次出现时都会被使用。

在使用嵌入时，请记住以下几点：

- **嵌入维度 (`embed_dim`)**: 这是一个重要的超参数 (hyperparameter)。更大的维度允许更具表达力的表示，但会增加模型大小和计算成本。维度过小可能无法捕获足够的信息。
- **词汇量大小 (`vocab_size`)**: 这直接影响嵌入矩阵的大小（$词汇量大小 \times 嵌入维度$）。大词汇量需要更多内存。
- **词汇表外（OOV）词元 (token)**: 数据中经常包含训练期间未曾见过的词元。一种常见策略是将所有 OOV 词元映射到一个特殊的“未知”词元（例如，`<UNK>`），该词元有其自己学习到的嵌入。
- **预训练 (pre-training)嵌入**: 对于自然语言处理等常见任务，你可以使用从海量文本语料库中学习到的预训练嵌入（例如 Word2Vec, GloVe, FastText）。这些可以提供良好的初始化，甚至可以用作固定特征，特别是在你自己的数据集较小时。Flux 可以支持将这些预训练权重 (weight)加载到 `Embedding` 层中。这个话题，连同迁移学习 (transfer learning)，将在第 5 章中更详细地讨论。

嵌入是现代深度学习 (deep learning)模型处理序列数据的重要组成部分。通过将离散符号转换为有意义的连续向量空间，它们使神经网络 (neural network)能够高效地处理和理解复杂的序列。当你构建更复杂的架构，如文本的 CNN 或高级 RNN 时，`Embedding` 层通常会是你处理符号输入的起始点。

## 参考资料

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013)
  Journal: International Conference on Learning Representations (ICLR 2013); DOI: [10.48550/arXiv.1301.3781](https://doi.org/10.48550/arXiv.1301.3781)
  介绍了Word2Vec模型，这是学习能够捕捉语义和句法关系的密集低维词嵌入的重要进展。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的深度学习教材，涵盖了理论基础和实际应用，包括嵌入在处理序列数据中的作用。
- [Layers: Flux.Embedding](https://fluxml.ai/Flux.jl/stable/layers/basic/#Flux.Embedding) — The Flux.jl Contributors (2024)
  官方文档，详细介绍了Flux.jl深度学习框架中`Flux.Embedding`层的用法、初始化和行为。
- [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/) — Jeffrey Pennington, Richard Socher, Christopher Manning (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1532-1543; DOI: [10.3115/v1/D14-1162](https://doi.org/10.3115/v1/D14-1162)
  介绍了GloVe，一种无监督模型，用于学习结合了全局矩阵分解和局部上下文窗口方法优势的全局词向量。
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Christopher Manning, Abigail See, John Hewitt, and others (2023)
  Publisher: Stanford University
  综合课程材料，包括讲座和作业，广泛涵盖词嵌入、循环神经网络及其在自然语言处理中的应用。
