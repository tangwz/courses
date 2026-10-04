---
course: "julia-deep-learning"
chapter: "constructing-nn-architectures"
lesson: "recurrent-neural-networks-lstms-flux"
sourceId: 6905
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/recurrent-neural-networks-lstms-flux"
title: "Flux 中的循环神经网络（RNN）与长短期记忆网络（LSTM）"
description: "使用 Flux.jl 开发循环神经网络，包括长短期记忆网络和门控循环单元，用于序列建模任务。"
order: 5
plots: []
sourceHash: "29d1c7486a6325f14aaf781f27c42a52c8c7759ff08a32b644e9e93fd4e42c65"
sourceCorrections: []
---

“虽然多层感知器（MLP）非常适合表格数据，卷积神经网络 (neural network)（CNN）擅长处理图像等网格状数据，但许多问题都涉及序列。比如，句子、随时间变化的股价或传感器读数。对于这些情况，我们需要能够理解时间或序列步骤中的顺序和前后关系的模型。这正是循环神经网络（RNN）及其长短期记忆（LSTM）单元等更高级变体发挥作用的地方。”

与前馈网络不同，循环神经网络具有循环结构，使信息能够从序列的一个步骤保留到下一个步骤。这种“记忆”使它们能够学习序列元素间的依赖关系。

### 核心思想：具有记忆的序列处理

循环神经网络 (neural network) (RNN)的核心是一个循环单元。该单元处理当前时间步（或序列位置）的输入，并将其与前一时间步的隐藏状态结合。这个隐藏状态充当网络的记忆，携带序列早期部分的信息。然后，该单元生成当前时间步的输出，并更新其隐藏状态，以便传递给下一个时间步。

> 一个循环神经网络单元处理当前输入和前一隐藏状态，以生成输出和更新的隐藏状态。

在 Flux.jl 中，您可以使用 `RNNCell` 定义一个基本的循环神经网络单元。对于处理整个序列，通常会用 `Recur` 来封装这个单元。

```julia
using Flux

# 定义输入特征大小和隐藏状态大小
input_size = 10
hidden_size = 20

# 创建一个基本的循环神经网络层
rnn_layer = Flux.RNN(input_size, hidden_size, σ) # σ 是激活函数，例如 tanh

# 示例输入：一个包含 5 个项目（每个项目有 10 个特征）的序列，批处理大小为 1
# Flux 对序列层期望的形状是 (特征数, 序列长度, 批处理大小)
# 或者，如果逐步处理，则是 (特征数, 批处理大小)
sample_sequence_batch = [rand(Float32, input_size) for _ in 1:5] # 矩阵向量（如果批处理大小为 1，则为向量）
# 对于批处理大小为 3 的序列，每个序列长度为 5，特征数为 10：
# sample_sequence_batch = [rand(Float32, input_size, 3) for _ in 1:5]

# 要处理单个步骤（如果您有 RNNCell）
# rnn_cell = Flux.RNNCell(input_size, hidden_size, tanh)
# initial_hidden_state = rnn_cell.state0(1) # 对于批处理大小为 1
# output_step1, next_hidden_state = rnn_cell(initial_hidden_state, sample_sequence_batch[1])

# 通过循环神经网络层处理整个序列
# 注意：循环神经网络层在处理序列时，内部管理隐藏状态。
# 要在每个步骤获取隐藏状态，您需要手动迭代或使用不同的方法。
output_sequence = rnn_layer.(sample_sequence_batch)
final_hidden_state = rnn_layer.state # 访问最终隐藏状态

# 为新的序列批次重置隐藏状态
Flux.reset!(rnn_layer)

println("最后一步的输出（批次中第一个项目）：", output_sequence[end][:, 1])
println("最终隐藏状态的形状：", size(final_hidden_state))
```

在处理整个序列时，为 Flux 的循环层（如 `RNN`、`LSTM` 或 `GRU`）构造输入的一种常见方式是使用矩阵向量 (vector)。向量中的每个矩阵代表所有批次的一个时间步，维度为 `(特征数, 批处理大小)`。向量本身的长度等于序列长度。或者，对于某些层或自定义循环，您可能会使用形状为 `(特征数, 序列长度, 批处理大小)` 的三维数组。

### 长期依赖的难点

简单的循环神经网络 (neural network) (RNN)，尽管设计巧妙，但难以学习长序列中的依赖关系。这是由于梯度消失或梯度爆炸问题。在反向传播 (backpropagation)过程中，梯度在通过许多时间步反向传播时会指数级缩小（消失）或指数级增长（爆炸）。梯度消失使网络难以学习序列中远距离元素之间的连接，而梯度爆炸可能导致训练不稳定。

### 长短期记忆（LSTM）网络

长短期记忆网络（LSTM）专门设计用于处理梯度消失问题并更好地捕获长期依赖。它们通过包含多个控制信息流动的“门”的更复杂单元结构来实现这一点。

一个长短期记忆单元除了隐藏状态（$h_t$）外，还保持一个*细胞状态*（$c_t$）。这个细胞状态就像一个传送带，使信息能够相对不变地流过，这有助于在长时间内保留梯度。这些门包括：

1. **遗忘门（$f_t$）：** 决定从细胞状态中丢弃哪些信息。它接收 $h_{t-1}$ 和 $x_t$，并为细胞状态 $c_{t-1}$ 中的每个数字输出一个介于 0 和 1 之间的值。1 表示“完全保留”，而 0 表示“完全丢弃”。
   $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. **输入门（$i_t$）：** 决定将哪些新信息存储到细胞状态中。这包括两个部分：
   - 输入门层（$i_t$）决定哪些值将被更新。
     $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
   - 一个 tanh 层创建一个新的候选值向量 (vector) $\tilde{C}_t$，这些值可以添加到状态中。
     $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
     这两者结合起来更新细胞状态：$c_t = f_t * c_{t-1} + i_t * \tilde{C}_t$。
3. **输出门（$o_t$）：** 决定将什么作为隐藏状态 $h_t$ 输出。输出基于细胞状态 $c_t$，但它是经过筛选的版本。
   - 首先，一个 sigmoid 层决定细胞状态的哪些部分将被输出。
     $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
   - 然后，细胞状态经过 $\tanh$（将值推向 -1 到 1 之间）并乘以 sigmoid 门的输出，这样只有之前决定的部分才会被输出。
     $h_t = o_t * \tanh(c_t)$

> 长短期记忆单元的简化结构，展示了门和细胞状态的交互。细胞状态（$c_t$）像传送带一样，被遗忘门和输入门修改。输出门筛选细胞状态以生成隐藏状态（$h_t$）。

在 Flux.jl 中，创建一个长短期记忆层简单直接：

```julia
using Flux

input_size = 10
hidden_size = 20 # 这是 h_t 和 c_t 的大小

# 创建一个长短期记忆层
lstm_layer = Flux.LSTM(input_size, hidden_size)

# 示例输入序列（由 5 个矩阵组成的向量，每个矩阵对应一个时间步）
# 每个矩阵：(特征数, 批处理大小)
# 这里，为了简化，批处理大小为 1
sample_sequence = [rand(Float33, input_size, 1) for _ in 1:5]

# 处理序列
output_lstm_sequence = lstm_layer.(sample_sequence)

# lstm_layer.state 包含一个元组 (h, c)，表示最终的隐藏状态和细胞状态
final_hidden_state_h, final_cell_state_c = lstm_layer.state

println("长短期记忆网络在最后一步的输出：", size(output_lstm_sequence[end]))
println("最终隐藏状态 (h) 的形状：", size(final_hidden_state_h))
println("最终细胞状态 (c) 的形状：", size(final_cell_state_c))

# 为下一个批次/序列重置
Flux.reset!(lstm_layer)
```

Flux 的 `LSTM` 层，与 `RNN` 类似，在处理表示为输入向量（每个元素对应一个时间步）的序列时，会自动管理状态。

### 门控循环单元（GRU）

门控循环单元（GRU）是 Cho 等人在 2014 年提出的一种新一代循环单元。它们与长短期记忆网络 (LSTM)类似，但具有更简单的架构，将遗忘门和输入门合并为一个“更新门”，并融合了细胞状态和隐藏状态。尽管结构简单，门控循环单元在许多任务上表现通常与长短期记忆网络相当，并且计算速度可能更快。

一个门控循环单元有两个主要门：

1. **重置门（$r_t$）：** 决定遗忘前一隐藏状态的多少信息。
2. **更新门（$z_t$）：** 决定保留前一隐藏状态的多少信息，以及整合多少新的候选隐藏状态。

Flux 提供 `GRU` 和 `GRUCell` 用于构建基于门控循环单元的模型：

```julia
using Flux

input_size = 10
hidden_size = 20

# 创建一个门控循环单元层
gru_layer = Flux.GRU(input_size, hidden_size)

# 示例输入序列
sample_sequence = [rand(Float32, input_size, 1) for _ in 1:5] # 批处理大小为 1

# 处理序列
output_gru_sequence = gru_layer.(sample_sequence)

# gru_layer.state 包含最终的隐藏状态
final_hidden_state = gru_layer.state

println("门控循环单元在最后一步的输出：", size(output_gru_sequence[end]))
println("最终隐藏状态的形状：", size(final_hidden_state))

# 为下一个批次/序列重置
Flux.reset!(gru_layer)
```

### 构建序列模型

循环神经网络 (neural network) (RNN)、长短期记忆网络 (LSTM)和门控循环单元 (GRU)构成为序列数据设计的模型的核心。它们通常与其他类型的层结合使用：

- **嵌入 (embedding)层：** 对于文本或类别序列数据，通常首先使用 `Flux.Embedding` 层（在“处理序列数据的嵌入”一节中介绍）将离散令牌转换为密集向量 (vector) (dense vector)表示。
- **全连接层：** 在循环层处理完序列后，通常使用一个或多个 `Flux.Dense` 层将最终隐藏状态（或隐藏状态的组合）转换为所需的输出格式（例如，分类的类别概率，回归的连续值）。
- **堆叠循环层：** 您可以堆叠多个循环层（例如，长短期记忆层堆叠在另一个长短期记忆层之上），以创建更深的模型，从而能够学习序列中更复杂的层次化特征。一个循环层的输出序列会成为下一个循环层的输入序列。

这是一个简单序列到一模型结构的例子，可能用于情感分类，其中输入是词嵌入序列，输出是单个情感分数：

```julia
using Flux

vocab_size = 1000  # 词汇表中不重复的词语数量
embed_size = 50   # 词嵌入的维度
hidden_size = 64  # 长短期记忆网络隐藏状态的大小
output_size = 1   # 回归的单个输出（或分类的类别数量）

model = Chain(
    Embedding(vocab_size, embed_size), # 输入：整数词索引
    LSTM(embed_size, hidden_size),
    # 仅获取最后一个时间步的输出，供下一层使用：
    x -> x[end], # 这从隐藏状态序列中选择最后一个隐藏状态
    # 如果长短期记忆网络只返回最后一个状态，则不需要此行。
    # Flux 的长短期记忆层在处理序列时，会返回一个输出序列。
    # 一种常见的模式是获取最后一个输出 h_T。
    # 另一种方法是使用 Flux.Recur(LSTMCell(...))，然后提取最终状态。
    Dense(hidden_size, output_size),
    # sigmoid # 对于二元分类或输出应在 [0,1] 之间的情况
)

# 示例：单个批次项的 10 个词索引序列
sample_input_indices = [rand(1:vocab_size) for _ in 1:10] # 整数向量

# Flux 的嵌入层期望整数向量或矩阵。
# 如果传递单个序列（批处理大小为 1）：
# 对于一个序列批次，每个步骤都应是一个矩阵 (词汇索引, 批处理大小)
# 或者，如果直接输入到 Chain，它应该将其作为单个批次项处理。
# 但是，`Embedding` 层需要对序列进行特殊处理。
# 通常，您会把 Embedding 应用于序列的每个元素。

# 让我们调整以符合长短期记忆网络期望的输入方式（矩阵向量）
# 1. 嵌入每个词索引
embedded_sequence = [model[1]([idx]) for idx in sample_input_indices] # (嵌入大小, 1) 矩阵向量

# 2. 通过长短期记忆网络
lstm_output_sequence = model[2].(embedded_sequence)
Flux.reset!(model[2]) # 处理后重置长短期记忆网络状态

# 3. 获取最后一个输出
last_lstm_output = model[3](lstm_output_sequence)

# 4. 通过全连接层
final_output = model[4](last_lstm_output)

# 要训练这个模型，您通常会使用序列批次。
# 来自 MLUtils.jl 的 DataLoaders（在“处理数据集”中讨论）在这里非常重要。

println("输出形状：", size(final_output)) # 应该是 (输出大小, 1)
```

这个例子展示了一种组合事物的方式。序列输入和输出的具体处理方式（例如，只取最后一个隐藏状态，还是使用所有隐藏状态）取决于具体任务。对于序列到序列的任务（如机器翻译），其架构会更复杂，通常涉及编码器-解码器结构。

在使用这些 Flux 中的循环层时，请记住：

- **输入形状：** 密切注意期望的输入形状。对于 `Flux.LSTM` 或 `Flux.GRU` 等处理完整序列的层，输入通常是一个 `Vector`，其中每个元素是一个大小为 `(特征数, 批处理大小)` 的矩阵，代表一个时间步。
- **状态管理：** 在处理独立序列之间（例如批次之间），`Flux.reset!(layer)` 对于清除隐藏状态很重要。
- **数据迭代：** `MLUtils.jl` 将是您高效批处理和迭代序列数据的好帮手。

循环神经网络、长短期记忆网络和门控循环单元是建模序列模式的有效工具。尽管长短期记忆网络和门控循环单元因其处理更长依赖关系的能力，通常优于普通循环神经网络，但理解基本的循环机制是根本。随着您继续学习，您会遇到变体和更高级的架构，如 Transformer，但这些门控循环单元仍然是序列数据深度学习 (deep learning)工具包中的重要组成部分。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  这篇原始论文介绍了长短期记忆（LSTM）网络，解决了循环神经网络中的梯度消失问题。
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/) — Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio (2014)
  Journal: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP); Publisher: Association for Computational Linguistics; Pages: 1724–1734; DOI: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179)
  这篇论文介绍了门控循环单元（GRU），这是一种简化的循环网络结构，其性能通常与LSTM相似。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本标准的教科书，提供了深度学习概念的坚实基础，包括对RNN、LSTM和GRU的详细讨论。
- [Recurrent Layers](https://fluxml.ai/Flux.jl/stable/models/recurrence/) — The Flux.jl Community (2025)
  Publisher: The Flux.jl Community
  Flux.jl 循环神经网络层的官方文档，包含了 `RNN`、`LSTM` 和 `GRU` 的使用说明及示例。
