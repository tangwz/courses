---
course: "julia-deep-learning"
chapter: "constructing-nn-architectures"
lesson: "building-multilayer-perceptrons-flux"
sourceId: 6899
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/building-multilayer-perceptrons-flux"
title: "构建多层感知器（MLP）"
description: "使用 Flux.jl 构建并了解多层感知器（全连接网络）。"
order: 3
plots: []
sourceHash: "541190ad921799105cb10c36d9530938410ea1ff2ad90f92c0d8ddf49cc62279"
sourceCorrections: []
---

多层感知器（MLP）是一种基本的神经网络 (neural network)模型。它们由一个或多个神经元层构成，前一层中的每个神经元都连接到后一层中的所有神经元。这种紧密的连接方式使得它们常被称为“全连接网络”。尽管结构简单，MLP 仍有能力学习数据中复杂的非线性关系，这使其成为理解神经网络结构的一个优秀起点。Flux.jl 的 `Dense` 层和 `Chain` 构造器是构建 MLP 的主要工具。

### 多层感知器（MLP）的组成

一个 MLP 通常包含三种主要的层类型：输入层、一个或多个隐藏层以及输出层。

> 典型的 MLP 结构，展示了信息从输入层流经隐藏层，最后到达输出层的过程。层间的每个连接都是“稠密的”，表示前一层的所有神经元都连接到下一层的所有神经元。

我们来细分这些组成部分：

- **输入层**：这一层接收原始数据。输入层中的神经元数量直接对应于数据集中特征的数量。例如，如果您基于 5 个特征（如面积、卧室数量等）预测房价，您的输入层将有 5 个神经元。
- **隐藏层**：这些是输入层和输出层之间的中间层。它们执行大部分计算，并负责学习数据中复杂的模式。“深度学习 (deep learning)”中的“深度”通常指存在多个隐藏层。隐藏层中的每个神经元对其输入应用线性变换（加权和加上一个偏置 (bias)），然后应用一个非线性激活函数 (activation function)。
- **输出层**：这一层产生网络的最终预测。输出层中的神经元数量和所使用的激活函数取决于具体的任务：
  - 对于回归任务（预测连续值），通常有一个神经元，并且常不使用激活函数（或使用恒等激活）。
  - 对于二分类任务（预测两个类别之一），通常会有一个神经元，并使用 sigmoid 激活函数来输出概率。
  - 对于多分类任务（预测多个类别之一），将有 N 个神经元（其中 N 为类别数量），并使用 softmax 激活函数来输出跨类别的概率分布。
- **权重 (weight)和偏置**：这些是网络的可学习参数 (parameter)。在训练过程中，网络会调整这些权重和偏置，以最小化其预测与实际目标值之间的差异。
- **激活函数**：如第 2 章所述，激活函数为模型引入非线性。没有它们，无论 MLP 有多少层，都将表现得像一个简单的线性模型。常见选择包括用于隐藏层的 ReLU（修正线性单元），以及根据任务需求用于输出层的 sigmoid 或 softmax。

### 使用 Flux.jl 构建 MLP

使用 `Dense` 层并将它们组合成 `Chain`，在 Flux.jl 中构建 MLP 非常直接。一个 `Dense` 层，即 `Dense(in::Integer, out::Integer, σ)`，创建一个标准的完全连接层，它将大小为 `in` 的输入转换为大小为 `out` 的输出，然后应用一个激活函数 (activation function) `σ`。

我们来构建一个简单的 MLP。假设我们有一个包含 10 个输入特征的数据集，我们想构建一个包含两个隐藏层的网络：第一层有 64 个神经元，第二层有 32 个神经元。对于回归任务，我们将有一个单独的输出神经元。

```julia
using Flux

# 定义网络维度
num_features = 10
num_hidden1 = 64
num_hidden2 = 32
num_outputs = 1 # 用于回归任务

# 构建 MLP
mlp_model = Chain(
    Dense(num_features, num_hidden1, relu),    # 输入层（10个特征）到第一个隐藏层（64个神经元）
    Dense(num_hidden1, num_hidden2, relu),     # 第一个隐藏层（64个神经元）到第二个隐藏层（32个神经元）
    Dense(num_hidden2, num_outputs)            # 第二个隐藏层（32个神经元）到输出层（1个神经元）
                                               # 此处未指定输出层的激活函数；
                                               # 对于回归任务，这是常见做法（恒等激活）。
)

# 您可以打印模型以查看其结构
println(mlp_model)
```

在此代码中：

- `Chain(...)` 按顺序组织层。一层的输出成为下一层的输入。
- `Dense(num_features, num_hidden1, relu)` 定义了我们的第一层。它接收 `num_features` 个输入，产生 `num_hidden1` 个输出，并应用 `relu` 激活函数。
- 随后的 `Dense` 层遵循相同的模式，将前一层的输出连接到下一层的输入。
- 最终的 `Dense(num_hidden2, num_outputs)` 层没有明确指定激活函数。默认情况下，如果未提供激活函数，Flux 的 `Dense` 层会使用恒等激活（`x -> x`），这适用于回归任务。对于分类任务，您可以在此处添加 `sigmoid` 或 `softmax`，或更常见地，将其作为损失函数 (loss function)计算的一部分或作为模型之后的最后一步应用。

为了查看数据如何流经此模型（即“前向传播”），我们可以创建一些虚拟输入数据。Flux 模型通常要求输入数据将特征作为行，将观测值（样本）作为列。

```julia
# 创建 5 个虚拟数据样本批次，每个样本有 10 个特征
batch_size = 5
dummy_data = rand(Float32, num_features, batch_size) # 形状: (10, 5)

# 将数据通过模型
predictions = mlp_model(dummy_data)

println("Input data size: ", size(dummy_data))
println("Output predictions size: ", size(predictions)) # 预期: (1, 5)
```

输出 `predictions` 将是一个大小为 `(1, 5)` 的矩阵，其中每列是对应输入样本的回归输出。

### 前向传播：数据如何流动

当您将数据通过 `mlp_model(dummy_data)` 这样的 MLP 时，每个 `Dense` 层执行两个主要操作：

1. **线性变换**：它计算输入的加权和并添加一个偏置 (bias)项。数学上，对于输入向量 (vector) $X$，激活前的输出 $Z$ 是 $Z = W \cdot X + b$，其中 $W$ 是权重 (weight)矩阵，$b$ 是该层的偏置向量。
2. **激活函数 (activation function)**：结果 $Z$ 随后通过该层指定的激活函数（例如 `relu`）。因此，该层的最终输出是 $A = \sigma(Z)$，其中 $\sigma$ 是激活函数。

`Chain` 确保一个层的输出 $A$ 成为序列中下一层的输入 $X$，直到达到最终输出层。

### MLP 的设计考量

设计 MLP 时，有几个选择会影响其性能：

- **网络深度与宽度**：

  - **深度**指的是隐藏层的数量，而**宽度**指的是每个隐藏层中神经元的数量。
  - 更深、更宽的网络拥有更多参数 (parameter)，因此能够表示更复杂的功能。然而，它们也计算成本更高，且容易出现过拟合 (overfitting)（过度学习训练数据，包括其噪声，导致在未见过的数据上表现不佳）。
  - 选择最佳深度和宽度没有固定的方法。这通常需要反复试验，可以从一个较简单的结构开始，如果模型表现不佳，则逐步增加复杂性。
- **激活函数 (activation function)**：

  - **隐藏层**：`relu` 是隐藏层非常常用的选择。它计算效率高，并有助于减轻在更深层网络中使用 `sigmoid` 或 `tanh` 等其他激活函数时可能出现的“梯度消失”问题。`leakyrelu` 或 `elu` 等替代方案有时也能带来好处。
  - **输出层**：选择取决于任务：
    - 回归：如果输出可以是任何实数，通常不使用激活函数（恒等函数）。如果输出受限（例如，必须为正），则可能使用适当的激活函数，如 `softplus`。
    - 二分类：`sigmoid` 用于将输出压缩到 $(0, 1)$ 范围，表示一个概率。
    - 多分类：`softmax` 用于将输出转换为 $N$ 个类别的概率分布，其中每个输出在 $(0, 1)$ 之间，且所有输出之和为 1。Flux 提供了 `Flux.sigmoid` 和 `Flux.softmax`。
- **数据缩放**：

  - MLP 对输入特征的尺度敏感。值范围较大的特征可能会主导学习过程。
  - 在将输入数据馈送给 MLP 之前对其进行缩放是常规操作。常用技术包括：
    - **标准化**：将特征重新缩放到均值为 0，标准差为 1。
    - **归一化 (normalization)**：将特征重新缩放到特定范围，通常为 $[0, 1]$ 或 $[-1, 1]$。
  - 本章关于“Julia 中的数据准备和预处理”的部分将更详细地介绍这些技术。

### 何时使用（和不使用）MLP

**优点**：

- MLP 对于结构化或表格数据非常有效，这类数据中的特征没有固有的空间或序列顺序（例如，客户数据、特定时间戳的传感器读数、财务数据）。
- 它们相对容易实现和理解，可以作为良好的基准模型。

**局限性**：

- **缺乏空间/序列感知**：标准 MLP 独立处理所有输入特征，本身不理解空间关系（如图像中）或序列依赖性（如文本或时间序列中）。尽管您可以将图像展平为向量 (vector)并馈送给 MLP，但这会丢弃二维结构。对于此类数据，我们将随后介绍的卷积神经网络 (neural network)（CNN）和循环神经网络（RNN）通常更具能力。
- **高维输入的参数 (parameter)效率低**：对于非常高维的输入（例如高分辨率图像），全连接 MLP 会导致大量的参数，使其训练缓慢且容易出现过拟合 (overfitting)。

### 总结与后续步骤

多层感知器是多功能的前馈神经网络 (neural network)，它们是深度学习 (deep learning)的根本组成部分。您现在已经了解了如何使用 Flux.jl 的 `Dense` 层和 `Chain` 结构在 Julia 中构建它们，并理解了所涉及的设计考量。

尽管 MLP 对某些类型的问题（特别是涉及表格数据的问题）有效，本章接下来将介绍卷积神经网络（CNN）和循环神经网络（RNN）。这些结构分别专门设计用于处理具有空间和序列结构的数据，通过整合利用这些特性的专用层。理解 MLP 为掌握这些更复杂的结构提供了坚实的根基。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6: Deep Feedforward Networks
  一本全面性的教科书，提供了深度学习的理论基础和实践方面，详细解释了多层感知器、其架构、激活函数和训练。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press; Pages: Chapters 1-3
  一本易于理解的在线书籍，清晰直观地介绍了神经网络，包括感知器、多层感知器和反向传播算法的基础知识。
