---
course: "julia-for-machine-learning"
chapter: "julia-deep-learning-flux-jl"
lesson: "flux-jl-feedforward-networks"
sourceId: 6901
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/flux-jl-feedforward-networks"
title: "构建前馈神经网络"
description: "使用 Flux.jl 直观的语法构建简单的前馈神经网络（多层感知器）。"
order: 3
plots: []
sourceHash: "b5b46c5c78eb3022d7e142c61766527b98ea920b36cde6581792f6db1feb341b"
sourceCorrections: []
---

前馈神经网络 (neural network)（FNNs），也称多层感知器（MLPs），是一种基本的人工神经网络类型，其中节点之间的连接不形成环路。信息只沿一个方向流动：从输入节点，经过任何隐藏层，到达输出节点。使用 Flux.jl 构建这些网络的方法将进行演示。

### `Dense` 层：核心构建模块

在 FNN 中，构建大多数层的主要组成部分是 `Dense` 层。`Dense` 层实现运算 $output = activation(W \cdot input + b)$，其中 $W$ 是权重 (weight)矩阵，$b$ 是偏置 (bias)向量 (vector)，$activation$ 是一个逐元素的激活函数 (activation function)。

在 Flux 中创建 `Dense` 层时，您需要指定该层的输入特征数量和输出特征（神经元）数量。您通常还需要提供一个激活函数。

```julia
using Flux

# 一个具有10个输入特征和5个输出神经元的 Dense 层，使用 ReLU 激活
layer1 = Dense(10 => 5, relu)

# 另一个 Dense 层，可能用于二元分类输出，具有5个输入和1个输出，使用 sigmoid
# 此层的输入 (5) 必须与前一层的输出匹配。
output_layer = Dense(5 => 1, sigmoid)
```

在 `Dense(10 => 5, relu)` 示例中，`10 => 5` 是一个 `Pair`，表示该层将长度为10的输入向量转换为长度为5的输出向量。`relu` 是整流线性单元激活函数，定义为 $f(x) = \max(0, x)$。章开头部分提及了 sigmoid 函数 $\sigma(x) = \frac{1}{1 + e^{-x}}$，它常用于二元分类问题的输出层，以将输出压缩到0到1之间的概率。

Flux 提供了许多常用激活函数：

- `relu`: 整流线性单元，$f(x) = \max(0, x)$。
- `sigmoid`: Sigmoid 函数，$\sigma(x) = \frac{1}{1 + e^{-x}}$。
- `tanh`: 双曲正切，$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$。
- `softmax`: 用于多类别分类，将输出标准化为概率分布。
- `identity`: 无激活，$f(x) = x$。这是未指定激活函数时的默认值。

选择合适的激活函数取决于层在网络中的位置以及您正在解决的问题的性质。对于隐藏层，`relu` 是一个非常常见且有效的选择。对于输出层，`sigmoid` 常用于二元分类，`softmax` 用于多类别分类，而 `identity`（或无激活）用于回归任务。

### 使用 `Chain` 堆叠层

大多数有用的神经网络 (neural network)由多个层逐一堆叠组成。Flux.jl 使用 `Chain` 构造函数将多个层组合成一个可调用的模型。`Chain` 中一个层的输出自动成为下一个层的输入。

让我们构建一个包含输入层、一个隐藏层和一个输出层的简单前馈神经网络：假设我们有784个特征的输入数据（例如一个展平的28x28图像），我们想要一个使用 `relu` 激活的128个神经元的隐藏层，以及一个使用 `softmax` 激活的10个神经元的输出层（例如用于分类0-9数字）。

```julia
using Flux

model = Chain(
  Dense(784 => 128, relu),  # 输入：784个特征，输出：128个特征
  Dense(128 => 10, softmax) # 输入：128个特征（来自上一层），输出：10个特征（概率）
)
```

在构建 `Chain` 时，一个层的输出维度与后续层的输入维度匹配很重要。在上面的示例中：

1. 第一个 `Dense` 层接收784个输入并生成128个输出。
2. 第二个 `Dense` 层接收128个输入（匹配上一层的输出）并生成10个输出。

您可以通过向 `Chain` 添加更多 `Dense` 层（或我们稍后会看到其他类型的层）来创建更复杂的前馈神经网络：

```julia
using Flux

# 一个包含两个隐藏层的更深的前馈神经网络
model_deep = Chain(
  Dense(784 => 256, relu),    # 隐藏层1
  Dense(256 => 128, relu),    # 隐藏层2
  Dense(128 => 10, softmax)   # 输出层
)
```

`model` 和 `model_deep` 对象现在是可调用的 Flux 模型。您可以将数据通过它们以获得预测，并且它们的参数 (parameter)（权重 (weight)和偏置 (bias)）可以被训练。

### 可视化网络结构

了解神经网络 (neural network)的架构是有益的。虽然 Flux 本身没有内置的可视化工具，但您可以这样表示一个简单前馈神经网络的结构：

> 该图显示数据从输入特征流经一个或多个隐藏层到达输出层。每个层执行一个转换，通常包含一个 `Dense` 运算，然后是一个激活函数 (activation function)。

### 数据通过网络（前向传播）

模型构建完成后，您可以通过调用模型并传入输入数据来执行“前向传播”。输入数据应该是一个矩阵，其中列表示单个样本，行表示特征。如果您有一个单个样本，它应该是一个列向量 (vector)。

让我们假设使用前一个示例中的 `model_deep` 和一些随机输入数据，表示5张图像的批次，每张图像有784个特征：

```julia
# 示例：5个样本，每个具有784个特征
dummy_input = rand(Float32, 784, 5) 

# 执行前向传播
predictions = model_deep(dummy_input)

# `predictions` 将是一个10x5的矩阵， 
# 其中每列包含一个输入样本的10个概率（由于softmax）。
println("Output dimensions: ", size(predictions)) # 预期结果：(10, 5)
```

此前向传播计算给定输入的网络输出。深度学习 (deep learning)过程的下一步，我们将在后续章节中介绍，是定义一个损失函数 (loss function)来衡量这些预测与真实值之间的差距，然后使用优化器调整模型的权重 (weight)和偏置 (bias)以最小化此损失。

您现在已具备在 Flux.jl 中构建前馈神经网络 (neural network)架构的基本知识。通过在 `Chain` 中组合 `Dense` 层，您可以定义不同深度和宽度的网络，根据当前机器学习 (machine learning)任务的复杂性进行调整。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这是深度学习领域的标准教科书，提供了关于前馈神经网络、激活函数和基本架构的详细理论背景。
- [Flux.jl: High-Performance Machine Learning](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Community (2025)
  Flux.jl的官方文档，提供了在Julia中使用Dense层、Chain模型和各种激活函数的直接信息和示例。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  一本在线书籍，提供了神经网络基础知识的清晰易懂的解释，包括多层感知器、其结构和激活函数的作用。
