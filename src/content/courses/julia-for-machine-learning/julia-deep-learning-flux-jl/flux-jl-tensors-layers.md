---
course: "julia-for-machine-learning"
chapter: "julia-deep-learning-flux-jl"
lesson: "flux-jl-tensors-layers"
sourceId: 6900
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/flux-jl-tensors-layers"
title: "Flux.jl 入门：张量与层"
description: "Flux.jl 简介，Julia 的深度学习库。了解张量（多维数组）和神经网络层的定义。"
order: 2
plots: []
sourceHash: "5f4163887c5343564794c36943866ed2e2bd659a0ccdff4d3df8ad3be4c18d24"
sourceCorrections: []
---

Flux.jl 是 Julia 用于深度学习 (deep learning)的主要库。它在设计时就考虑了灵活性和可扩展性，让您能够简洁地表达复杂模型。与其他引入许多新数据类型的框架不同，Flux.jl 与 Julia 现有的生态系统平顺结合，尤其是其强大的数组操作功能。Flux.jl 的基本构成单元包括张量（本质上是多维数组）和层（神经网络 (neural network)中的操作单元）。

## 张量：深度学习 (deep learning)的数据

在深度学习中，张量是向量 (vector)和矩阵到任意维度的推广。Flux.jl 不引入特殊的“张量”类型；相反，它直接在 Julia 内置的 `Array` 类型（用于 CPU 计算）或 `CuArray`（来自 `CUDA.jl`，用于 GPU 计算）上操作。这种平顺的结合表示您可以充分利用 Julia 丰富的所有数组功能。

让我们看看如何使用这些数组表示数据：

- **0维张量（标量）**：单个数字。

  ```julia
  scalar_value = 5.0f0 # A Float32 scalar
  # To store it in a 0-dimensional array:
  scalar_array = fill(5.0f0) 
  # ndims(scalar_array) == 0
  ```

  神经网络 (neural network)通常使用32位浮点数（`Float32`，由 `f0` 后缀表示），以在精度和计算性能之间取得平衡。
- **1维张量（向量）**：一串数字。适用于表示具有多个特征的单个数据样本，或层中的偏置 (bias)。

  ```julia
  vector_data = [1.0f0, 2.0f0, 3.0f0, 4.0f0] # A 4-element vector
  # size(vector_data) will be (4,)
  ```
- **2维张量（矩阵）**：一个数字网格。对于表格数据，这可能表示一批样本，其中行是特征，列是单个样本。在 Flux.jl 中，全连接层的约定通常是 `(特征数, 批大小)`。

  ```julia
  # 3 features, 5 samples in a batch
  matrix_data = rand(Float32, 3, 5) 
  # size(matrix_data) will be (3, 5)
  ```
- **3维张量**：可用于序列数据，例如 `(特征数, 序列长度, 批大小)`，或用于一批灰度图像等数据 `(高, 宽, 批大小)`。
- **4维张量**：彩色图像批次的标准，通常维度为 `(宽, 高, 通道数, 批大小)`。

  ```julia
  # A batch of 16 color images, each 28x28 pixels
  # (width, height, color_channels, batch_size)
  images_batch = rand(Float32, 28, 28, 3, 16)
  # size(images_batch) will be (28, 28, 3, 16)
  ```

理解不同操作和层所需的张量形状，对于正确构建模型很要紧。Flux.jl 的操作通常旨在遵循这些标准约定。Julia 中的基本数组操作，例如加法、乘法和逐元素函数，都可以直接在这些张量上使用。

## 层：网络的构成单元

层是神经网络 (neural network)的核心组成部分。每个层都对其输入数据执行特定的转换，通常涉及可学习的参数 (parameter)（权重 (weight)和偏置 (bias)）。Flux.jl 提供了丰富的预定义层集合，使得构建常见的网络架构变得方便。

Flux 中的一个层本质上是一个可调用的 Julia 结构，这意味着它的行为类似于函数。它接受一个输入张量并产生一个输出张量。

### 全连接层 `Dense`

最基本的层是 `Dense` 层，也称为全连接层。它对输入应用线性变换，随后是一个可选的激活函数 (activation function)。其运算可以通过以下公式描述：$y = \sigma(Wx + b)$，其中 $W$ 是权重 (weight)矩阵，$b$ 是偏置 (bias)向量 (vector)，$x$ 是输入，而 $\sigma$ 是激活函数（例如 sigmoid 函数 $\sigma(z) = \frac{1}{1 + e^{-z}}$）。

您可以通过指定输入特征数和输出特征数来创建一个 `Dense` 层。也可以提供激活函数。

```julia
using Flux

# 创建一个 Dense 层:
# 接收 3 个输入特征，产生 2 个输出特征
# 使用 sigmoid 激活函数 (σ)
input_features = 3
output_features = 2
layer = Dense(input_features, output_features, sigmoid)

# layer.weight 是一个 (输出特征数 x 输入特征数) 矩阵，即 2x3
# layer.bias 是一个 (输出特征数) 元素的向量，即 2 元素
# layer.σ 是 sigmoid 函数

# 让我们创建一些示例输入数据（3 个特征，批次中 1 个样本）
# 输入维度: (特征数, 批大小)
input_data = rand(Float32, 3, 1)

# 将数据通过层
output_data = layer(input_data)

println("输入大小: ", size(input_data))     # 预期: (3, 1)
println("输出大小: ", size(output_data))    # 预期: (2, 1)
```

该层的可学习参数 (parameter)是 `layer.weight` 和 `layer.bias`。Flux.jl 会自动初始化它们，通常是从合适的分布中随机取值（例如，`Dense` 层默认使用 Glorot 均匀分布）。

> `Dense` 层对输入张量的转换。具有 $D_{in}$ 个特征的输入张量被转换为具有 $D_{out}$ 个特征的输出张量。该层内部执行线性变换，随后进行激活。

Flux 提供许多其他类型的层，例如：

- `Conv`：用于卷积神经网络 (neural network) (CNN)，主要用于图像处理。
- `RNN`, `LSTM`, `GRU`：用于循环神经网络 (RNN)，用于自然语言处理等序列建模任务。
- 池化层，如 `MaxPool`、`MeanPool`。
- 归一化 (normalization)层，如 `BatchNorm`。

激活函数本身（例如 `relu`、`tanh`、`softmax`）在 Flux 中也可用。如果 `Dense` 等层中未指定激活函数，则默认为 `identity`（表示不应用激活，即 $y = Wx + b$）。

您可以显式地在层之后应用激活函数，尽管通常将其包含在层定义中更方便：

```julia
layer_no_activation = Dense(5, 10) # 未指定激活函数，默认为 identity (σ=identity)
input_for_layer = rand(Float32, 5, 3) # 5 个特征，3 个样本
output_linear = layer_no_activation(input_for_layer)
# 逐元素应用 ReLU 激活
output_activated = relu.(output_linear) 
```

## 使用 `Chain` 组合层

单个层功能强大，但深度学习 (deep learning)的真正优势在于将它们堆叠起来以形成更深的网络结构。Flux.jl 使用 `Chain` 来按顺序组合多个层（或操作张量的其他函数）。一个 `Chain` 接受任意数量的层作为参数 (parameter)，并按照提供的顺序应用它们。

```julia
using Flux

# 定义一个简单的两层网络
model = Chain(
    Dense(10, 20, relu),  # 第 1 层：10 个输入，20 个输出，ReLU 激活
    Dense(20, 5, sigmoid) # 第 2 层：20 个输入，5 个输出，Sigmoid 激活
)

# model 现在也是一个可调用的结构体。
# 它期望的输入与第一层兼容（10 个特征）。

# 创建一些示例输入数据
# (features, batch_size)
input_to_model = rand(Float32, 10, 32) # 10 个特征，批次中 32 个样本

# 将数据通过整个模型
final_output = model(input_to_model)

println("模型输入大小: ", size(input_to_model))    # 预期: (10, 32)
println("模型输出大小: ", size(final_output))   # 预期: (5, 32)
```

`Chain` 中一个层的输出成为下一个层的输入。这种简洁优雅的模型组合方式是 Flux.jl 的一个特点。您甚至可以嵌套 `Chain`，或在 `Chain` 中包含任何执行数组操作的 Julia 函数，为模型设计提供了相当大的灵活性。

> Flux.jl 中的一个简单 `Chain` 模型，说明数据依次流经两个 `Dense` 层，从 10 个特征转换为 20 个，然后转换为 5 个特征。

## 参数 (parameter)与梯度：简要介绍

当您创建像 `Dense` 这样的层或将它们组合成 `Chain` 时，Flux.jl 会跟踪所有可学习参数（权重 (weight)和偏置 (bias)）。您可以使用 `Flux.params()` 来查看这些参数：

```julia
# 对于 Chain 示例中定义的 'model':
parameters = Flux.params(model)
# parameters 是一个对象，允许迭代模型中的所有权重和偏置。
# 例如，要访问 Chain 中第一个 Dense 层的权重:
# model[1].weight 
# 要访问第二个 Dense 层的偏置:
# model[2].bias
```

这些参数是您的优化算法在训练过程中将调整的内容。损失函数 (loss function)相对于这些参数的梯度是使用自动微分计算的，主要通过 Zygote.jl 完成，我们将在后续章节中介绍。目前，您只需要知道 Flux.jl 会为您管理这些参数。

张量和层面的这些知识为构建和训练神经网络 (neural network)奠定了铺垫。您已经了解了如何表示数据以及如何定义对数据进行的转换。接下来，您将学习如何将这些组合成完整的模型，并通过定义损失函数和使用优化器来训练它们。

## 参考资料

- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Developers (2025)
  Flux.jl 的官方文档，详细介绍了其架构、层以及如何与 Julia 的数组系统集成以进行张量表示。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，全面介绍了深度学习概念，包括张量、各种层类型和激活函数。
- [Multi-dimensional Arrays](https://docs.julialang.org/en/v1/manual/arrays/) — The Julia Language Developers (2024)
  官方 Julia 文档中关于多维数组的部分，它们构成了 Flux.jl 中张量的基础。
