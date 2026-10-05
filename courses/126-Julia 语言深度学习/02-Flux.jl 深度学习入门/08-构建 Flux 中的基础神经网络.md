# 构建 Flux 中的基础神经网络

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/constructing-basic-nn-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Flux.jl 的组成部分包括：`Dense` 层、用于序列操作的 `Chain` 构造器、实现非线性的激活函数 (activation function)、量化 (quantization)误差的损失函数 (loss function)以及指导学习的优化器。整合这些组件，可以构造一个完整的基础神经网络 (neural network)。该过程有助于构建更复杂的深度学习 (deep learning)模型。

### 定义网络结构

在 Flux 中定义前馈神经网络 (neural network)最直接的方式是使用 `Chain`。`Chain` 接受一系列层或任何可调用函数，并将它们依次应用于输入数据。

设想一个简单的回归任务：从例如三个输入特征中预测一个单一的连续值。我们可以设计一个带有一个隐藏层的网络。该网络将包含：

- 一个输入层，由第一个 `Dense` 层的维度隐式定义。
- 一个 `Dense` 隐藏层，包含选定数量的神经元（例如五个）和一个激活函数 (activation function)（例如 `relu`）。
- 一个 `Dense` 输出层，包含一个神经元，用于单一输出值。对于回归任务，此层通常没有显式激活函数，这意味着它默认使用线性激活。

### 代码实现

让我们把这个网络设计转化为 Flux 代码。我们将设定输入数据有三个特征，隐藏层有五个神经元，以及一个输出神经元。

```julia
using Flux

# 定义输入特征、隐藏单元和输出单元的数量
input_features = 3
hidden_units = 5
output_units = 1

# 使用 Chain 构造模型
model = Chain(
    Dense(input_features, hidden_units, relu),  # 带有 ReLU 激活的隐藏层
    Dense(hidden_units, output_units)           # 输出层（默认线性激活）
)
```

在这个 `model` 中，数据首先通过一个 `Dense` 层，将3个输入特征转换为5个隐藏特征。然后，`relu` 激活函数 (activation function)按元素应用于此层的输出。所得的5个值再通过另一个 `Dense` 层，将其转换为一个单一的输出值。

### 可视化网络结构

可视化网络通常很有用。虽然 Flux 没有直接在 REPL 中为 `Chain` 对象提供内置的可视化工具，但我们可以将其结构呈现为图表。

> 一个带有一个隐藏层的简单神经网络 (neural network)。数据从输入特征流向隐藏层（带有 ReLU 激活），再到输出层，生成一个单一值。

### 前向传播：进行预测

模型定义后，你可以向其传递数据以获得预测。输入数据应与预期的输入维度匹配。根据惯例，Flux 的 `Dense` 层在处理批次数据时，期望输入数据的每一列是一个样本，每一行代表特征。如果提供的是单个样本向量 (vector)，则将其视为单列。

让我们创建一些虚拟输入数据：

```julia
# 具有3个特征的单个数据点（作为列向量）
single_input_data = rand(Float32, input_features, 1) 
# 输出: 3×1 矩阵{Float32}

# 通过模型传递数据
prediction = model(single_input_data)
println("单个输入的预测: ", prediction)
# 输出: Prediction for single input: Float32[...]] (一个 1x1 矩阵)

# 一个包含10个数据点的批次
batch_input_data = rand(Float32, input_features, 10) # 3个特征，10个样本
batch_predictions = model(batch_input_data)
println("批次预测的形状: ", size(batch_predictions))
# 输出: Shape of batch predictions: (1, 10)
```

模型处理单个数据点（一个 $3 \times 1$ 矩阵）和一批数据（一个 $3 \times 10$ 矩阵）。输出形状反映了这一点：单个输入对应一个 $1 \times 1$ 矩阵，10个输入的批次对应一个 $1 \times 10$ 矩阵，其中输出中的每一列对应各自输入样本的预测。

### 模型参数 (parameter)

Flux 模型包含可学习的参数，即层的权重 (weight)和偏置 (bias)。你可以使用 `Flux.params` 来查看这些参数。这些参数会自动从 `Chain` 等结构中的所有层中收集。

```julia
# 获取模型的所有参数（权重和偏置）
parameters = Flux.params(model)

# 你可以遍历它们或查看特定的参数
# 例如，查看参数数组的数量：
println("参数数组的数量: ", length(parameters)) 
# 预期输出: Number of parameter arrays: 4 
# (它们对应于：weights_hidden, bias_hidden, weights_output, bias_output)

# 查看第一个参数数组（第一个全连接层的权重）的维度：
if !isempty(parameters)
    println("第一个全连接层权重的形状: ", size(first(parameters)))
    # 预期输出: Shape of weights for the first Dense layer: (5, 3)
end
```

这些参数是在训练过程中被更新的。优化器根据相对于损失函数 (loss function)计算的梯度来修改它们。

### 下一步：训练网络

构造模型是最初的步骤。要使其执行有用的任务，需要对其进行训练。训练过程通常包括：

1. **选择损失函数 (loss function)**：选择一个合适的损失函数（例如，回归任务使用 `Flux.mse`，二元分类使用 `Flux.logitcrossentropy`），以衡量模型预测与真实目标值之间的差异。
2. **选择优化器**：选择一个优化器（例如 `ADAM()`，`Descent()`），它将指定如何调整模型的参数 (parameter)以最小化此损失。
3. **迭代训练**：重复地向模型输入数据，计算损失，计算损失相对于模型参数的梯度，然后使用优化器更新参数。这种迭代过程被称为训练循环。

Zygote.jl 在这里扮演重要角色，它自动计算训练所需的梯度。当你定义一个包含模型和数据的损失函数时，Zygote 可以对这个函数相对于 `Flux.params(model)` 进行求导。我们之前简单讨论过 Zygote.jl，你将在本章理论讨论之后的“动手实践：使用 Flux 构建简单回归器”部分看到它被整合到完整的训练循环中。

## 参考资料

- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Contributors (2025)
  Flux.jl深度学习库的官方指南，涵盖其API、层和模型构建。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  为神经网络、深度学习架构和学习算法提供全面的学术基础。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  对神经网络的易懂介绍，解释网络设计和学习的核心原理。
- [Zygote.jl Documentation](https://fluxml.ai/Zygote.jl/stable/) — The Zygote.jl Contributors (2025)
  Zygote.jl的官方文档，详细说明Julia中用于梯度计算的自动微分。

---

[上一节](07-Zygote.jl%EF%BC%9AFlux%E4%B8%AD%E7%9A%84%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E4%BD%BF%E7%94%A8%20Flux%20%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%9B%9E%E5%BD%92%E5%99%A8.md)
