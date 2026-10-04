---
course: "julia-deep-learning"
chapter: "intro-flux-jl"
lesson: "flux-jl-primitives-layers-models-chains"
sourceId: 6879
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/flux-jl-primitives-layers-models-chains"
title: "Flux.jl 基本构成：层、模型和链"
description: "了解 Flux.jl 中的主要组成部分：用于转换的层、用于组合层的模型以及用于序列网络的链。"
order: 2
plots: []
sourceHash: "72613a56691679e5fee32f7294e31ce45c912ddb49ba3f96c75e6e38b9018738"
sourceCorrections: []
---

Flux.jl 提供了一个高度可组合的系统来构建神经网络 (neural network)。你无需使用固定的预定义架构，而是从基本组件中组合你的网络。你会遇到的主要构成部分是层、链以及整体模型结构，它可以是一个简单的链，也可以是一个更复杂的自定义设计。掌握这些元素是在Flux中定义任何神经网络的第一步。

### 层：基本运算单元

在最基本的层面，Flux 模型由层构成。层本质上是一个对其输入数据进行转换的函数。重要的是，大多数层还具有可学习的参数 (parameter)，通常是权重 (weight)和偏置 (bias)，这些参数在训练过程中会进行调整。Flux 会自动为你管理这些参数。

常见的也是可能最简单的层是 `Dense` 层，也称为全连接层。它对输入进行线性转换，然后可以选择性地应用激活函数 (activation function)。
一个 `Dense` 层由它接受的输入特征数量和它产生的输出特征数量定义。举例来说，`Dense(10, 5)` 创建了一个层，它接收一个（或一批）每个有 10 个特征的输入向量 (vector)，并将其转换为一个有 5 个特征的输出向量。

```julia
using Flux

# 创建一个Dense层：10个输入特征，5个输出特征
# 也可以指定一个激活函数，例如 Dense(10, 5, relu)
# 我们将在下一节详细讨论激活函数。
input_features = 10
output_features = 5
dense_layer = Dense(input_features, output_features)

# 生成一些虚拟输入数据：一个包含10个Float32元素的单列向量
dummy_input = randn(Float32, input_features, 1) 

# 将输入通过该层
output = dense_layer(dummy_input)

println("输入维度: ", size(dummy_input))
println("输出维度: ", size(output))
# 预期输出:
# 输入维度: (10, 1)
# 输出维度: (5, 1)
```

在内部，这个 `dense_layer` 包含一个大小为 `(output_features, input_features)` 的权重矩阵 $W$ 和一个大小为 `(output_features, 1)` 的偏置向量 $b$。当你将输入 $x$ 通过它时，该层会计算 $W \cdot x + b$。如果指定了激活函数 $\sigma$（例如 `relu`），计算将是 $\sigma(W \cdot x + b)$。这些参数 $W$ 和 $b$ 是 Flux 在训练过程中会优化的内容。

尽管 `Dense` 是基本层，但 Flux 为不同任务提供了许多其他类型的层，例如：

- `Conv` 和 `MaxPool` 用于卷积神经网络 (neural network) (CNN)。
- `RNN`、`LSTM` 和 `GRU` 单元用于循环神经网络 (RNN)。
- `Embedding` 层用于处理分类数据或词嵌入 (embedding)。

我们将在后续章节中介绍这些专用层。目前，你需要知道层是一个可调用对象（意味着你可以像函数一样使用它：`layer(input)`），它转换数据并通常管理自己的可学习参数。

### 链：前馈模型的层序列化

通常，神经网络 (neural network)包含一系列层，一个层的输出成为下一个层的输入。Flux 提供了一种方便的方法，使用 `Chain` 来构造此类序列模型。一个 `Chain` 接收一系列层，并按顺序将它们应用于输入数据。

```julia
using Flux

# 使用Chain定义一个简单的序列模型
# 输入 -> Dense(10->20) -> relu激活 -> Dense(20->5) -> softmax激活
model_chain = Chain(
    Dense(10, 20),   # 第一层：10个输入，20个输出
    relu,            # 激活函数（逐元素应用）
    Dense(20, 5),    # 第二层：20个输入，5个输出
    softmax          # 输出激活（例如，用于分类）
)

# 生成虚拟输入：包含3个样本的批次，每个样本有10个特征
dummy_batch_input = randn(Float32, 10, 3)

# 将输入通过整个链
predictions = model_chain(dummy_batch_input)

println("模型输出维度: ", size(predictions))
# 预期输出:
# 模型输出维度: (5, 3) 
# (批次中3个样本，每个有5个输出特征)
```

在这个例子中，`model_chain` 首先会将 `dummy_batch_input` 通过 `Dense(10, 20)` 层。该层的输出（将有 20 个特征）随后传递给 `relu` 激活函数 (activation function)。`relu` 的结果输入到 `Dense(20, 5)`，最后，第二个全连接层的输出通过 `softmax`。

`Chain` 本身也是一个可调用对象，就像单个层一样。它将一系列操作整齐地打包成一个单一的、可重用的组件。像 `relu` 和 `softmax` 这样的激活函数在 `Chain` 中也被视为层；它们是简单的函数，转换输入但没有自己的可学习参数 (parameter)。我们将在下一节详细介绍激活函数。

这里有一个图示说明了数据流通过一个典型的 `Chain`：

> 一个 `Chain` 顺序处理数据。输入通过一个初始的 `Dense` 层及其激活函数，接着是另一个 `Dense` 层及其激活函数，最后生成输出。每个 `Dense` 层都具有自己的可学习权重 (weight)和偏置 (bias)。

### 模型：从简单链到自定义架构

在 Flux 中，“模型”一词通常指任何可调用结构，它处理输入并产生输出，可能具有可学习参数 (parameter)。
对于许多常见的神经网络 (neural network)架构，尤其是像多层感知器 (MLP) 这样的前馈网络，一个 `Chain` *就是*你的模型。它包含了整个网络结构。

然而，Flux 的强大之处在于其灵活性。在定义模型时，你不仅限于使用 `Chain`。对于不遵循简单顺序流的更复杂架构，例如具有跳跃连接（如 ResNet）的网络、多个输入或输出分支，或其他自定义路由逻辑，你可以将模型定义为自定义的 Julia `struct`。

一个自定义模型 `struct` 通常包含各种层（可以是 `Dense`、`Conv`，甚至其他 `Chain`）作为其字段。为了使其成为一个功能性模型，你通过使 `struct` 可调用来定义它如何处理输入数据。这是通过为你的 `struct` 实现一个方法来完成的，该方法接受输入 `x` 并定义前向传播。

这是一个非常简单的示例：

```julia
using Flux

# 为我们的模型定义一个自定义结构体
struct MyCustomModel
    layer1::Dense
    layer2::Dense
    # 你可以添加其他层、链，甚至非Flux组件
end

# 使结构体可调用以定义前向传播
# 此函数定义输入'x'如何流经模型的组件。
# 在这里，我们应用layer1，然后是relu激活，接着是layer2。
(m::MyCustomModel)(x) = m.layer2(relu.(m.layer1(x)))

# 实例化自定义模型
custom_model = MyCustomModel(
    Dense(10, 20), # layer1: 10个输入，20个输出
    Dense(20, 5)   # layer2: 20个输入，5个输出
)

# 像使用其他Flux层或链一样使用自定义模型
dummy_input = randn(Float32, 10, 1) # 10个特征，1个样本
output = custom_model(dummy_input)

println("自定义模型输出维度: ", size(output))
# 预期输出:
# 自定义模型输出维度: (5, 1)
```

这种自定义结构体方法可以实现任意复杂性。前向传播 `(m::MyCustomModel)(x)` 可以实现你需要的任何逻辑，按任何顺序调用其构成层，组合它们的输出等。Flux 仍然能够找到并训练你的自定义模型中包含的层（如 `m.layer1` 和 `m.layer2`）的参数。

这些基本构成要素——作为核心运算单元的层、用于直接序列组合的 `Chain`，以及用于定制架构的自定义结构体——提供了一个多功能且功能强大的工具集。它们使你能够清晰高效地在 Julia 中表达各种神经网络设计。随着你的学习，你会明白这些基本组件是如何组合起来构建高级深度学习 (deep learning)模型的。

## 参考资料

- [Defining Models](https://fluxml.ai/Flux.jl/stable/) — Flux.jl Contributors (2025)
  Publisher: FluxML
  解释如何在 Flux.jl 中使用层、链和自定义结构构建神经网络。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  详细阐述神经网络的基本组件，包括层、参数和前馈模型。
