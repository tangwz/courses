---
course: "julia-deep-learning"
chapter: "constructing-nn-architectures"
lesson: "custom-layer-creation-flux"
sourceId: 6909
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/custom-layer-creation-flux"
title: "在 Flux 中创建自定义层"
description: "关于在 Flux.jl 中创建自定义神经网络层以实现新颖结构或操作的指引。"
order: 7
plots: []
sourceHash: "7b18b9f57acefa10cba6560a3b9fb87fcf2fed6c309116dd24da0e45baf38c1b"
sourceCorrections: []
---

虽然 Flux.jl 提供了丰富的预置层，足以应对许多常见的神经网络 (neural network)结构，但总会有需要更具体功能的时候。也许您正在实现一篇研究论文中的新颖层，设计一种独特的数据转换，或者需要一个具有特殊结构的可学习参数 (parameter)层。幸运的是，Flux.jl 在设计时考虑了可扩展性，使得自定义层的创建过程相对简单，这主要得益于 Julia 强大的多重派发和灵活的类型系统等特性。

从本质上看，Flux 层通常是一个 Julia `struct`，用于存放层的状态，主要是其可学习参数（如权重 (weight)和偏置 (bias)），以及任何固定的超参数 (hyperparameter)。为了与 Flux 的运行环境结合，这个 `struct` 需要是可调用的（即像函数一样执行其前向传播），并且其可学习参数必须能被 Flux 的优化机制识别。

接下来，我们一步步创建自定义层。我们将构建一个名为 `BiasedActivation` 的简单层，它会向输入添加一个可学习的偏置向量 (vector)，然后应用用户指定的激活函数 (activation function)。

### 定义层结构

首先，我们定义将存储层数据的 `struct`。对于 `BiasedActivation`，这包括可学习的 `bias` 向量 (vector)和 `activation_fn`（在构造后是固定的）。

```julia
using Flux, Functors

struct BiasedActivation
    bias         # 可学习的偏置向量
    activation_fn # 用户指定的激活函数（例如，relu, sigmoid）
end
```

### 初始化层

接下来，我们需要一个构造函数来创建层的实例。这个构造函数将初始化 `bias` 向量 (vector)。我们会用零来初始化它，这是偏置 (bias)的常见做法。偏置向量的大小将取决于该层旨在生成或匹配的输出特征数量。激活函数 (activation function)将作为参数 (parameter)传入。

```julia
# 构造函数
function BiasedActivation(output_dims::Int, activation_fn=identity)
    # 将偏置初始化为零的列向量
    bias_init = zeros(Float32, output_dims) 
    return BiasedActivation(bias_init, activation_fn)
end
```

这里，`output_dims` 决定了 `bias` 向量的大小。如果该层的输入 `x` 的维度是 `(features, batch_size)`，那么 `bias` 的维度应为 `(features, 1)` 或 `(features,)` 以便进行广播。我们的构造函数将 `bias` 初始化为一个包含 `output_dims` 个元素的向量。

### 使用 Functors.@functor 使参数 (parameter)可识别

为了让 Flux 识别并训练 `bias` 参数，我们需要通过 `Functors.jl` 包告知它。这通过使用 `@functor` 宏完成。我们明确地将 `bias` 列为可训练参数。`activation_fn` 未被列出，因此 Flux 会将其视为层结构中固定的部分，而不是在训练期间更新的内容。

```julia
Functors.@functor BiasedActivation (bias,)
```

通过指定 `(bias,)`，我们是在告知 Flux，`bias` 是一个包含应被管理参数（例如，移至 GPU，计算梯度）的字段。如果一个层有多个参数字段，比如 `weights` 和 `bias`，我们就会将它们列为 `(weights, bias)`。

### 实现前向传播

为了使我们的层可用，它需要是可调用的。这意味着我们定义一个方法，允许 `BiasedActivation` 的实例像函数一样被调用，接收输入 `x` 并返回转换后的输出。此方法实现了层的前向传播逻辑。

```julia
function (layer::BiasedActivation)(x::AbstractArray)
    # 添加偏置（元素级，必要时对批次进行广播）
    # 然后应用激活函数
    return layer.activation_fn.(x .+ layer.bias)
end
```

在这个前向传播中，`x .+ layer.bias` 执行元素级加法。如果 `x` 是大小为 `(features, batch_size)` 的矩阵，并且 `layer.bias` 是大小为 `(features,)` 的向量 (vector)（或列向量 `(features,1)`），Julia 的广播规则将正确处理加法。结果随后通过 `layer.activation_fn` 进行元素级处理。

### 使用 Base.show 增强可用性

定义您的层应如何显示是一个推荐做法，例如，在打印包含它的 `Chain` 时。我们可以通过重载 `Base.show` 来实现这一点。

```julia
function Base.show(io::IO, l::BiasedActivation)
    print(io, "BiasedActivation(output_dims=", size(l.bias, 1), 
          ", activation=", nameof(typeof(l.activation_fn)), ")")
end
```

这将提供更清晰的表示，例如：`BiasedActivation(output_dims=10, activation=relu)`。

### 综合应用：使用自定义层

现在，让我们看看 `BiasedActivation` 层的实际运行情况。

```julia
# 创建自定义层的实例
# 假设它处理 5 个特征并使用 relu 激活函数
custom_layer = BiasedActivation(5, relu)

# 检查其参数
params_found = Flux.params(custom_layer)
println("Flux 发现的参数: ", params_found)
# 输出应显示偏置向量

# 创建一些模拟输入数据
# (特征数量, 批次大小)
dummy_input = randn(Float32, 5, 3) 

# 执行前向传播
output = custom_layer(dummy_input)
println("输出形状: ", size(output))

# 自定义层可以是 Chain 的一部分
model = Chain(
    Dense(10, 5), # 标准全连接层
    custom_layer, # 我们的自定义层
    Dense(5, 2),
    softmax
)

println("\n模型结构:")
println(model)

# 用一些数据测试模型
test_data = randn(Float32, 10, 4) # 模型的输入
model_output = model(test_data)
println("模型输出形状: ", size(model_output))
```

当您运行此代码时，`Flux.params(custom_layer)` 将正确识别 `bias` 向量 (vector)为可训练参数 (parameter)。Zygote 作为 Flux 的默认自动微分引擎，只要前向传播中的操作（如 `.+` 和 `relu`）是可微分的，就能够为 `bias` 计算梯度，而它们确实是可微分的。

以下图表说明了定义和使用像我们 `BiasedActivation` 这样的自定义 Flux 层所涉及的主要组成部分。

> 在 Flux 框架中定义和使用自定义 `BiasedActivation` 层所包含的组成部分。

### 自定义层的重要事项

- **参数 (parameter)初始化**: 尽管我们对偏置 (bias)使用了 `zeros`，但权重 (weight)通常需要更精密的初始化（例如，Glorot/Xavier 或 He 初始化）以帮助训练。Flux 提供了 `Flux.glorot_uniform` 和 `Flux.kaiming_uniform` 等函数，您可以在构造函数中使用它们。
- **GPU 兼容性**: 为了使您的自定义层与 GPU 兼容，请确保其参数存储为 `CuArray`（或可以转换为 `CuArray`），并且前向传播中的所有操作都与 `CUDA.jl` 兼容。Flux 的 `gpu(layer)` 函数会尝试将通过 `@functor` 注册的任何参数移至 GPU。如果 `x` 变为 `CuArray`，则 Flux 或 NNlib 提供的 `.+` 等操作和标准激活函数 (activation function)将在 GPU 上运行。如果您的自定义层代码 `(l::MyLayer)(x)` 依赖于这些标准、重载的操作，通常不需要明确的 GPU 逻辑。
- **类型稳定性**: 为了在 Julia 中获得最佳性能，请在您的层的前向传播中争取类型稳定的代码。尽管 Flux 和 Zygote 相当强大，但类型不稳定性有时可能导致执行速度变慢或编译开销增加。
- **测试**: 全面地测试您的自定义层。这包括验证已知输入的输出维度和值，以及检查梯度是否正确计算。您可以使用 `Zygote.gradient` 来实现这一点：

  ```julia
  # 示例：测试 'bias' 的梯度
  layer_instance = BiasedActivation(3, tanh)
  input_data = randn(Float32, 3, 2)

  # 获取相对于第一个参数（bias）的梯度
  # 和输入 (x)
  grads = Zygote.gradient((l, x_val) -> sum(l(x_val)), layer_instance, input_data)

  println("偏置的梯度: ", grads[1].bias) # 访问 'bias' 字段的梯度
  # grads[1] 包含层参数的梯度
  # grads[2] 包含 input_data 的梯度
  ```

  确保 `grads[1].bias` 不是 `nothing` 并且具有预期的形状。

在 Flux 中创建自定义层，让您可以扩展框架以满足几乎所有建模需求。通过遵循定义 `struct`、构造函数、使用 `@functor` 注册参数以及将前向传播实现为可调用方法这一模式，您可以将自己的创新组件融入到 Flux 强大的深度学习 (deep learning)环境中。这种灵活性在研究新结构或使现有结构适应特定问题领域时是一个重要的优势。

## 参考资料

- [Flux: High-Performance ML - Custom Layers](https://fluxml.ai/Flux.jl/stable/models/layers/#Custom-Layers) — The Flux Community (2025)
  Flux.jl的官方文档，详细说明如何定义自定义层、注册参数并将其集成到Flux模型中。
- [Functors.jl Documentation](https://fluxml.ai/Functors.jl/stable/) — The Functors.jl Contributors (2024)
  Functors.jl包的文档，对于使自定义层参数可被Flux的优化机制发现和训练至关重要。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习概念的综合教材，包括神经网络架构、层类型、激活函数和参数初始化策略，这些都是设计自定义层的基本要素。
- [The Julia Language Manual - Types and Methods](https://docs.julialang.org/en/v1/manual/types/) — The Julia Language Contributors (2024)
  Julia语言的官方文档，解释了其类型系统和多重分派，这些是实现Flux中自定义层灵活可扩展设计的核心特性。
