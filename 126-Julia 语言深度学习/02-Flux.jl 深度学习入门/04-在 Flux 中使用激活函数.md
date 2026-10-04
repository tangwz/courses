# 在 Flux 中使用激活函数

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/activation-functions-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

神经网络 (neural network)的强大表达能力很大程度上源于非线性激活函数 (activation function)。如果没有它们，多层网络实质上会退化为单一的线性变换，这会严重限制其对数据中复杂关系的建模能力。激活函数引入了这些必要的非线性，通常在神经元或层内的线性变换（权重 (weight)和偏置 (bias)）之后应用。在构建 Flux.jl 模型时，理解并正确使用这些函数是实现有效网络设计的一个主要步骤。

Flux.jl 提供了一系列常用激活函数，它们易于使用且经过性能优化。这些函数是标准的 Julia 函数，可以逐元素应用于数组，也可以直接集成到层定义中。这种灵活性既能实现标准网络结构，也能支持更多创新架构。

让我们查看一些最常用的激活函数、它们的特性以及如何在 Flux 中使用它们。

## 常用激活函数 (activation function)介绍

每种激活函数都有其独特的特点，会影响神经网络 (neural network)的学习动态和表现。

### Sigmoid

Sigmoid 函数，也称为逻辑函数，将任意实数值映射到 0 到 1 的范围内。
其数学形式是：
$\sigma(x) = \frac{1}{1 + e^{-x}}$
历史上，Sigmoid 曾是隐藏层的流行选择，但由于其某些缺点，它已在很大程度上被 ReLU 等函数取代。

**特性：**

- **输出范围：** (0, 1)。这使得它适合用于二分类问题的输出层，其中输出可以被解释为概率。
- **非线性：** 引入非线性，使网络能够学习复杂函数。
- **梯度消失：** 该函数在两个极端（对于非常大或非常小的输入）会饱和，这意味着其导数变得非常接近零。这可能导致在反向传播 (backpropagation)期间出现梯度消失问题，从而减慢或停止深度网络的学习。
- **非零中心：** 其输出始终为正，这可能导致训练期间梯度更新的效率不高。

在 Flux.jl 中，您可以使用 `sigmoid`：

```julia
using Flux

input_data = randn(Float32, 5) # 示例输入
output_data = sigmoid.(input_data) # 逐元素应用 Sigmoid
println(output_data)

# 在层内
layer = Dense(10, 5, sigmoid) # 具有 5 个输出神经元并使用 Sigmoid 激活的全连接层
```



[交互图表：Sigmoid 激活函数](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/activation-functions-flux#plot-1i4e6ey)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "输入 (x)"
    },
    "yaxis": {
      "title": "输出 (σ(x))"
    },
    "title": "Sigmoid 激活函数"
  },
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        0.0067,
        0.018,
        0.0474,
        0.1192,
        0.2689,
        0.5,
        0.7311,
        0.8808,
        0.9526,
        0.982,
        0.9933
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "Sigmoid",
      "line": {
        "color": "#228be6"
      }
    }
  ]
}
```

</details>



> Sigmoid 激活函数 (activation function)，将输入映射到 (0, 1) 范围。

### 双曲正切 (tanh)

双曲正切函数，即 `tanh`，是另一个与 Sigmoid 类似的 S 形函数，但它将输入映射到 (-1, 1) 的范围。
其数学形式是：
$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$

**特性：**

- **输出范围：** (-1, 1)。
- **零中心：** 与 Sigmoid 不同，`tanh` 的输出是零中心的，这对于优化来说可能是有益的，因为梯度不太可能偏向一个方向。
- **梯度消失：** 与 Sigmoid 类似，`tanh` 也存在梯度消失问题，尤其对于非常大或非常小的输入，尽管其梯度通常比 Sigmoid 的更陡峭。

在 Flux.jl 中，您使用 `tanh`：

```julia
using Flux

input_data = randn(Float32, 5)
output_data = tanh.(input_data)
println(output_data)

layer = Dense(10, 5, tanh) # 具有 tanh 激活的全连接层
```



[交互图表：双曲正切 (tanh) 激活函数](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/activation-functions-flux#plot-zz8zng)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "输入 (x)"
    },
    "yaxis": {
      "title": "输出 (tanh(x))"
    },
    "title": "双曲正切 (tanh) 激活函数"
  },
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        -0.9999,
        -0.9993,
        -0.9951,
        -0.964,
        -0.7616,
        0,
        0.7616,
        0.964,
        0.9951,
        0.9993,
        0.9999
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "tanh",
      "line": {
        "color": "#12b886"
      }
    }
  ]
}
```

</details>



> 双曲正切 (tanh) 激活函数 (activation function)，将输入映射到 (-1, 1) 范围。其零中心输出通常是有益的。

### 整流线性单元 (ReLU)

整流线性单元，即 ReLU，由于其简单和高效，已成为许多类型神经网络 (neural network)中隐藏层的标准激活函数 (activation function)。
其数学形式是：
$\text{ReLU}(x) = \text{max}(0, x)$

**特性：**

- **输出范围：** [0, $\infty$)。
- **非线性：** 尽管它是分段线性形式，但仍引入了非线性。
- **计算效率：** 计算非常简单（一个阈值操作）。
- **缓解梯度消失（对于正输入）：** 对于正输入，梯度是常数 (1)，这有助于在反向传播 (backpropagation)期间有效传播梯度。
- **稀疏性：** 可能导致稀疏激活，因为对于负输入输出零的神经元实际上处于“关闭”状态。这可以提高计算效率和泛化能力。
- **ReLU 死亡问题：** 如果神经元的输入始终低于零，它将始终输出零。因此，其梯度将为零，并且其权重 (weight)将不会更新。该神经元实际上“死亡”并停止对学习做出贡献。
- **非零中心：** 与 Sigmoid 类似，其输出也不是零中心的。

在 Flux.jl 中，`relu` 是该函数：

```julia
using Flux

input_data = randn(Float32, 5)
output_data = relu.(input_data)
println(output_data)

layer = Dense(10, 5, relu) # 具有 ReLU 激活的全连接层
```



[交互图表：整流线性单元 (ReLU) 激活函数](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/activation-functions-flux#plot-1vxhzwv)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "输入 (x)"
    },
    "yaxis": {
      "title": "输出 (ReLU(x))"
    },
    "title": "整流线性单元 (ReLU) 激活函数"
  },
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "ReLU",
      "line": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> 整流线性单元 (ReLU) 激活函数。它在输入为正时直接输出输入，否则输出零。

### Leaky ReLU 及其变体

为了解决“ReLU 死亡”问题，人们提出了 ReLU 的几种变体。一个常见的是 Leaky ReLU。

**Leaky ReLU：** 允许在单元不活跃时存在一个小的非零梯度。
其数学形式是：
$\text{LeakyReLU}(x) = \begin{cases} x & \text{如果 } x > 0 \\ \alpha x & \text{如果 } x \le 0 \end{cases}$
这里 $\alpha$ 是一个小的正数常数，通常在 0.01 到 0.2 左右。

**特性：**

- **解决 ReLU 死亡问题：** 通过允许一个小的负斜率，它能防止神经元变得完全不活跃。
- **保持 ReLU 的优点：** 保留了计算效率和正输入的良好梯度流动。

Flux.jl 提供了 `leakyrelu`。可以指定 $\alpha$ 参数 (parameter)（默认为 0.01）。

```julia
using Flux

input_data = randn(Float32, 5)
output_alpha_default = leakyrelu.(input_data) # 默认 alpha = 0.01
output_alpha_custom = leakyrelu.(input_data, 0.2f0) # 自定义 alpha = 0.2
println("Leaky ReLU (alpha=0.01): ", output_alpha_default)
println("Leaky ReLU (alpha=0.2): ", output_alpha_custom)

layer = Dense(10, 5, x -> leakyrelu(x, 0.1f0)) # 在层中使用自定义 alpha 的 leakyrelu
```

其他变体包括：

- **参数化 ReLU (PReLU)：** $\alpha$ 是一个可学习参数。Flux 没有直接的 `prelu` 函数可以将 $\alpha$ 作为 `Dense` 层激活参数的一部分来学习，但可以通过自定义层来实现。
- **指数线性单元 (ELU)：** 对负输入使用指数函数。Flux 提供了 `elu(x, α=1.0f0)`。ELU 有时可以比 ReLU 带来更快的学习和更好的泛化能力。



[交互图表：ReLU 与 Leaky ReLU 对比 (α=0.1)](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/activation-functions-flux#plot-ugftr3)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "输入 (x)"
    },
    "yaxis": {
      "title": "输出"
    },
    "title": "ReLU 与 Leaky ReLU 对比 (α=0.1)"
  },
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "ReLU",
      "line": {
        "color": "#fd7e14"
      }
    },
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "type": "scatter",
      "mode": "lines",
      "name": "Leaky ReLU (α=0.1)",
      "line": {
        "color": "#be4bdb",
        "dash": "dot"
      }
    }
  ]
}
```

</details>



> ReLU 和 Leaky ReLU 的比较。Leaky ReLU 为负输入引入了一个小的斜率。

### Softmax

Softmax 函数通常用于神经网络 (neural network)的输出层，处理多分类问题。它将一个原始分数（logits）向量 (vector)转换为 $K$ 个不同类别的概率分布。
对于 $K$ 个 logits 的向量 $x$，第 $j$ 个元素的 softmax 是：
$\text{softmax}(x_j) = \frac{e^{x_j}}{\sum_{k=1}^{K} e^{x_k}}$

**特性：**

- **输出范围：** 每个输出值都在 (0, 1) 范围内。
- **概率分布：** 所有输出值的和为 1，使其适合表示概率。
- **多分类：** 非常适合从多个可能性中选择一个类别。通常将概率最高的类别作为预测。
- **通常不用于隐藏层。**

在 Flux.jl 中，`softmax` 对数组进行操作，如果处理批量数据，通常会沿着特定维度进行。对于单个实例（向量），它计算标准的 softmax。

```julia
using Flux

logits = randn(Float32, 3) # 3 个类别的示例 logits
probabilities = softmax(logits)
println("Logits: ", logits)
println("Probabilities: ", probabilities)
println("Sum of Probabilities: ", sum(probabilities)) # 应该接近 1.0

# 在一个用于 3 分类问题的模型中：
model = Chain(
  Dense(10, 3), # 具有 3 个神经元（每个类别一个）的输出层
  softmax       # 应用 softmax 获取概率
)
# 注意：当与交叉熵损失函数一起使用时，通常将 logits 直接传递给损失函数
# （例如 `Flux.logitcrossentropy`），它在内部应用 softmax 或其稳定等效形式。
# 然而，要从模型获取直接概率输出，则需要显式应用 softmax。
```

需要注意的是，当使用像 `Flux.logitcrossentropy` 这样的损失函数 (loss function)时，通常会将原始 logits（`Dense` 层在 softmax 之前的输出）直接传递给损失函数。这是因为 `logitcrossentropy` 将 softmax 操作与交叉熵计算结合起来，以获得更好的数值稳定性和效率。然而，如果在推理 (inference)期间需要从模型中获取实际的概率输出，则需要显式应用 `softmax`。

## 在 Flux 层中应用激活函数 (activation function)

如示例所示，Flux.jl 使将激活函数集成到层中变得简单直观。例如，`Dense` 层接受激活函数作为其第三个参数 (parameter)：

```julia
# 具有 10 个输入特征、20 个输出特征和 ReLU 激活的全连接层
hidden_layer = Dense(10, 20, relu)

# 用于二分类的输出层，20 个输入，1 个输出，sigmoid 激活
output_layer_binary = Dense(20, 1, sigmoid)
```

如果在 `Dense` 层中省略激活函数，它会默认为 `identity`，这意味着不应用任何激活（一个线性层）。

```julia
linear_layer = Dense(5, 5) # 等同于 Dense(5, 5, identity)
```

您还可以使用 Julia 的广播语法将激活函数直接应用于层的输出或任何数组：

```julia
using Flux

# 示例：在线性层计算后应用 relu
W = randn(Float32, 3, 5) # 权重矩阵
b = randn(Float32, 3)    # 偏置向量
x = randn(Float32, 5)    # 输入向量

z = W * x .+ b    # 线性变换
h = relu.(z)      # 逐元素应用 relu
println(h)
```

这种逐元素应用是激活函数如何作用于神经元输出的基础。

## 激活函数 (activation function)选择指南

选择合适的激活函数可以显著影响模型的表现，但没有普遍适用的规则。不过，存在一些通用的指导原则和常见做法：

- **ReLU 通常是隐藏层的良好默认选择。** 它计算高效且通常表现良好。从 ReLU 开始，如果遇到神经元死亡等问题，再考虑其变体（Leaky ReLU、ELU）。
- **对于输出层：**
  - **二分类：** 使用 `sigmoid` 来获取正类的概率输出。
  - **多分类：** 使用 `softmax` 来获取所有类别的概率分布。
  - **回归：** 通常，如果输出可以是任何实数值，输出层不使用激活函数（或 `identity`）。如果输出受限制（例如，始终为正），则可以考虑使用 `relu` 或 `softplus` 等适当的函数。
- **如果可能，避免在深层隐藏层中使用 Sigmoid 和 tanh**，因为它们存在梯度消失问题。通常更推荐使用 ReLU 及其变体。
- **尝试是常见做法。** 激活函数的选择可以被视为一个超参数 (parameter) (hyperparameter)。尝试不同的函数，查看哪种最适合您的具体问题和架构。

当您在 Flux.jl 中构建更复杂的网络时，您会越来越熟悉这些函数，并培养出选择哪种函数的直觉。请记住，Julia 和 Flux 的灵活性甚至允许您在应用程序需要独特功能时定义自定义激活函数。接下来的部分将介绍损失函数 (loss function)和优化器，它们与您的网络架构和激活函数配合工作，以有效训练您的模型。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖深度学习的理论基础和实践方面，其中有专门章节解释激活函数、它们的特性以及对网络训练的影响。
- [Activation Functions](https://fluxml.ai/Flux.jl/stable/reference/activations/) — The Flux.jl Community (2024)
  Flux.jl库中可用激活函数的官方文档，提供使用示例和与使用Flux构建模型直接相关的实现细节。
- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEo1odL7ikjncWzFLNTsN0Ed7Sj5s-7AiB7ItOfX2Tnq4TuF8-nlPpCDW8l_gcf_LT35ny0tj9oqPBOoAjq5JBWFsl4CuFX0qYDfhDxommNLZzdcOyL8aCFRgKxgFmRU4Zs5b-3m8GGNqwegBk6GCpAtrVer9SYx_nOav3-mU=) — Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng (2013)
  Journal: Proceedings of the 30th International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research; Volume: 28; Pages: 3; DOI: [10.5591/978-1-57735-611-3-A109](https://doi.org/10.5591/978-1-57735-611-3-A109)
  介绍并评估了Leaky修正线性单元（Leaky ReLU），展示了它们通过允许负输入有一个小的非零梯度来解决“神经元死亡”问题的有效性。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2024)
  Publisher: Cambridge University Press
  一本互动且易于理解的在线教材，提供深度学习模型的理论解释和实践实现，包括对各种激活函数及其应用的详细。

---

[上一节](03-%E5%AE%9A%E4%B9%89%E7%AE%80%E5%8D%95%E7%9A%84%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%B1%82.md) · [下一节](05-%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%9A%E8%A1%A1%E9%87%8F%E6%A8%A1%E5%9E%8B%E8%AF%AF%E5%B7%AE.md)
