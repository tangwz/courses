---
course: "julia-for-machine-learning"
chapter: "julia-deep-learning-flux-jl"
lesson: "zygote-jl-automatic-differentiation"
sourceId: 6906
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/zygote-jl-automatic-differentiation"
title: "使用 Zygote.jl 的自动微分"
description: "了解 Zygote.jl，Julia 的源代码到源代码自动微分包，以及它在与 Flux.jl 结合进行深度学习中的作用。"
order: 6
plots: []
sourceHash: "8e3d05812c60ab137df9bab373bdef8816a3689f0704b5063d98343f5e442419"
sourceCorrections: []
---

训练神经网络 (neural network)涉及调整模型参数 (parameter)（权重 (weight)和偏置 (bias)）以使损失函数 (loss function)最小化。这种调整由损失函数相对于这些参数的梯度指导。计算这些梯度，特别是对于拥有数百万参数的复杂模型，是一项重要的计算任务。自动微分（AD）在此发挥作用，Zygote.jl 是 Julia 中实现此功能的主要工具。

自动微分是一种技术，它能针对一个计算值的计算机程序，生成另一个计算该值导数的程序。它不同于：

- **数值微分：** 使用有限差分近似导数（例如， $f'(x) \approx \frac{f(x+h) - f(x)}{h}$）。它实现起来可能很简单，但存在近似误差，并且对于大量参数来说计算成本可能很高。
- **符号微分：** 使用微积分规则处理数学表达式（就像你手动操作一样）。虽然精确，但它可能导致非常复杂的表达式（“表达式膨胀”），评估效率低下，特别是对于具有控制流的复杂程序。

自动微分提供了符号微分的精度，同时计算效率接近于评估原始函数本身，这使其成为深度学习 (deep learning)的理想选择。

## Zygote.jl：Julia 的可微分编程引擎

Zygote.jl 是 Julia 生态系统中一个功能强大的包，提供自动微分功能。Zygote 之所以对 Julia 特别有效和适用，是因为它的“源代码到源代码”自动微分机制。Zygote 不是显式构建计算图或依赖于有限类型集的运算符重载（基于磁带的自动微分），而是直接将现有 Julia 代码转换为计算梯度的 Julia 新代码。

这意味着 Zygote 可以对 Julia 的一系列原生特性进行微分，包括控制流（循环、条件）、数据结构，甚至用户定义类型，通常无需或只需少量修改原始代码。它与语言深度集成，使得梯度计算非常灵活且高效。

> Zygote.jl 分析你的 Julia 函数并生成一个专门用于计算其梯度的新函数。

## Zygote.jl 的基本梯度计算

Zygote 中用于获取梯度的核心函数是 `Zygote.gradient`。我们来看它如何与一些简单的 Julia 函数配合使用。

首先，确保你已安装并加载 Zygote.jl：

```julia
using Pkg
Pkg.add("Zygote")
using Zygote
```

考虑一个简单的标量函数：
$f(x) = 3x^2 + 2x + 1$
导数是 $f'(x) = 6x + 2$。我们来求 $x=5$ 时的梯度：

```julia
f(x) = 3x^2 + 2x + 1
df_dx_at_5 = Zygote.gradient(f, 5)
println(df_dx_at_5) # 输出：(32.0,)
```

Zygote 返回一个元组，其中包含相对于每个输入参数 (parameter)的梯度。因为 `f` 接受一个参数，所以元组有一个元素，即 $6(5) + 2 = 32$。

现在，我们尝试一个具有多个参数的函数：
$g(x, y) = xy + \sin(x)$
偏导数是 $\frac{\partial g}{\partial x} = y + \cos(x)$ 和 $\frac{\partial g}{\partial y} = x$。
我们来计算 $x=\pi$ 和 $y=2$ 时的梯度：

```julia
g(x, y) = x*y + sin(x)
dg_dx_dy_at_pi_2 = Zygote.gradient(g, pi, 2.0) # 注意：pi 是 Float64 类型
println(dg_dx_dy_at_pi_2) # 输出：(1.0, 3.141592653589793)
```

结果是一个元组 $(y + \cos(x), x)$。当 $x=\pi, y=2$ 时，结果是 $(2 + \cos(\pi), \pi) = (2 - 1, \pi) = (1, \pi)$，这与 Zygote 的输出一致。

## Zygote.jl 与 Flux.jl：高效协作

虽然你可以直接使用 Zygote.jl，但 Flux.jl 在训练神经网络 (neural network)时会将其作为底层工具。当你定义一个损失函数 (loss function)并让 Flux 训练你的模型时，Zygote 就是计算所需梯度的引擎。

回想一下，训练循环涉及计算损失函数相对于模型参数 (parameter)的梯度。以下是 Zygote 通常如何融入其中：

1. **识别参数：** 你使用 `Flux.params()` 告诉 Flux 哪些参数需要梯度。

   ```julia
   using Flux

   # 一个简单模型
   model = Dense(10, 5, relu) # 10 个输入，5 个输出，ReLU 激活
   ps = Flux.params(model) # 收集模型的权重和偏置
   ```
2. **定义损失函数：** 你的损失函数接收模型输入和真实标签，并使用模型进行预测。

   ```julia
   # 示例输入和目标
   x_sample = randn(Float32, 10)
   y_target = randn(Float32, 5)

   # MSE 损失
   loss(x, y) = sum((model(x) .- y).^2)
   ```
3. **计算梯度：** 使用 `Zygote.gradient` 和一个调用你的损失函数的匿名函数。`gradient` 的第二个参数是你想计算梯度的参数集合 (`ps`)。

   ```julia
   # 计算损失函数相对于 ps 中参数的梯度
   grads = Zygote.gradient(() -> loss(x_sample, y_target), ps)
   ```

   `() -> loss(x_sample, y_target)` 部分创建一个零参数匿名函数。Zygote 对此函数相对于 `ps` 中的元素进行微分。`grads` 将是一个 `Zygote.Grads` 对象，其行为类似于将参数映射到其梯度的字典。

   例如，要获取模型权重 (weight)的梯度：

   ```julia
   # model.weight 是 Dense 层中权重的参数
   ∇_weights = grads[model.weight]
   ∇_bias = grads[model.bias]
   println("权重的梯度：", size(∇_weights))
   println("偏置的梯度：", size(∇_bias))
   ```

然后，优化器（如 `ADAM` 或 `SGD`）在 `Flux.update!` 中使用此 `grads` 对象来调整模型参数：

```julia
opt = ADAM()
# 在训练循环中，你会这样做：
# Flux.update!(opt, ps, grads)
```

## 编写 Zygote 友好的代码

Zygote 的优势之一是它能够对各种标准 Julia 代码进行微分。然而，为了获得最佳结果并避免问题：

- **首选纯函数：** 不修改其输入或具有其他副作用的函数通常更容易被 Zygote 处理。虽然 Zygote 改进了对变异的处理，但纯函数通常会带来更直接的微分。
- **`gradient` 的标量输出：** `Zygote.gradient(f, args...)` 最常见的用法是假设 `f` 返回一个标量（如损失值）。如果 `f` 返回非标量（例如，一个向量 (vector)），你需要指定如何将此输出“转换”为用于反向传播 (backpropagation)的标量，通常是通过提供初始的“敏感度”或“种子”梯度。对于 Flux 中典型的损失函数 (loss function)，这会自动处理，因为损失本身就是标量。
- **类型稳定性：** 与 Julia 的一般性能一样，类型稳定的代码在使用 Zygote 时通常表现更好。
- **原地操作：** Zygote 传统上在原地（变异）数组操作（例如，`A[1] = 0.0` 或 `A .+= B`）方面存在问题。虽然支持已有所改进，但显式的非变异版本（例如，`A = [0.0; A[2:end]]` 或 `A = A .+ B`）通常更安全。对于需要变异的性能关键代码，Zygote 提供了像 `Zygote.Buffer` 这样的工具。然而，对于 Flux 中大多数常见的神经网络 (neural network)层和操作，这些都已为你处理好。

Flux.jl 本身就被设计为与 Zygote 兼容，因此当你组合 Flux 提供的层和损失函数时，它们通常能正确微分。

## 优化器背后的能力

Zygote.jl 是 Julia 深度学习 (deep learning)生态系统中的核心组成部分。通过提供高效灵活的自动微分，它使得 Flux.jl 能够训练复杂的神经网络 (neural network)架构。理解 Zygote 在幕后计算梯度可以帮助你调试问题，甚至为你的模型编写自定义的可微分组件。随着学习进展，请记住，每当优化器更新模型权重 (weight)时，Zygote 很可能都参与了确定这些权重应如何变化的过程。这种强大的能力正是神经网络学习的动力。

## 参考资料

- [Automatic Differentiation in Machine Learning: A Survey](http://jmlr.org/papers/v18/17-468.html) — Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind (2018)
  Journal: Journal of Machine Learning Research; Publisher: JMLR; Volume: 18; Pages: 1−43
  全面回顾了自动微分、其各种模式以及对现代机器学习的重要性，包括与数值微分和符号微分的比较。
- [Zygote.jl Documentation](https://fluxml.ai/Zygote.jl/dev/) — The Zygote.jl Developers (2024)
  Publisher: FluxML
  Zygote.jl 的官方来源，详细介绍了其源到源自动微分机制、使用方法以及 Julia 中可微分编程的功能。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Developers (2025)
  Publisher: FluxML
  提供了关于 Zygote.jl 如何在 Flux.jl 框架内用于构建和训练深度学习模型的信息，特别是参数管理和梯度计算方面。
