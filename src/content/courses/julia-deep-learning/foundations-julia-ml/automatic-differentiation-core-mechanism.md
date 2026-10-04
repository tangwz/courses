---
course: "julia-deep-learning"
chapter: "foundations-julia-ml"
lesson: "automatic-differentiation-core-mechanism"
sourceId: 6870
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-1-foundations-julia-ml/automatic-differentiation-core-mechanism"
title: "自动微分：核心机制"
description: "介绍自动微分的基本原理，它是现代深度学习系统的重要组成部分。"
order: 5
plots: []
sourceHash: "e7ffec2940d8a7da57c77d8f931d66db96d7826fd37784a649ae8faed20757bc"
sourceCorrections: []
---

训练机器学习 (machine learning)模型，特别是深度神经网络 (neural network)，通常归结为优化问题：找到最小化损失函数 (loss function)的模型参数 (parameter)。解决这类优化问题最常见的方式是基于梯度的方法，例如梯度下降 (gradient descent)。这些方法需要损失函数相对于模型参数的导数（或在多参数情况下的梯度）。自动微分（AD）是一种强大的技术，能高效准确地计算这些导数。

AD并非获取导数的唯一方法。你可能熟悉以下方式：

1. **符号微分**：这通常是你手动或使用计算机代数系统（如SymPy或Mathematica）进行的操作。它涉及将微分规则应用于数学表达式，以推导出一个新的导数表达式。虽然精确，但符号微分可能导致非常复杂的表达式（“表达式膨胀”），尤其对于复杂的函数，求值速度会很慢。
2. **数值微分**：此方法使用有限差分近似导数，例如，当 $h$ 很小时，$f'(x) \approx \frac{f(x+h) - f(x)}{h}$。虽然易于实现，但它存在两个主要问题：
   - **截断误差**：由近似公式本身引起（泰勒级数中忽略的项）。
   - **舍入误差**：由浮点数的有限精度引起。
   - 对于具有许多输入（如神经网络参数）的函数来说，它的计算成本也很高，因为它需要至少 $N+1$ 次函数求值才能计算 $N$ 个变量的梯度。

另一方面，自动微分计算作为计算机程序指定的函数的导数。它通过将计算分解为一系列基本算术运算（加法、乘法等）和基本函数（sin、cos、exp、log等）来实现。然后，AD反复对这些运算应用微积分的链式法则，累积导数。主要优点是AD能以机器精度计算导数，就像符号微分一样，但对于复杂函数和大量变量而言，它在计算上通常更高效，尤其是在使用其“反向模式”时。

### 基本思路：计算图和链式法则

本质上，AD将程序计算的任何函数视为基本运算的组合。这些运算可以被视为计算图，其中节点表示中间变量或运算，边表示数据流。

考虑一个简单的函数：$y = f(x_1, x_2) = \ln(x_1) + x_1 x_2 - \sin(x_2)$。
我们可以将其分解为一系列运算：

1. $v_1 = \ln(x_1)$
2. $v_2 = x_1 \cdot x_2$
3. $v_3 = \sin(x_2)$
4. $v_4 = v_1 + v_2$
5. $y = v_5 = v_4 - v_3$

该序列可以用以下图表示：

> 一个用于 $y = \ln(x_1) + x_1 x_2 - \sin(x_2)$ 的计算图。输入节点 $x_1, x_2$ 进入中间运算，最终生成输出 $y$。

AD遍历此图，并在每一步应用链式法则。实现此目的有两种主要模式：前向模式和反向模式。

### 前向模式AD

在前向模式中，AD通过将导数与函数本身的求值一起，*向前*传播通过计算图来计算导数。对于每个基本运算，它计算其值以及其相对于输入变量的导数。

如果我们要计算 $\frac{\partial y}{\partial x_1}$，前向模式将追踪每个中间变量 $v_i$ 相对于 $x_1$ 的导数，记作 $\dot{v_i} = \frac{\partial v_i}{\partial x_1}$。
规则如下：

- $\dot{x_1} = 1$，$\dot{x_2} = 0$（如果对 $x_1$ 求导）。
- 如果 $u = c \cdot v$，则 $\dot{u} = c \cdot \dot{v}$。
- 如果 $u = v + w$，则 $\dot{u} = \dot{v} + \dot{w}$。
- 如果 $u = v \cdot w$，则 $\dot{u} = \dot{v} \cdot w + v \cdot \dot{w}$（乘积法则）。
- 如果 $u = g(v)$，则 $\dot{u} = g'(v) \cdot \dot{v}$（链式法则）。

如果输入变量少而输出变量多，或者一次只需要计算一个输入变量的导数，前向模式会很高效。对于具有 $N$ 个输入的函数，要获得完整梯度，通常需要运行前向传播 $N$ 次，每次将不同输入变量的导数设为1，其他设为0。

### 反向模式AD

反向模式AD，在神经网络 (neural network)中常被称为反向传播 (backpropagation)，通过从最终输出向后传播通过图来计算导数。它包含两个阶段：

1. **前向传播**：原始函数从输入到输出进行求值。所有中间变量（$v_i$）的值都被计算并存储。
2. **反向传播**：导数（称为“伴随量”或“敏感度”）从输出开始计算。变量 $v_i$ 的伴随量，记作 $\bar{v_i}$，是最终输出 $y$ 相对于 $v_i$ 的导数：$\bar{v_i} = \frac{\partial y}{\partial v_i}$。
   该过程从 $\bar{y} = \frac{\partial y}{\partial y} = 1$ 开始。然后，对于作为生成 $v_k$ 的运算的输入节点 $v_j$，其伴随量 $\bar{v_j}$ 根据 $\bar{v_k}$ 和局部偏导数 $\frac{\partial v_k}{\partial v_j}$ 使用链式法则进行更新：$\bar{v_j} = \sum_{k \text{ where } v_j \text{ is input to } v_k} \bar{v_k} \frac{\partial v_k}{\partial v_j}$。

反向模式对于具有许多输入变量（如深度神经网络中的数百万参数 (parameter)）和单个标量输出（如损失函数 (loss function)）的函数来说，效率非常高。它允许以大致与原始函数几次求值相同的计算成本计算整个梯度（相对于所有输入的导数），而与输入数量无关。这就是它成为训练大多数深度学习 (deep learning)模型的根本原因。

### Julia中的自动微分

Julia的语言特性，如其动态类型系统、多重派发和元编程能力，使其成为开发强大而灵活的AD系统的优秀平台。编译器根据类型进行代码优化的能力，使得AD工具通常能够达到与手写导数代码相当的性能。

Julia中有几个包提供了AD功能。一些著名的包括：

- **ForwardDiff.jl**：实现前向模式AD。对于输入数量较少的函数，或者计算输入少、输出多的函数的雅可比矩阵，它通常非常快。它通常通过对特殊“对偶数”类型进行算术运算重载来实现。
- **Zygote.jl**：一个主要专注于反向模式的源到源AD系统。它通过转换Julia代码本身来生成其梯度代码。Zygote.jl是Flux.jl（Julia主要的深度学习 (deep learning)库）的底层，旨在高度灵活并与Julia的特性良好配合。
- **ReverseDiff.jl**：提供反向模式AD，通常通过“录制”操作（在计算图上记录它们）然后在此录制上执行反向传播 (backpropagation)来实现。
- **FiniteDifferences.jl**：虽然并非严格意义上的AD，但它提供了数值微分工具，这对于测试从AD系统获得的梯度很有用。

你不需要自己实现AD。这些包允许你获取Julia函数的梯度，通常只需对现有代码进行最少的修改。例如，要使用ForwardDiff.jl获取 $f(x) = x^3 + 2x$ 在 $x=3$ 处的导数，你可以这样编写：

```julia
# 这是一个简短的例子。包的安装和详细用法
# 将在“设置你的Julia深度学习环境”
# 以及后续章节中介绍。

# import Pkg; Pkg.add("ForwardDiff") # 如果尚未安装
using ForwardDiff

f(x) = x^3 + 2x
x_val = 3.0

# 计算f在x_val处的导数
df_dx = ForwardDiff.derivative(f, x_val)

println("函数: f(x) = x^3 + 2x")
println("x的值: $x_val")
println("f'(x)在x = $x_val处的导数: $df_dx") # 预期结果：3*x^2 + 2 = 3*(3^2) + 2 = 27 + 2 = 29
```

这个简单的例子表明了AD工具是多么容易应用。在深度学习的背景下，像Flux.jl这样的库集成了AD（主要通过Zygote.jl），从而使训练神经网络 (neural network)所需的梯度在幕后自动计算。这让你能够专注于定义模型架构和训练过程，而AD系统则处理复杂的微积分。

理解自动微分的原理对你在机器学习 (machine learning)方面的进步很有价值，因为它有助于调试、性能优化，甚至在需要时设计自定义模型组件。当我们开始构建神经网络时，你将看到AD实际运行，它默默而高效地驱动着学习过程。

## 参考资料

- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717761) — Andreas Griewank, Andrea Walther (2008)
  Publisher: SIAM; DOI: [10.1137/1.9780898717761](https://doi.org/10.1137/1.9780898717761)
  一本关于自动微分的综合教材，涵盖其理论基础、模式（前向和反向）及应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本深度学习的标准教材，详细解释了反向传播作为反向模式自动微分的应用。
- [ForwardDiff.jl Documentation](https://juliadiff.org/ForwardDiff.jl/stable/) — Jarrod Taylor and contributors (2025)
  ForwardDiff.jl 的官方文档，提供了其基于对偶数的前向模式自动微分实现及使用方法的说明。
