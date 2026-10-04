# Zygote.jl：Flux中的自动微分

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/zygote-jl-automatic-differentiation-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了有效训练我们的神经网络 (neural network)，我们需要一种方法来计算每个模型参数 (parameter)（权重 (weight)和偏置 (bias)）的微小变化如何影响整体损失。这种计算涉及求导数。导数会指引学习过程。手动推导复杂网络的这些导数既繁琐又容易出错。自动微分（AD）在此发挥作用，而在Flux.jl体系中，Zygote.jl是表现出色的部分。

Zygote.jl是Julia中一个先进的自动微分包。它通过对Julia代码执行源代码到源代码的转换来工作。这意味着Zygote会“读取”您的Julia函数，包括您的损失函数 (loss function)和模型定义，并生成计算导数的新Julia代码。它被设计为与Julia的多种功能高度兼容，使其使用起来很自然。您编写标准的Julia代码，而Zygote，在大多数情况下，能自动处理其微分。Flux.jl在导数计算方面非常依赖Zygote。

### 自动微分快速回顾

在第1章中，我们介绍了自动微分的基本原理。简要提醒一下，AD技术计算由计算机程序指定的函数的精确导数。与可能导致复杂表达式的符号微分不同，也与可能存在近似误差和计算成本的数值微分不同，AD（特别是反向模式AD，也称为反向传播 (backpropagation)）提供了一种高效且准确的获取导数的方法。Zygote主要使用反向模式AD，这非常适合深度学习 (deep learning)中常见的场景，即有许多输入参数 (parameter)（如模型权重 (weight)）和一个标量输出（损失）。

### Zygote.jl的实践：在Flux中

在我们了解Zygote如何与Flux紧密结合之前，我们先看一个独立示例来了解其核心功能。Zygote提供了一个`gradient`函数，它接收一个函数及其输入参数 (parameter)，并返回导数。

考虑一个简单的多项式函数：$f(x) = 3x^2 + 2x + 1$。导数为 $f'(x) = 6x + 2$。
让我们使用Zygote来求$x=5$处的导数：

```julia
using Zygote

# 定义函数
f(x) = 3x^2 + 2x + 1

# 求导的数值
x_val = 5

# 计算导数
# gradient(f, x_val)返回一个元组；第一个元素是相对于x_val的导数
grad_f_at_5 = gradient(f, x_val)[1]

println("函数 f(x) = 3x^2 + 2x + 1")
println("导数 f'(x) = 6x + 2")
println("当 x = $x_val 时, f'($x_val) = $(6*x_val + 2)")
println("Zygote.gradient(f, $x_val) 得到: $grad_f_at_5")

# 输出:
# 函数 f(x) = 3x^2 + 2x + 1
# 导数 f'(x) = 6x + 2
# 当 x = 5 时, f'(5) = 32
# Zygote.gradient(f, 5) 得到: 32
```

如您所见，Zygote正确地计算了导数。这种对任意Julia代码进行微分的能力是它如此强大的原因。

### Zygote.jl与Flux.jl：紧密合作

那么，这与训练我们的Flux模型有何关联呢？当我们定义一个依赖于模型输出和真实标签的损失函数 (loss function)时，我们需要找到此损失相对于模型中所有可训练参数 (parameter)（层中的权重 (weight)和偏置 (bias)）的导数。

Flux.jl提供了一个方便的函数`Flux.params()`，它从模型或层中收集所有可训练参数。然后，您将这些参数传递给Zygote的`gradient`函数，同时传递一个计算损失的匿名函数。Zygote会处理其余部分。

让我们用一个小的Flux模型来说明：

```julia
using Flux, Zygote

# 一个简单模型：一个全连接层
model = Dense(3 => 2, sigmoid) # 3个输入，2个输出，sigmoid激活

# 模拟输入数据（为简单起见，批处理大小为1）
input_data = randn(Float32, 3, 1)

# 模拟目标输出
target_output = [0.2f0, 0.8f0]

# 定义一个损失函数（均方误差）
# 注意：模型在这里作为参数传递
function compute_loss(m, x, y_true)
    y_pred = m(x)
    return Flux.mse(y_pred, y_true)
end

# 获取模型的可训练参数
parameters = Flux.params(model)
println("Flux.params追踪的参数：", parameters)

# 计算导数
# gradient的第一个参数是一个匿名函数 () -> ...
# Zygote将执行并进行微分。
grads = Zygote.gradient(() -> compute_loss(model, input_data, target_output), parameters)

# grads是一个类似字典的对象，将参数映射到它们的导数
println("\n模型权重的导数：\n", grads[model.weight])
println("\n模型偏置的导数：\n", grads[model.bias])
```

在此示例中：

1. 我们定义了一个`model`和一个`compute_loss`函数。
2. `Flux.params(model)`收集我们`Dense`层的权重矩阵和偏置向量 (vector)。这些是我们想要获取导数的参数。
3. `Zygote.gradient(() -> compute_loss(model, input_data, target_output), parameters)`完成了关键操作。
   - 匿名函数`() -> compute_loss(...)`被执行。在此执行过程中，Zygote会追踪所有涉及`parameters`的操作。
   - 损失计算完成后（前向传播），Zygote执行反向传播 (backpropagation)，以计算损失相对于`parameters`中每个元素的改变情况。
4. `grads`对象包含这些导数。例如，`grads[model.weight]`给出损失相对于模型权重矩阵的导数。这些正是优化器（如`ADAM`或`SGD`）更新模型参数所需要的。

以下图表说明了这种一般流程，包括优化器如何使用这些导数：

> 一个典型训练步骤中梯度计算和参数更新的数据流。Zygote.jl计算损失相对于模型参数的导数，优化器随后使用这些导数来调整参数。

### Zygote如何理解您的代码

Zygote对大部分“普通”Julia代码进行微分的能力是其最重要的优点之一。它通过源代码到源代码转换方法来实现这一点。它会解析您的Julia代码并生成计算导数的新Julia代码。这意味着：

- **最小干扰：** 您通常不需要以特殊的“可微分”风格重写函数，或为AD使用特定数据类型。标准Julia函数、控制流（循环、条件）和数据结构通常可以直接使用。
- **可组合性：** 因为Zygote理解Julia，它自然地处理函数的组合。如果您通过组合较小的、可微分的Julia函数来构建复杂的模型或损失函数 (loss function)，Zygote通常可以对整个函数进行微分。

### Zygote和`Flux.train!`

在前面的章节中，我们提到了`Flux.train!`，这是用于自动化训练循环的实用函数。当您使用`Flux.train!(loss, params, data, opt)`时，Zygote在`Flux.train!`的每一步中都在幕后勤奋工作。它计算您的`loss`函数相对于`params`的导数，这些导数随后被`opt`优化器用于更新参数 (parameter)。虽然`Flux.train!`抽象了对`Zygote.gradient`的直接调用，但了解Zygote是推动这一过程的引擎，这对于调试和更高级的自定义很有帮助。

### Julia体系中的优点

Zygote特别适合Julia，原因在于：

- **多重分派：** Julia的多重分派允许Zygote为特定的函数和类型组合定义自定义的微分规则（伴随式），这带来灵活性和效率。
- **可扩展性：** 对于Zygote无法自动微分的函数，或者已知更高效的手动导数的情况，用户可以定义自定义的“伴随式”。这允许AD系统根据特定需求进行扩展和优化，尽管这是一个更高级的话题。

### 使用Zygote时需注意的事项

虽然Zygote功能强大，但仍有一些需要注意的事项：

- **变动：** 对会变动数组或数据结构的代码进行自动微分可能比较复杂。Zygote对许多常见的变动操作（如`x .+= y`）有良好支持，但对于更复杂的场景，您可能需要使用非变动版本或`Zygote.Buffer`之类的工具来处理中间结果。通常，倾向于非变动风格可以使微分更直接。
- **不支持的代码：** 某些Julia操作可能（尚）无法被Zygote微分（例如，被微分代码中的某些I/O操作或外部函数调用）。如果Zygote遇到无法处理的情况，通常会报错。
- **性能：** Zygote通常性能很好，但您的代码结构有时会影响导数计算的效率。性能分析工具可以在需要时帮助查找瓶颈。
- **调试导数：** 如果您的模型训练不如预期，有时可能需要调试导数。Zygote中的工具以及对损失函数 (loss function)和模型结构的仔细检查是您的帮助。

Zygote.jl根本上简化了获取导数的任务，这是训练深度学习 (deep learning)模型的基础。通过与Flux.jl结合并借助Julia的表达能力，它让您能够更专注于设计模型架构和训练策略，而不是导数的计算。随着您构建更复杂的模型，Zygote将成为您的Julia深度学习工具包中不可或缺的部分。

## 参考资料

- [Zygote.jl: Automatic Differentiation in Julia](https://fluxml.ai/Zygote.jl/stable/) — Zygote.jl Contributors (2024)
  提供Zygote API、功能和用法的基本信息，包括自定义伴随和处理修改。
- [Flux.jl: An Elegant Deep Learning Library](https://fluxml.ai/Flux.jl/stable/) — Flux.jl Contributors (2025)
  描述了Flux深度学习框架及其架构，以及它如何在模型训练期间利用Zygote.jl进行梯度计算。
- [Zygote.jl: An Adjoint-Generating Compiler](https://dl.acm.org/doi/10.1145/3317559.3317666) — Mike Innes, Jesse Bettencourt, Alan Edelman, and Keno Fischer (2019)
  Journal: Proceedings of Machine Learning and Systems 1 (MLSys 2019); Publisher: ACM; Pages: 184-194; DOI: [10.1145/3317559.3317666](https://doi.org/10.1145/3317559.3317666)
  介绍了Zygote.jl的设计和实现，解释了其在Julia中进行自动微分的源到源转换方法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了反向传播的详细解释，这是深度学习中反向模式自动微分的一个特定实例，也是其基础。

---

[上一节](06-%E4%BC%98%E5%8C%96%E5%99%A8%EF%BC%9A%E5%BC%95%E5%AF%BC%E5%AD%A6%E4%B9%A0%E8%BF%87%E7%A8%8B.md) · [下一节](08-%E6%9E%84%E5%BB%BA%20Flux%20%E4%B8%AD%E7%9A%84%E5%9F%BA%E7%A1%80%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
