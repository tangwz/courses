---
course: "julia-for-machine-learning"
chapter: "julia-deep-learning-flux-jl"
lesson: "flux-jl-working-with-gradients"
sourceId: 6908
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/flux-jl-working-with-gradients"
title: "使用梯度"
description: "理解使用 Flux.jl 训练模型时，梯度在反向传播过程中如何计算和使用。"
order: 7
plots: []
sourceHash: "89609e9ab001cf264febb835228f000e015b7b37cb0c69972161f256d5095d3f"
sourceCorrections: []
---

Zygote.jl 等库提供的自动微分能力对于深度学习 (deep learning)非常重要。这使得梯度计算成为可能，这些梯度会在 Flux.jl 的训练流程中得到实际运用。梯度是训练神经网络 (neural network)的基础；它们量化 (quantization)了当模型参数 (parameter)（如权重 (weight)或偏置 (bias)）发生微小改变时，损失函数 (loss function)将变化多少。优化器随后会使用这些信息来调整参数，使其朝着最小化损失的方向移动。

### 使用 `Flux.gradient` 获取梯度

在 Flux.jl 中，梯度通常使用 `Zygote.gradient` 进行计算。虽然 Zygote 是底层自动微分引擎，但 Flux 通常会提供便捷封装或直接使用 Zygote 的函数。要计算梯度，您需要三个主要组成部分：

1. 一个计算损失的函数。此函数最终必须返回一个单一标量值。
2. 您想计算梯度的模型参数 (parameter)。
3. 损失函数 (loss function)所需的输入数据。

让我们来看一个典型使用模式：

```julia
using Flux, Zygote

# 定义一个简单模型
model = Dense(3, 2) # 3个输入特征，2个输出特征

# 样本输入数据和目标输出
x_sample = rand(Float32, 3)
y_target = rand(Float32, 2)

# 定义一个损失函数。它接收模型、输入和目标。
# 这是一个常见模式，但损失函数本身只需要知道
# 计算标量损失所必需的信息。
functionmse_loss(m, x, y_true)
    y_pred = m(x)
    return sum((y_pred .- y_true).^2) / length(y_true) # 均方误差
end

# 使用 Flux.params 收集模型参数
# 这告诉 Zygote 应该对哪些变量进行微分。
parameters = Flux.params(model)

# 计算梯度
# Zygote.gradient 的第一个参数是一个匿名函数
# 它调用我们的损失函数。
grads = Zygote.gradient(() -> mse_loss(model, x_sample, y_target), parameters)
```

在此片段中，`Zygote.gradient` 被调用时传入了一个匿名函数 `() -> mse_loss(model, x_sample, y_target)`。此函数不接收任何参数，并且在调用时执行我们的损失计算。第二个参数 `parameters` 是从 `Flux.params(model)` 获得的一个集合，它告诉 Zygote 应该计算哪些变量的梯度。

结果 `grads` 是一个 `Zygote.Grads` 对象。此对象的行为类似于字典，其键是原始参数数组（例如 `model.weight`、`model.bias`），其值是对应的梯度数组。

### 梯度的结构

`Zygote.Grads` 对象提供了一种简洁的方式来访问您传递给 `Flux.params` 的任何特定参数 (parameter)的梯度。例如，要获取损失相对于我们的 `Dense` 层 `model` 权重 (weight)的梯度：

```julia
# 假设 'model' 和 'grads' 来自上一个例子
gradient_weights = grads[model.weight]
gradient_bias = grads[model.bias]

println("权重的梯度:\n", gradient_weights)
println("偏置的梯度:\n", gradient_bias)
```

如果您的模型是一个包含多个层的 `Chain`，`Flux.params(model)` 将从链中的所有层收集参数。`grads` 对象随后将包含这些参数的条目。例如，如果 `model = Chain(Dense(3, 4), Dense(4, 2))`，那么 `Flux.params(model)` 将包括两个 `Dense` 层的权重和偏置 (bias)，并且 `grads` 将提供对其各自梯度的访问。

```julia
# Chain 示例
complex_model = Chain(
    Dense(10, 5, relu), # 层 1
    Dense(5, 2)         # 层 2
)
x_complex = rand(Float32, 10)
y_complex_target = rand(Float32, 2)

# 复杂模型的参数
params_complex = Flux.params(complex_model)

# 复杂模型的梯度
grads_complex = Zygote.gradient(() -> mse_loss(complex_model, x_complex, y_complex_target), params_complex)

# 访问第一层权重的梯度
# Chain 中的层通常通过索引访问
grad_layer1_weights = grads_complex[complex_model[1].weight]
grad_layer2_bias = grads_complex[complex_model[2].bias]

# println("第1层权重的梯度:\n", grad_layer1_weights)
# println("第2层偏置的梯度:\n", grad_layer2_bias)
```

参数与其梯度之间的这种直接映射对于调试或实现自定义训练逻辑非常方便。

### 训练循环中的梯度

在标准训练循环中，优化器会使用这些梯度来更新模型参数 (parameter)。对于参数 $W$ 的简单梯度下降 (gradient descent)更新的一般公式是：
$W_{new} = W_{old} - \eta \cdot \nabla_W L$
其中 $L$ 是损失，$\nabla_W L$ 是损失相对于 $W$ 的梯度，$\eta$ 是学习率。Flux.jl 优化器（如 `ADAM`、`SGD` 等）实现了此规则的变体。

虽然 `Flux.train!` 自动化了此过程，但理解这些步骤是有益的：

1. **前向传播**：计算模型预测，然后计算损失。
2. **反向传播 (backpropagation)**：计算损失相对于模型参数的梯度（使用 `Zygote.gradient`）。
3. **参数更新**：使用优化器根据梯度调整参数。

以下图表说明了此流程：

> 此图表展示了神经网络 (neural network)训练一个步骤中的操作循环，从输入数据到模型预测、损失计算、梯度计算，最后通过优化器进行参数更新。

当您使用 `Flux.train!(loss_function, params, data, optimizer)` 时，Flux 会在内部执行这些步骤。对于 `data` 中的每个数据批次，它会：

1. 调用 `loss_function`（该函数应执行前向传播并返回损失）。
2. 使用 `Zygote.gradient` 计算相对于 `params` 的梯度。
3. 将每个参数及其对应的梯度传递给 `optimizer`（例如，对于每个参数 `p` 和梯度 `g`，调用 `Flux.Optimise.update!(opt, p, g)`）。

### 检查与理解梯度

手动检查梯度会非常有益，尤其是在调试学习不正常的网络或尝试理解学习动态时。

- **零梯度**：如果某些参数 (parameter)的梯度持续为零，这可能表明这些参数不对损失计算做出贡献（网络中的“死区”），或者它们未包含在 `Flux.params` 中。
- **梯度消失**：在深层网络中，当梯度从输出层反向传播 (backpropagation)到早期层时，它们会变得极小。如果梯度持续微小（例如，$10^{-8}$ 或更小），早期层中的参数将更新非常缓慢，有效地阻碍学习。这在输入饱和时使用某些激活函数 (activation function)（如 Sigmoid）时很常见。
- **梯度爆炸**：反之，梯度可能会变得过大。这可能导致训练不稳定，其中损失剧烈震荡或由于数值溢出而变为 `NaN`（非数字）。大梯度会导致参数的剧烈更新，往往会越过最优值。

一种常见做法是监测梯度的范数。大范数可能表明梯度爆炸，而非常小的范数则可能提示梯度消失。

```julia
# 计算梯度后：
for p in parameters
    g = grads[p]
    if g !== nothing
        # println("参数梯度的范数：", norm(g))
    else
        # println("该参数没有梯度（或未用于损失计算）。")
    end
end
```

梯度裁剪（如果梯度的范数超过阈值，则将其缩小）等技术可以帮助处理梯度爆炸，而架构更改（例如，使用 ReLU 激活函数、残差连接）或谨慎初始化可以缓解梯度消失。这些是更进阶的主题，但它们强调了理解梯度的重要性。

### 关于 `Zygote.pullback` 的说明

虽然 `Zygote.gradient` 方便获取标量损失函数 (loss function)相对于一组参数 (parameter)的梯度，但 Zygote 还提供了一个更基础的函数，称为 `Zygote.pullback`。
“pullback”是一个函数，它接收来自“上游”的梯度（即某个后续计算相对于当前函数输出的梯度），并计算相对于当前函数输入的梯度。

`Zygote.gradient(f, args...)` 本质上是以下操作的简写：

1. 调用 `y, back = Zygote.pullback(f, args...)`。
2. 通过调用 `back(dy)` 计算梯度，其中 `dy` 是最终标量输出的梯度（对于标量损失函数，此值隐式地为 1.0，意味着 $\frac{dL}{dL} = 1$）。

在使用标量损失训练标准 Flux 模型时，您通常不需要直接使用 `Zygote.pullback`。然而，它对于更进阶的场景非常有用，例如：

- 当您需要输出非标量值（例如，向量 (vector)或矩阵）的函数的梯度，并希望手动链式微分时。
- 实现自定义层或操作，您需要对其反向传播 (backpropagation)进行精细控制时。

对于本课程涵盖的大多数深度学习 (deep learning)任务，`Zygote.gradient`（通常由 `Flux.train!` 隐式使用）将会足够。

理解梯度如何计算、结构化和使用是迈向掌握神经网络 (neural network)训练的重要一步。在即将进行的动手实践中，您将运用这些知识，在 Flux.jl 中构建并训练您的第一个神经网络。

## 参考资料

- [Flux.jl Documentation: Training Models](https://fluxml.ai/Flux.jl/stable/manual/training/) — The Flux.jl Contributors (2025)
  提供Flux.jl训练中梯度计算和应用的实用信息，涵盖Flux.params和Zygote.gradient。
- [Zygote.jl Documentation: Gradients](https://fluxml.ai/Zygote.jl/stable/gradients/) — The Zygote.jl Contributors (2023)
  阐释Zygote.jl的自动微分功能，包括Zygote.gradient和Zygote.pullback。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  介绍深度学习基础，包括自动微分、反向传播和基于梯度的优化。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  清晰、循序渐进地解释神经网络中的反向传播和梯度计算。
