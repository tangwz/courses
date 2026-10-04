---
course: "julia-for-machine-learning"
chapter: "julia-deep-learning-flux-jl"
lesson: "flux-jl-simple-nn-practice"
sourceId: 6910
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/flux-jl-simple-nn-practice"
title: "动手实践：搭建和训练一个简单的神经网络"
description: "使用 Flux.jl 搭建、训练和评估一个用于分类或回归任务的基础神经网络。"
order: 8
plots: ["plots/6910-0.json", "plots/6910-1.json"]
sourceHash: "213188b509e9d577fb055337c4741e8f75ecd32655eae874c06caf2eb6850340"
sourceCorrections: []
---

搭建和训练 Flux.jl 中用于二分类任务的简单神经网络 (neural network)，需要理解层、激活函数 (activation function)、损失函数 (loss function)和优化器等组成部分。您将学习如何生成合成数据集、定义模型、进行训练并评估其性能，以构建您的第一个神经网络。

### 环境准备

在开始之前，请确保您已安装 Flux.jl。如果您打算对数据或结果进行可视化，`Plots.jl` 是一个有益的补充。您可以使用 Julia 的包管理器来添加它们：

```julia
using Pkg
Pkg.add("Flux")
Pkg.add("Plots") # Optional, for visualization
Pkg.add("Random") # For data generation
```

本次练习中，我们将使用以下包：

```julia
using Flux
using Random
using Plots # Optional, if you want to run plotting code locally
println("Flux and supporting packages loaded.")
```

### 任务：合成数据上的二分类

我们将处理一个二分类问题：区分二维空间中两个非线性可分的数据点类别。这类问题非常适合神经网络 (neural network)。我们将生成一个“月亮”数据集，它由两个交错的新月形状组成。

#### 数据生成与准备

让我们创建一个函数来生成这些数据。

```julia
Random.seed!(42) # 用于结果重现

function generate_moons(n_samples::Int=200, noise::Float64=0.15)
    n_samples_per_moon = n_samples ÷ 2

    # 外层月亮形状
    t_outer = range(0, stop=pi, length=n_samples_per_moon)
    x_outer = 1.0 .* cos.(t_outer) .+ randn(n_samples_per_moon) .* noise
    y_outer = 1.0 .* sin.(t_outer) .+ randn(n_samples_per_moon) .* noise
    points_outer = [x_outer'; y_outer']

    # 内层月亮形状（偏移后）
    t_inner = range(0, stop=pi, length=n_samples_per_moon)
    x_inner = 1.0 .* cos.(t_inner) .- 0.5 .+ randn(n_samples_per_moon) .* noise
    y_inner = -1.0 .* sin.(t_inner) .+ 0.3 .+ randn(n_samples_per_moon) .* noise # 调整后的偏移
    points_inner = [x_inner'; y_inner']

    # 组合特征和标签
    X = hcat(points_outer, points_inner) # 特征：2xN 矩阵
    Y_labels = vcat(zeros(Int, n_samples_per_moon), ones(Int, n_samples_per_moon)) # 标签：N 向量

    # 打乱数据
    perm = randperm(n_samples)
    X = X[:, perm]
    Y_labels = Y_labels[perm]

    # 将 Y 重塑为 1xN，以兼容 Flux 的损失函数
    Y = reshape(Y_labels, 1, n_samples)

    return Float32.(X), Float32.(Y)
end

X_data, Y_data = generate_moons(300) # 生成 300 个数据点

println("Generated data X: $(size(X_data)), Y: $(size(Y_data))")
# X_data 将是一个 2x300 矩阵，Y_data 将是一个 1x300 矩阵
# X_data 中的每一列是一个数据点 [特征1; 特征2]
# Y_data 中每一列对应的标签是 (0.0 或 1.0)
```

Flux 中的神经网络通常期望输入数据为 `Float32`。我们的 `generate_moons` 函数处理了这种转换。特征 `X_data` 是一个矩阵，其中每列是一个样本，行是特征。标签 `Y_data` 是一个行向量 (vector)。

以下是类似合成数据集的可视化表示：



![合成月亮数据集示例](plots/6910-0.json)



> 散点图显示了两类数据点形成新月形状。类别 0 的点是红色圆圈，类别 1 的点是蓝色叉号，展示了一个非线性可分的模式。

### 定义神经网络 (neural network)架构

我们将使用 `Chain` 搭建一个前馈神经网络，它按顺序堆叠层。我们的网络将包含两个带有 `relu` 激活函数 (activation function)的隐藏层，以及一个带有 `sigmoid` 激活函数的输出层，这适用于二分类任务。

```julia
# 定义模型
model = Chain(
    Dense(2 => 16, relu),    # 输入层：2 个特征，输出：16 个特征，ReLU 激活函数
    Dense(16 => 8, relu),   # 隐藏层：16 个特征，输出：8 个特征，ReLU 激活函数
    Dense(8 => 1, sigmoid)   # 输出层：8 个特征，输出：1 个特征（概率），Sigmoid 激活函数
)

# 您可以查看模型的参数
# params(model)
```

- 第一个 `Dense` 层接收 2 个输入特征（我们的 x 和 y 坐标），并将其映射到 16 个特征。
- 第二个 `Dense` 层接收这 16 个特征，并将其映射到 8 个特征。
- 最后一个 `Dense` 层接收 8 个特征并输出一个值。`sigmoid` 函数将此输出压缩到 0 到 1 之间的概率，表示输入属于类别 1 的可能性。

### 选择损失函数 (loss function)和优化器

对于输出层使用 sigmoid 激活函数 (activation function)的二分类问题，`Flux.binarycrossentropy` 是一个合适的损失函数。它衡量预测概率与实际二元标签（0 或 1）之间的差异。单次预测的公式为 $L(y, \hat{y}) = - (y \log(\hat{y}) + (1-y) \log(1-\hat{y}))$，其中 $y$ 是真实标签，$\hat{y}$ 是预测概率。

我们将使用 `ADAM` 优化器，这是一种流行且高效的自适应学习率优化算法。

```julia
# 定义损失函数
loss(x, y) = Flux.binarycrossentropy(model(x), y)

# 定义优化器
optimizer = ADAM(0.01) # 学习率为 0.01 的 ADAM 优化器

# 获取用于训练的模型参数
ps = Flux.params(model)
```

### 训练循环

训练网络涉及多次（迭代轮次）遍历数据集。在每次迭代中，我们：

1. 计算当前模型参数 (parameter)在整个数据集上的损失（或者数据集的一个批次，尽管对于这个小型数据集，我们使用完整数据集）。
2. 使用 `Flux.gradient` 计算损失相对于模型参数的梯度。
3. 使用优化器和计算出的梯度，通过 `Flux.update!` 更新模型参数。

```julia
# 训练参数
epochs = 200
losses = [] # 用于存储损失值以供绘图

println("Starting training...")
for epoch in 1:epochs
    # 计算梯度
    grads = Flux.gradient(() -> loss(X_data, Y_data), ps)

    # 更新模型参数
    Flux.update!(optimizer, ps, grads)

    # 计算并存储当前损失
    current_loss = loss(X_data, Y_data)
    push!(losses, current_loss)

    if epoch % 20 == 0 || epoch == 1
        println("Epoch: $epoch, Loss: $current_loss")
    end
end
println("Training finished.")
```

训练循环每 20 个迭代轮次打印一次损失。您应该会观察到损失通常随时间降低，这表明模型正在学习。

### 模型评估

训练之后，我们应该评估模型的表现。

#### 绘制损失曲线

监控训练的一个常见方法是绘制损失函数 (loss function)随迭代轮次的变化图。



![训练损失随迭代轮次变化](plots/6910-1.json)



> 训练损失随迭代轮次递减，表明模型正在学习拟合数据。Y 轴显示二元交叉熵损失，X 轴显示迭代轮次。

损失曲线中的实际值将取决于随机初始化和具体数据。您可以使用 `Plots.jl` 和我们收集的 `losses` 数组生成类似的图表：

```julia
# using Plots
# plot(1:epochs, losses, xlabel="迭代轮次", ylabel="损失", label="训练损失", legend=:topright, title="训练损失曲线")
```

#### 计算准确率

准确率是另一个重要的衡量指标。它告诉我们被正确分类的数据点所占的比例。由于我们的模型输出的是概率，如果概率大于 0.5，我们将其视为类别 1 的预测；否则，视为类别 0。

```julia
# 在训练数据上进行预测
predictions_prob = model(X_data)
predictions_class = ifelse.(predictions_prob .> 0.5, 1.0, 0.0) # 将概率转换为类别标签（0 或 1）

# 计算准确率
# Y_data 是 1xN 矩阵，predictions_class 也是 1xN 矩阵
accuracy = sum(predictions_class .== Y_data) / size(Y_data, 2)
println("Training Accuracy: $(round(accuracy * 100, digits=2))%")
# 对于此合成问题，预期准确率应较高，例如 > 90%
```

#### 可视化决策边界（可选）

对于二维分类问题，可视化模型学到的决策边界非常有帮助。这包括创建一个覆盖数据空间的点网格，预测每个网格点的类别，然后将这些预测绘制为等高线图。

```julia
# # 可选：使用 Plots.jl 绘制决策边界的代码
# # 对于大型网格或大量点，这可能会占用大量计算资源
#
# # 创建网格点
# x_range = range(minimum(X_data[1,:]) - 0.2, maximum(X_data[1,:]) + 0.2, length=100)
# y_range = range(minimum(X_data[2,:]) - 0.2, maximum(X_data[2,:]) + 0.2, length=100)
# grid = Float32.(hcat([[x, y] for x in x_range for y in y_range]'...))
#
# # 获取模型在网格上的预测
# Z = model(grid)
# Z_reshaped = reshape(Z, length(x_range), length(y_range))
#
# # 绘制决策边界
# contourf(x_range, y_range, Z_reshaped', levels=1, color=[:red_alpha, :blue_alpha], aspect_ratio=:equal, title="决策边界和数据")
#
# # 叠加原始数据点
# class0_indices = Y_data[1,:] .== 0
# class1_indices = Y_data[1,:] .== 1
# scatter!(X_data[1, class0_indices], X_data[2, class0_indices], label="类别 0", color=:red, markershape=:circle, markerstrokewidth=0, alpha=0.7)
# scatter!(X_data[1, class1_indices], X_data[2, class1_indices], label="类别 1", color=:blue, markershape=:xcross, markerstrokewidth=1, alpha=0.7)
# xlims!(minimum(x_range), maximum(x_range))
# ylims!(minimum(y_range), maximum(y_range))
#
# # savefig("decision_boundary.png") # 保存图表
```

在 Julia 环境中运行上述代码（取消注释并安装 `Plots.jl`）将生成一张图片，显示两类数据点以及模型为每个类别划分的区域。这有助于视觉确认模型是否学到了合理的区分。

### 完整代码示例

以下是结合所有步骤的完整脚本：

```julia
using Flux
using Random
using Plots # 用于绘图，核心逻辑可选

# 1. 数据生成与准备
Random.seed!(42)

function generate_moons(n_samples::Int=300, noise::Float64=0.15)
    n_samples_per_moon = n_samples ÷ 2
    t_outer = range(0, stop=pi, length=n_samples_per_moon)
    x_outer = 1.0 .* cos.(t_outer) .+ randn(n_samples_per_moon) .* noise
    y_outer = 1.0 .* sin.(t_outer) .+ randn(n_samples_per_moon) .* noise
    points_outer = [x_outer'; y_outer']

    t_inner = range(0, stop=pi, length=n_samples_per_moon)
    x_inner = 1.0 .* cos.(t_inner) .- 0.5 .+ randn(n_samples_per_moon) .* noise
    y_inner = -1.0 .* sin.(t_inner) .+ 0.3 .+ randn(n_samples_per_moon) .* noise
    points_inner = [x_inner'; y_inner']

    X = hcat(points_outer, points_inner)
    Y_labels = vcat(zeros(Int, n_samples_per_moon), ones(Int, n_samples_per_moon))

    perm = randperm(n_samples)
    X = X[:, perm]
    Y_labels = Y_labels[perm]
    Y = reshape(Y_labels, 1, n_samples)

    return Float32.(X), Float32.(Y)
end

X_data, Y_data = generate_moons()

# 2. 模型定义
model = Chain(
    Dense(2 => 16, relu),
    Dense(16 => 8, relu),
    Dense(8 => 1, sigmoid)
)

# 3. 损失函数与优化器
loss(x, y) = Flux.binarycrossentropy(model(x), y)
optimizer = ADAM(0.01)
ps = Flux.params(model)

# 4. 训练循环
epochs = 200
losses = []

println("Starting training...")
for epoch in 1:epochs
    grads = Flux.gradient(() -> loss(X_data, Y_data), ps)
    Flux.update!(optimizer, ps, grads)

    current_loss = loss(X_data, Y_data)
    push!(losses, current_loss)

    if epoch % 20 == 0 || epoch == 1
        println("Epoch: $epoch, Loss: $current_loss")
    end

end
println("Training finished.")

# 5. 评估
# 绘制损失曲线
# plot(1:epochs, losses, xlabel="迭代轮次", ylabel="损失", label="训练损失", legend=:topright, title="训练损失曲线")
# savefig("loss_curve.png") # 保存图表示例

# 计算准确率
predictions_prob = model(X_data)
predictions_class = ifelse.(predictions_prob .> 0.5, 1.0, 0.0)
accuracy = sum(predictions_class .== Y_data) / size(Y_data, 2)
println("Training Accuracy: $(round(accuracy * 100, digits=2))%")

# 可选：绘制决策边界（代码已在前面提供）
# 确保使用 Plots.jl 进行此可视化。
```

### 总结

在此动手实践中，您已成功使用 Flux.jl 为二分类任务搭建、训练和评估了一个简单的神经网络 (neural network)。您了解了如何定义模型架构、选择合适的损失函数 (loss function)和优化器，并实现训练循环。合成“月亮”数据集上损失的下降和高训练准确率表明，即使是小型神经网络也能学习非线性决策边界。

在此基础上，您可以尝试更复杂的架构、不同的数据集，试验学习率和神经元数量等超参数 (parameter) (hyperparameter)，或将这些技术应用于回归问题。本示例为您进一步学习 Julia 深度学习 (deep learning)提供了一个良好的起点。

## 参考资料

- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Contributors (2025)
  Julia 深度学习库 Flux.jl 的官方文档，详细介绍了本节中使用的层、激活函数、损失函数、优化器和训练 API。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，涵盖了深度学习的理论和实践，包括神经网络基础、激活函数、损失函数和优化算法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: 3rd International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了ADAM优化器，一种广泛用于训练深度神经网络的自适应学习率算法。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press
  一本在线书籍，提供了神经网络的清晰、直观介绍，涵盖了层、激活函数和反向传播算法等基础概念。
