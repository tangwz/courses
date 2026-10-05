# 在 Flux.jl 中训练神经网络

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-6-julia-deep-learning-flux-jl/flux-jl-training-neural-networks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

神经网络 (neural network)训练在 Flux.jl 中侧重于迭代地调整模型参数 (parameter)。这一核心过程依赖于定义的网络架构、用于量化 (quantization)预测误差的合适损失函数 (loss function)以及指导学习的优化器。训练方式通常将对整个数据集的多次遍历组织起来，每次遍历称为一个 *epoch*（周期）。在每个周期内，数据通常会被分成更小的部分，称为 *mini-batches*（小批量），以便计算更易于管理，并常能改善学习的动态表现。

训练循环中的每一步或每次迭代基本上包含一系列操作：

1. **前向传播**：当前输入数据的小批量，我们称之为 $x_{\text{batch}}$，会被送入网络。网络使用其当前参数集 $\theta$ 产生预测结果，即 $\hat{y}_{\text{batch}} = \text{model}(x_{\text{batch}})$。
2. **损失计算**：这些预测结果 $\hat{y}_{\text{batch}}$ 会使用你所选的损失函数，与来自小批量的真实目标值 $y_{\text{batch}}$ 进行比较。这会得到一个标量损失值 $L = \text{loss\_function}(\hat{y}_{\text{batch}}, y_{\text{batch}})$，它衡量了模型对于此特定批量的预测偏差程度。
3. **梯度计算**：这正是 Julia 生态系统中由 Zygote.jl 驱动的自动微分发挥作用的地方。Zygote 计算损失 $L$ 对模型所有可训练参数 $\theta$ 的梯度。这些梯度，表示为 $\nabla_{\theta} L$（或对于每个参数 $\theta_i$ 为 $\frac{\partial L}{\partial \theta_i}$），指示了每个参数需要改变的方向和大小，以最大程度地降低损失。
4. **参数更新**：优化器使用这些计算出的梯度来调整模型参数。目标是朝着损失最小化的方向迈进。例如，基本的随机梯度下降 (gradient descent)（SGD）更新规则是：
   
   $$
   \theta_{\text{新}} = \theta_{\text{旧}} - \eta \nabla_{\theta} L
   $$
   
   其中 $\eta$ 是学习率，一个控制步长的超参数 (hyperparameter)。Adam 或 RMSProp 等更复杂的优化器有更复杂的更新规则，但都遵循使用梯度来指导参数改变的相同原理。

### 在 Flux.jl 中实现训练循环

Flux.jl 提供了有效实现此训练循环的工具。让我们查看这些操作如何转化为 Flux.jl 代码。

首先，你需要一个优化器及其关联的状态。Flux.jl 为优化器使用一套明确的状态管理系统，通过 `Flux.setup` 进行初始化。此状态跟踪动量或自适应学习率等信息，适用于更高级的优化器。

```julia
using Flux # 也引入了 Optimisers.jl 的功能

# 假设 'model' 是你定义的 Flux 模型（例如，一个 Chain）
# model = Chain(Dense(10, 5, relu), Dense(5, 1)) 

# 选择一个优化器规则
opt_rule = Adam(0.001) # Adam 优化器，学习率为 0.001

# 为模型设置优化器状态
opt_state = Flux.setup(opt_rule, model)
```

`opt_state` 现在包含所需的信息，以便 `Adam` 优化器能够与你的特定 `model` 协同工作。

为了同时计算损失和梯度，Flux.jl 提供了 `Flux.withgradient`。此函数接受模型（或其参数 (parameter)）以及一个计算损失的函数。它返回损失值和梯度。

```julia
# 假设 x_batch, y_batch 可用
# 假设 loss_fn(m, x, y) 已定义，例如：
# loss_fn(m, x, y) = Flux.mse(m(x), y)

# 在你的训练迭代中：
loss_value, grads = Flux.withgradient(model) do m
    # 此处传递的模型 'm' 是计算其梯度的模型
    y_hat = m(x_batch) # 前向传播
    loss_fn(m, x_batch, y_batch) # 计算损失
end
```

在这里，`loss_value` 是 `loss_fn` 的标量结果，而 `grads` 是一个梯度集合。具体来说，`grads[1]` 将包含与 `model` 参数对应的梯度。

获得梯度后，迭代中的最后一步是使用优化器更新模型参数：

```julia
Flux.update!(opt_state, model, grads[1])
```

`Flux.update!` 函数使用存储在 `opt_state` 中的优化器规则逻辑（例如 Adam）和计算出的 `grads[1]`，就地修改 `model` 的参数。

### 训练数据处理

神经网络 (neural network)训练要求数据采用适合处理的格式，通常是多维数组或张量。如前所述，训练通常在小批量上进行。Flux.jl 提供了 `Flux.Data.DataLoader` 作为一个方便的工具，用于对数据集进行分批和混洗。

```julia
using Flux.Data: DataLoader

# 假设 train_X（特征）和 train_Y（标签）是你的完整数据集数组
# 例如，train_X 是一个 10x1000 矩阵（10 个特征，1000 个样本）
#       train_Y 是一个 1x1000 矩阵（1 个输出，1000 个样本）
batch_size = 32
train_loader = DataLoader((train_X, train_Y), batchsize=batch_size, shuffle=true)

# 在你的训练循环中，你会遍历 train_loader：
# for (x_batch_from_loader, y_batch_from_loader) in train_loader
#   # ... 执行训练步骤 ...
# end
```

使用 `shuffle=true` 的 `DataLoader` 确保模型在每个周期看到数据时顺序不同，这有助于防止过拟合 (overfitting)并提升泛化能力。

### 完整的训练循环示例

让我们将所有这些部分组合成一个功能完整的训练循环。这个例子将定义一个简单的模型、损失函数 (loss function)和优化器，然后通过遍历周期和批次来训练模型。

```julia
using Flux
# DataLoader 经常使用，请确保它可以访问（例如，通过 using Flux.Data: DataLoader）

# 1. 定义模型
model = Chain(
    Dense(10 => 5, relu), # 10 个输入特征，5 个输出神经元，ReLU 激活函数
    Dense(5 => 1)         # 5 个输入特征，1 个输出神经元（例如，用于回归）
)

# 2. 定义损失函数
# 此版本的 loss_fn 接受模型、输入和目标
# 它适合与 Flux.withgradient(model) do m ... end 配合使用
loss_fn(m, x, y) = Flux.mse(m(x), y) # 均方误差

# 3. 定义优化器并设置其状态
opt_rule = Adam(0.001) # Adam 优化器，学习率为 0.001
opt_state = Flux.setup(opt_rule, model)

# 4. 准备数据
# 对于实际示例，请替换为你的实际数据加载
dummy_X = rand(Float32, 10, 100) # 10 个特征，100 个样本
dummy_Y = rand(Float32, 1, 100)  # 1 个输出，100 个样本
# 确保 DataLoader 可用，例如，如果未自动引入，则通过 `using Flux.Data: DataLoader`。
# 如果 Flux 直接重新导出它：
train_loader = Flux.DataLoader((dummy_X, dummy_Y), batchsize=32, shuffle=true)

# 5. 训练循环
num_epochs = 10
println("开始训练 $num_epochs 个周期...")

for epoch in 1:num_epochs
    epoch_cumulative_loss = 0.0
    batches_processed = 0

    for (x_batch, y_batch) in train_loader
        # 计算当前批次的损失和梯度
        # 此处的 loss_fn 将模型 'm_in_grad' 作为其第一个参数
        current_loss, grads = Flux.withgradient(model) do m_in_grad
            loss_fn(m_in_grad, x_batch, y_batch)
        end

        # 使用计算出的梯度更新模型参数
        Flux.update!(opt_state, model, grads[1])

        epoch_cumulative_loss += current_loss
        batches_processed += 1
    end

    avg_epoch_loss = epoch_cumulative_loss / batches_processed
    println("周期: $epoch, 平均批次损失: $avg_epoch_loss")
end

println("训练完成。")
```

在这个完整的循环中：

- 我们按照指定的 `num_epochs` 次数进行迭代。
- 在每个周期内，`train_loader` 提供数据小批量。
- `Flux.withgradient` 用于计算批次的 `current_loss` 和 `grads`（梯度）。模型本身 (`model`) 作为 `Flux.withgradient` 的第一个参数 (parameter)传递，匿名函数 `do m_in_grad ... end` 接收此模型（此处命名为 `m_in_grad`）以用于梯度计算上下文 (context)中的前向传播和损失计算。
- `Flux.update!` 应用优化器的逻辑，就地使用 `grads[1]` 调整 `model` 的参数。
- 每个周期的平均损失会被打印出来，提供一个简单的方式来监控训练进度。

Following diagram illustrates the flow of a single training iteration:

> 单次训练迭代的概览，显示了数据流向、损失和梯度的计算以及参数更新。

这种遍历数据、计算损失、推导梯度和更新参数的详细流程是训练大多数神经网络 (neural network)的基础。借助 Flux.jl，这些步骤可以非常直观地表示，为构建和训练你的深度学习 (deep learning)模型提供了清晰性和灵活性。

## 参考资料

- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The JuliaFlux Community (2025)
  介绍如何使用 Flux.jl 定义模型、选择优化器以及实现训练循环。
- [Zygote.jl Documentation](https://fluxml.ai/Zygote.jl/stable/) — The JuliaDiff Community (2024)
  详细说明 Julia 中的自动微分系统，这对于在 Flux.jl 中计算梯度至关重要。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教材，涵盖深度学习架构、训练算法和优化方法。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍 Adam 优化器的原始论文，该优化器广泛应用于深度神经网络的训练。

---

[上一节](04-%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E4%B8%8E%E4%BC%98%E5%8C%96%E5%99%A8.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20Zygote.jl%20%E7%9A%84%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86.md)
