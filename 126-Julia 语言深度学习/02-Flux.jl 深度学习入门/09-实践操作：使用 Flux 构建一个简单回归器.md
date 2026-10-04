# 实践操作：使用 Flux 构建一个简单回归器

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/hands-on-simple-regressor-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本实践练习将引导你使用 Flux.jl 构建一个简单的回归模型。回归任务涉及预测连续的数值。我们将生成一些合成数据，定义一个模型，选择一个损失函数 (loss function)和一个优化器，然后训练模型以适应数据。这个例子将加强你对这些部分在典型 Flux.jl 工作流程中如何共同作用的理解。

我们的目标是训练一个能学习 $y = mx + b$ 形式的简单线性关系的模型。我们将基于一个已知的线性函数生成数据点，并添加一些噪声，使学习任务更具挑战性。

### 生成回归用的合成数据

首先，我们来创建一些数据。我们将生成一组输入值 `X` 和相应的目标值 `Y`。Flux.jl 通常期望输入数据中每列是一个样本，每行是一个特征。对于我们简单的一维输入，`X` 将是一个 $1 \times N$ 矩阵，`Y` 也将是一个 $1 \times N$ 矩阵。

```julia
using Flux, Random

# 设置随机种子以保证结果可重现
Random.seed!(123)

# 生成输入特征 X (例如，从 0 到 5 的 100 个数据点)
# 我们需要 X 作为行向量 (1xN 矩阵) 以供 Flux 使用
X_data = hcat(collect(0.0:0.05:5.0)...) # 创建一个 1x101 矩阵

# 定义真实的基础关系 (例如，y = 2x + 1)
# 并添加一些噪声
true_slope = 2.0
true_intercept = 1.0
Y_data = (true_slope .* X_data) .+ true_intercept .+ (randn(Float32, size(X_data)) .* 0.5f0)

# Flux 倾向于使用 Float32 以提升性能，尤其是在 GPU 上，
# 尽管 Float64 也可以。我们确保数据是 Float32。
X_data = Float32.(X_data)
Y_data = Float32.(Y_data)

println("X_data 的大小: ", size(X_data))
println("Y_data 的大小: ", size(Y_data))
```

在这段代码中，`X_data` 代表我们的输入特征，而 `Y_data` 是我们想要预测的目标变量。我们向 `Y_data` 中添加了一些随机噪声，以模拟更常见的情形，即关系并非完全确定。

### 定义回归模型

对于简单的线性回归，一个 `Dense` 层就足够了。这一层执行线性变换：$输出 = W \cdot 输入 + b$。由于我们的输入 `X_data` 有一个特征（其值），并且我们想预测一个输出值 `Y_data`，所以 `Dense` 层将把 1 个输入特征映射到 1 个输出特征。

```julia
# 定义一个简单的线性模型：一个输入特征，一个输出特征
model = Dense(1 => 1)

# 我们可以查看初始（随机初始化）的参数
println("初始权重: ", model.weight)
println("初始偏置: ", model.bias)
```

`Dense(1 => 1)` 层创建了一个连接，其中第一个 `1` 是输入的维度，第二个 `1` 是输出的维度。Flux 用小的随机值初始化权重 (weight)和偏置 (bias)。我们的训练过程将调整这些参数 (parameter)以适应我们的数据。

### 选择损失函数 (loss function)

为了训练模型，我们需要一种方法来衡量其预测的“错误”程度。对于回归任务，均方误差（MSE）是一个常用选择。MSE 计算的是预测值与实际值之间平方差的平均值。


$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{\text{预测}}^{(i)} - y_{\text{实际}}^{(i)})^2
$$


Flux 在 `Flux.Losses.mse` 中提供了这个功能。

```julia
# 定义损失函数：均方误差
loss(x, y) = Flux.Losses.mse(model(x), y)

# 让我们用初始的随机模型测试损失
initial_loss = loss(X_data, Y_data)
println("初始损失: ", initial_loss)
```

`loss` 函数接受输入 `x` 和目标 `y`，将 `x` 通过 `model` 得到预测值，然后计算这些预测值与 `y` 之间的 MSE。

### 选择优化器

优化器负责根据损失函数 (loss function)的梯度来更新模型的参数 (parameter)（权重 (weight)和偏置 (bias)）。我们将使用 `Descent` 优化器，它实现了标准梯度下降 (gradient descent)。我们需要提供一个学习率，它控制着每一步中参数调整的幅度。

```julia
# 定义优化器：学习率为 0.01 的梯度下降
opt = Descent(0.01)

# 获取 Flux 将要训练的模型参数
params = Flux.params(model)
println("待训练参数: ", params)
```

`Flux.params(model)` 从我们的 `model` 中收集所有可训练参数。`Descent` 优化器将使用针对这些参数计算的梯度来更新它们。

### 训练循环

现在我们将实现训练循环。在循环的每次迭代（或周期）中，我们将：

1. 计算 `loss` 函数相对于模型 `params` 的梯度。这是由 Zygote.jl 驱动的自动微分发挥作用的地方。
2. 使用优化器和计算出的梯度更新模型的 `params`。

我们将运行固定数量的周期，并定期打印损失值，以观察模型是否正在学习。

```julia
# 训练参数
epochs = 200

# 训练循环
println("开始训练...")
for epoch in 1:epochs
    # 计算梯度
    grads = gradient(() -> loss(X_data, Y_data), params)

    # 更新模型参数
    Flux.update!(opt, params, grads)

    # 打印进度 (例如，每 20 个周期)
    if epoch % 20 == 0
        current_loss = loss(X_data, Y_data)
        println("周期: $epoch, 损失: $current_loss")
    end
end
println("训练完成。")

# 让我们看看学习到的参数
println("学习到的权重: ", model.weight)
println("学习到的偏置: ", model.bias)
```

在 `gradient(() -> loss(X_data, Y_data), params)` 内部，Zygote.jl 计算损失函数 (loss function)相对于 `params` 中每个参数 (parameter)的导数。接着 `Flux.update!(opt, params, grads)` 应用优化步骤（例如，$参数 = 参数 - \text{学习率} \times 梯度$）。

你应该观察到损失值随着周期数增加而减小，这表明模型在从 `X_data` 预测 `Y_data` 方面变得更好。学习到的权重 (weight)和偏置 (bias)应分别接近我们的 `true_slope` (2.0) 和 `true_intercept` (1.0)。

### 进行预测和结果可视化

训练完成后，我们可以使用 `model` 对输入数据进行预测，并将其与实际目标值进行比较。可视化简单回归器性能的一个好方法是，绘制原始数据点以及模型学习到的直线。

```julia
# 使用训练好的模型进行预测
Y_predicted = model(X_data)

# 为了绘图，我们需要一个绘图包。
# 如果你没有 Plots.jl 和像 GR 这样的后端，
# 可以安装它们：
# import Pkg; Pkg.add(["Plots", "GR"])
# 对于本示例，我们将直接提供 Plotly JSON。

# 如果需要，转换绘图数据 (例如，从 Flux 的 TrackedArrays 转换为普通数组)
# X_plot = X_data[1,:] # 获取第一行 (也是唯一一行) 作为向量
# Y_plot = Y_data[1,:] # 获取第一行 (也是唯一一行) 作为向量
# Y_pred_plot = Y_predicted[1,:] # 获取第一行 (也是唯一一行) 作为向量

# 如果使用 Plots.jl 绘图：
# using Plots
# scatter(X_plot, Y_plot, label="数据点", mc=:blue)
# plot!(X_plot, Y_pred_plot, label="学习到的回归线", lc=:red, lw=2)
# xlabel!("特征 (X)")
# ylabel!("目标 (Y)")
# title!("使用 Flux.jl 进行的简单线性回归")
```

下面是此类图表可能的样子。蓝色圆点代表我们的带噪声数据点，红色线条显示了 Flux 模型学习到的线性关系。



[交互图表：使用 Flux.jl 进行的简单线性回归](https://apxml.com/zh/courses/julia-deep-learning/chapter-2-intro-flux-jl/hands-on-simple-regressor-flux#plot-1ulzvi9)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0
      ],
      "y": [
        1.2,
        1.3,
        2.1,
        2.6,
        2.9,
        3.7,
        3.8,
        4.6,
        5.1,
        5.3,
        6.2,
        6.4,
        7.1,
        7.3,
        8.2,
        8.6,
        8.9,
        9.7,
        9.9,
        10.4,
        11.2
      ],
      "mode": "markers",
      "type": "scatter",
      "name": "数据点",
      "marker": {
        "color": "#228be6"
      }
    },
    {
      "x": [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0
      ],
      "y": [
        0.95,
        1.45,
        1.95,
        2.45,
        2.95,
        3.45,
        3.95,
        4.45,
        4.95,
        5.45,
        5.95,
        6.45,
        6.95,
        7.45,
        7.95,
        8.45,
        8.95,
        9.45,
        9.95,
        10.45,
        10.95
      ],
      "mode": "lines",
      "type": "scatter",
      "name": "学习到的回归线",
      "line": {
        "color": "#f03e3e",
        "width": 2
      }
    }
  ],
  "layout": {
    "title": "使用 Flux.jl 进行的简单线性回归",
    "xaxis": {
      "title": "特征 (X)"
    },
    "yaxis": {
      "title": "目标 (y)"
    },
    "legend": {
      "orientation": "h",
      "yanchor": "bottom",
      "y": 1.02,
      "xanchor": "right",
      "x": 1
    },
    "paper_bgcolor": "#f8f9fa",
    "plot_bgcolor": "#e9ecef",
    "autosize": true
  }
}
```

</details>



> 散点图显示了原始数据点，而连续线代表了训练后的线性回归模型所做的预测。理想情况下，这条线应穿过数据点的“中心”，捕捉到潜在的线性趋势。“学习到的回归线”的 y 值是示意性的，并在训练后从 `model(X_data)` 得到。数据点的具体 y 值也是围绕直线 $y \approx 2x+1$ 的带噪声数据的示意性例子。

### 完整代码示例

这是我们简单回归器的完整脚本：

```julia
using Flux, Random

# 设置随机种子以保证结果可重现
Random.seed!(123)

# 1. 生成合成数据
X_data = hcat(collect(0.0f0:0.05f0:5.0f0)...) # 1xN 矩阵
true_slope = 2.0f0
true_intercept = 1.0f0
Y_data = (true_slope .* X_data) .+ true_intercept .+ (randn(Float32, size(X_data)) .* 0.5f0)

println("X_data 的大小: ", size(X_data))
println("Y_data 的大小: ", size(Y_data))

# 2. 定义模型
model = Dense(1 => 1) # 一个输入特征，一个输出特征
println("初始权重: W=", model.weight, ", b=", model.bias)

# 3. 定义损失函数
loss(x, y) = Flux.Losses.mse(model(x), y)
println("初始损失: ", loss(X_data, Y_data))

# 4. 选择优化器
opt = Descent(0.01) # 学习率为 0.01 的梯度下降
params = Flux.params(model)

# 5. 训练循环
epochs = 200
println("\n开始训练 $epochs 个周期...")
for epoch in 1:epochs
    grads = gradient(() -> loss(X_data, Y_data), params)
    Flux.update!(opt, params, grads)

    if epoch % 20 == 0 || epoch == 1
        current_loss = loss(X_data, Y_data)
        println("周期: $epoch, 损失: $current_loss")
    end
end
println("训练完成。\n")

# 显示学习到的参数
println("学习到的权重: W=", model.weight, ", b=", model.bias)
println("真实参数: W_true=", [true_slope], ", b_true=", [true_intercept])

# 6. 进行预测 (可选：显示一些预测结果)
# 例如，预测前 5 个数据点
X_sample = X_data[:, 1:5]
Y_sample_actual = Y_data[:, 1:5]
Y_sample_predicted = model(X_sample)

println("\n样本预测结果:")
for i in 1:size(X_sample, 2)
    println("输入: $(round(X_sample[1,i], digits=2)), 实际值: $(round(Y_sample_actual[1,i], digits=2)), 预测值: $(round(Y_sample_predicted[1,i], digits=2))")
end
```

运行这段代码将显示模型的初始状态、训练过程中损失值的下降，以及最终学习到的参数 (parameter)，这些参数应近似于我们用来生成数据的真实斜率和截距。

本实践练习展示了使用 Flux.jl 构建、训练和预测一个非常基础的回归神经网络 (neural network)的端到端过程。你已经了解了如何定义层、将它们组合成模型、指定损失函数 (loss function)、选择优化器以及实现训练循环。随着我们研究更复杂的架构和任务，这些将是你在此基础上不断提升的核心技能。

## 参考资料

- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Contributors (2025)
  Flux.jl 的官方文档，提供其 API、层、损失函数、优化器和示例的详细信息。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习基本概念的教科书，包括线性回归、梯度下降和神经网络架构。
- [An Introduction to Statistical Learning, with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  统计学习方法的入门介绍，包含线性回归和模型评估的章节。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=2oY_EAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本构建机器学习系统的指南，涵盖回归、模型训练流程和优化技术。

---

[上一节](08-%E6%9E%84%E5%BB%BA%20Flux%20%E4%B8%AD%E7%9A%84%E5%9F%BA%E7%A1%80%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md) · [下一节](../03-%E6%9E%84%E5%BB%BA%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84/01-Julia%20%E4%B8%AD%E7%9A%84%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86.md)
