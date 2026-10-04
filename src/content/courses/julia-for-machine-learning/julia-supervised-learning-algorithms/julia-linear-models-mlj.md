---
course: "julia-for-machine-learning"
chapter: "julia-supervised-learning-algorithms"
lesson: "julia-linear-models-mlj"
sourceId: 6861
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-3-julia-supervised-learning-algorithms/julia-linear-models-mlj"
title: "使用 MLJ.jl 构建和训练线性模型"
description: "使用 MLJ.jl 实现线性回归和逻辑回归模型，用于预测和分类任务。"
order: 2
plots: ["plots/6861-0.json", "plots/6861-1.json"]
sourceHash: "8339b3cc6a93772a8ce2d5ba60976ece27ee35f815e40c1d76f5e66867eff8b2"
sourceCorrections: []
---

线性模型是监督机器学习 (machine learning)中的基本组成部分。它们因其简单、易于理解和计算高效而受到欢迎，通常为更复杂的建模任务提供可靠的基线。MLJ.jl（Julia 中的机器学习）框架在 Julia 中为实现和评估这些模型提供了统一且强大的界面。这里将指导您构建和训练两种常见的线性模型：用于连续目标变量的线性回归和用于分类任务的逻辑回归。

您将学习如何运用 MLJ.jl 的工作流程，这通常包括加载模型、准备数据、将模型和数据封装成一个“机器”、训练该机器，然后进行预测。

### 线性回归与 MLJ.jl

线性回归旨在通过将线性方程拟合到观测数据来建立输入特征集与连续输出变量之间的关系模型。单个特征的基本形式是 $y = \beta_0 + \beta_1 x + \epsilon$，其中 $y$ 是目标， $x$ 是特征，$\beta_0$ 是截距，$\beta_1$ 是特征的系数，而 $\epsilon$ 代表误差项。对于多个特征，这扩展为 $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon$。

MLJ.jl 提供了访问不同包中各种线性回归模型实现的方式。一个常见的选择是 `MLJLinearModels.jl` 包中的 `LinearRegressor`。

让我们通过一个例子来了解。首先，确保您已安装必要的包。您可以使用 Julia 的包管理器添加它们：

```julia
using Pkg
Pkg.add(["MLJ", "MLJLinearModels", "DataFrames", "Plots", "Random"]) # 如果您想进行可视化，添加 Plots
```

现在，我们来实现一个简单的线性回归模型。

```julia
using MLJ
using DataFrames
using Random

# 加载 LinearRegressor 模型
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

# 1. 生成一些合成数据
Random.seed!(123) # 为了可重复性
n_samples = 100
X_vec = rand(n_samples) .* 10
intercept = 2.0
coefficient = 3.5
noise = randn(n_samples) .* 2.0
y = intercept .+ coefficient .* X_vec .+ noise

# 将 X 转换为 DataFrame（MLJ 期望特征为表格形式）
# 并确保 y 是一个向量
X = DataFrame(feature1 = X_vec)

# 2. 实例化模型
model = LinearRegressor()

# 3. 通过将模型与数据绑定来创建一个机器
mach = machine(model, X, y)

# 4. 训练模型（拟合机器）
fit!(mach, verbosity=0)

# 5. 在训练数据上进行预测
y_pred = predict(mach, X)

# 6. 查看学习到的参数（可选）
fp = fitted_params(mach)
println("Learned intercept: $(fp.intercept)")
println("Learned coefficient: $(fp.coefs[1])") # 因为我们只有一个特征
```

在这个例子中：

1. 我们生成合成数据，其中 `y` 与 `X_vec`（我们的 `feature1`）具有线性关系，并加上一些随机噪声。
2. 我们使用 `@load` 宏从 `MLJLinearModels.jl` 加载 `LinearRegressor`。此宏动态加载模型代码并使其可用。
3. 创建了一个模型实例：`model = LinearRegressor()`。
4. 构建了一个“机器”（`mach`），它是 MLJ 的一个构造，用于将模型与数据绑定。这个机器包含模型、数据，并在训练后存储学习到的参数 (parameter)。
5. `fit!(mach)` 训练模型。`verbosity=0` 在拟合期间抑制输出。
6. `predict(mach, X)` 使用训练好的机器对新数据进行预测（此处为简化起见，我们使用原始 `X`）。
7. `fitted_params(mach)` 允许您查看模型学习到的参数，例如截距和系数。

对于这样的数据集，您可能希望可视化实际数据点和拟合的回归线。



![线性回归拟合](plots/6861-0.json)



> 显示合成数据点和相应拟合线性回归线的散点图。实际拟合线将从 `fp.intercept` 和 `fp.coefs` 导出。

### 逻辑回归与 MLJ.jl

逻辑回归用于分类问题，通常在目标变量是类别型（例如，真/假、猫/狗/鸟）时使用。它使用应用于输入特征线性组合的逻辑函数（或 S 形函数）来对输入属于特定类别的概率进行建模。对于二分类问题，正类（$y=1$）的概率由下式给出：


$$
P(y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
$$


MLJ.jl 为此提供了像 `MLJLinearModels.jl` 中的 `LogisticClassifier` 这样的模型。

在使用 MLJ.jl 进行分类时，一个重要步骤是确保您的目标变量 `y` 具有正确的科学类型，通常是 `Multiclass` 或 `Finite`。您可能需要对目标向量 (vector)进行 `coerce`（强制转换）。

让我们设置一个二分类例子：

```julia
using MLJ
using DataFrames
using Random
using CategoricalArrays # 用于创建类别型目标变量

# 加载 LogisticClassifier 模型
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

# 1. 生成用于二分类的合成数据
Random.seed!(456)
n_samples = 100
X1 = randn(n_samples)
# 创建 y 对 X1 的依赖，例如，如果 X1 > 0.5，则更可能是类别“B”
y_logical = (X1 .+ 0.5 .* randn(n_samples)) .> 0.5
y_categorical = categorical(map(x -> x ? "B" : "A", y_logical))

# 将 X1 转换为 DataFrame
X = DataFrame(feature1 = X1)

# 2. 实例化模型
model = LogisticClassifier()

# 3. 创建一个机器
# MLJ 自动将 y_categorical 的科学类型推断为 Multiclass
mach = machine(model, X, y_categorical)

# 4. 训练模型
fit!(mach, verbosity=0)

# 5. 进行预测
# 预测概率（为每个实例返回一个分布）
y_prob_dist = predict(mach, X)

# 获取特定类别（例如“B”）的概率
prob_B = pdf.(y_prob_dist, "B") # pdf 是概率密度函数

# 预测最可能的类别
y_pred_mode = predict_mode(mach, X)

# 6. 查看学习到的参数（可选）
fp = fitted_params(mach)
println("Learned coefficients: $(fp.coefs)")
println("Learned intercept: $(fp.intercept)")

# 比较实际值与预测值
# println("Actual classes: $(y_categorical[1:5])")
# println("Predicted classes: $(y_pred_mode[1:5])")
# println("Predicted probabilities for class 'B': $(round.(prob_B[1:5], digits=2))")
```

在这个逻辑回归例子中：

1. 我们生成数据，其中 `X1` 影响二元结果 `y_categorical`（“A”或“B”）。重要的一点是，`y_categorical` 是作为一个 `CategoricalVector` 创建的，MLJ.jl 将其理解为 `Multiclass`。如果您的目标是，例如 `Vector{String}`，您将需要使用 `y = coerce(y_raw, Multiclass)` 对其进行 `coerce`（强制转换）。
2. `LogisticClassifier` 被加载并实例化。
3. 机器是使用 `X` 和 `y_categorical` 创建的。
4. `fit!(mach)` 训练逻辑回归模型。
5. MLJ.jl 中分类器的 `predict(mach, X)` 通常返回一个概率分布向量。您可以使用 `pdf(distribution, class_label)` 提取特定类别的概率。
6. `predict_mode(mach, X)` 直接给出具有最高预测概率的类别。
7. `fitted_params(mach)` 再次显示学习到的模型参数 (parameter)。

可视化逻辑回归通常涉及在您有一个特征时绘制 S 形曲线，或者在有两个特征时绘制决策边界。



![逻辑回归概率](plots/6861-1.json)



> 示例性图表，显示两个类别的数据点和一条 S 形曲线，表示基于“特征 (X1)”属于“类别 B”的概率。实际曲线将取决于拟合的模型参数。

### MLJ.jl 中的数据类型和强制转换

MLJ.jl 大量依赖 `ScientificTypes.jl` 来确保数据以模型所需的正确格式存在。对于线性回归，目标 `y` 应该是 `Continuous`。对于分类，目标 `y` 应该是 `Finite`（例如，`Multiclass` 或 `OrderedFactor`）。`X` 中的特征通常是 `Continuous` 或 `Count`。

如果您的数据不是预期的科学类型，您将需要使用 `coerce` 或 `coerce!`。例如：

```julia
# 对于用于回归的目标向量 `y_raw`：
# y = coerce(y_raw, Continuous)

# 对于所有列都应该是连续的特征矩阵 `X_table`：
# X = coerce(X_table, Continuous)
# 或者，对于特定列：
# X = coerce(X_table, :col1 => Continuous, :col2 => Count)
```

如果您遇到问题，请始终检查数据的 `scitype`（`scitype(X)`、`scitype(y)`）和模型期望的类型（`input_scitype(model)`、`target_scitype(model)`）。

### 线性模型中的正则化 (regularization)

过拟合 (overfitting)可能是线性模型中的一个问题，尤其是在特征很多时。正则化技术在损失函数 (loss function)中添加惩罚项以缩小系数的值，这可以提高对未见数据的泛化能力。

- **岭回归（L2 惩罚）**：添加与系数平方值之和成比例的惩罚。它倾向于缩小所有系数，但不会将它们精确地设置为零。
- **Lasso 回归（L1 惩罚）**：添加与系数绝对值之和成比例的惩罚。Lasso 可以通过将某些系数缩小到精确的零来执行特征选择。

MLJ.jl 中的许多线性模型实现（例如 `MLJLinearModels.jl` 中的那些）允许您指定正则化。例如：

```julia
# 岭回归器
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels verbosity=0
ridge_model = RidgeRegressor(lambda=1.0) # lambda 是正则化强度

# Lasso 回归器
LassoRegressor = @load LassoRegressor pkg=MLJLinearModels verbosity=0
lasso_model = LassoRegressor(lambda=0.1)

# 带 L2 惩罚的逻辑分类器
# MLJLinearModels 中的 LogisticClassifier 通常默认包含 L2，
# lambda 控制强度。
logreg_model_reg = LogisticClassifier(lambda=0.5)
```

`lambda`（或有时是 `alpha`）参数 (parameter)控制正则化的强度。为 `lambda` 选择正确的值通常通过超参数 (hyperparameter)调优完成，这将在本章后面介绍。

线性模型，无论是用于回归还是分类，都是多功能的工具。它们在 Julia 中使用 MLJ.jl 的实现遵循一致的模式，简化了模型的构建、训练和预测。理解这个工作流程对于您转向更复杂的算法和构建全面的机器学习 (machine learning)流程非常重要。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer
  详细介绍了线性回归、逻辑回归和正则化技术。
- [MLJ.jl Documentation](https://juliaai.github.io/MLJ.jl/) — Anthony Blaom, Logan Kilpatrick, David Josephs (2025)
  Publisher: JuliaAI
  MLJ.jl框架的官方文档，解释了其工作流程、数据处理和模型组合。
- [MLJLinearModels.jl Documentation](https://juliaai.github.io/MLJLinearModels.jl/) — Anthony F. X. Nolan and others (2023)
  MLJLinearModels.jl中实现的线性模型的官方文档，包括`LinearRegressor`和`LogisticClassifier`。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本详细介绍线性模型、正则化及其他统计学习方法的理论基础的教材。
