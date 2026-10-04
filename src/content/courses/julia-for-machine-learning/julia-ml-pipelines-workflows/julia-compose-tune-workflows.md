---
course: "julia-for-machine-learning"
chapter: "julia-ml-pipelines-workflows"
lesson: "julia-compose-tune-workflows"
sourceId: 6892
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-5-julia-ml-pipelines-workflows/julia-compose-tune-workflows"
title: "组合与调整复杂工作流"
description: "构建更复杂的机器学习工作流和在MLJ.jl中调整整个管线的策略。"
order: 4
plots: []
sourceHash: "6bfd5211f7d7e70ce9c0099eaa7e622b0772338a74a9ceaebd57b3a3892cf488"
sourceCorrections: []
---

基础机器学习 (machine learning)管线通常将单一预处理步骤与模型有效连接，但实际问题常要求更复杂的流程。您可能需要填充缺失值，然后缩放特征，再执行特征选择，所有这些都在数据到达学习算法之前完成。MLJ.jl 提供构建和管理这些多阶段工作流的工具，确保每个步骤恰当衔接，并且整个过程可进行调整。

### 构建更精细的管线

在MLJ.jl中构建多阶段管线直接基于您已学习过的原则。您可以使用 `|>` 操作符或 `@pipeline` 宏，将多个转换器和一个最终的有监督模型连接起来。链中的每个元素都处理前一个的输出。

设想一个工作流，您首先需要使用均值填充来处理缺失的数值数据，然后标准化特征，最后应用K近邻回归器。以下是定义此类管线的方式：

```julia
using MLJ
using DataFrames

# 加载必要的模型类型
KNNRegressor = @load KNNRegressor pkg=NearestNeighborModels verbosity=0
Standardizer = @load Standardizer pkg=MLJModels verbosity=0
FillImputer = @load FillImputer pkg=MLJModels verbosity=0

# 定义管线
complex_pipe = @pipeline(
    imputer = FillImputer(),
    scaler = Standardizer(),
    regressor = KNNRegressor()
)
```

在这种结构中，`FillImputer` 将首先处理输入数据。其输出随后会传递给 `Standardizer`，标准化后的数据最终会输入到 `KNNRegressor`。MLJ.jl 会恰当地管理数据流和每个组件的拟合过程。

您可以可视化此类管线的结构，以更好地理解其流程。

> 一张图表，表示一个多阶段机器学习 (machine learning)管线，数据从原始输入开始，经过填充、缩放和回归模型处理，最终产生预测结果。

这种连接多个操作的能力不限于预处理。例如，您的管线可以包含特征选择作为中间步骤，甚至可以组合来自不同分支的输出，尽管后者涉及更高级的学习网络构建。

### 调整多阶段管线中的超参数 (parameter) (hyperparameter)

将工作流封装在管线中的一个重要优势在于，能够同时调整其所有阶段的超参数。`complex_pipe` 中的每个组件（`FillImputer`、`Standardizer` 和 `KNNRegressor`）都可能拥有可优化的超参数。例如，`FillImputer` 有 `features` 参数用于指定要填充的列（尽管它通常会推断），而 `KNNRegressor` 有 `K`（邻居数量）参数。

MLJ.jl 的 `TunedModel` 可以封装整个管线，允许您定义一个跨越不同组件超参数的搜索空间。在指定要调整的参数时，您将使用点号表示法来指明您指的是哪个组件的超参数。您在 `@pipeline` 宏中指定的名称（例如 `imputer`、`scaler`、`regressor`）将成为访问其各自超参数的前缀。

假设我们希望调整 `complex_pipe` 中 `KNNRegressor` 的邻居数量 (`K`)。

```julia
# 定义调整策略
tuning_strategy = Grid(resolution=5) # 简单网格搜索
k_range = range(complex_pipe, :(regressor.K); lower=1, upper=10, scale=:linear)

# 创建管线的自调整版本
tuned_complex_pipe = TunedModel(
    model = complex_pipe,
    resampling = CV(nfolds=3), # 3折交叉验证
    tuning = tuning_strategy,
    range = [k_range],
    measure = rms # 回归的均方根误差
)
```

在此示例中：

- `:(regressor.K)` 指明我们正在调整 `complex_pipe` 中名为 `regressor` 组件的 `K` 超参数。`:(...)` 语法会创建一个表示参数路径的符号。
- `range(complex_pipe, :(regressor.K); ...)` 有助于在管线的语境中定义此特定参数的搜索范围。
- `TunedModel` 随后将通过使用3折交叉验证评估整个管线的性能来寻找最佳 `K` 值。

如果您的 `scaler` 也有一个可调整的超参数，例如 `scaler.some_param`，您可以将其另一个范围添加到 `TunedModel` 的 `range` 数组中。这种调整方法确保了不同阶段及其设置之间的关系得到考量，通常比单独调整每个组件能带来更好的整体性能。

例如，如果您想同时调整回归器的 `K` 和 `Standardizer` 的参数 `delta`（假设它有此参数并在管线中命名为 `scaler`）：

```julia
# 如果 Standardizer 有一个可调整的 'delta' 参数
# delta_range = range(complex_pipe, :(scaler.delta); lower=0.1, upper=1.0)
#
# tuned_complex_pipe_multi = TunedModel(
#     model = complex_pipe,
#     resampling = CV(nfolds=3),
#     tuning = Grid(resolution=5),
#     range = [k_range, delta_range], # 调整多个参数
#     measure = rms
# )
```

### 管理复杂工作流的策略

随着管线复杂性增加，有效管理它们变得日益重要。以下是一些策略：

1. **清晰命名**：使用 `@pipeline` 定义管线时，为每个组件指定有意义的名称（例如，`imputer_numerical`、`one_hot_encoder`、`final_model`）。这些名称用于指定超参数 (parameter) (hyperparameter)的调整路径，因此此处清晰的命名使得调整设置更易读且不易出错。

   ```julia
   # 使用更具描述性名称的示例
   descriptive_pipe = @pipeline(
       num_imputer = FillImputer(),     # 用于数值特征
       cat_imputer = FillImputer(),     # 用于分类特征（如适用）
       scaler = Standardizer(),
       feature_selector = FeatureSelector(), # 特征选择器
       learner = KNNRegressor()
       # ... 如果不是纯线性，则指定连接 ...
   )
   # 调整时会使用类似 :(num_imputer.features) 或 :(learner.K) 的路径
   ```
2. **模块化设计**：在将较小的管线段组合成更大的工作流之前，独立构建和测试它们。例如，在附加最终学习算法之前，确保您的多步骤预处理管线能独立按预期运行。如果省略最终模型，MLJ.jl 允许管线输出转换后的数据而非预测结果，从而便利了这种模块化测试。
3. **迭代式改进**：不要一开始就尝试构建最复杂的管线。从一个更简单的版本开始，逐步添加组件或调整维度。在每个步骤评估性能，以理解新增加的复杂性带来的影响。
4. **用于非线性流程的学习网络**：尽管标准管线（`@pipeline` 或 `|>`）非常适合线性操作序列，但有些问题通过更复杂的安排会受益，例如特征子集经历不同转换的分支结构，或者模型堆叠。对于这些情况，MLJ.jl 提供**学习网络**。它们提供了一个更通用的框架，用于定义任意有向无环图（DAG）的操作。尽管学习网络的完整讨论属于更高级别的学习内容，但了解它们的存在在线性管线不足时会很有用。

通过巧妙地组合和调整这些更精细的管线，您可以创建功能强大、自动化且可复现的机器学习 (machine learning)系统。将整个多阶段工作流视为一个可调整的单一实体，是构建Julia中有效且可靠模型的重要一步。这种结构化方法有助于管理许多机器学习任务固有的复杂性，让您能够专注于优化整体解决方案。

## 参考资料

- [Learning Networks and Composing Models](https://alan-turing-institute.github.io/MLJ.jl/dev/learning_networks/) — The MLJ Development Team (2024)
  MLJ.jl官方文档，解释如何组合机器学习模型以及使用学习网络创建多阶段管道。
- [Tuning Models](https://alan-turing-institute.github.io/MLJ.jl/stable/model_tuning/) — The MLJ Development Team (2024)
  Publisher: The Alan Turing Institute
  MLJ.jl官方文档，详细介绍针对单个模型和完整管道的超参数调整策略。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，介绍机器学习管道、预处理技术和超参数调整概念，适用于各种机器学习框架。
- [Applied Predictive Modeling](https://www.springer.com/gp/book/9781461468486) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  一本关于构建和评估预测模型的综合资源，包含数据预处理、特征选择和稳健工作流中系统模型调整的详细章节。
