---
course: "julia-for-machine-learning"
chapter: "julia-ml-pipelines-workflows"
lesson: "julia-save-load-models-pipelines"
sourceId: 6891
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-5-julia-ml-pipelines-workflows/julia-save-load-models-pipelines"
title: "保存和加载训练好的模型和管道"
description: "用于在Julia中序列化和反序列化训练好的机器学习模型及管道，以便后续使用或部署的技术。"
order: 3
plots: []
sourceHash: "41fa0329f8271d49dd16ebb8bf0b2b30bc4310bb9ee0006b87b709b567eed977"
sourceCorrections: []
---

训练好的机器学习 (machine learning)管道的用途远不止一次会话。持久化这些训练好的模型和管道非常重要，以便无需昂贵的重新训练即可重用，将其部署到应用程序中，与协作者共享，或确保后续工作能够可靠重现。这个过程在MLJ.jl生态系统中很简单。

### 保存训练好的机器

MLJ.jl提供`MLJ.save`函数，用于将训练好的机器序列化到文件。MLJ中的“机器”是一个对象，它将模型（可以是单个学习器或复杂管道）绑定到数据，并在训练后存储学习到的参数 (parameter)（即`fitresult`）。`MLJ.save`函数通常在底层使用JLD2.jl包，将机器保存为`.jlso`（Julia序列化对象）文件。这种格式对于存储Julia对象很高效。

假设您已经训练了一个机器`mach`，它可以包含从简单的决策树到多步骤预处理和建模管道的任何内容。保存它就像这样简单：

```julia
using MLJ
import JLD2 # MLJ.save 内部使用 JLD2.jl

# 假设 'mach' 是一个训练好的机器
# 例如：
# ModelType = @load DecisionTreeClassifier pkg=DecisionTree
# model = ModelType()
# X, y = @load_iris
# mach = machine(model, X, y)
# fit!(mach)

# 保存训练好的机器
MLJ.save("my_trained_model.jlso", mach)
```

此命令保存`mach`的整个状态，包括模型的超参数 (hyperparameter)及其学习到的参数，到文件“my\_trained\_model.jlso”中。

### 加载训练好的机器

要将您保存的机器重新带回Julia会话，您可以使用`machine`构造函数本身，但不是提供模型和数据，而是提供`.jlso`文件的路径：

```julia
# 在新会话中，或在 'mach' 不再内存中时：
loaded_mach = machine("my_trained_model.jlso")
```

`loaded_mach`现在是您保存的机器的精确复制品。它已准备好进行预测或检查，无需重新训练。您可以通过使用它进行预测来验证这一点：

```julia
# 假设 X_new 是与模型兼容的新数据
# y_pred = predict(loaded_mach, X_new)
# @info "使用加载模型进行的预测：", y_pred[1:5]
```

这种流畅的保存和加载能力对于管道特别有用。如果`mach`是管道机器，`MLJ.save`会存储整个管道结构以及每个组件的拟合状态。

### 究竟保存了什么？

当您使用`MLJ.save(filename, mach)`时，您正在序列化MLJ `Machine`对象。这个对象包含：

1. **模型：** 您的学习器或管道的蓝图（例如，`DecisionTreeClassifier(max_depth=3)`或`@pipeline`对象）。
2. **拟合结果 (Fitresult)：** 在`fit!`操作期间学习到的参数 (parameter)。这是您训练好的模型的“智能”。对于线性模型，这会是系数；对于树模型，这是树结构。
3. **报告 (Report)：** 训练期间生成的任何诊断信息或副产品。
4. **数据绑定（部分）：** 有关机器训练所用数据的信息，例如特征名称，通常会保留，但原始训练数据本身通常不存储在机器内部，以保持文件大小可管理。机器主要存储进行新预测所需的内容。

底层的`JLD2.jl`序列化器设计用于处理各种Julia类型，使其非常适合构成MLJ机器的复杂对象。

### 实际例子：保存和加载管道

让我们看一个涉及管道的更完整的例子。我们将定义一个简单管道，训练它，保存它，加载它，然后用它进行预测。

```julia
using MLJ
import RDatasets: dataset
import JLD2 # 用于 MLJ.save

# 加载必要的模型类型
Standardizer = @load Standardizer pkg=MLJModels
OneHotEncoder = @load OneHotEncoder pkg=MLJModels
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0

# 准备一些数据（为简洁起见使用子集）
data = dataset("datasets", "iris")
X = data[:, [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
y = data[:, :Species]
train_rows, test_rows = partition(eachindex(y), 0.7, shuffle=true, rng=123)

X_train = X[train_rows, :]
y_train = y[train_rows]
X_test = X[test_rows, :]

# 定义一个管道
@pipeline MyPipeline(
    std = Standardizer(),
    ohe = OneHotEncoder(),
    tree = DecisionTreeClassifier(max_depth=3)
) is_probabilistic=true prediction_type=:probabilistic

# 实例化并训练管道机器
pipe_model = MyPipeline()
pipe_mach = machine(pipe_model, X_train, y_train)
fit!(pipe_mach, rows=train_rows) # 再次使用 train_rows，相对于提供给机器的 X_train, y_train

# 保存训练好的管道机器
MLJ.save("my_pipeline_machine.jlso", pipe_mach)
@info "管道机器已保存。"

# 模拟在新环境中加载
loaded_pipe_mach = machine("my_pipeline_machine.jlso")
@info "管道机器已加载。"

# 使用加载的管道进行预测
y_pred_loaded = predict(loaded_pipe_mach, X_test)
@info "从加载管道获得的前5个预测：" first(y_pred_loaded, 5)

# 您还可以检查拟合参数或报告
# 例如，查看决策树组件的报告：
# report(loaded_pipe_mach).tree 
# 或者标准化器的拟合参数：
# fitted_params(loaded_pipe_mach).std
```

这个例子展示了整个工作流程，从预处理（标准化、独热编码）到建模（决策树），是如何被封装、训练、保存并作为一个单元重新加载的。

MLJ中保存和加载模型和管道的一般工作流程如下所示：

> 这个图表概述了典型步骤：定义和训练一个MLJ机器，保存它，然后后续加载并使用持久化的机器进行新预测。

### 重要注意事项

保存和加载模型时，请记住以下几点：

- **文件扩展名：** 尽管`.jlso`是`MLJ.save`使用的JLD2序列化文件的约定，但请确保一致使用它。
- **版本兼容性：** 序列化的Julia对象有时在尝试使用显着不同版本的Julia、MLJ.jl或特定模型包（例如DecisionTree.jl）加载时，可能会遇到兼容性问题。
  - **最佳做法：** 为了长期重现性，请将项目的`Project.toml`和`Manifest.toml`文件与您的序列化模型一同保存。这些文件会锁定所用所有包的精确版本，以便您在需要时重新创建环境。
- **环境：** 您加载模型的环境必须能访问保存的机器中使用的任何自定义类型或模型类型的定义。通常，这意味着加载相同的包（例如`using MLJDecisionTreeInterface`）。如果包在您的环境中，MLJ通常会处理加载必要的模型代码。
- **可移植性：** JLD2文件通常可以在不同操作系统之间移植，前提是Julia版本和相关包版本兼容。
- **文件大小：** 具有大量拟合结果的复杂模型或管道可能导致大型`.jlso`文件。请注意存储空间，特别是当您版本化许多模型时。
- **安全性：** 与任何反序列化过程（如Python的`pickle`）一样，只从受信任的来源加载`.jlso`文件。恶意制作的文件在加载时可能会执行任意代码。对于您自己训练和保存的模型，这通常不是一个大问题。

通过使用`MLJ.save`和`machine`进行加载，您可以有效地管理Julia中训练好的机器学习 (machine learning)资产的生命周期，使您的工作流程可重现并准备好部署。这种做法是机器学习系统从实验迈向生产就绪阶段的根本。

## 参考资料

- [JLD2.jl: Julia Hierarchical Data Format](https://github.com/JuliaIO/JLD2.jl) — JuliaIO/JLD2.jl Developers (2024)
  JLD2.jl的仓库和文档，它是MLJ.jl用于将对象序列化到磁盘的Julia包。
- [Pkg.jl Manual: Environments](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html#Environments-1) — The Julia Language Developers (2024)
  Julia Pkg.jl官方手册中解释包环境、Project.toml和Manifest.toml的部分，用于实现可重现的环境。
