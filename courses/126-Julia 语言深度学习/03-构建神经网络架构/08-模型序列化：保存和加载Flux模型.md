# 模型序列化：保存和加载Flux模型

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/model-serialization-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

无论是构建和训练多层感知器（MLP）、卷积神经网络 (neural network)（CNN）还是循环神经网络（RNN），您一旦投入精力，自然会希望保存您的成果。模型序列化是将模型训练所得的参数 (parameter)以及通常其结构保存到文件的过程。这样，您日后便能再次载入它，用于推断、继续训练或与他人分享。若无序列化，您训练好的模型将仅存在于计算机内存中，一旦Julia会话结束便会丢失。

## 保存和加载模型的重要性

保存已训练的模型是机器学习 (machine learning)工作流程中的常规做法，其有几个重要原因：

- **持久性**：深度学习 (deep learning)模型的训练可能需要数小时、数天甚至数周。保存模型可确保这些计算工作不会因系统重启或其他中断而白费。
- **部署**：要让模型在生产环境或其他应用程序中进行预测，您需要方法载入已训练的版本。
- **可复现性**：保存模型及其版本和训练细节有助于在日后或由他人复现结果。
- **迁移学习 (transfer learning)**：一个已保存的模型可以作为新任务训练的起点，有可能节省大量训练时间。
- **检查点**：对于长时间的训练运行，定期保存模型（检查点）可让您在出现问题时能从上次保存的状态继续训练，而非从头开始。

本质上，序列化使您的模型成为有形的资产，可以被管理、版本化，并长期运用。

## 在Julia中使用BSON.jl进行序列化

对于Flux模型，推荐用于序列化的包是`BSON.jl`。BSON代表Binary JSON（二进制JSON），它是一种用于类JSON文档的二进制编码序列化格式。它被设计为轻量、可遍历且高效。`BSON.jl`非常适合Julia对象，包括构成Flux模型的复杂结构和函数。

要使用`BSON.jl`，您首先需要将其添加到您的Julia环境并导入：

```julia
# 如果尚未安装：
# import Pkg; Pkg.add("BSON")

using Flux
using BSON
```

## 保存Flux模型

使用`BSON.jl`保存对象的主要宏是`BSON.@save`。假设您有一个名为`my_cnn_model`的已训练Flux模型。

### 保存单个模型

要将模型保存到文件，您可以使用`BSON.@save`，后接文件名和包含您模型的变量：

```julia
# 假设 'my_cnn_model' 是一个已训练的Flux模型
# 比如：
# my_cnn_model = Chain(
#     Conv((3, 3), 1=>16, relu),
#     MaxPool((2,2)),
#     Conv((3, 3), 16=>32, relu),
#     MaxPool((2,2)),
#     Flux.flatten,
#     Dense(32*5*5, 10), # 假设输入为28x28，请相应调整大小
#     softmax
# )
# ... 在此处进行训练 ...

# 保存模型
BSON.@save "my_trained_cnn.bson" my_cnn_model
println("模型已保存到 my_trained_cnn.bson")
```

此命令会将`my_cnn_model`对象（包括其架构和训练所得的权重 (weight)）保存到名为`my_trained_cnn.bson`的文件中。

### 保存多个对象

通常，您会希望保存的不仅仅是模型。例如，您可能想保存优化器的状态、当前的训练轮次编号或性能指标。`BSON.@save`支持将多个变量保存到同一个BSON文件：

```julia
# 假设 'model', 'optimizer_state', 'epoch' 和 'history' 已定义
# model = Chain(...)
# opt = Adam()
# optimizer_state = Flux.setup(opt, model) # 获取优化器状态
# epoch = 100
# history = Dict("loss" => [0.5, 0.4, ...], "accuracy" => [0.8, 0.85, ...])

# BSON.@save "training_checkpoint.bson" model optimizer_state epoch history
# 或者，为了加载时的清晰度，使用关键字参数：
BSON.@save "training_checkpoint.bson" trained_model=model opt_state=optimizer_state current_epoch=epoch training_history=history
println("检查点已保存到 training_checkpoint.bson")
```

使用关键字参数 (parameter)（例如`trained_model=model`）进行保存是一种好做法，因为它使得加载过程更为明确，正如您接下来将看到的。

## 加载Flux模型

要从BSON文件加载已保存的模型（或其他对象），您可以使用`BSON.@load`宏。

### 恢复模型

`BSON.@load`宏会将指定文件中的对象加载到您当前的Julia工作空间。变量将沿用保存时的名称，或者采用保存时使用的关键字命名。

```julia
using Flux, BSON

# 加载之前保存的模型
BSON.@load "my_trained_cnn.bson" my_cnn_model # 假设它保存时名为 'my_cnn_model'
# 如果使用关键字保存：BSON.@load "my_trained_cnn.bson" loaded_model_alias
# 那么 'loaded_model_alias' 将包含该模型。

# 'my_cnn_model' 现在在您的工作空间中可用
# 您可以检查它或用于预测
# 比如：
# dummy_input = rand(Float32, 28, 28, 1, 1) # CNN的示例输入
# predictions = my_cnn_model(dummy_input)
# println("模型已加载并完成预测。")

# 如果您使用关键字保存了多个对象：
BSON.@load "training_checkpoint.bson" trained_model opt_state current_epoch training_history
# 现在 'trained_model', 'opt_state', 'current_epoch' 和 'training_history' 都已可用。

println("模型及相关数据加载成功。")
```

以下图表说明了基本的保存和加载周期：

> 模型序列化工作流程：将Flux模型保存到BSON文件，随后再将其加载回Julia环境。

### 加载时的环境考量

当您加载模型时，Julia需要理解被加载对象的结构。

- **标准Flux层**：如果您的模型由标准Flux层（如`Dense`、`Conv`、`Chain`）构成，`BSON.jl`和Flux能顺利处理此情况。
- **自定义层/结构体**：如果您的模型包含自定义层或结构体，这些类型的定义必须在您加载模型的Julia环境中可用。这意味着定义您自定义结构体的Julia代码需要在调用`BSON.@load`之前运行。
- **包版本**：保存和加载环境之间包版本（例如Flux.jl本身，或涉及GPU时的CUDA.jl）的较大差异有时会导致问题。通常，最好在包版本相近的环境中加载模型。

## 进阶：仅保存和加载模型参数 (parameter)

有时，您可能更倾向于只保存模型训练所得的参数（权重 (weight)和偏置 (bias)），而非整个模型对象。这种方法可以带来更大的灵活性，特别是当您想将权重加载到稍作修改的架构中，或者担心不同Flux版本间模型结构本身的兼容性问题时。

Flux提供`Flux.params(model)`来提取所有可训练参数，以及`Flux.loadparams!(model, params_array)`来将它们加载回去。

### 保存参数 (parameter)

```julia
using Flux, BSON

# 假设 'model' 是您已训练的Flux模型
# model = Chain(Dense(10, 5, relu), Dense(5, 1))
# ... 训练 ...

# 提取参数
model_parameters = Flux.params(model)

# 注意：Flux.params(model) 返回一个 Zygote.Params 对象。
# 为了用BSON有效保存它们，通常最好将它们收集到标准数组的数组中。
# 然而，BSON通常可以直接处理 Zygote.Params。如果需要更通用的保存方式，特别是
# 如果您想在Flux外部检查或操作权重，转换为常规数组更安全。
# 对于直接用BSON保存并加载到另一个Flux模型中，保存 Zygote.Params 通常可行。

# 保存原始权重数组：
# weights_arrays = [copy(p) for p in model_parameters] # 如果需要，copy() 可以将它们从GPU移出并制成普通数组
# BSON.@save "model_just_weights.bson" trained_weights=weights_arrays

# 为简单起见，如果BSON在您的设置中能很好地处理 Zygote.Params：
BSON.@save "model_params_object.bson" model_ps=model_parameters
println("模型参数已保存。")
```

### 将参数 (parameter)加载到新的模型实例中

要加载这些参数，您首先需要构建一个具有相同架构的模型实例。然后，加载已保存的参数并填充到新的模型实例中。

```julia
using Flux, BSON

# 1. 重建模型架构
# 这必须与保存参数的模型架构相匹配。
new_model_instance = Chain(Dense(10, 5, relu), Dense(5, 1))

# 2. 加载已保存的参数
BSON.@load "model_params_object.bson" model_ps # 加载 'model_ps'

# 3. 将参数加载到模型中
Flux.loadparams!(new_model_instance, model_ps)
println("参数已加载到新的模型实例中。")

# 'new_model_instance' 现在已准备好，带有训练所得的权重。
```

此方法将模型的结构（由您在代码中定义）与其训练所得的状态（参数）解耦，实现了更清晰的分离。

## 使用GPU训练的模型

如果您的模型在训练期间使用`model |> gpu`移至GPU，其参数 (parameter)很可能是`CuArray`s（来自`CUDA.jl`）。`BSON.jl`可以保存这些`CuArray`s。

加载时：

- 如果您在具有兼容GPU和`CUDA.jl`的环境中加载模型，模型可能会直接作为GPU模型加载。
- 如果您在仅有CPU的环境中加载它，或者想明确地在CPU上使用它，您可以在加载后将其移动：

```julia
using Flux, BSON, CUDA # 假设CUDA可用

# 假设 "gpu_trained_model.bson" 包含一个在GPU上训练的模型
BSON.@load "gpu_trained_model.bson" loaded_model

# 检查模型是否在GPU上（其参数将是 CuArray 类型）
# 您可能需要检查，例如：first(Flux.params(loaded_model)) isa CuArray

# 确保它在CPU上：
cpu_model = loaded_model |> cpu
println("模型已移至CPU。")

# 确保它在GPU上（如果GPU可用）：
# if CUDA.functional()
#   gpu_model = loaded_model |> gpu
#   println("模型已移至GPU。")
# else
#   println("没有可用的GPU，模型保留在CPU上。")
# end
```

加载后明确管理设备传输（`|> cpu`或`|> gpu`）是很好的做法，特别是如果您计划将模型部署到与训练时不同的环境中。

## 模型管理的实用提示

- **版本控制**：将您保存的模型文件（`.bson`文件）视为重要产物。如果您正在使用Git，请考虑为模型文件使用Git LFS（大型文件存储），因为它们可能会变得很大。包含版本号或时间戳的命名约定也很有用（例如，`my_model_v1.2_epoch50.bson`）。
- **包含元数据**：保存模型，尤其是检查点时，保存相关元数据。这可以包括您的训练脚本版本、数据集标识符、使用的超参数 (parameter) (hyperparameter)、达到的性能指标以及训练轮次编号。此信息对于跟踪实验和复现结果非常有价值。

  ```julia
  # BSON.@save "model_with_metadata.bson" model=my_model version="1.1" dataset="CIFAR10" epoch=50 acc=0.85
  ```
- **测试加载**：保存模型后，明智的做法是立即尝试将其加载回一个干净的会话或脚本中，以验证序列化是否成功以及模型行为是否符合预期。

## 关于替代方案的说明：JLD2.jl

尽管`BSON.jl`是Flux模型的常见选择，但Julia中另一个通用的序列化包是`JLD2.jl`。JLD2以基于HDF5的格式保存数据。对于某些Julia对象，`JLD2.jl`可能非常有效。然而，`BSON.jl`通常对Flux模型中经常遇到的闭包和自定义结构类型有更好的支持，使其成为此特定用途更直接的选择。如果您在使用`BSON.jl`处理非常特殊的自定义类型时遇到问题，`JLD2.jl`可作为选择，尽管它在模型保存和加载方式上可能需要更多关注。

通过掌握模型序列化，您可保证自己构建和训练的神经网络 (neural network)架构得到保存、可分享，并可供后续使用，无论是用于新数据预测，还是作为进一步实验的依据。这项本领在您进行实际练习时会特别有用，例如即将到来的构建用于图像分类的CNN的练习；保存您训练好的分类器将是自然的最后一步。

## 参考资料

- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — Flux.jl Contributors (2025)
  Flux深度学习框架的官方文档，详细介绍了模型构建、参数处理以及模型持久化的推荐实践。
- [BSON.jl Documentation](https://github.com/JuliaIO/BSON.jl) — JuliaIO/BSON.jl Contributors (2024)
  BSON.jl的官方仓库和文档，它是用于高效序列化Julia对象的Julia包，包括复杂的Flux模型。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，全面涵盖深度学习原理，包括在机器学习工作流中模型持久性、部署和检查点等关键方面。
- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/) — Tim Besard, Valentin Churavy, Mike Innes, Katharine Hyatt, Simon Danisch (2025)
  CUDA.jl的官方文档，支持Julia中的GPU加速，与理解GPU训练模型在序列化和设备传输中的处理方式相关。

---

[上一节](07-%E5%9C%A8%20Flux%20%E4%B8%AD%E5%88%9B%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E5%B1%82.md) · [下一节](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%BA%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
