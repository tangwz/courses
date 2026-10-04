---
course: "julia-deep-learning"
chapter: "constructing-nn-architectures"
lesson: "handling-datasets-mlutils-jl"
sourceId: 6897
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/handling-datasets-mlutils-jl"
title: "处理数据集：使用 MLUtils.jl 的迭代器和加载器"
description: "使用 MLUtils.jl 在深度学习训练流程中实现高效的数据加载、批处理和迭代。"
order: 2
plots: []
sourceHash: "402118147b8e946159d881933309e827497141f3128aa46e6ef22e1eb303ed25"
sourceCorrections: []
---

有效管理并将数据送入神经网络 (neural network)，对模型训练的成功来说必不可少，尤其是在处理大型数据集时。在构建卷积神经网络（CNN）和循环神经网络（RNN）等更复杂的架构时，数据迭代、批处理和转换的机制会变得越发重要。Julia 的 `MLUtils.jl` 包提供了一系列工具，旨在提升数据处理效率，让深度学习 (deep learning)工作流程更高效、更易于管理。

本节将指导你如何使用 `MLUtils.jl` 来准备用于训练的数据集。我们将介绍如何迭代观察值、创建小批量、打乱数据以获得更好的泛化能力、分割数据集，以及应用即时转换。这些实用工具旨在与 Flux.jl 以及你的 GPU 加速训练流程良好地结合。

### MLUtils.jl 在数据处理流程中的作用

训练深度学习 (deep learning)模型通常涉及分块处理数据，这些分块称为小批量，而不是一次处理一个样本或整个数据集。这种方法在计算效率和优化过程中梯度更新的稳定性之间取得了平衡。`MLUtils.jl` 简化了这些小批量的创建和管理，以及其他常见的数据准备步骤。

`MLUtils.jl` 提供的主要功能，对深度学习特别有帮助的包括：

- **观察值迭代**：一种访问数据集中单个样本的稳定方式。
- **批处理**：将观察值分组为小批量。
- **打乱**：随机化数据顺序，以防止模型学习到与数据顺序相关的虚假模式。
- **数据分割**：将数据集划分为训练集、验证集和测试集。
- **数据转换**：对观察值应用函数，通常用于数据增强或预处理。

我们来看看这些功能是如何实现的。

### 使用 `eachobs` 迭代观察值

`MLUtils.jl` 的核心思想是“观察值”。一个观察值通常是一个单一数据点，通常包含特征和对应的标签。`eachobs` 函数提供了一种遍历数据集中观察值的通用方法。

如果你的数据以数组元组的形式存储（例如，`(features, labels)`），`eachobs` 将生成元组，其中每个元素都是原始数组中对应单个观察值的一个切片或视图。

```julia
using MLUtils

# 示例数据：10 个特征，5 个样本
X = rand(Float32, 10, 5)
# 对应 5 个样本的标签
Y = [1, 0, 1, 1, 0]

# 迭代每个观察值
for (x_obs, y_obs) in eachobs((X, Y))
    # x_obs 将是包含 10 个特征的向量，代表一个样本
    # y_obs 将是一个单一标签
    println("Features: ", size(x_obs), ", Label: ", y_obs)
end
```

这种简单的迭代是构建批处理等更复杂操作的基础。`MLUtils.jl` 还定义了 `nobs` 来获取观察值数量，以及 `getobs` 来按索引检索特定观察值。

### 使用 `DataLoader` 和 `eachbatch` 进行数据批处理

训练神经网络 (neural network)时，你几乎总是希望以小批量方式处理数据。`MLUtils.jl` 提供了两种主要方法来实现这一点：`DataLoader` 类型和 `eachbatch` 函数。

**`DataLoader`**：它构建一个迭代器，用于生成数据批次。它很灵活，允许你指定批次大小、是否打乱数据，以及如何处理大小小于指定值的最后一个批次。

```julia
using MLUtils

# 示例数据：2 个特征，100 个样本
features = rand(Float32, 2, 100)
labels = rand(Int, 100) # 100 个整数标签
dataset = (features, labels)

# 创建 DataLoader
# batchsize: 每个批次的样本数量
# shuffle: 如果为 true，则在每个 epoch（遍历整个数据集）开始时打乱数据
# partial: 如果为 false（默认），则丢弃大小小于 batchsize 的最后一个批次。如果为 true，则包含它。
loader = DataLoader(dataset, batchsize=32, shuffle=true, partial=false)

# 迭代批次
for epoch in 1:3 # 示例：3 个 epoch
    println("Epoch: ", epoch)
    for (x_batch, y_batch) in loader
        # x_batch 将是一个 2x32 矩阵（32 个样本的特征）
        # y_batch 将是一个包含 32 个标签的向量
        # 在此示例中，有 100 个样本和 batchsize 为 32，
        # 且 partial=false，我们将得到 3 个大小为 32 的批次。100 = 3*32 + 4。最后的 4 个样本被丢弃。
        println("Batch sizes: ", size(x_batch), ", ", size(y_batch))
        # 在这里，你通常会执行一个训练步骤：
        # 1. 将批次移至 GPU（如果适用）
        # 2. 前向传播：model(x_batch)
        # 3. 计算损失：loss_function(predictions, y_batch)
        # 4. 反向传播（计算梯度）
        # 5. 更新模型参数
    end
end
```

**`eachbatch`**：这是一个更直接的函数，对于更简单的用例，它通常与 `DataLoader` 实现相同的目的。它也接受数据、`batchsize` 和像 `shuffle` 这样的选项。

```julia
using MLUtils

features = rand(Float32, 2, 100)
labels = rand(Int, 100)
dataset = (features, labels)

# 使用 eachbatch
for (x_batch, y_batch) in eachbatch(dataset, batchsize=16, shuffle=true)
    # x_batch 将是一个 2x16 矩阵
    # y_batch 将是一个包含 16 个标签的向量
    # 处理批次...
    # println("使用 eachbatch 的批次大小：", size(x_batch), ", ", size(y_batch))
end
```

对于训练数据，通常建议使用 `shuffle=true`。这有助于防止模型学习到基于数据呈现顺序的模式，并能带来更好的泛化能力。对于验证或测试数据，通常会关闭打乱功能 (`shuffle=false`)，以确保评估的一致性。

### 使用 `splitobs` 分割数据集

一个常见需求是将数据集划分为训练集、验证集，有时还包括测试集。`MLUtils.jl` 为此提供了 `splitobs` 函数。

```julia
using MLUtils

# 1000 个样本，每个样本 20 个特征
X_all = rand(Float32, 20, 1000)
Y_all = rand(Float32, 1, 1000) # 回归任务，1 个输出
full_dataset = (X_all, Y_all)

# 分割成训练集 (80%) 和验证集 (20%)
# 在分割前 shuffle=true 是一个好习惯
train_data, val_data = splitobs(full_dataset, at=0.8, shuffle=true)

# train_data 和 val_data 本身就是元组：(X_train, Y_train) 和 (X_val, Y_val)
X_train, Y_train = train_data
X_val, Y_val = val_data

println("训练样本数：", nobs(train_data)) # nobs(X_train) 或 nobs(Y_train)
println("验证样本数：", nobs(val_data))

# 现在你可以为每个集合创建 DataLoader
train_loader = DataLoader(train_data, batchsize=64, shuffle=true)
val_loader = DataLoader(val_data, batchsize=64, shuffle=false) # 验证集不打乱
```

`at` 参数 (parameter)指定了分割第一部分的比例。你也可以向 `at` 传递一个比例元组以进行多次分割，例如，`at=(0.7, 0.15)` 将得到 70% 的训练集、15% 的验证集以及剩余 15% 的测试集。

### 使用 `mapobs` 进行即时数据转换

通常，在数据被批处理并送入模型之前，你需要对其应用转换。这可以是归一化 (normalization)、数据增强（如翻转或旋转图像），或转换数据类型。`MLUtils.jl` 的 `mapobs` 函数允许你惰性地对每个观察值应用函数。

```julia
using MLUtils

# 假设 raw_images 是一个包含 100 张小灰度图像（例如，28x28）的数组
raw_images = [rand(Float32, 28, 28) for _ in 1:100]
labels = rand(0:9, 100) # 数字 0-9 的标签

# 一个简单的数据增强：水平翻转图像
function augment_image(image::Matrix{Float32})
    # 添加通道维度 (Flux.jl CNN 期望 WHC 或 WHCN 格式)
    img_with_channel = reshape(image, size(image)..., 1)
    if rand() > 0.5
        return reverse(img_with_channel, dims=2) # 水平翻转
    else
        return img_with_channel
    end
end

# 将数据增强应用于 raw_images 中的每张图像
# mapobs 将创建一个新的数据源，其中每个观察值都是 augment_image 的结果
augmented_dataset = mapobs(augment_image, raw_images)

# 然后我们可以为 (augmented_images, labels) 创建一个 DataLoader
# 注意：要将增强特征与标签配对，你通常会向 mapobs 传递一个元组
# 或者只将 mapobs 应用于 (features, labels) 元组的 features 部分。

# 如果标签也需要处理或应该直接传递：
full_dataset_to_transform = (raw_images, labels) # 假设 raw_images 是一个矩阵向量

# 用于 (图像, 标签) 对的转换函数
# 这里，只转换图像，标签直接传递
function transform_observation(obs_tuple)
    image, label = obs_tuple
    augmented_img = augment_image(image) # 你之前定义的数据增强
    # Flux 期望特征为 WHCN 格式（宽度、高度、通道、批次）
    # DataLoader 将整理 N（批次）维度。
    return (augmented_img, label)
end

# 正确地将 mapobs 应用于数据集元组
processed_dataset = mapobs(transform_observation, full_dataset_to_transform)

# DataLoader 现在将获取增强后的图像
loader = DataLoader(processed_dataset, batchsize=10, shuffle=true)

for (img_batch, label_batch) in loader
    # img_batch 将是 (28, 28, 1, 10)
    # label_batch 将是一个包含 10 个标签的向量
    # println("增强图像批次大小：", size(img_batch))
    # println("标签批次大小：", size(label_batch))
end
```

`mapobs` 功能强大，因为转换是在数据被请求时应用的，这可以节省内存，特别是对于复杂的增强操作。

以下图表展示了这些 `MLUtils.jl` 组件如何融入典型的深度学习 (deep learning)数据处理流程：

> 使用 `MLUtils.jl` 的数据处理流程。原始数据通常会进行分割，然后可以使用 `mapobs` 对每个观察值应用转换。接着 `DataLoader` 或 `eachbatch` 会创建打乱过的小批量，这些小批量最后被准备好（例如，移至 GPU）并送入神经网络 (neural network)模型。打乱操作可以在 `splitobs` 阶段进行初始分割时发生，也可以在每个 epoch 的 `DataLoader` 内部发生。

### 与 GPU 工作流集成

训练深度学习 (deep learning)模型时，特别是大型模型，使用 GPU 来加速计算很常见。`MLUtils.jl` 本身不处理 GPU 内存传输。它的作用是提供数据批次。然后，你通常会在训练循环中使用 `Flux.jl` 的函数（如 `Flux.gpu` 或 `|> gpu`）或 `CUDA.jl` 的函数（如 `CUDA.cu`）将这些批次移至 GPU。

```julia
using Flux # 用于 gpu 函数和模型定义
using CUDA # 如果需要，用于 cu 函数，并检查 GPU 是否可用
# 假设：
# model = Chain(...) |> gpu # 你的 Flux 模型已移至 GPU
# loader = DataLoader(train_data, batchsize=64, shuffle=true)

if CUDA.functional() # 检查 GPU 是否可用且功能正常
    model = model |> gpu
    println("正在 GPU 上训练。")
    for (x_batch, y_batch) in loader
        x_batch_gpu = x_batch |> gpu
        y_batch_gpu = y_batch |> gpu

        # 使用 GPU 数据的训练步骤
        # grads = gradient(params(model)) do
        #     loss(model(x_batch_gpu), y_batch_gpu)
        # end
        # update!(opt, params(model), grads)
    end
else
    println("CUDA GPU 不可用或功能不正常。正在 CPU 上训练。")
    for (x_batch, y_batch) in loader
        # 使用 CPU 数据的训练步骤
        # ...
    end
end
```

这种模式确保只有当前批次的数据驻留在 GPU 上，这对于管理有限的 GPU 内存很重要。

通过提供这些用于数据迭代、批处理、打乱、分割和转换的多功能工具，`MLUtils.jl` 大大简化了你在 Julia 中深度学习项目的数据准备方面。数据得到高效管理后，你就可以更专注于设计和训练神经网络 (neural network)架构本身，我们也将继续了解这些内容。

## 参考资料

- [MLUtils.jl Documentation](https://github.com/JuliaML/MLUtils.jl) — The MLUtils.jl Contributors (2025)
  该Julia库为机器学习提供数据工具的官方文档。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，解释了小批量处理、数据分割和增强等数据处理技术的理论基础。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Contributors (2025)
  Flux.jl 深度学习框架的官方文档，MLUtils.jl 与其集成用于模型训练和GPU工作流程。
