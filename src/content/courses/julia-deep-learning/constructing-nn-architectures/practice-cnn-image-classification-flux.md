---
course: "julia-deep-learning"
chapter: "constructing-nn-architectures"
lesson: "practice-cnn-image-classification-flux"
sourceId: 6912
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/practice-cnn-image-classification-flux"
title: "实践：为图像分类实现一个卷积神经网络"
description: "动手项目，使用Flux.jl构建、训练和评估一个卷积神经网络以完成图像分类任务。"
order: 9
plots: []
sourceHash: "a2e104c44c6dc0cd6ed0f25b835a98df4029fa39739a3597cd57fddd8c0512ea"
sourceCorrections: []
---

构建一个卷积神经网络 (neural network)（CNN）进行图像分类时，会用到Flux.jl中可用的神经网络构建块。在Flux.jl中，单个层被组合以形成完整的模型。本练习将引导您完成数据集加载、CNN架构定义以及为训练做准备的步骤。

我们将使用知名的MNIST数据集，它包含手写数字（0-9）的灰度图像。我们的目标是建立一个CNN，能够查看这些图像并正确预测它所代表的数字。

### 1. 配置与加载依赖

首先，请确保您拥有所需的Julia包。如果您一直跟着学习，`Flux.jl`应该已在您的环境中。对于本练习，我们还需要`MLDatasets.jl`来便捷获取MNIST，`MLUtils.jl`用于数据处理实用工具，如果处理原始图像文件，可能还需要`Images.jl`用于图像特有变换（尽管`MLDatasets`通常提供的数据格式已相当可以直接使用）。我们还将使用`Statistics`进行平均值计算，并使用`Random`来保证结果的可复现性。

```julia
using Flux
using MLDatasets
using MLUtils: DataLoader, flatten, onehotbatch, unsqueeze
using Statistics: mean
using Random: MersenneTwister # 为了结果的可复现性
using Printf # 用于格式化输出
```

设置一个随机种子是很好的做法，尤其在开发和调试期间，如果您希望结果可复现的话。

```julia
const T = Float32 # 为数据和模型定义默认浮点类型
Random.seed!(MersenneTwister(123)); # 为了结果的可复现性
```

### 2. 加载和准备MNIST数据集

`MLDatasets.jl` 让加载像MNIST这样的标准数据集变得简单直接。

```julia
# 加载MNIST数据集
train_x_raw, train_y_raw = MNIST(T, :train)[:]
test_x_raw,  test_y_raw  = MNIST(T, :test)[:]
```

原始图像数据`train_x_raw`通常是一个3D数组，大小为`(宽度, 高度, 样本数量)`，例如，MNIST训练图像是`(28, 28, 60000)`。Flux中的CNN通常期望输入数据采用`(宽度, 高度, 通道, 批量大小)`格式，常被称为WHCN。由于MNIST图像是灰度图，它们有一个通道。

让我们重塑并预处理数据：

```julia
# 将数据重塑为WHCN格式（宽度、高度、通道、批量）
# MNIST图像为28x28，灰度（1通道）
train_x = unsqueeze(train_x_raw, dims=3) # 添加通道维度：28x28x1x60000
test_x  = unsqueeze(test_x_raw, dims=3)  # 添加通道维度：28x28x1x10000

# 将像素值归一化到[0, 1]（MLDatasets已为MNIST完成此步骤，但了解它仍是好的做法）
# 默认情况下，MLDatasets加载的MNIST数据为[0,1]范围内的Float32类型。
# 如果不是，您可能会这样做：train_x = train_x ./ T(255.0)

# 对标签进行独热编码
# 标签为0-9，共10个类别
train_y = onehotbatch(train_y_raw, 0:9) # 输出：10x60000
test_y  = onehotbatch(test_y_raw, 0:9)  # 输出：10x10000

# 创建用于批量处理的DataLoader
batch_size = 128
train_loader = DataLoader((train_x, train_y), batchsize=batch_size, shuffle=true)
test_loader  = DataLoader((test_x, test_y), batchsize=batch_size) # 测试数据无需打乱
```

这里，`unsqueeze(data, dims=3)` 在第三个位置增加一个新维度，将我们的`(28, 28, 60000)`数组转换为`(28, 28, 1, 60000)`，这是Flux期望的`WHCN`格式。`onehotbatch` 将整数标签（例如 `5`）转换为独热向量 (vector)（例如 `[0,0,0,0,0,1,0,0,0,0]`）。`MLUtils.jl`中的`DataLoader`是一个方便的实用工具，用于以小批量方式遍历数据，这对于高效训练神经网络 (neural network)非常重要。

### 3. 定义CNN架构

现在，让我们定义CNN的层。用于图像分类的常见CNN结构包含：

- 卷积层（`Conv`）用于检测特征。
- 激活函数 (activation function)（如`relu`）用于引入非线性。
- 池化层（`MaxPool`）用于降采样并减少维度。
- 展平步骤（`Flux.flatten`）用于将二维特征图转换为一维向量 (vector)。
- 全连接层（`Dense`）用于最终分类。

以下是一个架构示例：

```julia
model = Chain(
    # 第一个卷积块
    Conv((5, 5), 1=>6, relu),   # 28x28x1 -> 24x24x6 (假设无填充)
    MaxPool((2, 2)),          # 24x24x6 -> 12x12x6

    # 第二个卷积块
    Conv((5, 5), 6=>16, relu),  # 12x12x6 -> 8x8x16
    MaxPool((2, 2)),          # 8x8x16  -> 4x4x16

    # 展平输出并送入全连接层
    Flux.flatten,             # 4x4x16  -> 256
    Dense(256, 120, relu),    # 256     -> 120
    Dense(120, 84, relu),     # 120     -> 84
    Dense(84, 10)             # 84      -> 10 (10个类别的 logits)
)
```

让我们详细说明一下：

- `Conv((5, 5), 1=>6, relu)`: 一个带有5x5核的卷积层，接受1个输入通道（灰度）并生成6个输出通道（特征图）。`relu` 作为激活函数应用。
- `MaxPool((2, 2))`: 一个带有2x2窗口的最大池化层，将空间维度减半。
- 第二个`Conv`和`MaxPool`块进一步处理特征。
- `Flux.flatten`: 将卷积/池化层的多维输出（例如，`4x4x16`）转换为一个展平的向量（例如，`256`个元素），适合于全连接层。
- `Dense`层根据学到的特征执行分类。最终的`Dense`层有10个输出，对应于10个数字类别。我们这里不应用`softmax`，因为我们将使用一个需要原始 logits 的损失函数 (loss function)（`logitcrossentropy`）。

您可以将此架构可视化为一系列变换：

> 数据流经我们CNN的过程，从输入图像到每个类别的输出logits。N表示批量大小。

### 4. 准备训练：损失函数 (loss function)与优化器

为了训练我们的模型，我们需要一个损失函数来衡量其预测的错误程度，以及一个优化器来调整其参数 (parameter)。

```julia
# 损失函数：logitcrossentropy 适合于处理原始logit输出的分类任务
loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

# 优化器：ADAM 是一个常用选项
opt_state = Flux.setup(Adam(0.001), model) # Adam 学习率 0.001
```

`Flux.logitcrossentropy` 高效且数值稳定，适用于模型输出原始分数（logits）而非概率（这将在`softmax`后获得）的分类任务。`Adam` 是一种自适应学习率优化算法，通常表现良好。`Flux.setup` 为给定模型参数初始化优化器状态。

我们还可以定义一个辅助函数来计算准确率：

```julia
# 计算准确率的函数
function accuracy(data_loader, model)
    acc = 0
    num_samples = 0
    for (x_batch, y_batch) in data_loader
        # 获取模型预测（logits）
        y_hat_batch = model(x_batch)
        # 将 logits 转换为类别预测（最大 logit 的索引）
        # 并将独热标签转换为类别索引
        acc += sum(Flux.onecold(y_hat_batch) .== Flux.onecold(y_batch))
        num_samples += size(x_batch, ndims(x_batch)) # 计算批次中的样本数量
    end
    return acc / num_samples
end
```

`Flux.onecold` 将独热编码向量 (vector)转换回整数标签，或找到预测向量（logits或概率）中最大值的索引。

让我们检查一下未经训练的模型的准确率。它应该在10%左右（10个类别的随机猜测）。

```julia
@printf("Initial test accuracy: %.2f%%\n", accuracy(test_loader, model) * 100)
```

您应该会看到类似以下的输出：
`Initial test accuracy: 9.80%` （由于随机权重 (weight)初始化，具体值可能略有不同）。

### 5. 训练模型（概览）

实际的训练循环包含多次（epochs）遍历数据、计算损失、计算梯度，并使用优化器更新模型参数 (parameter)。第4章将更详细地介绍训练循环、评估和微调 (fine-tuning)。现在，让我们看看一个使用`Flux.train!`的非常基本的训练步骤：

```julia
# 一个单轮训练的简化示例
# 在第4章中，我们将构建更全面的训练循环。

# 仅为演示，让我们训练几个批次，以观察损失的减少。
# 这里我们将使用一个手动循环来阐明各个组成部分。
# 实际上，在完整训练中，会使用 Flux.train! 或自定义的循环来迭代多个 epoch。

num_epochs = 1 # 对于本次练习，只进行一个 epoch 以作说明
@printf("Starting training for %d epoch(s)...\n", num_epochs)

for epoch in 1:num_epochs
    epoch_loss = T(0.0)
    batches_processed = 0
    for (x_batch, y_batch) in train_loader
        # 计算损失和梯度
        batch_loss, grads = Flux.withgradient(model) do m
            loss(m, x_batch, y_batch)
        end

        # 更新模型参数
        Flux.update!(opt_state, model, grads[1])

        epoch_loss += batch_loss
        batches_processed += 1

        # 可选：打印少数批次的进度
        if batches_processed % 100 == 0
            @printf("Epoch %d, Batch %d/%d: Batch Loss: %.4f\n", 
                    epoch, batches_processed, length(train_loader), batch_loss)
        end
    end
    avg_epoch_loss = epoch_loss / batches_processed
    train_acc = accuracy(train_loader, model)
    test_acc = accuracy(test_loader, model)
    @printf("Epoch %d finished. Avg Loss: %.4f, Train Acc: %.2f%%, Test Acc: %.2f%%\n", 
            epoch, avg_epoch_loss, train_acc*100, test_acc*100)
end
```

运行这个简化训练片段一个 epoch 应该会显示准确率提升和损失下降，表明模型正在学习。例如，一个 epoch 后，您可能会看到类似：
`Epoch 1 finished. Avg Loss: 0.2831, Train Acc: 94.75%, Test Acc: 94.99%`

这些结果对于仅仅一个 epoch 来说表现很好！随着 epoch 数量的增加以及可能的超参数 (hyperparameter)调整（稍后会介绍），性能可以进一步提升。

### 6. 保存模型

一旦您有了训练好的模型，通常会想保存其结构和学到的参数 (parameter)。在Julia生态系统中，`BSON.jl`常用于此目的。

```julia
using BSON: @save, @load

# 保存模型：
# 注意：只保存 `model` 会保存模型结构和参数。
# 对于某些优化器或更复杂的场景，您可能还需要保存 `opt_state`。
@save "mnist_cnn_model.bson" model

# 稍后加载模型：
# @load "mnist_cnn_model.bson" loaded_model
# `loaded_model` 现在将包含您已保存模型的结构和参数。
```

这让您可以重用您训练好的模型进行推断或进一步训练，而无需从头开始重新训练。

本次实践引导您完成了实现CNN的核心步骤：使用`MLUtils.jl`进行数据加载和准备，定义一个由`Conv`、`MaxPool`和`Dense`层组成的`Chain`，以及设置损失函数 (loss function)和优化器。虽然我们只是初步接触了训练过程，但您现在已拥有构建各种神经网络 (neural network)架构的坚实基础。下一章将更详细地讲解这些模型的训练、评估和改进机制。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍了卷积神经网络（CNN）的基础架构，包括卷积层、池化层和全连接层，并通过MNIST手写数字识别数据集进行了演示。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合性教材，涵盖了神经网络架构、训练方法、损失函数和Adam等优化算法。其中关于CNN的章节与本节相关。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Community (2025)
  Flux.jl的官方文档，提供了详细的API参考和指南，用于在Julia中构建神经网络、使用MLUtils处理数据以及设置训练循环。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: International Conference for Learning Representations (ICLR) 2015; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam优化算法，这是一种广泛采用的自适应学习率方法，用于高效训练深度神经网络。
