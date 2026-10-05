# 实践：使用GPU加速训练

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/practice-accelerating-training-gpus)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本实践重点在于将神经网络 (neural network)训练过程从仅在CPU上运行调整为使用NVIDIA GPU，通过CUDA.jl和Flux.jl实现。目标是完成一个简单的图像分类任务，首先在CPU上训练模型，然后修改脚本以在GPU上加速训练，并观察其性能差异。

为完成此练习，您需要：

- 一个安装了Flux.jl、CUDA.jl、MLDatasets.jl和Statistics.jl并能正常运行的Julia环境。
- 一块与CUDA兼容且驱动程序配置正确的NVIDIA GPU。
- MNIST数据集，我们将使用MLDatasets.jl获取。

### 我们的起点：一个受CPU限制的图像分类器

让我们从一个脚本开始，该脚本定义了一个简单的卷积神经网络 (neural network)（CNN），准备MNIST数据集，并在CPU上训练模型。

```julia
using Flux, MLDatasets, Statistics, Random
using Flux: onehotbatch, onecold, @epochs, glorot_uniform
using Base.Iterators: partition # For creating minibatches
using Printf

# 设置随机种子以保证结果可复现
Random.seed!(123)

# 1. 加载MNIST数据
println("正在加载MNIST数据集...")
# 完整数据集
# imgs_train_raw, labels_train_raw = MNIST.traindata();
# imgs_test_raw, labels_test_raw = MNIST.testdata();

# 为了更快地演示，我们使用一个子集
train_n = 5000 # Number of training samples
test_n = 1000  # Number of test samples
imgs_train_raw, labels_train_raw = MNIST.traindata(1:train_n);
imgs_test_raw, labels_test_raw = MNIST.testdata(1:test_n);

# 2. 数据预处理
println("正在预处理数据...")
# 为CNN重塑数据（WHCN格式：宽度、高度、通道数、样本数）
# 转换为Float32并归一化像素值到[0,1]
preprocess_images(imgs) = reshape(Float32.(imgs), 28, 28, 1, :) |> Flux.normalize

X_train = preprocess_images(imgs_train_raw);
X_test = preprocess_images(imgs_test_raw);

# 进行独热编码
Y_train = onehotbatch(labels_train_raw, 0:9);
Y_test = onehotbatch(labels_test_raw, 0:9);

# 3. 定义CNN模型（CPU）
println("正在定义CPU模型...")
model_cpu = Chain(
    Conv((3, 3), 1=>16, relu, init=glorot_uniform), # 输出：26x26x16
    MaxPool((2,2)),                               # 输出：13x13x16
    Conv((3, 3), 16=>32, relu, init=glorot_uniform),# 输出：11x11x32
    MaxPool((2,2)),                               # 输出：5x5x32
    Flux.flatten,                                 # 输出：800
    Dense(5*5*32, 128, relu, init=glorot_uniform),
    Dense(128, 10, init=glorot_uniform)           # 原始分数（logits）
    # softmax通常在损失函数中或之后应用以得到概率
)

# 4. 定义损失函数和优化器（CPU）
loss_cpu(x, y) = Flux.logitcrossentropy(model_cpu(x), y) # 使用logitcrossentropy以提高数值稳定性
opt_cpu = ADAM(0.001)

# 5. 准备小批量数据（CPU）
batch_size = 128
# 数据已在CPU上
train_loader_cpu = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)

# 6. CPU训练函数
function train_epoch_cpu!(model, loader, opt_state, loss_fn)
    ps = Flux.params(model)
    for (x_batch, y_batch) in loader
        # x_batch和y_batch已经是CPU数组
        gs = gradient(() -> loss_fn(x_batch, y_batch), ps)
        Flux.update!(opt_state, ps, gs)
    end
end

# 7. 在CPU上训练和计时
println("正在开始CPU单轮训练...")
# 为获得更精确的基准测试结果，请考虑使用BenchmarkTools.jl的@btime
# 这里，@time提供简单的计时。
# 热身（可选，但有助于获得更稳定的@time结果）
# train_epoch_cpu!(model_cpu, first(train_loader_cpu,1) , opt_cpu, loss_cpu)
cpu_training_time = @elapsed train_epoch_cpu!(model_cpu, train_loader_cpu, opt_cpu, loss_cpu)
@printf "CPU单轮训练完成，耗时 %.2fs。\n" cpu_training_time

# 快速检查CPU上的准确率
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))
acc_cpu = accuracy(X_test, Y_test, model_cpu)
@printf "CPU模型单轮训练后的准确率: %.2f%%\n" acc_cpu*100
```

此脚本搭建了一个标准的训练流程。模型、数据和计算都位于CPU上。请留意`Flux.DataLoader`的使用，它便于进行批量处理和数据打乱。

### 迁移到GPU：`gpu`函数

现在，我们将CUDA.jl引入。Flux为GPU加速提供的主要工具是`gpu`函数。此函数将模型和数据移动到当前活跃的GPU上。反之，`cpu`函数将它们移回CPU。

首先，请确保CUDA可用并正常运行：

```julia
using CUDA

if !CUDA.functional()
    println("此系统上CUDA不可用或无法正常运行。将跳过GPU实践。")
    # 您可能希望退出或确保不执行GPU特定的代码路径。
else
    println("CUDA运行正常。GPU加速可用。")
    # 继续执行GPU特定代码
end
```

假设`CUDA.functional()`返回`true`：

1. **将模型移至GPU**：
   为在GPU上运行模型，我们使用`gpu`函数传输其参数 (parameter)和结构。

   ```julia
   # 这会创建一个新的模型结构，其参数在GPU上
   if CUDA.functional()
       model_gpu = gpu(model_cpu)
       # 您也可以这样做：model_gpu = model_cpu |> gpu
       println("模型已传输到GPU。")
   end
   ```

   重要的是要明白，`model_gpu`现在是一个新的模型实例。它的参数（权重 (weight)和偏置 (bias)）位于GPU内存中。对`model_gpu`进行的操作，若数据也在GPU上，将会在GPU上执行。
2. **将数据移至GPU**：
   类似地，输入数据（特征`x`和标签`y`）在馈送给`model_gpu`之前必须位于GPU上。这通常在训练循环中逐批完成。

   ```julia
   # 示例：移动单个批次（来自CPU加载器的x_batch_cpu, y_batch_cpu）
   # x_batch_gpu = gpu(x_batch_cpu)
   # y_batch_gpu = gpu(y_batch_cpu)
   ```

### 调整GPU训练循环

模型移至GPU后，训练循环需要稍作修改，以确保数据批次在每次前向和反向传播 (backpropagation)之前也被移动到GPU。

```julia
if CUDA.functional()
    # 1. 模型已在GPU上：model_gpu

    # 2. 定义GPU模型的损失函数和优化器
    # 损失函数现在使用model_gpu
    loss_gpu(x, y) = Flux.logitcrossentropy(model_gpu(x), y)
    opt_gpu = ADAM(0.001) # GPU模型参数的优化器

    # 3. GPU数据加载器
    # Flux.DataLoader可以与自定义的collate函数一起使用以移动数据，
    # 或者我们可以在训练循环内部移动数据。为了此处简洁起见，我们将
    # 修改训练循环来移动批次数据。
    # train_loader_cpu仍提供基于CPU的批次数据。

    # 4. GPU训练函数
    function train_epoch_gpu!(model, loader, opt_state, loss_fn)
        ps = Flux.params(model) # 参数已在GPU上
        for (x_batch_cpu, y_batch_cpu) in loader
            # 将当前批次数据移至GPU
            x_batch_gpu = gpu(x_batch_cpu)
            y_batch_gpu = gpu(y_batch_cpu)

            # 在GPU上计算梯度
            gs = gradient(() -> loss_fn(x_batch_gpu, y_batch_gpu), ps)
            Flux.update!(opt_state, ps, gs)

            # CUDA.synchronize() # 解除注释以进行精确的步长计时/调试
                                # 通常在训练循环的正确性上不需要
        end
    end

    println("正在开始GPU单轮训练...")
    # 如果您想从头开始进行公平比较，请从model_cpu重新初始化model_gpu
    # model_gpu = gpu(deepcopy(model_cpu)) # Or re-define model_cpu then gpu(model_cpu)
    # 为了比较速度，理想情况下，我们会重置model_cpu的权重或使用一个全新副本
    # 对于此脚本，我们将只训练一个从原始model_cpu结构创建的新model_gpu。
    # 让我们为GPU重新创建一个全新的模型结构，以避免使用已训练的model_cpu参数
    model_gpu_fresh = Chain(
        Conv((3, 3), 1=>16, relu, init=glorot_uniform),
        MaxPool((2,2)),
        Conv((3, 3), 16=>32, relu, init=glorot_uniform),
        MaxPool((2,2)),
        Flux.flatten,
        Dense(5*5*32, 128, relu, init=glorot_uniform),
        Dense(128, 10, init=glorot_uniform)
    ) |> gpu # 创建并移至GPU

    loss_gpu_fresh(x,y) = Flux.logitcrossentropy(model_gpu_fresh(x),y)
    opt_gpu_fresh = ADAM(0.001)

    # GPU热身
    # train_epoch_gpu!(model_gpu_fresh, first(train_loader_cpu,1), opt_gpu_fresh, loss_gpu_fresh)
    gpu_training_time = @elapsed train_epoch_gpu!(model_gpu_fresh, train_loader_cpu, opt_gpu_fresh, loss_gpu_fresh)
    @printf "GPU单轮训练完成，耗时 %.2fs。\n" gpu_training_time

    # 快速检查GPU模型的准确率
    # 测试数据也需要放在GPU上进行评估
    X_test_gpu = gpu(X_test)
    Y_test_gpu = gpu(Y_test) # onecold支持CuArray进行比较
    acc_gpu = accuracy(X_test_gpu, Y_test_gpu, model_gpu_fresh) # accuracy函数需要处理GPU数据

    # 如果需要，可为GPU定制准确率函数，或确保onecold和模型输出可以比较
    # 如果model(X_test_gpu)返回CuArray，则提供的准确率函数应能正常工作
    # 并且onecold能处理CuArray（Flux最新版本支持此功能）。
    @printf "GPU模型单轮训练后的准确率: %.2f%%\n" acc_gpu*100

    # 将结果移回CPU（示例）
    # 如果您需要在CPU上检查预测结果：
    # sample_preds_gpu = model_gpu_fresh(X_test_gpu[:,:,:,1:5]) # 获取前5张测试图像的预测结果
    # sample_preds_cpu = cpu(sample_preds_gpu)
    # println("从GPU模型得到的样本预测结果（logits），已移至CPU：")
    # display(sample_preds_cpu)
end
```

主要改动是循环中的`x_batch_gpu = gpu(x_batch_cpu)`和`y_batch_gpu = gpu(y_batch_cpu)`。Flux的优化器和损失函数 (loss function)通常设计为可与`gpu()`生成的`CuArray`（CUDA数组）透明地协同工作。

### 观察速度提升

运行CPU和GPU训练脚本（请确保GPU部分的`CUDA.functional()`返回`true`）后，比较单轮训练的报告时间。



[交互图表：每轮训练的示意性时间（MNIST CNN）](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/practice-accelerating-training-gpus#plot-1mii5sv)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "CPU时间",
        "GPU时间"
      ],
      "y": [
        10.5,
        0.8
      ],
      "marker": {
        "color": [
          "#4dabf7",
          "#20c997"
        ],
        "line": {
          "color": "#1c7ed6",
          "width": 1.5
        }
      },
      "text": [
        10.5,
        0.8
      ],
      "textposition": "auto"
    }
  ],
  "layout": {
    "title": {
      "text": "每轮训练的示意性时间（MNIST CNN）"
    },
    "xaxis": {
      "title": {
        "text": "计算平台"
      }
    },
    "yaxis": {
      "title": {
        "text": "时间（秒）"
      },
      "range": [
        0,
        12
      ]
    },
    "font": {
      "family": "Arial, sans-serif"
    }
  }
}
```

</details>



> 针对MNIST任务和简单CNN的单轮训练示意性时间。GPU时间（绿色）明显低于CPU时间（蓝色）。实际的加速效果取决于具体的GPU、CPU、模型复杂度和批量大小。

您应该会看到GPU带来的明显加速，尤其是当模型复杂度和数据量增加时。对于非常小的模型或微小批次，数据传输到GPU的开销可能会抵消一部分收益，但对于大多数深度学习 (deep learning)任务，GPU加速效果显著。

### GPU训练的重要考量

使用GPU时，还需要注意以下几点：

- **数据传输开销**：在CPU内存和GPU显存 (VRAM)之间移动数据需要时间。因此，高效的做法是批量传输数据，并在将结果移回CPU之前，尽可能多地在GPU上完成计算。对于小到足以完全放入GPU内存的数据集，您可以在开始时一次性将其全部移动。
- **GPU内存管理**：GPU拥有自己的专用内存（显存），其容量通常比系统内存小。
  - 使用`CUDA.memory_status()`检查可用的GPU内存。
  - 如果您遇到`OutOfMemoryError`，常见的解决办法包括减小`batch_size`、简化模型或使用梯度累积等技巧（这是一个更高级的主题）。
  - 显式调用`gc()`或`CUDA.reclaim()`有时可以释放内存，但代码设计应尽量减少对它们的依赖。如果需要，`empty!(CUDA.pool)`是一种更强制性的方式来清除CUDA内存池。
- **异步操作**：GPU计算通常与CPU操作异步运行。`CUDA.synchronize()`可用于等待所有待处理的GPU任务完成。这对于对特定GPU内核进行精确的微基准测试至关重要。Flux的梯度计算以及通过`cpu()`进行的数据传输在需要结果时会隐式同步。
- **类型一致性**：确保您的模型和数据类型兼容（例如，`Float32`在深度学习 (deep learning)中很常见）。Flux和CUDA.jl在这方面表现不错，但类型不稳定性可能导致性能下降或错误。

### GPU工作流程故障排除

如果您遇到问题：

- **CUDA安装问题**：仔细检查您的CUDA工具包和NVIDIA驱动安装。确保`CUDA.functional()`返回`true`。
- **数组类型不匹配**：“无法将AbstractArray{Float32}转换为CuArray...”或反之的错误通常意味着某些数据（或模型的一部分）未正确移动到GPU（或CPU）。追踪您的数据流和`gpu`/`cpu`调用。
- **`CuArray`上不支持的操作**：虽然大多数Flux层和常用操作都支持`CuArray`，但自定义层或特定的Julia函数可能不支持。您可能需要寻找与GPU兼容的替代方案，或自行实现自定义CUDA内核（这是一个更高级的话题）。
- **内核错误**：这些错误源自GPU代码本身。它们可能难以理解，但通常指向诸如层输入维度不正确、内存越界访问或训练期间数值不稳定（例如`NaN`）等问题。

本实践环节为您提供了使用GPU加速Flux.jl模型训练的基本技能。随着您处理更大、更复杂的深度学习 (deep learning)问题，GPU计算将成为您工具箱中不可或缺的一部分。请尝试使用您自己的模型和数据集进行实践，以加深理解并亲身体验性能提升。

## 参考资料

- [Flux.jl Documentation: GPU Acceleration](https://fluxml.ai/Flux.jl/stable/gpu/) — The Flux.jl contributors (2025)
  Flux.jl与GPU结合使用的官方文档，详细说明了如何使用`gpu`函数将模型和数据移动到GPU，并管理GPU特定操作。
- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/) — The CUDA.jl contributors (2025)
  将NVIDIA CUDA功能集成到Julia程序中的官方指南，涵盖Julia生态系统中的`CuArray`、内存管理和直接GPU计算。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA并行计算平台和编程模型的全面指南，详细介绍了GPU架构、内存层次结构和并行处理的执行模型。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本广泛认可的教科书，提供了对深度学习算法和架构的透彻理解，并讨论了使GPU对训练至关重要的计算考虑因素。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2023)
  Publisher: Cambridge University Press
  一本开源的交互式书籍，提供深度学习概念的实践示例和讨论，包括利用GPU进行高效模型训练的最佳实践。

---

[上一节](08-Julia%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F%E7%9A%84%E9%83%A8%E7%BD%B2%E6%96%B9%E5%BC%8F.md)
