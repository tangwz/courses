# 在GPU上管理数据

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/managing-data-gpu-julia)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当您的Flux.jl模型使用`model |> gpu`移至GPU后，下一个重要步骤是确保其处理的数据也位于GPU上。涉及CPU和GPU内存之间数据分离的计算要么无法进行，要么效率非常低。因此，高效的数据管理是实现高性能GPU加速深度学习 (deep learning)的根本所在。

现代GPU拥有其专用的高速内存，通常称为VRAM（视频随机存取存储器）。这种内存与CPU使用的计算机主RAM（随机存取存储器）是独立的。要在GPU上执行计算，数据必须从CPU RAM显式传输到GPU VRAM。同样，如果需要在CPU上使用GPU计算的结果（例如，将它们保存到文件或绘图），数据也必须传回。这些传输虽然必要，但会引入额外开销，因此高效地管理它们非常重要。

### 介绍 CuArray：数据进入GPU的凭证

在Julia生态系统中，`CUDA.jl`提供了`CuArray`类型。这类似于Julia的标准`Array`，但它表示一个数据存储在GPU内存中的数组。您对标准`Array`执行的大多数操作也可以在`CuArray`上执行，但它们将在GPU上运行，充分利用其并行处理能力。

### 使用 `cu()` 将数据移至 GPU

将数据从CPU传输到GPU的主要函数是`cu()`。它接受一个标准Julia数组（或兼容的数据类型，如数字），并返回其`CuArray`版本。

让我们看看实际操作：

```julia
using CUDA

# 确保 CUDA 可用
println("CUDA 功能正常: ", CUDA.functional())

# 在 CPU 上创建一个标准 Julia 数组
cpu_vector = rand(Float32, 5)
println("原始 CPU 向量: ", cpu_vector)
println("cpu_vector 的类型: ", typeof(cpu_vector))

# 将向量移至 GPU
gpu_vector = cu(cpu_vector)
println("GPU 向量: ", gpu_vector) # 打印可能只显示一个占位符
println("gpu_vector 的类型: ", typeof(gpu_vector))

# 也可以移动单个数字或矩阵
cpu_scalar = Float32(10.5)
gpu_scalar = cu(cpu_scalar)
println("gpu_scalar 的类型: ", typeof(gpu_scalar))

cpu_matrix = rand(Float32, 2, 3)
gpu_matrix = cu(cpu_matrix)
println("gpu_matrix 的类型: ", typeof(gpu_matrix))
```

当您打印`gpu_vector`时，`CUDA.jl`通常显示一个概要而非所有元素，因为直接访问以供打印需要将数据缓慢地传回CPU。重要部分是它的类型，对于`Float32`的1D向量 (vector)，类型将类似于`CuArray{Float32, 1}`。

对于深度学习 (deep learning)，通常会将数据和模型参数 (parameter)使用`Float32`。这种精度通常足以训练模型，并且与`Float64`相比，可以明显节省内存并通常在GPU上实现更快的计算。因此，如果您的输入数据尚未是`Float32`类型，请在将其移至GPU之前务必进行转换。

```julia
# 示例：移动前确保为 Float32
cpu_vector_f64 = rand(5) # 默认为 Float64
gpu_vector_f32 = cu(Float32.(cpu_vector_f64))
println("转换后的 GPU 向量类型: ", typeof(gpu_vector_f32))
```

### 从 GPU 取回数据

有时您需要将数据从GPU传回CPU。例如：

- 为了调试，检查特定值。
- 使用CPU特定的库进行绘图或分析。
- 将结果保存到磁盘。

您可以使用`Array()`构造函数或`Flux.jl`提供的`cpu()`函数（通常封装了`CUDA.jl`的功能以方便使用）将`CuArray`传回CPU上的标准`Array`。

```julia
using CUDA
using Flux # 为 cpu() 便捷函数

# 假设 gpu_vector 是之前操作生成的 CuArray
gpu_vector = cu(rand(Float32, 3))
println("传输前 gpu_vector 的类型: ", typeof(gpu_vector))

# 方法 1: 使用 Array()
cpu_vector_from_gpu_A = Array(gpu_vector)
println("Array() 后类型: ", typeof(cpu_vector_from_gpu_A))
println("值: ", cpu_vector_from_gpu_A)

# 方法 2: 使用 Flux.cpu()
# 如果您顺序运行，请确保 gpu_vector 仍然是 CuArray
gpu_vector_B = cu(rand(Float32, 3))
cpu_vector_from_gpu_B = cpu(gpu_vector_B)
println("cpu() 后类型: ", typeof(cpu_vector_from_gpu_B))
println("值: ", cpu_vector_from_gpu_B)
```

这两种方法都能达到相同的结果。请选择适合您编码风格或当前工作环境的方法。

> 此图展示了使用`cu()`将数组数据移至GPU，以及使用`Array()`或`cpu()`将其传回CPU，实现CPU RAM和GPU VRAM之间的数据传输。

### 训练循环中的数据处理

训练神经网络 (neural network)时，通常以小批量方式处理数据。如果您的模型在GPU上，那么在将每个小批量数据输入模型之前，也必须将其移至GPU。

考虑使用数据加载器（如`MLUtils.jl`中的加载器）的典型训练循环结构：

```julia
using Flux
using CUDA
using MLUtils # 用于 DataLoader
using Optimisers # 用于 Optimisers.setup 和 Optimisers.update!

# 0. 确保 CUDA 可用并功能正常
if !CUDA.functional()
    @warn "CUDA 功能不正常。将在 CPU 上进行训练。"
    # 如果没有 GPU，则回退到 CPU
    global gpu = identity 
    # 或者，如果严格需要 GPU，则抛出错误
    # error("需要 CUDA GPU 但不可用。")
el
    # 为方便起见，定义 gpu 转换函数
    global gpu = Flux.gpu # 对于 Flux 模型/数据，与 x -> cu(x) 相同
end

# 1. 示例数据 (CPU)
X_train_cpu = rand(Float32, 784, 1000) # 1000 个样本，784 个特征
Y_train_cpu = Flux.onehotbatch(rand(0:9, 1000), 0:9) # 1000 个标签，10 个类别

# 2. 创建一个 DataLoader (在 CPU 数据上迭代)
batch_size = 64
train_loader = DataLoader((X_train_cpu, Y_train_cpu), batchsize=batch_size, shuffle=true)

# 3. 定义一个简单模型并将其移至 GPU
model = Chain(
    Dense(784, 128, relu),
    Dense(128, 10)
) |> gpu # 将模型移至 GPU

# 4. 定义损失函数和优化器
loss(x, y) = Flux.logitcrossentropy(model(x), y)
opt_state = Optimisers.setup(Optimisers.Adam(0.001), model)

# 5. 训练循环
epochs = 5
for epoch in 1:epochs
    total_loss = 0.0
    num_batches = 0
    for (x_batch_cpu, y_batch_cpu) in train_loader
        # 重要提示: 将当前批次移至 GPU
        x_batch_gpu = x_batch_cpu |> gpu
        y_batch_gpu = y_batch_cpu |> gpu

        # 在 GPU 上计算梯度
        grads = gradient(model, x_batch_gpu, y_batch_gpu) do m, x, y
            loss(x, y)
        end

        # 更新模型参数 (也在 GPU 上)
        Optimisers.update!(opt_state, model, grads[1])

        total_loss += loss(x_batch_gpu, y_batch_gpu) # 损失是一个标量，如果需要，通常会隐式或显式地将其带到 CPU
        num_batches += 1
    end
    avg_loss = total_loss / num_batches
    println("轮次: $epoch, 平均损失: $avg_loss")
end

# 要在 CPU 上获取预测或评估，请适当移动数据和模型
# 例如，在新 CPU 数据点上进行预测:
# new_sample_cpu = rand(Float32, 784, 1)
# model_cpu = model |> cpu # 将模型移至 CPU
# prediction = model_cpu(new_sample_cpu)

# 或者，在 GPU 上使用 GPU 数据进行预测:
# new_sample_gpu = cu(rand(Float32, 784, 1))
# prediction_gpu_output = model(new_sample_gpu) # 模型已在 GPU 上
# prediction_cpu_readable = prediction_gpu_output |> cpu # 将结果带到 CPU
```

在此循环中，`x_batch_cpu`和`y_batch_cpu`是您的数据集仍在CPU内存中的切片。`x_batch_gpu = x_batch_cpu |> gpu`和`y_batch_gpu = y_batch_cpu |> gpu`这两行代码非常必要。它们在当前小批量数据用于前向传播和梯度计算之前，将其传输到GPU。这里的`gpu`函数是`Flux.jl`提供的一个便捷功能，对于数组来说，它会调用`cu()`。

请注意，损失值本身通常是一个标量。当计算`loss(x_batch_gpu, y_batch_gpu)`时，如果损失函数 (loss function)涉及将结果传回CPU的操作（某些归约操作可能会），则这种传输通常影响很小。但是，如果您明确需要在CPU上获取损失值（例如，用于记录日志），则可以使用`loss_value = cpu(loss(x_batch_gpu, y_batch_gpu))`。

### 关注 GPU 内存

GPU的VRAM数量有限，通常少于系统的主RAM。了解您的数据和模型消耗了多少GPU内存非常重要。`CUDA.jl`提供了`CUDA.memory_status()`来查看内存使用情况：

```julia
using CUDA

if CUDA.functional()
    # 将一些数据和模型移至 GPU 后
    # model = ... |> gpu
    # data = ... |> gpu

    CUDA.memory_status() # 打印 GPU 的总内存、空闲内存、已用内存
else
    println("CUDA 不可用，无法检查内存状态。")
end
```

这对于调试`OutOfMemoryError`问题非常有帮助。如果GPU内存不足，常见的方法包括：

- 减小批次大小。
- 使用更小的模型结构。
- 确保任何不再需要的`CuArray`由Julia的垃圾收集器进行收集。Julia的垃圾收集器(GC)通常能很好地处理`CuArray`。当`CuArray`超出作用域并被GC收集时，相应的GPU内存会被归还。
- 在少数更高级的情况下，您可能需要使用`CUDA.unsafe_free!(cu_array)`手动归还GPU内存，但使用此方法务必极其谨慎，因为它可能导致内存仍在使用或由其他地方管理时程序崩溃。通常最好让Julia的GC管理内存。

### 数据传输的性能考量

高效管理数据传输对于最大限度地发挥GPU性能至关重要。以下是一些指导原则：

1. **减少传输频率**：每次CPU到GPU的传输都会有延迟。如果可能，避免在循环内反复来回传输数据。数据在GPU计算中被主动使用时，应尽可能长时间地保留在GPU上。
2. **批量传输**：以更大的、连续的块（如小批量）传输数据比传输许多小的、单个元素更有效率。这就是`DataLoader`模式表现良好的原因。
3. **数据类型**：如前所述，在GPU上进行深度学习 (deep learning)时，通常首选`Float32`而非`Float64`。它能将数据的内存占用和带宽需求减半，通常能加快处理速度，且不会造成模型精度的明显降低。
4. **固定内存（进阶内容）**：对于要求高性能的情况，`CUDA.jl`支持固定主机内存，这可以加速传输。这是一个较深层的话题，通常除非数据传输已被证实是明显的瓶颈，否则不需要它。
5. **异步传输（进阶内容）**：`CUDA.jl`允许异步数据传输（`CUDA.CuStream`），这可以将数据移动与计算重叠进行。这可以掩盖一部分传输延迟，但会增加代码的复杂性。

通过了解如何将数据移入和移出GPU，并通过构建训练循环以高效处理批次传输，您可以高效使用GPU加速来完成您的Julia深度学习项目。请记住，目的是让GPU持续进行计算，按需为其提供数据，并最大程度地减少因等待数据传输而导致的空闲时间。

## 参考资料

- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/) — Tim Besard, Valentin Churavy, Mike Innes, Katharine Hyatt, Simon Danisch (2025)
  提供使用Julia与NVIDIA GPU的详细信息，包括CuArray和内存管理。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl contributors (2025)
  描述了Flux.jl在深度学习生态系统中用于GPU加速和数据移动的功能。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  一本解释CUDA编程模型、GPU内存结构和数据传输原理的指南。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2023)
  Publisher: Cambridge University Press
  提供深度学习实现的实践指导，涵盖数据加载、训练循环和GPU使用等方面的有效计算。

---

[上一节](01-GPU%20%E5%8A%A0%E9%80%9F%EF%BC%9A%E4%BD%BF%E7%94%A8%20CUDA.jl%20%E5%92%8C%20Flux.md) · [下一节](03-Flux%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96.md)
