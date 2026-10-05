# GPU 加速：使用 CUDA.jl 和 Flux

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-5-advanced-topics-gpu-computing/gpu-acceleration-cudajl-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练深度学习 (deep learning)模型，特别是那些拥有大量数据集的大型模型，可能是一个耗时的过程。缩短此训练时间的一种非常有效的方法是，通过使用图形处理单元（GPU）的并行处理能力。提供了在 Julia 中通过 CUDA.jl 包使用 NVIDIA GPU 的详细信息，并展示了它如何与 Flux.jl 结合以加速深度学习工作流程。

如果您的机器配备了兼容的 NVIDIA GPU，您可以发挥其强大的计算能力。Julia 的 CUDA.jl 包为 NVIDIA CUDA 平台提供了一个全面的接口，允许直接进行 GPU 编程，对我们来说更重要的是，它使 Flux.jl 等深度学习框架能够在 GPU 上运行操作。

### 开始使用 CUDA.jl

在将 GPU 与 Flux 结合使用之前，您需要确保系统已正确设置并安装了 CUDA.jl。

1. **NVIDIA 驱动程序**：请确保您已为 GPU 安装最新的 NVIDIA 驱动程序。您通常可以从 NVIDIA 网站下载这些驱动程序。
2. **CUDA 工具包（可选但建议）**：尽管 CUDA.jl 通常可以下载其所需 CUDA 工具包组件的兼容版本（作为工件），但拥有一个系统范围的 CUDA 工具包安装有时会带来好处，特别是在开发时或如果您计划使用其他启用 CUDA 的应用程序。
3. **安装 CUDA.jl**：像其他任何 Julia 包一样，您可以使用 Julia 包管理器添加 CUDA.jl：

   ```julia
   using Pkg
   Pkg.add("CUDA")
   ```
4. **验证 GPU 功能**：安装后，最好检查 Julia 是否可以识别和使用您的 GPU。

   ```julia
   using CUDA

   if CUDA.functional()
       println("CUDA 正常工作，GPU 可用！")
       CUDA.versioninfo() # 打印有关 CUDA 工具包和驱动程序的信息
       # 您也可以查询设备属性
       # 例如，查看可用设备：
       # for (i, dev) in enumerate(CUDA.devices())
       #     println("$i: $(CUDA.name(dev))")
       # end
   else
       println("CUDA 功能不正常。请确保驱动程序和工具包已正确设置。")
   end
   ```

   如果 `CUDA.functional()` 返回 `true`，则可以继续。如果不是，您可能需要排查 NVIDIA 驱动程序或 CUDA 工具包的安装问题。CUDA.jl 的错误消息通常会提供线索。

### 将 Flux 模型和数据移至 GPU

Flux.jl 在设计时就考虑了 GPU 计算，使得从 CPU 到 GPU 的转换非常顺畅。将模型或数据移至 GPU 的主要机制是 Flux 提供的 `gpu` 函数（它通常在底层使用 CUDA.jl 的 `cu` 函数来处理 NVIDIA GPU）。相反地，`cpu` 函数将它们移回 CPU。

**移动模型：**
要将整个 Flux 模型移至 GPU，您只需对其应用 `gpu` 函数：

```julia
using Flux

# 定义一个简单模型
model_cpu = Chain(
    Dense(10, 20, relu),
    Dense(20, 5, relu),
    Dense(5, 1)
)

# 检查 GPU 是否可用并移动模型
if CUDA.functional()
    model_gpu = model_cpu |> gpu
    println("模型已移至 GPU。")
else
    model_gpu = model_cpu # 回退到 CPU
    println("GPU 不可用，模型保留在 CPU 上。")
end
```

当您将 `gpu` 应用于 `Chain` 或任何自定义 Flux 模型结构时，Flux 会智能地遍历模型的层和参数 (parameter)，将它们转换为与 GPU 兼容的结构（例如，权重 (weight)和偏置 (bias)的 `CuArray`）。

**移动数据：**
同样，您的输入数据和目标标签需要位于 GPU 上，模型才能在那里处理它们。数据通常存储在 CPU 上的标准 Julia `Array` 中。要将它们移至 GPU，您也使用 `gpu` 函数（或直接使用 `CUDA.cu`）：

```julia
# 示例 CPU 数据
x_cpu = rand(Float32, 10, 128) # 10 个特征，128 个样本
y_cpu = rand(Float32, 1, 128)

# 移动数据到 GPU
if CUDA.functional()
    x_gpu = x_cpu |> gpu # 或 CUDA.cu(x_cpu)
    y_gpu = y_cpu |> gpu # 或 CUDA.cu(y_cpu)
    println("数据已移至 GPU。")

    # 检查类型
    # println(typeof(x_gpu)) # 应显示 CuArray{Float32, 2}
else
    x_gpu = x_cpu
    y_gpu = y_cpu
    println("GPU 不可用，数据保留在 CPU 上。")
end
```

移至 GPU 的数据通常表示为 CUDA.jl 的 `CuArray` 对象。这些数组驻留在 GPU 的内存中。

### 调整您的训练循环

一旦您的模型和数据可以移至 GPU，调整训练循环将涉及一些重要更改：

1. 在训练循环开始前，将模型**一次性移至 GPU**。
2. 在训练循环内部，将数据批次**移至 GPU**，就在将它们馈送给模型之前。这很重要，因为通常您无法将整个数据集一次性放入 GPU 内存中。
3. **确保损失计算和梯度**也在 GPU 上执行。如果模型和输入数据在 GPU 上，Flux 和 Zygote.jl 会自动处理此问题。
4. 如果需要用于日志记录、打印或存储，请将**损失值移回 CPU**，因为过度进行 GPU 到 CPU 的数据传输可能会很慢。

让我们看一个简化的训练循环结构：

```julia
using Flux, CUDA, Optimisers
using MLUtils: DataLoader # 用于批处理

# 0. 定义模型和数据（如前所示）
features = 10
outputs = 1
n_samples = 1000

model_cpu = Chain(Dense(features => 64, relu), Dense(64 => outputs))
X_train_cpu = rand(Float32, features, n_samples)
Y_train_cpu = rand(Float32, outputs, n_samples)

# 1. 如果可用，则为 GPU 设置
use_gpu = CUDA.functional()
if use_gpu
    model = model_cpu |> gpu
    println("在 GPU 上训练。")
else
    model = model_cpu
    println("在 CPU 上训练。")
end

# 优化器
opt_state = Optimisers.setup(Optimisers.Adam(1e-3), model)

# 损失函数
loss(m, x, y) = Flux.mse(m(x), y)

# 用于批处理的数据加载器
batch_size = 64
train_loader = DataLoader((X_train_cpu, Y_train_cpu), batchsize=batch_size, shuffle=true)

# 2. 训练循环
epochs = 10
for epoch in 1:epochs
    epoch_loss = 0.0
    num_batches = 0
    for (x_batch_cpu, y_batch_cpu) in train_loader
        # 将当前批次移至 GPU
        x_batch = use_gpu ? (x_batch_cpu |> gpu) : x_batch_cpu
        y_batch = use_gpu ? (y_batch_cpu |> gpu) : y_batch_cpu

        # 计算损失和梯度
        val, grads = Flux.withgradient(model) do m
            loss(m, x_batch, y_batch)
        end

        # 更新模型参数
        Optimisers.update!(opt_state, model, grads[1])

        # 累加损失（如果在 GPU 上，则移至 CPU 进行聚合）
        epoch_loss += use_gpu ? cpu(val) : val # 确保 'val' 在 CPU 上进行累加
        num_batches += 1
    end
    avg_loss = epoch_loss / num_batches
    println("周期: $epoch, 平均损失: $avg_loss")
end

# 训练结束后，如果您需要 CPU 上的模型：
# model_final_cpu = model |> cpu
```

在此循环中：

- 模型一次性移至 GPU。
- 来自 `DataLoader` 的每个小批次 `(x_batch_cpu, y_batch_cpu)`（产生 CPU 数组）在传递给模型之前，都使用 `x_batch_cpu |> gpu` 和 `y_batch_cpu |> gpu` 明确地移至 GPU。
- 损失 `val` 在 GPU 上计算。为了聚合或打印，最好使用 `cpu(val)` 将其移回 CPU。Flux 和 Zygote 将确保梯度也在 GPU 上计算。

### GPU 计算的重要考虑事项

虽然使用 GPU 可以大幅提升速度，但为了高效利用 GPU，有几个因素需要记住：

- **数据传输开销**：与 GPU 上的计算相比，CPU 和 GPU 之间（RAM 和 GPU 显存 (VRAM)）的数据传输相对较慢。尽量减少这些传输。
  - 在 GPU 上处理合理大小的批次数据。
  - 避免频繁地来回移动小块数据。
- **GPU 内存管理**：GPU 有其专用的内存（显存），通常比系统 RAM 更有限。
  - 确保您的模型参数 (parameter)和数据批次适合显存。`CuArray` 占用 GPU 内存。
  - 您可以使用 `CUDA.memory_status()` 查看 GPU 内存状态。
  - 如果您遇到“内存不足”错误，请尝试减少批次大小、模型大小或数据精度（例如，如果合适，使用 `Float32` 而不是 `Float64`）。
- **内核启动开销**：每次分派到 GPU 的操作都会产生少量开销。对于非常小的计算，此开销可能会抵消 GPU 并行的好处。GPU 在对大量数据执行许多并行操作时表现出色。
- **模型和操作兼容性**：大多数标准 Flux 层和操作都开箱即用，与 GPU 兼容。如果您定义自定义层或使用不感知 GPU 的函数，您可能需要调整它们以与 `CuArray` 配合使用，或者实现自定义 CUDA 内核（这是一个更高级的主题）。
- **浮点精度**：GPU，特别是消费级 GPU，通常在单精度浮点数（`Float32`）方面比双精度（`Float64`）具有明显更好的性能。大多数深度学习 (deep learning)模型使用 `Float32` 也能很好地训练，因此这是常见的选择。在面向 GPU 时，请确保您的数据和模型参数为 `Float32`。Flux 层通常默认为 `Float32` 或适应输入数据类型。

> 使用 Flux.jl 进行 GPU 加速时的数据流和模型放置。重要步骤包括将模型一次性移至 GPU，然后在训练循环中将数据批次传输到 GPU。

通过理解这些原则，您可以有效地修改您的 Flux.jl 训练脚本以使用 GPU，减少复杂模型的训练时间，并使您能够更快地迭代深度学习项目。接下来的章节将介绍如何分析您的模型以找到瓶颈和进一步的优化技巧。

## 参考资料

- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/) — Tim Besard, Valentin Churavy, Mike Innes, Katharine Hyatt, Simon Danisch (2025)
  CUDA.jl包的官方文档，提供安装、验证GPU功能以及在Julia中编程NVIDIA GPU的说明。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — Flux.jl developers (2025)
  Flux.jl的官方文档，详细说明了如何定义模型、将模型及其相关数据移动到GPU，以及如何调整训练循环以适应GPU加速的工作流程。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  NVIDIA CUDA平台的主要编程指南，概述了GPU架构、CUDA编程模型以及高性能GPU计算的实践。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola (2024)
  Publisher: Cambridge University Press
  一本交互式的开放获取书籍，提供了关于深度学习的广泛教学，包括计算性能和GPU在加速模型训练中的作用的信息。

---

[上一节](../04-%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B/09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%B0%83%E4%BC%98.md) · [下一节](02-%E5%9C%A8GPU%E4%B8%8A%E7%AE%A1%E7%90%86%E6%95%B0%E6%8D%AE.md)
