---
course: "pytorch-for-tensorflow-developers"
chapter: "advanced-pytorch-features-tf-users"
lesson: "profiling-pytorch-code"
sourceId: 5777
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-6-advanced-pytorch-features-tf-users/profiling-pytorch-code"
title: "剖析 PyTorch 代码以查找性能瓶颈"
description: "使用 PyTorch 内置的性能分析器，找出模型和数据管道中的性能瓶颈并进行处理。"
order: 4
plots: ["plots/5777-0.json"]
sourceHash: "867711b3a8ba823821340068083bdd06804cc8a34c23b496b286138e3eb6f1a2"
sourceCorrections: []
---

随着您使用更复杂的模型结构和更大的数据集，优化性能变得日益重要。缓慢的训练迭代或低效的推断会严重阻碍开发进度并增加计算成本。如果您使用过 TensorFlow，可能熟悉用于找出此类性能问题的 TensorFlow Profiler。PyTorch 提供了其内置的强大性能分析器 `torch.profiler`，旨在帮助您了解 PyTorch 操作的时间和内存消耗。

本节将指导您使用 `torch.profiler` 准确定位 PyTorch 代码中的性能瓶颈，从而让您的模型更快、内存效率更高。

## 为何要对 PyTorch 代码进行性能分析？

在讨论具体方法之前，我们先思考一下为什么性能分析是一项有价值的做法：

1. **找出瓶颈**: 性能分析的主要目标是找到代码中消耗时间或内存最多的部分。是数据加载吗？网络中的某个特定层？还是 GPU-CPU 同步？性能分析器提供数据来回答这些问题。
2. **优化资源使用**: 性能分析帮助您了解代码如何有效使用 CPU、GPU 和内存等可用资源。您可以判断 GPU 是否利用不足，是否存在过多的内存分配，或者数据传输是否拖慢性能。
3. **加速训练和推断**: 通过处理已发现的瓶颈，您可以大幅加速模型的训练过程及其部署时的推断时间。
4. **做出明智决策**: 性能分析器的输出为您提供具体数据，以指导您的优化工作。您无需猜测瓶颈所在，而是可以进行有针对性的改进。

## 介绍 torch.profiler

PyTorch 的 `torch.profiler` 模块是收集性能指标的标准工具。它可以在 CPU 和 CUDA (GPU) 设备上追踪事件，跟踪内存分配，并将操作与其源代码关联起来。它的设计开销相对较低，尤其是在分析短代码段时。

性能分析器通过记录代码执行期间发生的各种“事件”信息来工作。这些事件包括：

- 操作符执行（例如，卷积、矩阵乘法）。
- 内存分配和释放。
- GPU 上的核函数启动。
- 同步原语。

## 使用 `torch.profiler.profile` 进行基本性能分析

使用性能分析器最直接的方法是配合 `torch.profiler.profile` 上下文 (context)管理器。将您想要分析的代码段包裹在此上下文中即可。

```python
import torch
import torchvision.models as models
import torch.profiler

# 示例模型和输入
model = models.resnet18().cuda()
inputs = torch.randn(16, 3, 224, 224).cuda()

# 热身迭代（对准确的 GPU 性能分析很重要）
for _ in range(5):
    model(inputs)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    record_shapes=True,      # 记录操作符的输入形状
    profile_memory=True,     # 启用内存性能分析
    with_stack=True          # 记录调用堆栈
) as prof:
    with torch.profiler.record_function("model_inference"): # 可选的自定义标签
        for _ in range(10): # 分析几次迭代
            model(inputs)

# 打印聚合统计信息
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出追踪数据以供 Chrome Trace Viewer 或 TensorBoard 使用
prof.export_chrome_trace("resnet18_trace.json")
# 对于 TensorBoard，您通常会使用一个处理器，例如：
# prof.export_to_tensorboard("tb_logs/resnet18_profile") # 如果使用 tensorboard_trace_handler
```

我们来细致讲解 `torch.profiler.profile` 的重要参数 (parameter)：

- `activities`: 一个列表，用于指定要分析的活动。常用选项是 `ProfilerActivity.CPU` 和 `ProfilerActivity.CUDA`。
- `schedule`: （上面未展示）可用于更精细的控制，例如使用 `torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)` 仅分析循环的某些迭代。这会分析迭代 3-5 和 9-11（等待 1 次，热身 1 次，活动 3 次，然后重复此模式）。
- `on_trace_ready`: （上面未展示）一个可调用对象，用于处理追踪数据。对于自定义追踪处理很有用，例如用于直接 TensorBoard 输出的 `torch.profiler.tensorboard_trace_handler`。
- `record_shapes`: 如果为 `True`，性能分析器会记录被分析操作符的输入形状。这对于了解不同输入大小是否影响性能非常有帮助。
- `profile_memory`: 如果为 `True`，启用内存性能分析，跟踪 CPU 和 GPU 上的分配和释放。
- `with_stack`: 如果为 `True`，性能分析器会记录被分析操作的 Python 调用堆栈。这有助于将性能数据映射回您的源代码，但可能会增加一些开销。
- `with_flops`: （实验性）如果为 `True`，则估计相关操作符的 FLOPS（每秒浮点运算次数）。
- `with_modules`: 如果为 `True`，性能分析器会尝试将操作符调用归因于模型中的特定 `torch.nn.Module` 实例。

`torch.profiler.record_function("label_name")` 上下文管理器允许您在分析区域内向某些代码块添加自定义标签，使追踪数据更易于理解。

## 分析性能分析器输出

性能分析器运行后，您可以通过多种方式分析收集到的数据。

### 1. 使用 `key_averages()` 获取聚合统计信息

`prof.key_averages()` 方法返回一个对象，允许您查看聚合统计信息。在此对象上调用 `.table()` 会打印一份人类可读的摘要。

```
CPU 时间：
-------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
名称                                   自身 CPU 总百分比   自身 CPU 总时间   CPU 总百分比      CPU 总时间        CPU 平均时间     调用次数         CUDA 总时间      自身 CUDA 总时间   CUDA 平均时间    输入形状
-------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
model_inference                        2.62%            6.658ms          100.00%          254.414ms        25.441ms         10               230.184ms        0.000us          23.018ms         []
aten::convolution                      0.01%            30.000us         0.01%            30.000us         30.000us         1                30.000us         30.000us         30.000us         [[16, 3, 224, 224], [64, 3, 7, 7], [64], [2, 2], [3, 3], [1, 1], False, [0, 0], 1]
...（更多行）
-------------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
自身 CPU 总时间：254.414ms
CUDA 总时间：230.184ms
```

列包括：

- **名称**: 操作或 `record_function` 标签的名称。
- **自身 CPU/CUDA 总时间**: 在操作符本身中花费的时间，不包括子操作符中的时间。
- **CPU/CUDA 总时间**: 在操作符及其所有子项中花费的时间。
- **CPU/CUDA 平均时间**: 每次调用的平均时间。
- **调用次数**: 此操作符被调用的次数。
- **输入形状**: 如果 `record_shapes=True`，显示输入张量的形状。

您可以对该表进行排序（例如，`sort_by="cuda_time_total"`、`sort_by="cpu_time_total"`）并限制行数（`row_limit`），以便着重查看最耗时的操作。您也可以以不同方式对结果进行分组，例如 `prof.key_averages(group_by_input_shape=True)` 或 `prof.key_averages(group_by_stack_n=5)`。

### 2. Chrome 追踪查看器 (`.export_chrome_trace()`)

`prof.export_chrome_trace("filename.json")` 方法将追踪数据保存为 JSON 格式，该格式可以加载到 Chrome 追踪查看器中（打开 Chrome，导航到 `chrome://tracing`，然后加载文件）。

这提供了详细的时间线可视化：

- **CPU 操作符**: 显示 CPU 侧操作、Python 函数调用和用户定义的 `record_function` 块。
- **GPU 核函数**: 对于 CUDA 活动，它显示在 GPU 上运行的核函数、它们的持续时间以及它们的名称（例如，Volta GPU 上矩阵乘法的 `volta_sgemm_...`）。
- **关联**: 您通常可以看到 CPU 操作如何启动 GPU 核函数。
- **内存事件**: 如果 `profile_memory=True`，您将看到内存分配/释放事件。

追踪查看器对于了解操作序列、找出 GPU 上的空闲时间以及发现意外长时间运行的核函数是极有价值的。

下面是一个简化的 Chrome 追踪片段示例，显示了几项操作。



![Interactive chart](plots/5777-0.json)



> 性能分析器输出的简化视图，显示 CPU 操作启动相应的 GPU 核函数。“model\_inference”是一个用户定义的块。

### 3. TensorBoard 集成

为了获得更集成化的体验，特别是如果您已经将 TensorBoard 用于其他日志记录，您可以使用 `torch.profiler.tensorboard_trace_handler`。

```python
# ...（模型和输入设置）
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

# 确保日志目录存在
log_dir = "tb_logs/my_model_profile"
import os
os.makedirs(log_dir, exist_ok=True)

# 使用 schedule 进行有针对性的性能分析
my_schedule = schedule(wait=1, warmup=1, active=2, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    on_trace_ready=tensorboard_trace_handler(log_dir),
    record_shapes=True,
    with_stack=True
) as prof_tb:
    for step in range(10): # 模拟训练步骤
        model(inputs)
        prof_tb.step() # 重要：通知性能分析器一个步骤已完成

# 运行后，启动 TensorBoard：tensorboard --logdir tb_logs
```

然后，启动 TensorBoard (`tensorboard --logdir tb_logs`) 并导航到“PyTorch Profiler”选项卡。TensorBoard 提供概览页、操作符视图、核函数视图以及类似于 Chrome 的追踪视图，通常包含更多 PyTorch 相关的详细信息和更轻松的导航。

## 常见性能瓶颈及其识别

以下是一些您可能遇到的常见性能问题以及性能分析器如何帮助识别它们：

1. **数据加载瓶颈**：

   - **症状**: GPU 通常处于空闲状态，而 CPU 繁忙。性能分析器显示大量时间花费在数据加载器 worker、`next(iter(dataloader))` 或数据转换函数中。
   - **识别方法**: 在追踪视图中，查找在 GPU 活动恢复之前发生的数据加载相关的长 CPU 块。`key_averages()` 可能会显示数据相关函数消耗了大量 CPU 时间。
   - **解决方案**: 增加 `DataLoader` 中的 `num_workers`，使用 `pin_memory=True`，优化自定义的 `Dataset.__getitem__` 方法，或者在可行的情况下在 GPU 上执行转换。
2. **CPU-GPU 同步开销**：

   - **症状**: 诸如 `tensor.item()`、`tensor.cpu()` 或显式 `torch.cuda.synchronize()` 等操作可能导致 CPU 等待 GPU，从而停滞执行。
   - **识别方法**: 性能分析器追踪显示 CPU 活动中存在间隙，随后是这些同步调用，或者这些调用本身显示大量“自身”时间。
   - **解决方案**: 最小化显式同步。在 GPU 上累积损失，并减少向 CPU 传输的频率。避免设备间不必要的数据传输。
3. **过多的小型 GPU 核函数**：

   - **症状**: GPU 看起来很忙，但整体吞吐量 (throughput)很低。启动许多小型核函数的开销可能超过计算时间。
   - **识别方法**: Chrome/TensorBoard 追踪显示 GPU 时间线上有大量非常短的条形。
   - **解决方案**: 尝试向量 (vector)化操作（例如，使用批量矩阵乘法代替单个矩阵乘法的循环）。查找合并操作符（例如，`torch.nn.Fused আমাকেAdamW`），或者考虑对模型部分使用 `torch.jit.script` 进行 JIT 编译，这可以合并操作。
4. **低效的模型操作或层**：

   - **症状**: 模型的某个部分（例如，自定义层、一系列标准层）耗费了不成比例的时间。
   - **识别方法**: 按 `cuda_time_total` 或 `cpu_time_total` 排序的 `key_averages()` 将显示耗时的操作符。如果使用了 `with_modules=True`，您有时可以看到是哪个 `nn.Module` 造成的问题。追踪视图可以显示哪个层的前向传播很慢。
   - **解决方案**: 重新评估层的实现。是否可以使用更高效的算法？是否有内置的 PyTorch 操作可以以更好的性能实现相同功能？
5. **内存瓶颈**：

   - **症状**: `OutOfMemoryError` (OOM)，或高内存搅动（频繁分配/释放）导致执行速度变慢。
   - **识别方法**: 使用 `profile_memory=True`。性能分析器输出（特别是在 TensorBoard 中或通过 `prof.export_memory_timeline()`）将显示随时间变化的内存使用情况，并突出显示大额分配。如果按内存指标排序，`key_averages()` 也会显示内存使用情况。
   - **解决方案**: 减小批处理大小，使用梯度累积，应用混合精度训练（AMP，在另一节中介绍），使用 `torch.utils.checkpoint` 进行梯度检查点，或优化模型结构以节省内存。使用 `del tensor_name` 删除不再需要的张量，并调用 `torch.cuda.empty_cache()`（尽管后者应谨慎使用，因为它可能导致同步）。

## 与 TensorFlow Profiler 的比较

如果您使用过 TensorFlow 中的 `tf.profiler`，您会发现 `torch.profiler` 的目标和一般工作流程非常相似：

- **数据收集**: 两者都涉及在某个代码区域周围启用性能分析器。
- **分析工具**: 两者都提供查看聚合统计信息和详细时间线追踪的方法（TensorFlow Profiler 与 TensorBoard 紧密结合，很像 PyTorch 的 `tensorboard_trace_handler`）。
- **关注点**: 两者都旨在找出 CPU/GPU 瓶颈、内存问题和低效操作。

主要的区别在于具体的 API 和底层机制，这反映了 TensorFlow 基于图的执行（尤其是在 TF1.x 或使用 `tf.function` 时）与 PyTorch 更即时、按运行定义本质之间的差异。PyTorch 的性能分析器非常适合其动态环境，允许对任意代码块进行灵活的性能分析。

## 高效性能分析的实用建议

- **热身**: 在启动性能分析器之前，务必运行几次代码的“热身”迭代，特别是对于 GPU 性能分析。最初的几次迭代通常涉及一次性成本（如 CUDA 上下文 (context)创建、核函数编译缓存），这些成本您不希望包含在常规性能测量中。
- **分析代表性负载**: 分析代码中能代表您实际关注负载的某个部分。对于训练而言，这通常意味着训练循环的几次迭代，包括数据加载、前向传播、反向传播 (backpropagation)和优化器步骤。
- **最小化性能分析器开销**: 尽管 `torch.profiler` 旨在高效，但它仍然会增加一些开销。如果您只需要某个部分的精细细节，请避免一次性分析过长的执行时间。如果需要，可以使用 `schedule` 参数 (parameter)实现更复杂的性能分析模式。对于对性能非常敏感的内部循环，请考虑 `with_stack=True` 和 `record_shapes=True` 比仅进行基本活动分析会增加更多开销。
- **在目标硬件上分析**: 不同 CPU 和 GPU 之间的性能特征可能明显不同。务必在反映您的部署或训练环境的硬件上进行性能分析。
- **迭代**: 性能分析是一个迭代过程。找出瓶颈，应用优化，然后再次进行性能分析以确认改进并查看是否出现了新的瓶颈。
- **使用标签**: 使用 `torch.profiler.record_function("your_label")` 在您的追踪中创建自定义注释。这使得将性能分析器输出与某些代码部分关联起来变得容易得多。

通过系统地应用 `torch.profiler`，您可以获得对 PyTorch 程序执行的透彻了解，从而带来显著的性能提升，并更好地了解模型如何利用系统资源。这项能力在处理更具挑战性的机器学习 (machine learning)任务时非常重要。

## 参考资料

- [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) — PyTorch Documentation (2020)
  Publisher: PyTorch Foundation
  `torch.profiler`的官方文档，详细说明其API、用法和在PyTorch中进行性能分析的功能。
- [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes_optimization.html) — Szymon Migacz (2024)
  本指南提供多种优化PyTorch模型的策略，通过展示如何解决已识别的瓶颈来补充性能分析。
- [Visualizing PyTorch Profiler Output in TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html) — PyTorch Documentation (2024)
  一个实用教程，解释如何利用TensorBoard对性能分析器跟踪进行可视化分析，从而有助于识别瓶颈。
