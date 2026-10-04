---
course: "advanced-tensorflow"
chapter: "high-performance-tensorflow"
lesson: "profiling-tensorboard"
sourceId: 2819
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-2-high-performance-tensorflow/profiling-tensorboard"
title: "使用 TensorBoard Profiler 分析 TensorFlow 代码性能"
description: "了解如何使用 TensorBoard Profiler 识别 TensorFlow 模型和数据管道中的性能瓶颈。"
order: 1
plots: ["plots/2819-0.json"]
sourceHash: "20cd1dc512f155007ed7829a27f817982dd84ad9d594b80bdbacbba5116185a6"
sourceCorrections: []
---

在优化之前，您必须先进行衡量。运行复杂的 TensorFlow 模型却不了解计算时间都花在哪里，就像没有地图地航行。您可能进行凭直觉的改动，但这可能无法解决实际的性能瓶颈，导致白费力气且收效甚微。准确找出这些瓶颈是高效模型训练和推理 (inference)的第一步，也是重要的一步。

**TensorBoard Profiler** 是集成在 TensorBoard 中的一个重要工具，用于分析您的 TensorFlow 代码性能。它捕获关于在 CPU、GPU 或 TPU 上运行的操作的执行时间与资源消耗的详细信息，提供对您的训练或推理过程各个方面的了解。

### 启用性能分析器

有几种方法可以收集性能分析数据。在使用 Keras 训练模型时，最常用的方法是使用 `tf.keras.callbacks.TensorBoard` 回调。您需要使用 `profile_batch` 参数 (parameter)来指定要进行性能分析的批次。

```python
import tensorflow as tf
import datetime

# 定义您的模型和数据集（假设已在其他地方定义）
# model = ...
# train_dataset = ...
# val_dataset = ...

# 定义 Keras TensorBoard 回调。
log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # 可选：记录直方图
    profile_batch='100,120' # 对批次 100 到 119 进行性能分析
    # 另一种选择：profile_batch=100 # 仅对批次 100 进行性能分析
    # 另一种选择：profile_batch=(100, 120) # 与 '100,120' 相同
)

model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback]
)
```

在此示例中，性能分析器将在训练过程中第 100 到第 120 个批次之间（不包含第 120 个批次）激活。选择在初始步骤之后的一段范围通常很有用，因为最初的几个批次可能涉及一次性设置成本（如函数追踪或资源分配），这些不代表典型的步骤性能。

为了进行更精细的控制，尤其是在 `model.fit` 之外（例如，对自定义训练循环或推理 (inference)函数进行性能分析），您可以在 `with` 块中使用 `tf.profiler.experimental.start` 和 `tf.profiler.experimental.stop` API：

```python
import tensorflow as tf

log_dir = "logs/profile_custom/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tf.profiler.experimental.start(log_dir)

# --- 您要进行性能分析的代码段 ---
# 示例：执行一些训练步骤或推理调用
# for step, batch in enumerate(train_dataset):
#     if step >= 5 and step < 10: # 对步骤 5 到 9 进行性能分析
#         # 执行您的计算（例如，training_step(batch)）
#         pass
#     elif step >= 10:
#         break
# --- 性能分析代码段结束 ---

tf.profiler.experimental.stop()
```

一旦您在 `log_dir` 中收集了性能分析数据，从您的终端启动 TensorBoard：

```bash
tensorboard --logdir logs/profile
```

访问提供的 URL（通常是 `http://localhost:6006`），并在 TensorBoard UI 的下拉菜单中选择“Profile”（性能分析）。

### 使用性能分析器 UI 分析性能

TensorBoard Profiler 提供多种工具，帮助您了解性能：

1. **概览页：** 这是您的起始点。它提供在性能分析期间的性能高级概述。

   - **步长耗时图：** 显示在已分析批次中步长持续时间的变化。高波动可能表示存在问题。
   - **性能摘要：** 概述了时间花在哪里（例如，内核启动、主机计算、设备计算、输入）。它通常会提供具体的建议，比如，如果输入管道被识别为瓶颈，则会建议优化输入管道。
   - **建议：** TensorBoard 会尝试自动检测常见的性能问题并提出可能的解决方案（例如，“您的程序受限于输入”、“考虑使用混合精度”）。
2. **输入管道分析器：** 如果概览页提示输入管道问题，此工具会提供详细分类。

   - 它分析您的 `tf.data` 管道的性能，显示在不同阶段花费的时间（例如，读取文件、数据预处理、批处理）。
   - 有助于识别诸如以下瓶颈：从磁盘读取数据慢、预取不足、或在 CPU 上运行的计算密集型预处理步骤。查找执行时间较长的阶段，或设备（GPU/TPU）等待数据的步骤。
3. **TensorFlow 统计：** 此工具显示单个 TensorFlow 操作 (Op) 在主机 (CPU) 或设备 (GPU/TPU) 上的执行时间。

   - 您可以按自用时间（操作自身所花费的时间，不包括对其他操作的调用）或总时间（包括被调用的操作）对操作进行排序。
   - 有助于找出占用大量计算时间的特定操作。这可能指导关于使用 XLA 的决策，或研究是否存在数学上等效但速度更快的操作。
4. **GPU 内核统计：** 如果您使用 GPU，此视图提供关于 TensorFlow 操作启动的 CUDA 内核的详细统计信息。

   - 显示内核持续时间、占用率，以及关于 Tensor Core 使用的信息（与混合精度相关）。
   - 有助于确定 GPU 资源是否被有效使用，或特定内核是否异常缓慢。在此处也可识别出高内核启动开销。
5. **追踪查看器：** 这无疑是最强大的工具，尽管也最复杂。它提供时间线可视化，显示操作在不同线程和设备（CPU、GPU 流）上的执行情况。

   - 水平条表示操作或活动的持续时间。
   - 设备时间线上的操作之间存在间隔通常表明设备处于空闲状态，经常等待来自主机 CPU 的数据（输入瓶颈）。
   - 您可以放大查看事件的精确顺序和时间，有助于理解依赖关系和并行性。在此处，识别长时间运行的操作或 CPU 调度与 GPU 执行之间的延迟是常见情况。

### 解读性能分析结果

您的目标是找出模型性能的主要限制因素。常见情况包括：

- **输入受限：** 概览页显示输入管道耗时较高，或追踪查看器显示 GPU 时间线上存在显著间隔（空闲时间），而与 `tf.data` 相关的 CPU 线程正在活跃运行。这意味着您的 GPU 经常在等待数据。请专注于优化您的 `tf.data` 管道（本章稍后会介绍）。
- **CPU 受限（主机受限）：** GPU 利用率较低，但 CPU 却非常忙于执行 TensorFlow 操作（在 TensorFlow 统计或追踪查看器中的 CPU 线程上可见）。如果本应在 GPU 上运行的复杂计算却在 CPU 上运行，或者存在过多的 Python 开销，就可能发生这种情况。考虑确保操作放置在 GPU 上、优化基于 CPU 的操作，或减少 `tf.function` 内的 Python 逻辑。
- **GPU 受限（设备受限）：** CPU 和 GPU 利用率都可能很高，但步长耗时主要由 GPU 计算占据（在 GPU 内核统计或追踪查看器中的设备流中可见）。这通常是大型模型期望的状态，但仍有优化的空间。查找运行缓慢的 GPU 内核、使用混合精度的机会，或启用 XLA 编译（本章稍后会介绍）。



![示例步长耗时分解 (%)](plots/2819-0.json)



> 每次训练步骤耗时的示例分布。在此示例中，设备计算 (GPU) 占用时间最多，但输入管道也占用大量时间，这表明这两个方面都存在潜在瓶颈。

系统地使用 TensorBoard Profiler 可让您从猜测转变为数据驱动的优化。通过识别程序时间花费所在，您可以有效地应用本章其余部分讨论的性能提升技术，例如优化输入管道、采用混合精度以及启用 XLA。

## 参考资料

- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2024)
  Journal: tensorflow.org
  官方指南，提供了使用 tf.data API 优化输入管道的技术和最佳实践。
- [Optimize TensorFlow performance](https://www.tensorflow.org/guide/profiler) — TensorFlow Authors (2020)
  Journal: tensorflow.org
  关于各种 TensorFlow 性能优化技术的综合官方指南，包括混合精度和 XLA。
